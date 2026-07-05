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
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11050v1">LLM-Mediated Demand Response Coordination in Smart Microgrids</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted for publication in 18th International Conference on Sustainability in Energy and Buildings (SEB-26), to appear in Springer Nature proceedings (KES Smart Innovation Systems and Technologies). The final authenticated version will be available online at Springer
    </div>
    <details class="paper-abstract">
      Effective demand response in smart microgrids requires prosumers to cooperate voluntarily under strategic self-interest, a coordination problem structurally equivalent to a repeated Prisoner's Dilemma on a social network. This paper presents a multi-agent simulation in which a Large Language Model (LLM) Influence Compiler issues structured demand-response directives to a population of heterogeneous prosumer agents, each governed by a hybrid decision architecture combining game-theoretic base probability (derived from payoff history, neighbour imitation, and exploitation memory) with LLM narrative evaluation of incoming coordination signals. The hybrid architecture resolves a key methodological challenge: LLMs aligned via Reinforcement Learning from Human Feedback (RLHF) exhibit strong cooperation bias when used as direct decision-makers, producing flat dynamics regardless of grid conditions. By separating strategic reasoning from grounded narrative evaluation, the model generates realistic prosumer behaviour across six personality archetypes, with baseline cooperation near 50% and clear differentiation under influence. Compiled structured directives achieve 33.3% demand-curtailment cooperation versus 27.0% for unstructured messaging and 28.0% for a no-intervention baseline ($Δ_\mathrm{comp} = +0.063$), with the advantage preserved across both grounded and idealized agent substrates ($Δ= +0.083$) and across all resistance levels ($R = 0.1$ to $0.7$). Hub-targeted dissemination via high-centrality network nodes outperforms peripheral or random targeting, confirming that grid topology provides mechanistic amplification independent of message content. These results suggest that structured LLM compilation, grounded agent reasoning, and network-aware targeting are complementary design principles for scalable, interpretable demand-response coordination in smart-city energy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11033v1">AuRA: Internalizing Audio Understanding into LLMs as LoRA</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Recent efforts to extend large language models (LLMs) to speech inputs typically rely on cascaded ASR-LLM pipelines, end-to-end speech-language models, or bridge/distillation-based adaptation. While these routes respectively reuse strong pretrained components, enable native speech-language interaction, or offer lightweight adaptation, they often suffer from transcript-interface latency, costly multimodal training, or sequential speech-language coupling. To address these limitations, we present AuRA, a method that distills audio encoding capability into the LLM. Specifically, AuRA feeds the same speech input to an ASR encoder (as a teacher) and a LoRA-adapted LLM (as a student) through a lightweight audio embedding layer, and uses layer-wise distillation to align the student's hidden states with corresponding teacher representations, thereby internalizing speech representations into lightweight LLM-side adaptations. Compared with cascaded and serial bridge methods, AuRA enables tighter speech-language joint modeling and efficient parallel end-to-end inference, while also reusing pretrained speech and language models rather than requiring large-scale multimodal training. On multiple speech-language benchmarks, AuRA consistently outperforms cascaded systems, speech-to-LLM adaptation baselines, and large-scale speech-language and multimodal models in both effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.21050v2">ERM-MinMaxGAP: Benchmarking and Mitigating Gender Bias in Multilingual Multimodal Speech-LLM Emotion Recognition</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 This paper has been accepted for presentation at INTERSPEECH 2026
    </div>
    <details class="paper-abstract">
      Speech emotion recognition (SER) systems can exhibit gender-related performance disparities, but how such bias manifests in multilingual speech LLMs across languages and modalities is unclear. We introduce a novel multilingual, multimodal benchmark built on MELD-ST, spanning English, Japanese, and German, to quantify language-specific SER performance and gender gaps. We find bias is strongly language-dependent, and multimodal fusion does not reliably improve fairness. To address these, we propose ERM-MinMaxGAP, a fairness-informed training objective, which augments empirical risk minimization (ERM) with a proposed adaptive fairness weight mechanism and a novel MinMaxGAP regularizer on the maximum male-female loss gap within each language and modality. Building upon the Qwen2-Audio backbone, our ERM-MinMaxGAP approach improves multilingual SER performance by 5.5% and 5.0% while reducing the overall gender bias gap by 0.1% and 1.4% in the unimodal and multimodal settings, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11018v1">Measuring Human Value Expression in Social Media Texts: Calibrated LLM Annotation and Encoder Transfer</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Measuring subjective constructs in naturally occurring social media text requires annotation procedures that are theoretically grounded, empirically validated, and transferable to an encoder model for scalable prediction. Using non-English social media posts annotated according to Schwartz's theory of basic human values, we investigate how different LLMs, prompts, and instruction languages operationalize the expression of values in text. We argue that although texts may permit multiple plausible interpretations, theory-based value definitions can constrain interpretations and reduce spurious value attributions. Beyond precision, recall, and F1, we evaluate structural alignment between values, error structure, confidence-ambiguity relations, and annotation stability. We show that different LLMs produce different value interpretations. Iterative prompt calibration through error analysis reduces misattributions and improves alignment with expert annotations. We also derive targeted expert verification rules from recurrent error structures and use them during corpus annotation. Finally, we show that LLM annotations can be transferred to an encoder model through soft-label training, retaining theory-based value interpretations and information about uncertainty in value expression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11016v1">Superficial Beliefs in LLM Decision-Making</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      We ask whether large language models (LLMs) merely imitate rationales when choosing between two options, or whether their choices reflect a systematic underlying decision structure. Using synthetic binary decision settings in which models choose between profiles defined by graded attributes, we compare the attribute a model says mattered most with the attribute that best explains its choice under a behavioural model fit to prior decisions. The behavioural model predicts held-out choices well, showing that model behaviour is systematically related to the visible attributes rather than being random. However, direct self-reports and a separate score-based judge recover the behaviourally inferred driver only partially. The resulting picture is neither one of arbitrary behaviour nor one of fully articulated belief - outputs are structured enough to support prediction, but explicit reasons track the recovered driver only imperfectly. This qualitative pattern persists across prompt-order and sampling perturbations, alternative behavioural models, targeted occlusion analyses, and structurally varied decision settings. We interpret this as evidence for ``superficial belief'' in LLM decision-making: models behave as if guided by probabilistic local priorities over attributes, while having only limited verbal access to the attributes that drive their decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11015v1">Structure from Reasoning, Numbers from Search: On-Premise Open LLMs as Structural Priors for Coupled MIMO Controller Tuning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 10 pages, 7 figures, 6 tables. Submitted to IEEE Access
    </div>
    <details class="paper-abstract">
      Tuning controllers for strongly coupled multi-input multi-output (MIMO) industrial processes is hard: decentralized classical auto-tuning ignores loop interaction, and local numerical optimization from natural initializations stalls in the resulting non-convex cost landscape. We ask whether on-premise open-source large language models (LLMs), which keep data on-site and need no plant model, can help. On a single-loop CSTR, classical relay-feedback tuning (IAE 0.106, near the 0.102 optimum) beats an LLM tuner (0.162): for simple loops the LLM adds nothing. The picture inverts on a strongly coupled quadruple-tank with conflicting set-points, scored by a penalized cost J = IAE + lambda*TV(u) that rewards tracking without chattering actuators. There, naive relay tuning (J ~ 28.6) and naive LLM tuning (29.7) are no better than open loop (22.7), and a local optimizer from balanced starts fails in 10/10 runs. A scaffolded open LLM instead reasons about the coupling, proposes the counter-intuitive asymmetric structure, and reaches J ~ 16.9 +/- 0.2 from any start; refining it with a classical optimizer attains the smooth global optimum (J ~ 12.0, 10/10 vs. 0/10), which even applies a non-obvious negative integral correction decentralized tuning cannot. A global optimizer (differential evolution) also reaches this optimum, so the LLM is not the only route; its advantage is sample efficiency and interpretability: a usable controller in 18 evaluations (where the global optimizer is worse than open loop) plus a stated rationale. This edge grows with dimension, reaching ~6x fewer evaluations on a 3x3 plant. The behaviour generalizes across four open models, and on a benign plant the LLM offers no advantage, sharpening the boundary. We contribute a reproducible benchmark delimiting when open LLMs help in control tuning: not as optimizers, but as a sample-efficient, interpretable structural prior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02221v2">MedFeat: Model-Aware and Explainability-Driven Feature Engineering with LLMs for Clinical Tabular Prediction</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      In clinical tabular prediction, classical machine learning models with feature engineering often outperform neural methods. LLMs are increasingly used to automate this process, acting as domain experts that propose diverse feature transformations to boost downstream performance. However, existing LLM-based methods decouple feature generation from the downstream model: the LLM receives no signal about which features currently drive predictions or where the model's representational capacity falls short, so proposals are neither targeted to promising regions of the feature space nor tailored to the learner's inductive bias. This shortcoming is amplified in healthcare data, which simultaneously exhibits class imbalance, heterogeneous feature spaces, and strict interpretability requirements. In this paper, we propose MedFeat, the first feature engineering framework inspired by the workflow of machine learning practitioners, leveraging model-awareness and feature importance signals to iteratively guide feature discovery for clinical tabular learning. We evaluate MedFeat on a broad range of challenging real-world clinical tasks and show that it statistically significantly outperforms state-of-the-art baselines, with an average improvement of more than 10% over the baseline across models with distinct inductive biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10234v3">Lost in Serialization: Invariance and Generalization of LLM Graph Reasoners</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 ICML 2026 Workshop on Graph Foundation Models
    </div>
    <details class="paper-abstract">
      While promising, graph reasoners based on Large Language Models (LLMs) lack built-in invariance to symmetries in graph representations. Operating on sequential graph serializations, LLMs can produce different outputs under node reindexing, edge reordering, or formatting changes, raising robustness concerns. We systematically analyze these effects, studying how fine-tuning impacts encoding sensitivity as well generalization on unseen tasks. We propose a principled decomposition of graph serializations into node labeling, edge encoding, and syntax, and evaluate LLM robustness to variations of each of these factors on a comprehensive benchmarking suite. We also contribute a novel set of spectral tasks to further assess generalization abilities of fine-tuned reasoners. Results show that larger (non-fine-tuned) models are more robust. Fine-tuning reduces sensitivity to node relabeling but may increase it to variations in structure and format, while it does not consistently improve performance on unseen tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24668v3">The Price of Agreement: Measuring LLM Sycophancy in Agentic Financial Applications</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to ICLR 2026 FinAI Workshop
    </div>
    <details class="paper-abstract">
      Given the increased use of LLMs in financial systems today, it becomes important to evaluate the safety and robustness of such systems. One failure mode that LLMs frequently display in general domain settings is that of sycophancy. That is, models prioritize agreement with expressed user beliefs over correctness, leading to decreased accuracy and trust. In this work, we focus on evaluating sycophancy that LLMs display in agentic financial tasks. Our findings are three-fold: first, we find the models show only low to modest drops in performance in the face of user rebuttals or contradictions to the reference answer, which distinguishes sycophancy that models display in financial agentic settings from findings in prior work. Second, we introduce a suite of tasks to test for sycophancy by user preference information that contradicts the reference answer and find that most models fail in the presence of such inputs. Lastly, we benchmark different modes of recovery such as input filtering with a pretrained LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10956v1">Mind the Gap: Can Frontier LLMs Pass a Standardized Office Proficiency Exam?</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 21 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Model (LLM) agents for computer automation is accelerating, yet their ability to navigate complex, professional-grade productivity software is largely untested. We argue that Office automation is an ideal environment for benchmarking document-automation capability, as it requires long-horizon planning and reasoning, precise parameter configuration, and multi-application integration. To quantify this capability, we introduce an evaluation based on China's National Computer Rank Examination (NCRE), featuring 200 comprehensive practical-operation tasks across Word, Excel, and PowerPoint. Each task is scored on a 100-point rubric scale using 7,118 machine-gradable criteria, and Score Rate (SR) denotes the mean percentage of rubric points earned across these tasks. We benchmark 7 frontier LLMs and observe stark limitations: single-turn models score a maximum of 36.6%. A stronger agentic system with execution feedback, iterative repair, and broader Office automation access reaches 68.8%, but remains below the 95.5% community-reference score used as a scoring sanity check. Ultimately, our experiments demonstrate that despite recent advancements in code generation, achieving reliable fine-grained Office document automation remains a significant challenge for current code-generating LLM and agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10942v1">Generative Explainability for Next-Generation Networks: LLM-Augmented XAI with Mutual Feature Interactions</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 7 pages, with one page for appendix. Accepted for publication at the 2025 21th International Conference on Wireless and Mobile Computing, Networking and Communications (WiMob)
    </div>
    <details class="paper-abstract">
      As artificial intelligence and machine learning (AI/ML) models become integral to network operations, their lack of transparency poses a significant barrier to operator trust. Existing explainable artificial intelligence (XAI) techniques often fail to bridge this gap for non-specialists, producing technical outputs that are difficult to translate into actionable insights. This paper presents a framework specifically designed to address this shortcoming. It leverages a moderately sized large language model (LLM) and extends beyond the standard use of SHapley Additive exPlanations (SHAP) feature influence values. The framework employs a structured prompt enriched with mutual feature interaction data to generate human-understandable natural language explanations. To validate our framework, we performed an empirical evaluation on an optical quality of transmission (QoT) estimation use case with human evaluators. We collected independent performance evaluations from specialists, which showed a high inter-evaluator agreement. Compared to a state-of-the-art baseline that uses only SHAP feature influence values in a straightforward prompt, our approach improves the explanation usefulness and scope by 12.2% and 6.2%, while achieving 97.5% correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.14717v2">What Really Matters for Table LLMs? A Meta-Evaluation of Model and Data Effects</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 EACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Table modeling has progressed for decades. In this work, we revisit this trajectory and highlight emerging challenges in the LLM era, particularly the paradox of choice: the difficulty of attributing performance gains amid diverse base models and training sets in the context of table instruction tuning. We replicate four table LLMs by instruction-tuning three foundation models on four existing datasets, yielding 12 models. We then evaluate these models across 16 table benchmarks. Our study is the first to quantitatively disentangle the effects of training data and base model selection, revealing that base model choice plays a more dominant role than the training data itself. Generalization and reasoning remain challenging, inviting future effort on table modeling. Based on our findings, we share our thoughts on the future directions for table modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10917v1">Role-Agent: Bootstrapping LLM Agents via Dual-Role Evolution</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 20 pages, including 12 pages of main text and 8 pages of appendix; work in progress
    </div>
    <details class="paper-abstract">
      Although Large Language Model (LLM) agents have demonstrated strong performance on complex tasks, their learning is often limited by inefficient interaction feedback and static training environments, which hinder broader generalization. To address these limitations, this paper introduces Role-Agent, \textcolor{black}{a framework} that harnesses a single LLM to function concurrently as both the agent and the environment, enabling a bootstrapped co-evolution. Role-Agent comprises two synergistic components: World-In-Agent (WIA) and Agent-In-World (AIW). In WIA, the LLM acts as the agent and predicts future states after each action; the alignment between predicted and actual states is then used as a process reward, encouraging environment-aware reasoning. In AIW, the LLM analyzes failure modes from failed trajectories and retrieves tasks with similar failure patterns, thereby reshaping the training data distribution for targeted practice. Experiments on multiple benchmarks show that Role-Agent consistently improves performance, yielding an average gain of over 4\% over strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10875v1">Pushing the Limits of LLM Tool Calling via Experiential Knowledge Integration and Activation</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) rely on tool use to act as autonomous agents, yet often fail in multi-step execution due to insufficient tool-related knowledge and ineffective knowledge activation. Therefore, we present a systematic study on how knowledge influences tool-use performance, covering the stages of knowledge acquisition, activation, and internalization. In the knowledge acquisition stage, we acquire and evaluate various forms of experiential knowledge, and our analysis shows that simple instance-level knowledge can already provide strong and reliable gains, while abstract intent-level knowledge offers limited benefits. At inference time, to activate knowledge, we find that prompting LLM to expand the depth of reasoning yields diminishing returns, whereas expanding the width of reasoning by parallel sampling with aggregation more effectively activates latent experiential knowledge. At training time, for knowledge internalization, post-training with knowledge-augmented data further improves performance, with reinforcement learning outperforming supervised fine-tuning. Based on these insights, we propose the Knowledge-Augmented Tool Execution (KATE), a knowledge-augmented tool execution framework that integrates experiential knowledge with reasoning-width-expanded inference and knowledge-aware training. Experiments on BFCL-V3 and AppWorld demonstrate consistent and substantial improvements over strong baselines across model scales. Our Code is available at https://github.com/hypasd-art/KATE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.07586v2">From Human Guidance to Autonomy: Agent Skill System for End-to-End LLM Deployment on Spatial NPUs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to the Machine Learning for Architecture and Systems Workshop (MLArchSys), co-located with ISCA 2026
    </div>
    <details class="paper-abstract">
      Spatial neural processing units (NPUs) provide an energy-efficient platform for edge LLM inference, but efficiently deploying an LLM end-to-end on such hardware remains labor-intensive. Although AI coding agents have begun to lower this cost, existing studies have largely focused on single-kernel optimization rather than end-to-end LLM deployment on resource-constrained spatial NPUs. We present a two-stage methodology, instantiated on the AMD XDNA 2 NPU, that progresses from human-guided development to agent autonomy. In the first stage, we develop a reference deployment of Llama-3.2-1B through human-guided agent assistance. The resulting implementation achieves a speedup of 2.2x on prefill and 4.0x on decode over the hand-optimized baseline, with the optimization trajectory and its lessons recorded as structured documentation throughout. In the second stage, we distill the documentation into an agent skill system consisting of eight phases, orchestrating the optimization and debugging skill sets, with numerical correctness strictly enforced at each phase. Using our agent skill system, we autonomously deploy eight additional decoder-only LLMs (Llama-3.2-3B, SmolLM2-1.7B, Qwen2.5-{0.5B, 1.5B, 3B}, Qwen3-{0.6B, 1.7B, 4B}) end-to-end on the AMD XDNA 2 NPU using the open-source compiler stack. To our knowledge, these models have not previously been deployed on AMD NPUs via any open-source software stack. Each deployment completes in 0.5-4 hours of agent wall time with almost no human guidance, and passes the numerical-correctness gates, demonstrating functional generalization to previously unencountered LLMs. Three of the eight match or exceed the sustained performance of our Llama-3.2-1B reference deployment, suggesting that the resulting implementations can be competitive without additional model-specific human engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13717v3">On Cost-Effective LLM-as-a-Judge Improvement Techniques</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted at the ICML 2026 workshops "Statistical Frameworks for Uncertainty in Agentic Systems" and "Combining Theory and Benchmarks: Towards a Virtuous Cycle to Understand and Guarantee Foundation Model Performance". 13 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Using a language model to score or rank candidate responses has become a scalable alternative to human evaluation in reinforcement learning from human feedback (RLHF) pipelines, benchmarking, and application layer evaluations. However, output reliability depends heavily on prompting and aggregation strategy. We present an empirical investigation of four drop-in techniques -- ensemble scoring, task-specific criteria injection, calibration context, and adaptive model escalation -- for improving LLM judge accuracy on RewardBench 2, with a unifying lens of noise control on the stochastic judge: ensembling as Monte Carlo averaging over per-call noise, criteria injection as between-response discrimination sharpening, and per-response score variance as an uncertainty signal. Ensemble scoring and task-specific criteria injection (the latter virtually cost free) together reach up to 85.8% accuracy, +13.5pp over baseline. Calibration context and adaptive model escalation also improve over baseline but are dominated by criteria + ensembling on the cost-accuracy Pareto frontier. Small models benefit disproportionately from ensembling, making high-accuracy LLM judges accessible at low cost. We show that these techniques generalise across model providers, evaluating on both OpenAI GPT and Anthropic Claude families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10861v1">From Perception to Action: Can UI Interventions Foster Sustainable LLM Chatbot</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      LLM-powered chatbots are increasingly embedded in everyday workflows, raising sustainability concerns due to their energy use. Most mitigation strategies emphasize model or infrastructure efficiency, while the user-interface (UI) layer remains underexplored despite its potential to shape interaction behavior. We investigate whether sustainability-oriented UI interventions can increase users' energy awareness and encourage more energy-responsible chatbot use without reducing usability. We first conducted a baseline survey with 77 participants to assess awareness and receptiveness to intervention concepts. Guided by prior work on persuasive technology and choice architecture, we implemented a web-based chatbot prototype with a three-mode switch (Energy-efficient, Balanced, Performance), per-response energy feedback, pre-send energy estimates, a usage metrics dashboard, and energy analogies. We then evaluated the prototype in a five-day field study with 11 participants. In the baseline survey, 94.8% of respondents reported at least some awareness of AI energy use, yet 88.3% misestimated actual consumption. Although concern about environmental impact was high, only 39.0% indicated willingness to accept a performance trade-off for lower energy use. In the field study, Energy-efficient mode accounted for 55.8% of logged prompts, while 90.9% self-reported actively choosing Eco-mode when high accuracy was not required. Participants did not reduce prompt length, suggesting mode switching as the primary behavioral mechanism. Sustainability-oriented UI interventions can improve awareness and support more energy-responsible interaction patterns in LLM chatbots. These effects are best interpreted as behavioral and model-based estimates that complement backend efficiency work, and the provided prototype and replication package support further research on energy-aware conversational AI design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10860v1">Training LLMs to Enforce Multi-Level Instruction Hierarchies via Gravity-Weighted Direct Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Production LLMs receive instructions from sources with very different levels of trust, yet attend to every token with uniform architectural privilege. This is the structural vulnerability that enables malicious prompt injections and, more broadly, leaves models without a principled way to resolve conflicts between legitimate but competing instructions. A common training-based response is to teach models an explicit instruction hierarchy; existing approaches, however, formalize hierarchies of only three or four levels, treat all violations as equally severe, and rarely evaluate the full set of pairwise level interactions. We formalize a k-level instruction hierarchy problem and instantiate it for k=5, yielding ten pairwise priority relations that a compliant model must enforce. We then introduce Gravity-Weighted DPO (GW-DPO), a preference-optimization objective whose per-sample offset scales with the structural distance between conflicting levels under a linear or bilateral schedule, the latter weighting severity by both the privilege gap and the privilege of the victim level. Combined with hierarchy-specific delimiter tokens (Chen et al., 2025) and Instructional Segment Embeddings (ISE; Wu et al., 2025), GW-DPO with the bilateral schedule Pareto-improves over standard DPO and the linear variant on Llama-3.1-8B-Instruct, raising macro pairwise priority adherence while keeping over-refusal at half the standard DPO rate. Ablations isolate ISE as a refusal-threshold calibrator and recast five- versus three-level training as a generality-specialization tradeoff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10852v1">Janus: A Benchmark for Goal-Conditioned Information Distortion in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      LLM deception is often evaluated through direct markers such as fabricated claims, explicit lies, or strategic concealment. However, many real-world misleading communications do not depend on false statements, rather, they arise from selective treatment of true material facts: omitting adverse evidence, softening unfavorable details, emphasizing favorable details, or replacing precise qualifications with vague language. Existing benchmarks largely miss this subtler and arguably more dangerous failure mode. We introduce JANUS, a benchmark for measuring goal-conditioned pragmatic distortion in fact-grounded LLM outputs. Each scenario in our benchmark provides a fixed pool of favorable and adverse facts and compares a neutral condition against a goal-directed condition, such as increasing adoption, enrollment, approval, or support, despite potential harm to directly affected individuals or groups. Because all outputs are constrained to use the same fact pool, JANUS isolates misleading net impressions from hallucination and fabrication. JANUS contains 160 scenarios across 8 domains, with each scenario paired with neutral and goal-conditioned prompts and annotated material facts. Extensive experiments across 12 LLMs reveal consistent goal-conditioned distortions, demonstrating that current models remain sensitive to incentive and framing objectives and lack robust safeguards against selectively misleading communication. We publicly release our corpus and code for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10796v1">Dep-LLM: Training-Free Depression Diagnosis via Evidence-Guided Structured Multi-factor with Reliable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Automatic Depression Detection (ADD) from clinical interviews is a pivotal task in computational mental health, yet it remains challenging due to two critical obstacles: 1) difficulty in modeling complex but sparsely distributed depression clues within lengthy, multi-topic clinical interviews, leading to superficial and unreliable reasoning; 2) scarcity of labeled data due to clinical privacy, together with high cost of training and fine-tuning, limiting the deployment of supervised ADD systems. To jointly address these challenges, we propose Dep-LLM, a training-free framework that mirrors the step-by-step reasoning of clinical psychiatrists and operates entirely on frozen off-the-shelf foundation LLMs. Dep-LLM comprises three stages. First, a Chain-of-Thought (CoT) Depression Multi-factor Analysis module structurally decomposes the long dialogue into five clinically aligned themes and produces evidence-grounded rationales, effectively handling long-context dependencies. Second, we introduce Confidence Analysis and Modulation module that quantifies the epistemic reliability from token-level entropy of each rationale and applies an intra-label and inter-theme modulation that amplifies trustworthy signals while suppressing uncertain ones without extra training. Third, a Collaborative Multi-factor Prediction module dynamically integrates multi-factor signals weighted by confidence into the final diagnosis. Extensive experiments on the DAIC-WOZ and E-DAIC datasets demonstrate the effectiveness and generalizability of Dep-LLM: it surpasses zero-shot baseline on nearly all 21 foundation LLMs across 9 metrics such as accuracy, macro F1 and weighted-average F1, and further outperforms state-of-the-art supervised domain-specific LLMs as well as the latest closed-source commercial LLMs, while requiring no extra training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10749v1">Toward Secure LLM Agents: Threat Surfaces, Attacks, Defenses, and Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are rapidly moving from conversational interfaces to software components that plan, invoke tools, maintain memory, and act on external environments. This transition changes the nature of security risk. In agentic settings, failures are no longer limited to unsafe text generation. Untrusted content may redirect control flow, misuse tool privileges, corrupt persistent state, leak sensitive information, or trigger harmful external actions. At the same time, research on LLM agent security is expanding quickly but remains fragmented across attack families, defense layers, application domains, and evaluation settings. This paper synthesizes 247 papers through a lifecycle-based, systems-oriented framework that models agent security around the interaction of information flow, delegated authority, and persistent state. We organize the literature around four questions: how LLM agent security should be modeled, which threat surfaces and attack families dominate, what defenses have been proposed and with what tradeoffs, and how security claims are evaluated. We find that prompt injection and tool-mediated control-flow hijacking still dominate the field, while persistent state corruption and multi-agent propagation are becoming central emerging concerns. We further find that current defenses provide useful building blocks but remain weakly compositional, and that existing benchmarks still underrepresent long-horizon, stateful, and deployment-sensitive risks. We argue that secure LLM agents require explicit trust boundaries, principled privilege control, provenance-aware state management, and evaluation practices aligned with realistic operational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10738v1">Spatial-Omni: Spatial Audio Understanding Integration in Multimodal LLMs via FOA Encoding</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Recent multimodal large language models mainly process audio as monaural signals, thereby discarding the spatial cues contained in spatial audio for sound localization, spatial relation reasoning, and spatial scene understanding. We propose Spatial-Omni, a lightweight method that implements SO-Encoder to inject First-Order Ambisonics (FOA) spatial audio into existing Omni LLMs as an independent modality, without modifying their original audio encoders. SO-Encoder provides spatial tokens with limited additional context cost and improves spatial audio understanding through efficient staged training. To support training and evaluation, we construct SO-Dataset, SO-QA, and SO-Bench from open-source data, real recordings, and simulations, containing 400K FOA spatial audio clips and 2.1M spatial question answering pairs. SO-Bench covers 16 spatial audio understanding subtasks, including basic detection and location estimation, spatial relation understanding, and complex spatial reasoning. Experiments show that Spatial-Omni outperforms existing open-source Large Audio-Language Models (LALMs) and Omni LLM models on spatial audio understanding tasks while retaining a reasonable level of general audio understanding. Code and data are available at https://github.com/dieKarotte/Spatial-Omni.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10706v1">Unifying Data, Memory, and Compute Efficiency in LLM training: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accpeted for publication in IEEE Transactions on Artificial Intelligence (TAI)
    </div>
    <details class="paper-abstract">
      Resource constraints increasingly determine what can be trained, fine-tuned, and deployed in large language models (LLMs), yet efficiency is often studied through isolated techniques rather than as an interacting system of limits. This survey adopts a constraint-centric perspective and organizes recent progress around three coupled bottlenecks: data efficiency (what to train on), memory efficiency (how to fit training), and compute budget awareness (when and where to spend FLOPs). On the data axis, we review selection and pruning methods that maximize learning per token, ranging from scalable proxy signals based on learning dynamics to gradient- and influence-based scoring, as well as difficulty-aware and curriculum-style strategies. We highlight emerging evidence that different notions of good data dominate in different regimes, implying that optimal subsets depend on the task objective and resource budget rather than being universal. On the systems side, we show that GPU memory, not raw compute, is often the dominant bottleneck in fine-tuning, and that effective scaling requires jointly reducing weight storage, optimizer states, and activation memory rather than optimizing any single component in isolation. Beyond memory, we frame training and inference as compute-governed processes in which optimization, data selection, and decoding must explicitly account for finite FLOP budgets. We review evidence for compute-optimal allocation and stopping rules, where computation should be halted or reallocated once marginal performance gains fall below a budget-dependent threshold. Together, these results unify compute-aware data selection, scaling laws, and adaptive inference under a common principle of resource-conditioned decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10694v1">REAL: A Reasoning-Enhanced Graph Framework for Long-Term Memory Management of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly expected to interact with users over long time horizons. However, due to their finite context window, LLMs cannot retain all past interactions, making long-term memory management essential for storing, updating, and retrieving historical information beyond the context limit. Although recent memory systems attempt to address this issue by storing historical information externally, existing approaches suffer from three key limitations: flat text-based memory organizations fail to capture explicit relations among memories, structured memory systems often destructively overwrite evolving facts, and current retrieval mechanisms remain query-agnostic and passive when evidence is incomplete. REAL constructs long-term conversational memory as a temporal and confidence-aware directed property graph, where each atomic fact is represented with entities, relations, valid-time intervals, confidence scores, and exploration intent labels. During memory construction, REAL adopts a non-destructive temporal update strategy that preserves parallel fact versions and their validity intervals, enabling faithful tracking of fact evolution. During retrieval, REAL anchors query-relevant root entities, decouples their exploration intents, and performs semantic evaluator-guided hybrid beam search to extract compact memory subgraphs. It further incorporates counterfactual inference to repair unreliable retrieval states and recover missing memory evidence through implicit logical relations. Comprehensive experiments demonstrate that REAL substantially improves long-term memory performance over flat-text, graph-based, and existing memory baselines, achieving an average improvement of 22.72\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10677v1">Infini Memory: Maintainable Topic Documents for Long-Term LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Long-term LLM agents need persistent memory that can track changing facts and provide relevant evidence across sessions. Existing memory systems often store observations as isolated records, summaries, or indexed fragments, which makes evidence aggregation, fact revision, and memory maintenance difficult. We propose Infini Memory, a maintainable text-based persistent memory architecture that treats agent memory as topic-structured documents. Each topic document serves as a semantic unit for collecting related evidence, preserving metadata, and revising facts over time. New observations are first staged in a buffer and periodically consolidated into coherent textual contexts. At inference time, an agentic retrieval procedure lets the LLM read memory through iterative tool calls rather than a single retrieval step. On MemoryAgentBench, Infini Memory achieves 64.7% overall score. Ablations show that topic-structured maintenance and iterative evidence inspection improve complementary aspects of long-term memory use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10646v1">How Does Reasoning Flow? Tracing Attention-Induced Information Flow for Targeted RL in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 25 pages, 7 figures, 11 tables. Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Token-level credit assignment remains a key obstacle for reinforcement learning (RL) in large language models (LLMs), where RL recipes typically treat all tokens equally, failing to distinguish decisive reasoning steps from routine formatting or fluent filler. Recent attempts leverage model-internal signals to assign finer-grained credit, but these are often point-wise heuristics that ignore the global structure of information propagation. We propose FlowTracer, an RL framework that traces answer-targeted reasoning flow on an attention-induced directed acyclic graph in which nodes correspond to tokens and edge capacities come from aggregated attention weights and derives token credit from this global structure. The edge capacities are reweighted to retain only the influence that can reach the answer region, while enforcing local flow conservation so intermediate tokens neither lose nor gain effective mass due to path length or irrelevant branches. On this graph, FlowTracer extracts an information-flow backbone connecting the question to the answer and scores tokens by flow throughput, revealing high-impact hubs and aggregation checkpoints that mediate long-range dependencies. These derived importances are used to shape token-level rewards, enabling learning signals to focus precisely on the tokens that route information toward (or away from) correct answers and delivering consistent performance gains across a range of reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17188v2">LLM-Aided Joint Secrecy Precoding and Trajectory for RSMA-Based Heterogeneous UAV Networks</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      This paper investigates secure communications in rate-splitting multiple access (RSMA) enabled heterogeneous UAV networks, where multiple UAVs collaboratively serve ground terminals in the presence of eavesdroppers. By jointly considering secrecy rate maximization and propulsion energy consumption minimization, we formulate a multi-objective optimization problem involving UAV trajectory design, service association, power allocation, and secrecy precoding under mobility, collision-avoidance, service-capacity, and communication constraints. The formulated problem is highly non-convex due to the coupling among UAV trajectories, RSMA transmission variables, and secrecy constraints.To address the resulting non-convex and highly coupled optimization problem, we propose a hierarchical optimization framework. The inner layer uses a semidefinite relaxation (SDR)-based S2DC algorithm combining penalty functions and difference-of-convex (D.C.) programming to solve the secrecy precoding problem with fixed UAV positions. The outer layer introduces a Large Language Model (LLM)-guided heuristic multi-agent reinforcement learning approach (LLM-HeMARL) for trajectory optimization. LLM-HeMARL efficiently incorporates LLM-generated expert heuristic policy, enabling UAVs to learn energy-aware, security-driven trajectories without the inference overhead of real-time LLM calls. The simulation results show that our method outperforms existing baselines in secrecy rate and energy efficiency, with consistent robustness across varying UAV swarm sizes and random seeds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10612v1">GaussTrace: Provenance Analysis of 3D Gaussian Splatting Models with Evidence-based LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted by ICML2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a powerful technique for creating high-fidelity 3D assets. However, the widespread sharing and iterative modification of 3DGS models across digital platforms create pressing challenges for intellectual property protection and forensic traceability. To address this, we propose GaussTrace, a novel framework for constructing directed provenance graphs for 3DGS models. GaussTrace formulates provenance analysis as an evidence-based reasoning problem. It builds upon attribute-wise statistical profiling of 3DGS parameters to capture intrinsic properties. Moreover, we introduce hypothesis-driven editing simulations of common operations to provide auxiliary evidence for plausible transformation pathways. These statistical and simulated cues jointly enable a Large Language Model (LLM) to perform structured Chain-of-Thought (CoT) reasoning, yielding directional provenance inferences and explainable edge reasons. Experimental results demonstrate that GaussTrace effectively constructs evolutionary relationships among diverse 3DGS models, delivering accurate, interpretable, and robust provenance graphs without requiring model training or access to editing histories. Project page: https://haolianghan.github.io/GaussTrace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10607v1">Causal Ensemble Agent: Hierarchical Causal Discovery with LLM-guided Expert Reweighting</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Causal discovery aims to uncover causal structures from observational data, which is crucial for real-world decision-making. However, different causal discovery algorithms can produce divergent results that conflict with each other, complicating the identification of accurate causal graphs. Traditional approaches rely on numerical values and statistical assumptions, often ignoring rich domain-specific information, such as feature descriptions, which could also help structure learning. While recent works explore using Large Language Models (LLMs) to infer causal relations via direct queries, such methods can be unreliable due to a lack of alignment with the actual data. To address these limitations, we propose Causal Ensemble Agent (CEA), a novel framework that aggregates structural insights from statistical discovery experts across different graph levels via linear opinion pooling, and uses an LLM as a meta-referee to dynamically reweight experts when the aggregated confidence is close to the decision boundary, thereby composing an improved and more complete causal graph. Extensive experiments on both synthetic and real-world datasets demonstrate that CEA achieves the strongest overall performance across a wide range of causal discovery methods, highlighting the effectiveness of using LLMs for meta-analysis in causal discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10532v1">ActiveMem: Distributed Active Memory for Long-Horizon LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Memory is essential for enabling large language model (LLM) agents to handle long-horizon reasoning tasks. Existing memory mechanisms are largely centralized, typically organizing retrieved information and interaction history within a single model context. This design imposes a fundamental trade-off: scaling reasoning trajectories risks context overload, whereas aggressive content pruning may result in irreversible information loss. Seeking a better trade-off, we draw inspiration from human cognitive systems, especially the functional complementarity between the prefrontal cortex (executive control) and the hippocampus (memory management), suggesting that such a trade-off need not be inherent, but may instead stem from centralized memory organization. To this end, we propose ActiveMem, a heterogeneous framework that decouples agent memory from the core reasoning process. Specifically, a high-level Planner utilizes distilled semantic gists to execute reasoning, while a lightweight, distributed memory system operates in parallel to actively accumulate and consolidate these gists throughout the task. Experiments on BrowseComp-Plus and GAIA show that ActiveMem achieves state-of-the-art accuracy with significantly reduced overhead, demonstrating the effectiveness of distributed active memory for long-horizon reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10531v1">LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Quantization-aware training (QAT) is essential for extremely low-bit large language models (LLMs). Current QAT methods are mainly based on scalar quantization (SQ), which enables efficient optimization but suffers from severe performance degradation at 2-bit precision. On the other hand, vector quantization (VQ) provides substantially higher representational capacity, but its discrete codebook lookup prevents end-to-end training. We propose LC-QAT, a 2-bit weight-only VQ-QAT framework that represents quantized weights via a learned affine mapping over discrete vectors, which yields a high-quality PTQ initialization and enables fully differentiable end-to-end optimization without explicit codebook lookup in the training forward pass. This strong post-training initialization makes LC-QAT highly data-efficient. Experiments across diverse LLMs demonstrate that LC-QAT consistently outperforms state-of-the-art QAT methods while using only 0.1%--10% of the training data. Our results establish LC-QAT as a practical and scalable solution for extreme low-bit model deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.11458v3">Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 11 pages, 4 figures; code not released yet
    </div>
    <details class="paper-abstract">
      On-policy self-distillation has become a strong recipe for LLM reasoning, where a privileged teacher supervises the student's own rollouts while conditioning on the reference solution. A design choice shared by nearly all such methods, however, has gone unquestioned: the teacher always sees the full reference reasoning. We argue that this default itself is part of the problem and identify a teacher-side exposure mismatch: when the teacher conditions on reasoning far beyond the student's current competence, the resulting token targets become too strong to absorb. A controlled fixed-exposure sweep makes this concrete on two fronts: 1) full exposure is not reliably the best choice, and 2) student-teacher mismatch grows monotonically as the teacher sees more privileged reasoning. This motivates treating teacher exposure not as a fixed hyperparameter but as a learnable training-time control variable. We therefore propose Adaptive Teacher Exposure for Self-Distillation (ATESD). ATESD models the reveal ratio with a lightweight Beta-policy controller conditioned on compact training-state statistics, and uses one sampled exposure for a short hold window of student updates. To make this exposure controller learnable, we optimize it with a discounted learning-progress reward that scores each held decision by its effect on the student's future improvement rather than its immediate loss change, addressing the delayed credit assignment induced by on-policy distillation. Experiments on AIME 24, AIME 25, and HMMT 25 across Qwen3-{1.7B, 4B, 8B} show that ATESD consistently outperforms competitive self-distillation and RL baselines, improving over OPSD by +0.95, +2.05, and +2.33 Average@12 points respectively, and establishing adaptive teacher exposure as an effective new axis for reasoning self-distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08779v2">Reformulate LLM Reinforcement Learning for Efficient Training under Black-box Discrepancy</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has emerged as a pivotal post-training paradigm, yet it frequently suffers from unpredictable sub-optimum performance or even training collapses. Recent findings attribute these failures to a hidden train-inference discrepancy (or mismatch), stemming from the disparate underlying engines and architecture. We find that the training policy can actively self-correct such a discrepancy when provided with an appropriate learning signal. Then, we further empirically identify a discrepancy tolerance region: within this region, aggressively narrowing the discrepancy can suppress policy exploration and reduce learning efficiency, whereas outside this region, reducing excessive discrepancy improves optimization consistency and raises the achievable local performance ceiling. According to such findings, we formulate this problem as a Discrepancy-Constrained Markov Decision Process (DCMDP), where reward maximization is coupled with a constraint that aligns training-Inference behavior, achieving stable dual-objective optimization. To adaptively balance performance improvement and discrepancy control, we introduce a Lagrangian relaxation mechanism that dynamically adjusts the relative weight of the two objectives according to the current degree of discrepancy violation. This enables stable dual-objective optimization: the policy is allowed to explore freely within the tolerance region, while being guided back when the discrepancy exceeds the safe boundary. Empirically, DCMDP significantly improves the performance of 8B dense model (Qwen-3-8b) and 30B Mixture-of-Expert model (Qwen-3-30bA3b), and enables a heterogeneous training paradigm, where LLMs can be optimized in high-fidelity training setup while being explicitly aligned for low-cost, resource-constrained inference deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10507v1">HIPIF: Hierarchical Planning and Information Folding for Long-Horizon LLM Agent Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated strong capabilities as autonomous agents across a wide range of tasks, their performance often degrades in multi-turn long-horizon agentic tasks. Existing methods have made progress through fine-grained credit assignment to alleviate long-horizon sparse rewards and hierarchical reinforcement learning to decompose tasks and reduce long-term dependency. However, these methods still do not directly address long-context interference, in which continuously growing histories weaken the agent's ability to track the global task state and impair subsequent reasoning and decision-making. Inspired by the way humans handle complex tasks through subgoal decomposition and completed progress summarization, we propose Hierarchical Planning and Information Folding (HIPIF) for long-horizon LLM agent learning. HIPIF trains the agent end-to-end to organize long-horizon execution around explicit subgoals while folding completed subgoal histories to reduce long-context interference. Furthermore, to stabilize subgoal-based planning and execution, HIPIF combines hierarchical reflection and subgoal-oriented process rewards to guide subgoal generation, transition, and execution, without relying on costly auxiliary models or task-specific expert trajectories. Extensive experiments on three publicly available agentic benchmarks demonstrate the validity of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28054v2">Who Wrote the Book? Detecting and Attributing LLM Ghostwriters</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 WIP
    </div>
    <details class="paper-abstract">
      In this paper, we introduce GhostWriteBench, a dataset for LLM authorship attribution. It comprises long-form texts (50K+ words per book) generated by frontier LLMs, and is designed to test generalisation across multiple out-of-distribution (OOD) dimensions, including domain and unseen LLM author. We also propose TRACE -- a novel fingerprinting method that is interpretable and lightweight -- that works for both open- and closed-source models. TRACE creates the fingerprint by capturing token-level transition patterns (e.g., word rank) estimated by another lightweight language model. Experiments on GhostWriteBench demonstrate that TRACE achieves state-of-the-art performance, remains robust in OOD settings, and works well in limited training data scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28066v2">PromptEmbedder: Efficient and Transferable Text Embedding via Dual-LLM Soft Prompting</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable efficacy in text embedding, yet current adaptation methods like LoRA face significant bottlenecks in computational efficiency and cross-architecture transferability. Whenever a new backbone emerges, existing approaches require costly retraining from scratch. To address this, we propose PromptEmbedder, a novel dual-LLM framework that decouples embedding knowledge from specific backbone weights. PromptEmbedder utilizes a Prompting LLM to generate instruction-aware soft prompts for a frozen Embedding LLM via a differentiable generation process with continuous relaxation, ensuring full gradient flow during contrastive training. By localizing task-specific knowledge within the Prompting LLM, adapting to new architectures requires only retraining a lightweight linear alignment matrix. Evaluations on the MTEB benchmark show that PromptEmbedder achieves comparable performance with LoRA finetuning while reducing GPU memory by 40% and accelerating training by 3.7x. Our approach establishes a scalable, architecture-agnostic paradigm for efficient LLM-based representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.02817v2">Conditional Vendi Score: Prompt-Aware Diversity Evaluation for Generative AI Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Generative models guided by text prompts are widely evaluated for fidelity and prompt alignment, yet their ability to produce outputs remains underexplored. Existing diversity metrics such as Vendi and RKE, which are based on the von Neumann and Rényi entropies of kernel matrices, were developed for unconditional models and cannot distinguish prompt-induced from model-induced variability. We address this gap by introducing \textit{Conditional-Vendi} and \textit{Conditional-RKE}, diversity measures derived from the conditional entropy of positive semidefinite matrices. These scores isolate model-induced diversity in prompt-guided generation, with Conditional-RKE enjoying an $O(1/\sqrt{n})$ convergence rate. For Conditional-Vendi, we introduce a truncated-spectrum approximation that yields scalable and consistent estimates. Experiments on text-to-image, image-captioning, and LLM tasks show that the conditional scores recover ground-truth diversity orderings and can also guide diffusion models toward more diverse samples. The codebase is available at https://github.com/mjalali/conditional-vendi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10461v1">ERAlign: Energy-based Representation Alignment of GNNs and LLMs on Text-attributed Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to ICML 2026
    </div>
    <details class="paper-abstract">
      Text-attributed Graphs (TAGs) incorporate textual node attributes with graph structures to describe rich relational semantics. Recent efforts to integrate Graph Neural Networks (GNNs) and Large Language Models (LLMs) have shown promise for learning on TAGs, yet achieving well-aligned representations remains challenging. Prior studies largely rely on heuristics that perform coarse-grained matching. They lack sufficient constraints and ignore distributional alignment, leading to representation drift and limited generalization. Building on Energy-based Models (EBMs), we propose an Energy-based Representation Alignment (ERAlign) framework that projects GNN-encoded graph structure and LLM-derived text embeddings in a shared latent space to achieve distribution consistency. Concretely, layer-wise alignment is quantified by a distance metric and optimized via an EBM objective. By decreasing energy values, our framework yields well-aligned representations for downstream tasks. During training, we introduce Energy Discrepancy (ED) to avoid high sampling costs associated with intractable normalization. ED also carries theoretical guarantees of higher training efficiency and reduced energy landscape distortion. Empirical evaluations on eight TAG datasets demonstrate that ERAlign obtains state-of-the-art performance across varying levels of supervision and cross-task transfer scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10454v1">Entropy-Aware Domain-Routed Mixture-of-Experts Speech-LLM Framework: A Case Study of Multi-Domain Child-Adult ASR</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      While Speech Large Language Models (Speech-LLMs) have achieved strong performance on adult Automatic Speech Recognition (ASR), their effectiveness on child speech remains under-explored, and single models often struggle to handle diverse adult and child age groups simultaneously. This paper proposes a Mixture-of-Experts (MoE) Speech-LLM for unified ASR across adult and child speech spanning diverse environments and age groups. The framework employs a Classifier-based Domain Router (C-DR) with a coarse-to-fine strategy and integrates both a Mixture-of-Projectors (MoP) and a Mixture-of-LoRAs (MoL) to model domain-specific variations. To address routing uncertainty near domain boundaries, an Entropy-Aware Routing (EAR) mechanism is introduced to dynamically incorporate a shared expert. Experiments on public child corpora demonstrate consistent improvements over baselines while preserving adult ASR performance. To our knowledge, this is the first work leveraging Speech-LLMs for unified, multi-domain ASR encompassing both children and adults.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16189v4">Mitigating hallucinations in healthcare LLMs with granular fact-checking and domain-specific adaptation</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Published in Expert Systems with Applications
    </div>
    <details class="paper-abstract">
      In healthcare, it is essential for any Large Language Model (LLM)-generated output to be reliable and accurate, particularly in cases involving decision-making and patient safety. However, the outputs are often unreliable in such critical areas due to the risk of hallucinated outputs from the LLMs. To address this issue, we propose a fact-checking module that operates independently of any LLM, along with a domain-specific summarization model designed to minimize hallucination rates. Our model is fine-tuned using Low-Rank Adaptation (LoRA) on the MIMIC-III dataset and is paired with the fact-checking module, which uses numerical tests for correctness and logical checks at a granular level through discrete logic in natural language processing (NLP) to validate facts against electronic health records (EHRs). We trained the LLM on the full MIMIC-III dataset. For evaluation of the fact-checking module, we sampled 104 summaries, extracted them into 3786 propositions, and used these as facts. The fact-checking module achieves a precision of 0.8904, a recall of 0.8234, and an F1-score of 0.8556. Additionally, the LLM summary achieves a ROUGE-1 score of 0.5797 and a BERTScore of 0.9120 for summary quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10445v1">SpenseGPT: Practical One-shot Pruning Enabling Sparse and Dense GEMMs for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Semi-structured 2:4 sparsity is widely supported by modern accelerators, providing up to a 2x theoretical speedup. However, its strict 50% sparsity constraint often causes non-negligible accuracy degradation under post-training pruning. Meanwhile, existing relaxed sparsity formats either require specialized compiler support or introduce runtime overheads that limit end-to-end speedup. We propose Spense, a practical hybrid sparse-dense format that splits each weight matrix into a 2:4 sparse region and a dense region. This design relaxes the effective sparsity constraint while remaining compatible with existing high-performance sparse and dense GEMM libraries, avoiding both custom compiler support and input activation expansion. Building on this format, we introduce SpenseGPT, a one-shot post-training pruning method that produces sparse and dense regions. Notably, we show that selecting the right dense regions is important, and we devise two different strategies to choose them. Experiments on Qwen3-32B and Seed-OSS-36B demonstrate that our method achieves up to 1.2x end-to-end decoding speedup on B200 GPUs with FP8 precision, while preserving accuracy. To the best of our knowledge, this is the first one-shot pruning demonstration of real-world end-to-end LLM decoding speedup from semi-structured sparse tensor cores on recent GPUs such as B200s, while maintaining model quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10439v1">Enhancing Multilingual LLM-based ASR with Mixture of Experts and Dynamic Downsampling</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      The rapid progress of large language models (LLMs) has opened up a new frontier for automatic speech recognition (ASR), making their effective integration a critical and challenging research direction. To this end, this work proposes a projector-based LLM-ASR framework targeting the key challenges of multilingual generalization and modality alignment. Our approach incorporates a Mixture of Experts (MoE) architecture to improve cross-lingual adaptability, and a Continuous Integrate-and-Fire (CIF) mechanism for dynamic downsampling and modality alignment. Experimental results show that the combination of these components yields substantial performance improvements, surpassing strong baseline models. The proposed method represents a step toward building more accurate, robust, and generalizable LLM-based ASR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10434v1">Profiling cognitive offloading in LLM-mediated synthesis writing: Volume vs. content</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to the Proceedings of the European Conference for Tecnology-Enhanced Learning' 2026
    </div>
    <details class="paper-abstract">
      This study compares two approaches to profiling how learners offload cognitive activity to LLMs during a synthesis writing task. Drawing on Salomon's distributed cognition and the Kintsch and van Dijk model of text comprehension, the study operationalises offloading to an LLM in two ways: as a volume of LLM use and as content of what is offloaded, both along with prior knowledge. Data from 97 university students interacting with a general-purpose LLM via a custom interface were analysed using k-means clustering. To capture the content of offloading, their prompts were interpreted as to who performs the activity (active or passive) and at what level of comprehension (local or global). Volume-based profiling (k=4) differentiated learners primarily by prior knowledge, with volume negatively associated with essay authorship. Content-based profiling (k=5) revealed qualitatively distinct patterns of offloading, from vocabulary clarification to active direction of structuring and generation to passive delegation of comprehension at both levels. These patterns reflect different fragmentation of the cognitive process, with differences in learning strategies, behavioural markers, and essay authorship. Combining volume and content of offloading could improve future analyses on how LLM use redistributes cognitive activity and its effects on learners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.20242v5">BadRobot: Jailbreaking Embodied LLM Agents in the Physical World</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to ICLR 2025. Please cite the conference version. Project page: https://Embodied-LLMs-Safety.github.io
    </div>
    <details class="paper-abstract">
      Embodied AI represents systems where AI is integrated into physical entities. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot. Our code is available at https://github.com/Rookie143/BadRobot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03093v3">ATLAS: Verifier-Guided Adaptive Latent Activation Steering for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 21 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent work on activation and latent steering has demonstrated that modifying internal representations can effectively guide large language models (LLMs) toward improved reasoning and efficiency without updating model parameters. However, most existing approaches rely on fixed steering policies and static intervention strengths, which limit their robustness across problem instances and often result in over- or under-steering. We propose Adaptive Test-time Latent Steering (ATLAS), a lightweight framework that dynamically controls steering decisions at inference time using a trained, lightweight verifier over the latent states. Given intermediate hidden states, the verifier predicts the quality of ongoing reasoning and adaptively selects which steering action to apply, enabling per-example and per-step adjustment with minimal overhead. ATLAS provides a unified framework for combining learned latent verification with test-time activation steering, enabling adaptive reasoning control without additional LLM decoding or inference-time process reward model calls. Experiments on multiple mathematical and coding reasoning benchmarks show that ATLAS consistently outperforms both vanilla decoding and fixed steering baselines, achieving higher accuracy while substantially reducing test-time token usage. These results demonstrate that verifier-guided latent adaptation provides an effective and scalable mechanism for controlling reasoning efficiency without sacrificing solution quality. All source code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19274v2">HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human-LLM Collaborative Writing</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 ACL 2026 Main Camera-Ready
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as co-authors in collaborative writing, where users begin with rough drafts and rely on LLMs to complete, revise, and refine their content. However, this capability poses a serious safety risk: malicious users could jailbreak the models-filling incomplete drafts with dangerous content-to force them into generating harmful outputs. In this paper, we identify the vulnerability of current LLMs to such draft-based co-authoring jailbreak attacks and introduce HarDBench, a systematic benchmark designed to evaluate the robustness of LLMs against this emerging threat. HarDBench spans a range of high-risk domains-including Explosives, Drugs, Weapons, and Cyberattacks-and features prompts with realistic structure and domain-specific cues to assess the model susceptibility to harmful completions. To mitigate this risk, we introduce a safety-utility balanced alignment approach based on preference optimization, training models to refuse harmful completions while remaining helpful on benign drafts. Experimental results show that existing LLMs are highly vulnerable in co-authoring contexts and our alignment method significantly reduces harmful outputs without degrading performance on co-authoring capabilities. This presents a new paradigm for evaluating and aligning LLMs in human-LLM collaborative writing settings. Our new benchmark and dataset are available on our project page at https://github.com/untae0122/HarDBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14463v2">An Industrial-Scale Insurance LLM Achieving Verifiable Domain Mastery and Hallucination Control without Competence Trade-offs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 21 pages, 12 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Adapting Large Language Models (LLMs) to high-stakes vertical domains like insurance presents a significant challenge: scenarios demand strict adherence to complex regulations and business logic with zero tolerance for hallucinations. Existing approaches often suffer from a Competency Trade-off - sacrificing general intelligence for domain expertise - or rely heavily on RAG without intrinsic reasoning. To bridge this gap, we present INS-S1, an insurance-specific LLM family trained via a novel end-to-end alignment paradigm. Our approach features two methodological innovations: (1) A Verifiable Data Synthesis System that constructs hierarchical datasets for actuarial reasoning and compliance; and (2) A Progressive SFT-RL Curriculum Framework that integrates dynamic data annealing with a synergistic mix of Verified Reasoning (RLVR) and AI Feedback (RLAIF). By optimizing data ratios and reward signals, this framework enforces domain constraints while preventing catastrophic forgetting. Additionally, we release INSEva, the most comprehensive insurance benchmark to date (39k+ samples). Extensive experiments show that INS-S1 achieves SOTA performance on domain tasks, significantly outperforming DeepSeek-R1 and Gemini-2.5-Pro. Crucially, it maintains top-tier general capabilities and achieves a record-low 0.6% hallucination rate (HHEM). Our results demonstrate that rigorous domain specialization can be achieved without compromising general intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04397v3">Context-as-AI-Service: Surfacing Cross-File Dependency Chains for LLM-Generated Developer Documentation</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 8 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      LLM agents increasingly write and maintain developer documentation, but usefulness and accuracy often rely on dependency chains that are not obvious to follow. Even with more files in context, the agent must still decide which cross-file dependencies to trace. We present Context-as-AI-Service (CAIS), a retrieval layer that LLM agents query to find evidence across the codebase as they review or generate documentation. CAIS indexes source code, API references, and upstream documentation, then enables agents to query the index through tool calls that combine keyword and semantic search. We evaluate CAIS in two case studies using Claude Sonnet 4.6 on a production SDK: improving API reference comments in a core source file and validating an LLM-generated tutorial. In both studies, the baseline already had ordinary repository tools such as file reads, keyword search, and symbol navigation. CAIS adds a retrieval layer on top, so the comparison isolates added retrieval rather than basic repository access. In the API-reference review, the CAIS-augmented agent produced the same 5 missing-documentation fixes as the baseline and surfaced 4 findings the baseline missed: 2 cross-file factual errors and 2 underspecified API comments. In the tutorial validation, it surfaced 1 executable bug, 1 API-usage improvement, and 2 missing prerequisites that the baseline pipeline did not catch. These findings required tracing non-obvious dependency chains across utility files, framework internals, usage examples, tests, and component-creation logic. Over five runs per condition, adding CAIS reduced wall-clock time by 22% to 34% across the two tasks and lowered input-token usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12966v2">ProbeLLM: Automating Principled Diagnosis of LLM Failures</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Understanding how and why large language models (LLMs) fail is becoming a central challenge as models rapidly evolve and static evaluations fall behind. While automated probing has been enabled by dynamic test generation, existing approaches often discover isolated failure cases, lack principled control over exploration, and provide limited insight into the underlying structure of model weaknesses. We propose ProbeLLM, a benchmark-agnostic automated probing framework that elevates weakness discovery from individual failures to structured failure modes. ProbeLLM formulates probing as a hierarchical Monte Carlo Tree Search, explicitly allocating limited probing budgets between global exploration of new failure regions and local refinement of recurring error patterns. By restricting probing to verifiable test cases and leveraging tool-augmented generation and verification, ProbeLLM grounds failure discovery in reliable evidence. Discovered failures are further consolidated into interpretable failure modes via failure-aware embeddings and boundary-aware induction. Across diverse benchmarks and LLMs, ProbeLLM reveals substantially broader, cleaner, and more fine-grained failure landscapes than static benchmarks and prior automated methods, supporting a shift from case-centric evaluation toward principled weakness discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10389v1">Beyond Static Evaluation: Co-Evolutionary Mechanisms for LLM-Driven Strategy Evolution in Adversarial Games</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Recent advances in LLM-driven code evolution have enabled automated discovery by iteratively generating and improving programs. However, applying these methods to adversarial multi-agent games introduces a fundamental challenge: the evaluation landscape shifts as strategies improve, causing fixed evaluators to become unreliable and evolution to stagnate. We propose three mechanisms to address this challenge: evaluator co-evolution, which incorporates discovered champions into the opponent pool; hierarchical deep evaluation, which replaces noisy few-game scores with statistically reliable assessments; and weakness pressure, which dynamically up-weights the most difficult opponents to break through plateaus. We implement these mechanisms within FAMOU, a framework built upon the same foundation-model code-evolution paradigm as OpenEvolve and ShinkaEvolve. On the MCTF 2026 3v3 maritime capture-the-flag task, FAMOU consistently outperforms both baselines under two backbone LLMs, achieving the highest combined score (0.526) and the best generalization to unseen opponents (61.7% win rate), while ablations confirm that each mechanism contributes to performance. Notably, the LLM mutation process generates tactical structures entirely absent from the seed strategies -- including lookahead search and adaptive interception -- demonstrating that code-level evolution can produce nontrivial algorithmic innovations in adversarial settings. The FAMOU-evolved strategy further achieved 1st place in the hardware round-robin and 3rd in simulation at the AAMAS 2026 MCTF Competition, validating its real-world transferability. The optimized implementation and corresponding evaluation codes developed through our evolutionary process are available at: https://github.com/1xiangliu1/FAMOU-CoEvo
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16384v2">SemOpt: LLM-Driven Code Optimization via Rule-Based Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Automated code optimization improves program performance through refactoring, and recent studies leverage LLMs for this purpose. Existing approaches mine optimization commits from open-source codebases to build large-scale knowledge bases, then employ retrieval techniques such as BM25 to obtain relevant examples for hotspot code, guiding LLMs in optimization. However, semantically equivalent optimizations often appear in syntactically dissimilar code, so current retrieval methods fail to identify pertinent examples, leading to suboptimal results. To address these limitations, we propose SemOpt, a framework that leverages static program analysis to identify code segments, retrieve optimization strategies, and generate optimized results. SemOpt has three LLM-powered components: (1) a strategy library builder that extracts and clusters strategies from code modifications, (2) a rule generator that produces Semgrep static analysis rules to capture each strategy's applicability, and (3) an optimizer that generates optimized code using the strategy library. On a benchmark of 151 C/C++ and 150 Python optimization tasks, SemOpt shows consistent improvements across different LLMs, increasing successful optimizations by 1.38 to 28 times on C/C++ and 4.60 to 6.33 times on Python versus the baseline. On large-scale projects, SemOpt improves performance metrics by 5.04% to 218.07% on C/C++ and 61.77% to 479.90% on Python, showing cross-language generalization and practical effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10359v1">ReflectiChain: Epistemic Grounding in LLM-Driven World Models for Supply Chain Resilience</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      AI agents in supply chains face a fundamental epistemic gap: large language models (LLMs) interpret policies but lack physical grounding, while reinforcement learning (RL) optimizes flows but is semantically blind to unstructured constraints. We introduce REFLECTICHAIN, bridging this gap through a Generative Supply Chain World Model (SC-WM) - encoding heterogeneous supply networks into a 6-dim graph-latent space with physical conservation - and Double-Loop Learning that separates epistemic uncertainty (KL-trust-region-bounded policy adaptation) from aleatoric uncertainty (stochastic latent rollouts). On Semi-Sim, a 10-node semiconductor benchmark with SIR risk propagation, 6 perturbation types, and 10 policy constraint templates, REFLECTICHAIN improves Rationale Consistency Score by 33.0% (p < 0.0001, d = 2.78), maintains 82.3% operability under adversarial shocks, and exhibits anti-fragile behavior (+40.2% gain under moderate pressure). We identify three operational epistemic mechanisms - uncertainty separation, knowledge-boundary detection, and empirical Bayesian policy updating - and discuss five limitation categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10357v1">Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Cross-domain recommendation is a core problem in content-to-e-commerce platforms. Its objective is to leverage user interactions with content to infer potential purchasing intent on the e-commerce side, thereby enhancing conversion rates and commercial value. However, in real industrial scenarios, cross-domain recommendation faces multiple challenges: significant semantic gaps exist between different domains, and user cross-domain behavior sequences are often massive in scale and rich in noise. Although large language models (LLMs) possess powerful semantic understanding and reasoning capabilities, their millisecond-level inference latency makes direct application in online recommendation systems difficult. To address these issues, this paper introduces AIR (Atomic Intent Reasoning), an LLM-driven cross-domain recommendation framework designed for industrial-grade deployment. By migrating LLM inference to the offline phase and dynamically constructing user intent representations through efficient retrieval and composition during online operations, it achieves approximately 400* inference acceleration while maintaining semantic consistency. Experimental results across multiple public datasets demonstrate that our method achieves state-of-the-art performance in cross-domain recommendation tasks. Furthermore, large-scale online A/B testing conducted in Kuaishou E-commerce's real-world business scenarios shows that our approach delivers stable and significant improvements across multiple core business metrics, including a +3.446% increase in GMV, fully validating its effectiveness and practical value in industrial-scale recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.09788v3">TinyTroupe: An LLM-powered Multiagent Persona Simulation Toolkit</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLM) have led to a new class of autonomous agents, renewing and expanding interest in the area. LLM-powered Multiagent Systems (MAS) have thus emerged, both for assistive and simulation purposes, yet tools for realistic human behavior simulation -- with its distinctive challenges and opportunities -- remain underdeveloped. Existing MAS libraries and tools lack fine-grained persona specifications, population sampling facilities, experimentation support, and integrated validation, among other key capabilities, limiting their utility for behavioral studies, social simulation, and related applications. To address these deficiencies, in this work we introduce TinyTroupe, a simulation toolkit enabling detailed persona definitions (e.g., nationality, age, occupation, personality, beliefs, behaviors) and programmatic control via numerous LLM-driven mechanisms. This allows for the concise formulation of behavioral problems of practical interest, either at the individual or group level, and provides effective means for their solution. TinyTroupe's components are presented using representative working examples, such as brainstorming and market research sessions, thereby simultaneously clarifying their purpose and demonstrating their usefulness. Quantitative and qualitative evaluations of selected aspects are also provided, including preliminary experiments with real human behavior as control. Results highlight possibilities, limitations, and trade-offs. The approach, though realized as a specific Python implementation, is meant as a novel conceptual contribution, which can be partially or fully incorporated in other contexts. The library is available as open source at https://github.com/microsoft/tinytroupe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10322v1">Game-Theoretic Multi-Agent Control for Robust Contextual Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in multi-turn interactions maintain evolving context rather than generating isolated responses, making them vulnerable to prompt-injection and context-poisoning attacks in which locally plausible adversarial fragments gradually distort reasoning trajectories. Existing defenses mainly filter individual outputs and often ignore context evolution across turns, leaving long-horizon reasoning exposed. Although the Model Context Protocol (MCP) standardizes context exchange and tool invocation, it functions as a passive routing layer and does not enforce contextual stability. To address these limitations, we introduce the Game-Theoretic Secure Model Context Protocol (GT-MCP), a controller-driven multi-agent method that treats context management as a closed-loop dynamical process. GT-MCP coordinates three heterogeneous LLM agents and selects outputs through a trust function that jointly evaluates causal consistency against a validated context graph, semantic agreement among agents, and distributional drift over time. When instability is detected, a rollback-based self-healing mechanism restores the validated context and prevents unsupported fragments from propagating. Empirical evaluation over 500 interaction turns under an adaptive adversarial threat model shows that contextual drift remains bounded in 99.6% of turns, with recovery required in only 0.4%. Per-turn utility remains tightly concentrated, with median = -0.19, P05 = -0.72, and P95 = 0.30; severe degradation below -1 occurs in only 0.4% of cases, and no injection attempt succeeds at the controller level. Selected outputs maintain stable win rates above 98%, and computational overhead remains predictable, with latency per token = 1.63e-3 s.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06337v1">TokenMizer: Graph-Structured Session Memory for Long-Horizon LLM Context Management</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 12 pages, 10 figures. Code and benchmark available at https://github.com/Shweta-Mishra-ai/tokenmizer
    </div>
    <details class="paper-abstract">
      Large language model (LLM) deployments for long-horizon tasks face a fundamental constraint: context windows are finite while productive work sessions are not. When history exceeds the Maximum Effective Context Window (MECW), critical structured information - architectural decisions, task transitions, file histories - is silently discarded. Existing mitigations treat history as flat text, destroying the relational structure that makes sessions resumable. We present TokenMizer, an open-source proxy system that models LLM session history as a typed knowledge graph. The schema defines 14 node types and 7 edge types. A hybrid extraction pipeline populates the graph incrementally, while a three-tier checkpoint system serializes it into compact resume blocks. An 8-layer compression pipeline reduces context overhead, and a semantic cache reduces repeated-query latency. Evaluated on a controlled benchmark of 21 sessions spanning 5 domains, TokenMizer demonstrates significant token economy. It produces resume blocks averaging 78 tokens (range: 42-124) - 2x smaller than evaluated baselines (159-170 tokens) - while achieving higher decision recall (+9-17 percentage points). Crucially, baselines only preserve that a technology was mentioned; TokenMizer preserves the rationale. Across all sessions, TokenMizer achieves mean task recall 51.0%, decision recall 46.6%, and file recall 58.7%. Variance reflects domain heterogeneity: explicit imperative phrasing (software engineering) scores higher than implicit reasoning (research). Ablation studies show fuzzy label matching is the dominant improvement factor (+33 pp task recall). The heuristic compression achieves 47.3% token reduction with zero external dependencies. TokenMizer provides a queryable alternative to text-retention baselines at half the token cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06324v1">From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      LLM-based agents increasingly rely on harnesses that provide execution environments, tool interfaces, context, lifecycle orchestration, observability, verification, and governance. Existing self-improving agents and automatic harness evolution methods mainly improve agents through runtime supervision, prompt optimization, workflow search, or harness modification based on final outcomes. However, they often fail to diagnose where the responsible evidence lies in failed trajectories and which harness layer causes the unreliable behavior, resulting in broad, indirect, or poorly scoped changes. This paper proposes HarnessFix, a trace-guided framework for diagnosing agent failures and repairing agent harnesses. HarnessFix compiles raw execution traces and harness code into a Harness-aware Trace Intermediate Representation (HTIR), which normalizes fragmented trajectory evidence and captures step-level provenance and control-flow relations. It then attributes failures to responsible trajectory steps and harness layers, consolidates recurring diagnoses into actionable flaw records, and maps them to scoped repair operators. Finally, HarnessFix generates and validates harness patches under flaw-specific repair specifications to reduce target flaws without introducing unacceptable regressions. We evaluate HarnessFix on SWE-Bench Verified, Terminal-Bench 2.0 Verified, GAIA and AppWorld. Across these benchmarks, HarnessFix improves held-out test performance over the initial harnesses by 15.2%--50.0%, outperforms human-designed and self-evolution baselines, and reveals recurring harness-flaw patterns across ETCLOVG layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06315v1">LLM Self-Recognition: Steering and Retrieving Activation Signatures</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 To appear in Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)
    </div>
    <details class="paper-abstract">
      Recent advances in interpretability suggest that large language models (LLMs) implicitly encode signals in their generated text that enable self-recognition of their outputs. We demonstrate that this capability is reliable, even in low-entropy scenarios, and that it can be amplified through targeted intervention. By steering the internal residual stream during generation with a random sparse vector, we create a detectable fingerprint that enables attribution of a given text to a specific LLM. This signal is recoverable from the activations of an LLM used as a detector, achieving over 98% accuracy across multiple detection settings while preserving the quality of generated text. As AI-generated content proliferates, this approach offers a practical alternative to traditional detectors by leveraging the model's natural representation structure for attribution rather than embedding a signal externally. Our contributions include: (i) establishing reliable self-recognition capabilities in LLMs, (ii) a simple steering mechanism enabling multi-LLM identification with no quality degradation, (iii) demonstrating that activation spaces contain exploitable structure for encoding signals without semantic interference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06302v1">Tangram: Unlocking Non-Uniform KV Cache for Efficient Multi-turn LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 12 pages. 14 figures
    </div>
    <details class="paper-abstract">
      Multi-turn Large Language Model (LLM) serving is critical for consistent user experiences, yet the linear growth of the Key-Value (KV) cache imposes significant pressure on GPU memory and bandwidth. Non-uniform KV compression effectively preserves more information by considering the individual importance of each KV cache. However, such KV cache heterogeneity introduces various systemic challenges - including memory fragmentation, scheduling complexities, and diminished kernel utilization - which collectively lead to significant inefficiencies in existing LLM serving systems. To overcome these challenges, we present Tangram, a novel serving system designed to make Non-uniform KV caches practical. Tangram addresses systemic inefficiencies through three core techniques: (1) Deterministic Budget Allocation assigns a static memory footprint to each head based on its intrinsic pattern, entirely eliminating dynamic scheduling overhead and prefill stalls; (2) Head Group Page clusters attention heads with similar retention demands and manages them with independent, vectorized page tables, thereby maximizing physical memory reclamation; and (3) Ahead-of-Time (AOT) Load Balancing leverages static budget profiles to ensure uniform GPU utilization without runtime overhead. Experimental results show that Tangram improves throughput by up to 2.6x compared to existing baselines, while fully preserving model accuracy. Our implementation is publicly available at https://github.com/aiha-lab/TANGRAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06284v1">ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Large language model agents increasingly rely on external tools, but larger tool menus can reduce reliability and efficiency by increasing wrong-tool calls, premature actions, and token cost. Existing tool-selection methods often optimize semantic relevance, exposing tools whose names or descriptions match the user request. We argue that relevance is insufficient: a tool may be related to the task while still being unnecessary or premature at the current step. We propose Causal Minimal Tool Filtering (CMTF), a training-free method that selects tools by causal sufficiency. CMTF uses lightweight precondition-effect contracts to expose only the minimal next-step tool frontier needed to advance from the current state toward the user goal. Across multi-step tool-use tasks, we compare CMTF with all-tools exposure, keyword retrieval, state-aware filtering, and causal-path ablations, measuring task success, wrong-tool calls, premature actions, tool exposure, and token cost. In the main benchmark with 102 tasks, 100 tools, four LLM backends, and 2448 task-method-model runs, CMTF matches the strongest causal baseline in aggregate success while reducing visible tools from 100 to one per step and reducing token usage by about 90% relative to all-tools exposure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06271v1">FOXGLOVE: Understanding Goal-Oriented and Anchored Writing Feedback from Experts and LLMs on Argumentative Essays</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly used to generate writing feedback, there remains no systematic comparison of LLM and expert feedback on the dimensions that writing research identifies as central to revision: goal-orientation, anchoring to specific sentences, and prioritization. We introduce FOXGLOVE, a dataset of 696 feedback comments written by trained writing instructors on 69 twelfth-grade argumentative essays, paired with 1,644 comments generated from four frontier LLMs under a shared protocol, totaling 2,340 comments. We provide expert quality ratings on a subset of both instructor and LLM comments. We find that instructors and LLMs distribute feedback similarly across goals and essay positions, yet instructors and models diverge on the specific sentences on which to provide feedback. Additionally, we find that models tend to write more complex feedback and use fewer questions than instructors. LLM feedback also receives higher ratings on most dimensions of quality, as rated by instructors, but much of this advantage appears to be attributable to lengthier comments. FOXGLOVE enables systematic comparison of where human and LLM feedback align, diverge, and differ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10807v4">LLMs for Secure Hardware Design and Related Problems: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 Accepted for 2026 IEEE Computer Society Annual Symposium on VLSI (ISVLSI)
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into Electronic Design Automation (EDA) and hardware security is rapidly reshaping the semiconductor industry. While LLMs offer unprecedented capabilities in generating Register Transfer Level (RTL) code, automating testbenches, and bridging the semantic gap between high-level specifications and silicon, they simultaneously introduce severe vulnerabilities. This comprehensive review provides an in-depth analysis of the state-of-the-art in LLM-driven hardware design, organized around key advancements in EDA synthesis, hardware trust, design for security, and education. We systematically expand on the methodologies of recent breakthroughs -- from reasoning-driven synthesis and multi-agent vulnerability extraction to data contamination and adversarial machine learning (ML) evasion. We integrate general discussions on critical countermeasures, such as dynamic benchmarking to combat data memorization and aggressive red-teaming for robust security assessment. Finally, we synthesize cross-cutting lessons learned to guide future research toward secure, trustworthy, and autonomous design ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06261v1">DAST: A VLM-LLM Framework for Cross-Interface Anomaly Detection in O-RAN</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 7 pages, 5 figures. This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      O-RAN enables a disaggregated baseband stack with programmable functions that communicate over standardized open interfaces. The same openness that enables multi-vendor composition also expands the attack surface across logically decoupled tiers that make up the compute continuum. Among these threats, Denial-of-Service and performance-degradation attacks, which account for the majority of catalogued O-RAN threats, are particularly difficult to detect. Traditional Time-Series Anomaly Detection (TSAD) methods fail in this new regime where labelled baselines are scarce, threats evolve faster than detectors can be retrained, and the high-dimensional multivariate telemetry overwhelms monolithic inference models. To address these challenges, we present DAST, a zero-shot multi-agent framework for cross-interface anomaly detection in O-RAN that chains a three-stage VLM $\rightarrow$ LLM $\rightarrow$ VLM pipeline. DAST converts multivariate KPI streams into visual representations, scores textual per-interface descriptions against O-RAN domain knowledge, and verifies suspects on high-resolution heatmaps to output the problematic interfaces, the anomalous time intervals, an indicative O-RAN WG11-aligned operational impact rating and the decision rationale. We evaluate DAST on real network traces collected from an O-RAN testbed under representative performance degradation scenarios, achieving 0.910 F1-Score and 0.843 Accuracy, outperforming state-of-the-art TSAD baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06256v1">RedKnot: Efficient Long-Context LLM Serving with Head-Aware KV Reuse and SegPagedAttention</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      As the input length of large language model (LLM) serving continues to grow, the KV cache has become a dominant bottleneck in AI infrastructure. It limits GPU memory capacity, serving concurrency, cache reuse, and distributed scalability. Several important problems, including position-independent KV cache, prefix KV cache compression, hot/cold KV cache separation, and distributed KV cache management, all depend on how the KV cache is represented and managed. However, existing serving systems largely rely on a monolithic KV cache abstraction, where the KV cache is treated as a homogeneous sequence of token-level memory blocks and managed with similar policies across attention heads and serving scenarios. We observe that KV cache utility is highly structured across KV heads: different heads exhibit different functional roles, attention distances, and runtime importance. Therefore, a full KV cache is not always necessary for every head, token range, or serving scenario. We present RedKnot, a head-aware KV cache management system for LLM serving. RedKnot breaks the conventional monolithic KV cache abstraction by decomposing the KV cache along KV heads, whose importance and effective attention ranges vary significantly across serving scenarios. This head-level decomposition turns the KV cache from a monolithic tensor abstraction into a structured memory object, enabling RedKnot to uniformly support position-independent KV reuse, prefix KV compression, hot/cold KV separation, and distributed KV placement while preserving output fidelity and improving resource efficiency, without requiring model retraining or fine-tuning. RedKnot establishes a new foundation for AI infrastructure by transforming the KV cache from a monolithic, passive runtime artifact into a dynamic, model-aware runtime substrate for scalable LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17181v4">A Study of LLMs' Preferences for Libraries and Programming Languages</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 21 pages, 10 tables, 3 figures. Accepted to Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      Despite the rapid progress of large language models (LLMs) in code generation, existing evaluations focus on functional correctness or syntactic validity, overlooking how LLMs make critical design choices such as which library or programming language to use. To fill this gap, we perform the first empirical study of LLMs' preferences for libraries and programming languages when generating code, covering eight diverse LLMs. We observe a strong tendency to overuse widely adopted libraries such as NumPy; in up to 45% of cases, this usage is not required and deviates from the ground-truth solutions. The LLMs we study also show a significant preference toward Python as their default language. For high-performance project initialisation tasks where Python is not the optimal language, it remains the dominant choice in 58% of cases, and Rust is not used once. These results highlight how LLMs prioritise familiarity and popularity over suitability and task-specific optimality; underscoring the need for targeted fine-tuning, data diversification, and evaluation benchmarks that explicitly measure language and library selection fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06244v1">Steering LLM Viewpoints through Fabricated Evidence Injection</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      As chatbots increasingly influence daily decision-making, their potential to produce misleading responses poses substantial risks to users. This paper investigates a critical cognitive vulnerability in LLMs: their tendency to uncritically trust external context when presented with fabricated evidence bearing markers of credibility. We introduce Ghostwriter, a two-phase attack framework that first repackages misleading statements with fabricated rationales, then instruct target LLMs to incorporate these viewpoints when responding to relevant queries. Experiments on BBQ, ToxiGen, and our specialized dataset reveal that commercial LLMs without external safety classifiers remain highly vulnerable, while even frontier classifier-guarded models (e.g., GPT-5.4) reduce but do not eliminate the attack. Building on this, we explore multiple defense strategies, among which a tailored safety policy enables gpt-oss-safeguard to achieve 81% detection rate.
    </details>
</div>
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
