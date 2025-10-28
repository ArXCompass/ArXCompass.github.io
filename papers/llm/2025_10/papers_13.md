# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
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
- Part 13
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05520v1">CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) are confronted with overwhelming information volume when comprehending long-form documents. This challenge raises the imperative of a cohesive memory module, which can elevate vanilla LLMs into autonomous reading agents. Despite the emergence of some heuristic approaches, a systematic design principle remains absent. To fill this void, we draw inspiration from Jean Piaget's Constructivist Theory, illuminating three traits of the agentic memory -- structured schemata, flexible assimilation, and dynamic accommodation. This blueprint forges a clear path toward a more robust and efficient memory system for LLM-based reading comprehension. To this end, we develop CAM, a prototype implementation of Constructivist Agentic Memory that simultaneously embodies the structurality, flexibility, and dynamicity. At its core, CAM is endowed with an incremental overlapping clustering algorithm for structured memory development, supporting both coherent hierarchical summarization and online batch integration. During inference, CAM adaptively explores the memory structure to activate query-relevant information for contextual response, akin to the human associative process. Compared to existing approaches, our design demonstrates dual advantages in both performance and efficiency across diverse long-text reading comprehension tasks, including question answering, query-based summarization, and claim verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13360v2">What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Prompt underspecification is a common challenge when interacting with LLMs. In this paper, we present an in-depth analysis of this problem, showing that while LLMs can often infer unspecified requirements by default (41.1%), such behavior is fragile: Under-specified prompts are 2x as likely to regress across model or prompt changes, sometimes with accuracy drops exceeding 20%. This instability makes it difficult to reliably build LLM applications. Moreover, simply specifying all requirements does not consistently help, as models have limited instruction-following ability and requirements can conflict. Standard prompt optimizers likewise provide little benefit. To address these issues, we propose requirements-aware prompt optimization mechanisms that improve performance by 4.8% on average over baselines. We further advocate for a systematic process of proactive requirements discovery, evaluation, and monitoring to better manage prompt underspecification in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05497v1">Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with Mixture of Experts (MoE) architectures achieve remarkable performance improvements, but their random expert selection mechanism introduces significant data movement overhead that becomes the dominant bottleneck in multi-unit serving systems. To forecast the patterns underlying this data movement, we conduct comprehensive data-movement-centric profiling across three state-of-the-art large-scale MoE models (200B- 671B) using over 24,000 requests spanning diverse workloads. With the resulting 150GB+ trace files, we perform systematic analysis from both temporal and spatial perspectives and distill six key insights to guide the design of diverse future serving systems. Taking wafer-scale GPUs as a case study, we demonstrate that minor architectural modifications leveraging our insights achieve substantial performance gains, delivering 6.3X and 4.0X average speedups on DeepSeek V3 and Qwen3, respectively. Our work provides the first comprehensive data-centric analysis of MoE models at scale. Our profiling traces and analysis results are publicly available at {https://huggingface.co/datasets/core12345/MoE_expert_selection_trace. We will also release our simulation framework shortly to facilitate future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05484v1">Evaluating LLM Safety Across Child Development Stages: A Simulated Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly becoming part of tools used by children; however, existing benchmarks fail to capture how these models manage language, reasoning, and safety needs that are specific to various ages. We present ChildSafe, a benchmark that evaluates LLM safety through simulated child agents that embody four developmental stages. These agents, grounded in developmental psychology, enable a systematic study of child safety without the ethical implications of involving real children. ChildSafe assesses responses across nine safety dimensions (including privacy, misinformation, and emotional support) using age-weighted scoring in both sensitive and neutral contexts. Multi-turn experiments with multiple LLMs uncover consistent vulnerabilities that vary by simulated age, exposing shortcomings in existing alignment practices. By releasing agent templates, evaluation protocols, and an experimental corpus, we provide a reproducible framework for age-aware safety research. We encourage the community to expand this work with real child-centered data and studies, advancing the development of LLMs that are genuinely safe and developmentally aligned.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05480v1">Vul-R2: A Reasoning LLM for Automated Vulnerability Repair</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 13 pages, 8 figures. This paper is accepted by ASE 2025
    </div>
    <details class="paper-abstract">
      The exponential increase in software vulnerabilities has created an urgent need for automatic vulnerability repair (AVR) solutions. Recent research has formulated AVR as a sequence generation problem and has leveraged large language models (LLMs) to address this problem. Typically, these approaches prompt or fine-tune LLMs to generate repairs for vulnerabilities directly. Although these methods show state-of-the-art performance, they face the following challenges: (1) Lack of high-quality, vulnerability-related reasoning data. Current approaches primarily rely on foundation models that mainly encode general programming knowledge. Without vulnerability-related reasoning data, they tend to fail to capture the diverse vulnerability repair patterns. (2) Hard to verify the intermediate vulnerability repair process during LLM training. Existing reinforcement learning methods often leverage intermediate execution feedback from the environment (e.g., sandbox-based execution results) to guide reinforcement learning training. In contrast, the vulnerability repair process generally lacks such intermediate, verifiable feedback, which poses additional challenges for model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18356v2">The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 MRL Workshop at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) still struggle across tasks outside of high-resource languages. In this work, we investigate cross-lingual transfer to lower-resource languages where task-specific post-training data is scarce. Building on prior work, we first validate that the subsets of model parameters that matter most for mathematical reasoning and multilingual capabilities are distinctly non-overlapping. To exploit this implicit separability between task and target language parameterization, we develop and analyze numerous modular frameworks to improve the composition of the two during fine-tuning. These methods generally employ freezing parameters or post hoc model merging to assign math and language improvement to different key parts of the LLM. In the absence of in-language math data, we demonstrate that the modular approaches successfully improve upon baselines across three languages, four models, and two fine-tuning paradigms (full and LoRA). Furthermore, we identify the most consistently successful modular method to be fine-tuning separate language and math experts and model merging via Layer-Swapping, somewhat surprisingly. We offer possible explanations for this result via recent works on the linearity of task vectors. We further explain this by empirically showing that reverting less useful fine-tuning updates after training often outperforms freezing them from the start.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07218v2">LLM Unlearning via Neural Activation Redirection</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      The ability to selectively remove knowledge from LLMs is highly desirable. However, existing methods often struggle with balancing unlearning efficacy and retain model utility, and lack controllability at inference time to emulate base model behavior as if it had never seen the unlearned data. In this paper, we propose LUNAR, a novel unlearning method grounded in the Linear Representation Hypothesis and operates by redirecting the representations of unlearned data to activation regions that expresses its inability to answer. We show that contrastive features are not a prerequisite for effective activation redirection, and LUNAR achieves state-of-the-art unlearning performance and superior controllability. Specifically, LUNAR achieves between 2.9x and 11.7x improvement in the combined unlearning efficacy and model utility score (Deviation Score) across various base models and generates coherent, contextually appropriate responses post-unlearning. Moreover, LUNAR effectively reduces parameter updates to a single down-projection matrix, a novel design that significantly enhances efficiency by 20x and robustness. Finally, we demonstrate that LUNAR is robust to white-box adversarial attacks and versatile in real-world scenarios, including handling sequential unlearning requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01213v2">Show or Tell? Modeling the evolution of request-making in Human-LLM conversations</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Designing user-centered LLM systems requires understanding how people use them, but patterns of user behavior are often masked by the variability of queries. In this work, we introduce a new framework to describe request-making that segments user input into request content, roles assigned, query-specific context, and the remaining task-independent expressions. We apply the workflow to create and analyze a dataset of 211k real-world queries based on WildChat. Compared with similar human-human setups, we find significant differences in the language for request-making in the human-LLM scenario. Further, we introduce a novel and essential perspective of diachronic analyses with user expressions, which reveals fundamental and habitual user-LLM interaction patterns beyond individual task completion. We find that query patterns evolve from early ones emphasizing sole requests to combining more context later on, and individual users explore expression patterns but tend to converge with more experience. From there, we propose to understand communal trends of expressions underlying distinct tasks and discuss the preliminary findings. Finally, we discuss the key implications for user studies, computational pragmatics, and LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06477v1">Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Attention sinks and compression valleys have attracted significant attention as two puzzling phenomena in large language models, but have been studied in isolation. In this work, we present a surprising connection between attention sinks and compression valleys, tracing both to the formation of massive activations in the residual stream. We prove theoretically that massive activations necessarily produce representational compression and establish bounds on the resulting entropy reduction. Through experiments across several models (410M-120B parameters), we confirm that when the beginning-of-sequence token develops extreme activation norms in the middle layers, both compression valleys and attention sinks emerge simultaneously. Targeted ablation studies validate our theoretical predictions. This unified view motivates us to propose the Mix-Compress-Refine theory of information flow, as an attempt to explain how LLMs organize their computation in depth by controlling attention and representational compression via massive activations. Specifically, we posit that Transformer-based LLMs process tokens in three distinct phases: (1) broad mixing in the early layers, (2) compressed computation with limited mixing in the middle layers, and (3) selective refinement in the late layers. Our framework helps explain why embedding tasks perform best at intermediate layers, whereas generation tasks benefit from full-depth processing, clarifying differences in task-dependent representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20309v2">Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Controlling undesirable Large Language Model (LLM) behaviors, such as the generation of unsafe content or failing to adhere to safety guidelines, often relies on costly fine-tuning. Activation steering provides an alternative for inference-time control, but existing methods typically lack fine-grained, adaptive mechanisms. We introduce a novel approach using a lightweight, trainable controller network integrated during inference. This controller network observes specific intermediate LLM activations and predicts both a global scaling factor and layer-specific weights. The predicted global scaling factor and layer-specific weights then dynamically modulate the intensity of a steering patch, derived from a pre-computed "refusal direction" vector, applied across the LLM's layers during generation. Trained on activations from both harmful and benign prompts, our controller learns to discriminatively apply nuanced, layer-aware interventions, activating steering primarily for harmful inputs. Experiments using safety benchmarks like ToxicChat & In-The-Wild Jailbreak Prompts demonstrate that our weighted steering controller significantly increases refusal rates compared to the base LLM, achieving targeted behavioral modification without altering the original model parameters. Our experiments with Llama-3.1-8B, Llama-3.2-1B & Mistral-7B show our approach outperforms existing methods, presenting an efficient and adaptive method for fine-grained control over LLM behavior at inference time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03540v2">Improving Factuality in LLMs via Inference-Time Knowledge Graph Construction</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with producing factually consistent answers due to limitations in their parametric memory. Retrieval-Augmented Generation (RAG) paradigms mitigate this issue by incorporating external knowledge at inference time. However, such methods typically handle knowledge as unstructured text, which reduces retrieval accuracy, hinders compositional reasoning, and amplifies the influence of irrelevant information on the factual consistency of LLM outputs. To overcome these limitations, we propose a novel framework that dynamically constructs and expands knowledge graphs (KGs) during inference, integrating both internal knowledge extracted from LLMs and external knowledge retrieved from external sources. Our method begins by extracting a seed KG from the question via prompting, followed by iterative expansion using the LLM's internal knowledge. The KG is then selectively refined through external retrieval, enhancing factual coverage and correcting inaccuracies. We evaluate our approach on three diverse Factual QA benchmarks, demonstrating consistent gains in factual accuracy over baselines. Our findings reveal that inference-time KG construction is a promising direction for enhancing LLM factuality in a structured, interpretable, and scalable manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06426v1">FinLFQA: Evaluating Attributed Text Generation of LLMs in Financial Long-Form Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently hallucinate to long-form questions, producing plausible yet factually incorrect answers. A common mitigation strategy is to provide attribution to LLM outputs. However, existing benchmarks primarily focus on simple attribution that retrieves supporting textual evidence as references. We argue that in real-world scenarios such as financial applications, attribution goes beyond reference retrieval. We introduce FinLFQA, a benchmark designed to evaluate the ability of LLMs to generate long-form answers to complex financial questions with reliable and nuanced attributions. FinLFQA evaluates three critical aspects of attribution through human annotations: (1) supporting evidence extracted from financial reports, (2) intermediate numerical reasoning steps, and (3) domain-specific financial knowledge that informs the reasoning process. We further provide an automatic evaluation framework covering both answer quality and attribution quality. Through extensive experiments on eight LLMs across multiple attribution-generation paradigms, we find that fine-grained metrics are important to distinguish model capabilities, that end-to-end generation achieves comparable performance to post-hoc approaches, and that iterative refinement only helps when guided by external feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02016v2">Mind the (Belief) Gap: Group Identity in the World of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Accepted to ACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Social biases and belief-driven behaviors can significantly impact Large Language Models (LLMs) decisions on several tasks. As LLMs are increasingly used in multi-agent systems for societal simulations, their ability to model fundamental group psychological characteristics remains critical yet under-explored. In this study, we present a multi-agent framework that simulates belief congruence, a classical group psychology theory that plays a crucial role in shaping societal interactions and preferences. Our findings reveal that LLMs exhibit amplified belief congruence compared to humans, across diverse contexts. We further investigate the implications of this behavior on two downstream tasks: (1) misinformation dissemination and (2) LLM learning, finding that belief congruence in LLMs increases misinformation dissemination and impedes learning. To mitigate these negative impacts, we propose strategies inspired by: (1) contact hypothesis, (2) accuracy nudges, and (3) global citizenship framework. Our results show that the best strategies reduce misinformation dissemination by up to 37% and enhance learning by 11%. Bridging social psychology and AI, our work provides insights to navigate real-world interactions using LLMs while addressing belief-driven biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06411v1">Instructional Goal-Aligned Question Generation for Student Evaluation in Virtual Lab Settings: How Closely Do LLMs Actually Align?</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Virtual Labs offer valuable opportunities for hands-on, inquiry-based science learning, yet teachers often struggle to adapt them to fit their instructional goals. Third-party materials may not align with classroom needs, and developing custom resources can be time-consuming and difficult to scale. Recent advances in Large Language Models (LLMs) offer a promising avenue for addressing these limitations. In this paper, we introduce a novel alignment framework for instructional goal-aligned question generation, enabling teachers to leverage LLMs to produce simulation-aligned, pedagogically meaningful questions through natural language interaction. The framework integrates four components: instructional goal understanding via teacher-LLM dialogue, lab understanding via knowledge unit and relationship analysis, a question taxonomy for structuring cognitive and pedagogical intent, and the TELeR taxonomy for controlling prompt detail. Early design choices were informed by a small teacher-assisted case study, while our final evaluation analyzed over 1,100 questions from 19 open-source LLMs. With goal and lab understanding grounding questions in teacher intent and simulation context, the question taxonomy elevates cognitive demand (open-ended formats and relational types raise quality by 0.29-0.39 points), and optimized TELeR prompts enhance format adherence (80% parsability, >90% adherence). Larger models yield the strongest gains: parsability +37.1%, adherence +25.7%, and average quality +0.8 Likert points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06410v1">Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectory?</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Reasoning LLMs are trained to verbalize their reasoning process, yielding strong gains on complex tasks. This transparency also opens a promising direction: multiple reasoners can directly collaborate on each other's thinking within a shared trajectory, yielding better inference efficiency and exploration. A key prerequisite, however, is the ability to assess the usefulness and build on another model's partial thinking -- we call this off-trajectory reasoning. Our paper investigates a critical question: can standard solo-reasoning training pipelines deliver desired off-trajectory behaviors? We propose twin tests that capture the two extremes of the off-trajectory spectrum, namely Recoverability, which tests whether LLMs can backtrack from "distractions" induced by misleading reasoning traces, and Guidability, which tests their ability to build upon correct reasoning from stronger collaborators. Our study evaluates 15 open-weight LLMs (1.5B-32B) and reveals a counterintuitive finding -- "stronger" LLMs on benchmarks are often more fragile under distraction. Moreover, all models tested fail to effectively leverage guiding steps from collaborators on problems beyond their inherent capabilities with solve rates remaining under 9.2%. Finally, we conduct control studies to isolate the effects of three factors in post-training on these behaviors: the choice of distillation teacher, the use of RL, and data selection strategy. Our results provide actionable insights for training natively strong reasoning collaborators; e.g., we find that suboptimal recoverability behaviors of teacher models are transferred to distilled students even if the distillation trajectories are correct. Taken together, this work lays the groundwork for evaluating multi-model collaborations in shared reasoning trajectories and highlights the limitations of off-the-shelf reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06378v1">Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Automated interpretability aims to translate large language model (LLM) features into human understandable descriptions. However, these natural language feature descriptions are often vague, inconsistent, and require manual relabeling. In response, we introduce semantic regexes, structured language descriptions of LLM features. By combining primitives that capture linguistic and semantic feature patterns with modifiers for contextualization, composition, and quantification, semantic regexes produce precise and expressive feature descriptions. Across quantitative benchmarks and qualitative analyses, we find that semantic regexes match the accuracy of natural language while yielding more concise and consistent feature descriptions. Moreover, their inherent structure affords new types of analyses, including quantifying feature complexity across layers, scaling automated interpretability from insights into individual features to model-wide patterns. Finally, in user studies, we find that semantic regex descriptions help people build accurate mental models of LLM feature activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06354v1">LLM Bias Detection and Mitigation through the Lens of Desired Distributions</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Although prior work on bias mitigation has focused on promoting social equality and demographic parity, less attention has been given to aligning LLM's outputs to desired distributions. For example, we might want to align a model with real-world distributions to support factual grounding. Thus, we define bias as deviation from a desired distribution, which may be an equal or real-world distribution, depending on application goals. We propose a weighted adaptive loss based fine-tuning method that aligns LLM's gender-profession output distribution with the desired distribution, while preserving language modeling capability. Using 3 profession sets -- male-dominated, female-dominated, and gender-balanced -- derived from U.S. labor statistics (2024), we assess both our adaptive method for reflecting reality and a non-adaptive variant for equality. Across three masked language models, bias is observed under both distributions. We achieve near-complete mitigation under equality and 30-75% reduction under real-world settings. Autoregressive LLMs show no bias under equality but notable bias under real-world settings, with the Llama Instruct models (3.2-3B, 3.1-8B) achieving a 50-62% reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06214v1">Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly rely on external tools such as search engines to solve complex, multi-step problems, and reinforcement learning (RL) has become a key paradigm for training them. However, the trajectories of search agents are structurally heterogeneous, where variations in the number, placement, and outcomes of search calls lead to fundamentally different answer directions and reward distributions. Standard policy gradient methods, which use a single global baseline, suffer from what we identify and formalize as cross-stratum bias-an "apples-to-oranges" comparison of heterogeneous trajectories. This cross-stratum bias distorts credit assignment and hinders exploration of complex, multi-step search strategies. To address this, we propose Stratified GRPO, whose central component, Stratified Advantage Normalization (SAN), partitions trajectories into homogeneous strata based on their structural properties and computes advantages locally within each stratum. This ensures that trajectories are evaluated only against their true peers. Our analysis proves that SAN eliminates cross-stratum bias, yields conditionally unbiased unit-variance estimates inside each stratum, and retains the global unbiasedness and unit-variance properties enjoyed by standard normalization, resulting in a more pure and scale-stable learning signal. To improve practical stability under finite-sample regimes, we further linearly blend SAN with the global estimator. Extensive experiments on diverse single-hop and multi-hop question-answering benchmarks demonstrate that Stratified GRPO consistently and substantially outperforms GRPO by up to 11.3 points, achieving higher training rewards, greater training stability, and more effective search policies. These results establish stratification as a principled remedy for structural heterogeneity in RL for LLM search agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04573v2">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06175v1">VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      The Key-Value (KV) cache introduces substantial memory overhead during large language model (LLM) inference. Although existing vector quantization (VQ) methods reduce KV cache usage and provide flexible representational capacity across bit-widths, they suffer severe performance degradation at ultra-low bit-widths due to key cache outliers that hinder effective codebook utilization. To address this challenge, we propose VecInfer, a novel VQ method for aggressive KV cache compression while enabling efficient inference. By applying smooth and Hadamard transformations, VecInfer suppresses outliers in the key cache, enabling the codebook to comprehensively cover the original data distribution and thereby reducing quantization difficulty. To facilitate efficient deployment, we design an optimized CUDA kernel that fuses computation with dequantization to minimize memory access overhead. Extensive evaluations demonstrate that VecInfer consistently outperforms existing quantization baselines across both long-context understanding and mathematical reasoning tasks. With only 2-bit quantization, VecInfer achieves performance comparable to full precision, while delivering up to $\mathbf{2.7\times}$ speedup in large-batch self-attention computation and $\mathbf{8.3\times}$ reduction in single-batch end-to-end latency on Llama-3.1-8B with a 196k sequence length.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06151v1">LLMs as Policy-Agnostic Teammates: A Case Study in Human Proxy Design for Heterogeneous Agent Teams</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 This is a preprint of a paper presented at the \textit{European Conference on Artificial Intelligence (ECAI 2025)}. It is made publicly available for the benefit of the research community and should be regarded as a preprint rather than a formally reviewed publication
    </div>
    <details class="paper-abstract">
      A critical challenge in modelling Heterogeneous-Agent Teams is training agents to collaborate with teammates whose policies are inaccessible or non-stationary, such as humans. Traditional approaches rely on expensive human-in-the-loop data, which limits scalability. We propose using Large Language Models (LLMs) as policy-agnostic human proxies to generate synthetic data that mimics human decision-making. To evaluate this, we conduct three experiments in a grid-world capture game inspired by Stag Hunt, a game theory paradigm that balances risk and reward. In Experiment 1, we compare decisions from 30 human participants and 2 expert judges with outputs from LLaMA 3.1 and Mixtral 8x22B models. LLMs, prompted with game-state observations and reward structures, align more closely with experts than participants, demonstrating consistency in applying underlying decision criteria. Experiment 2 modifies prompts to induce risk-sensitive strategies (e.g. "be risk averse"). LLM outputs mirror human participants' variability, shifting between risk-averse and risk-seeking behaviours. Finally, Experiment 3 tests LLMs in a dynamic grid-world where the LLM agents generate movement actions. LLMs produce trajectories resembling human participants' paths. While LLMs cannot yet fully replicate human adaptability, their prompt-guided diversity offers a scalable foundation for simulating policy-agnostic teammates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06143v1">RoSE: Round-robin Synthetic Data Evaluation for Selecting LLM Generators without Human Test Sets</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      LLMs are powerful generators of synthetic data, which are used for training smaller, specific models. This is especially valuable for low-resource languages, where human-labelled data is scarce but LLMs can still produce high-quality text. However, LLMs differ in how useful their outputs are for training. Selecting the best LLM as a generator is challenging because extrinsic evaluation requires costly human annotations (which are often unavailable for low-resource languages), while intrinsic metrics correlate poorly with downstream performance. We introduce Round robin Synthetic data Evaluation (RoSE), a proxy metric for selecting the best LLM generator without human test sets. RoSE trains a small model on the outputs of a candidate generator (LLM) and then evaluates it on generated synthetic examples from all other candidate LLMs. The final RoSE score is the mean performance of this small model. Across six LLMs, eleven languages, and three tasks (sentiment, topic, intent), RoSE identifies the optimal generator more often than any other intrinsic heuristics. RoSE outperforms intrinsic heuristics and comes within 0.76 percentage points of the optimal generator baseline. This result is measured in terms of downstream performance, obtained by training a small model on the chosen generator's outputs (optimal vs. proxy metric selected) and evaluating it on human-labelled test data. Additionally, RoSE is the only metric to achieve a positive correlation with performance on human test data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06105v1">Moloch's Bargain: Emergent Misalignment When LLMs Compete for Audiences</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly shaping how information is created and disseminated, from companies using them to craft persuasive advertisements, to election campaigns optimizing messaging to gain votes, to social media influencers boosting engagement. These settings are inherently competitive, with sellers, candidates, and influencers vying for audience approval, yet it remains poorly understood how competitive feedback loops influence LLM behavior. We show that optimizing LLMs for competitive success can inadvertently drive misalignment. Using simulated environments across these scenarios, we find that, 6.3% increase in sales is accompanied by a 14.0% rise in deceptive marketing; in elections, a 4.9% gain in vote share coincides with 22.3% more disinformation and 12.5% more populist rhetoric; and on social media, a 7.5% engagement boost comes with 188.6% more disinformation and a 16.3% increase in promotion of harmful behaviors. We call this phenomenon Moloch's Bargain for AI--competitive success achieved at the cost of alignment. These misaligned behaviors emerge even when models are explicitly instructed to remain truthful and grounded, revealing the fragility of current alignment safeguards. Our findings highlight how market-driven optimization pressures can systematically erode alignment, creating a race to the bottom, and suggest that safe deployment of AI systems will require stronger governance and carefully designed incentives to prevent competitive dynamics from undermining societal trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06104v1">Explaining Code Risk in OSS: Towards LLM-Generated Fault Prediction Interpretations</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Open Source Software (OSS) has become a very important and crucial infrastructure worldwide because of the value it provides. OSS typically depends on contributions from developers across diverse backgrounds and levels of experience. Making safe changes, such as fixing a bug or implementing a new feature, can be challenging, especially in object-oriented systems where components are interdependent. Static analysis and defect-prediction tools produce metrics (e.g., complexity,coupling) that flag potentially fault-prone components, but these signals are often hard for contributors new or unfamiliar with the codebase to interpret. Large Language Models (LLMs) have shown strong performance on software engineering tasks such as code summarization and documentation generation. Building on this progress, we investigate whether LLMs can translate fault-prediction metrics into clear, human-readable risk explanations and actionable guidance to help OSS contributors plan and review code modifications. We outline explanation types that an LLM-generated assistant could provide (descriptive, contextual, and actionable explanations). We also outline our next steps to assess usefulness through a task-based study with OSS contributors, comparing metric-only baselines to LLM-generated explanations on decision quality, time-to-completion, and error rates
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06096v1">The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The objectives that Large Language Models (LLMs) implicitly optimize remain dangerously opaque, making trustworthy alignment and auditing a grand challenge. While Inverse Reinforcement Learning (IRL) can infer reward functions from behaviour, existing approaches either produce a single, overconfident reward estimate or fail to address the fundamental ambiguity of the task (non-identifiability). This paper introduces a principled auditing framework that re-frames reward inference from a simple estimation task to a comprehensive process for verification. Our framework leverages Bayesian IRL to not only recover a distribution over objectives but to enable three critical audit capabilities: (i) Quantifying and systematically reducing non-identifiability by demonstrating posterior contraction over sequential rounds of evidence; (ii) Providing actionable, uncertainty-aware diagnostics that expose spurious shortcuts and identify out-of-distribution prompts where the inferred objective cannot be trusted; and (iii) Validating policy-level utility by showing that the refined, low-uncertainty reward can be used directly in RLHF to achieve training dynamics and toxicity reductions comparable to the ground-truth alignment process. Empirically, our framework successfully audits a detoxified LLM, yielding a well-calibrated and interpretable objective that strengthens alignment guarantees. Overall, this work provides a practical toolkit for auditors, safety teams, and regulators to verify what LLMs are truly trying to achieve, moving us toward more trustworthy and accountable AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06093v1">Classical AI vs. LLMs for Decision-Maker Alignment in Health Insurance Choices</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 15 pages, 3 figures. Accepted at the Twelfth Annual Conference on Advances in Cognitive Systems (ACS 2025)
    </div>
    <details class="paper-abstract">
      As algorithmic decision-makers are increasingly applied to high-stakes domains, AI alignment research has evolved from a focus on universal value alignment to context-specific approaches that account for decision-maker attributes. Prior work on Decision-Maker Alignment (DMA) has explored two primary strategies: (1) classical AI methods integrating case-based reasoning, Bayesian reasoning, and naturalistic decision-making, and (2) large language model (LLM)-based methods leveraging prompt engineering. While both approaches have shown promise in limited domains such as medical triage, their generalizability to novel contexts remains underexplored. In this work, we implement a prior classical AI model and develop an LLM-based algorithmic decision-maker evaluated using a large reasoning model (GPT-5) and a non-reasoning model (GPT-4) with weighted self-consistency under a zero-shot prompting framework, as proposed in recent literature. We evaluate both approaches on a health insurance decision-making dataset annotated for three target decision-makers with varying levels of risk tolerance (0.0, 0.5, 1.0). In the experiments reported herein, classical AI and LLM-based models achieved comparable alignment with attribute-based targets, with classical AI exhibiting slightly better alignment for a moderate risk profile. The dataset and open-source implementation are publicly available at: https://github.com/TeX-Base/ClassicalAIvsLLMsforDMAlignment and https://github.com/Parallax-Advanced-Research/ITM/tree/feature_insurance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06092v1">Learning from Failures: Understanding LLM Alignment through Failure-Aware Inverse RL</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) aligns Large Language Models (LLMs) with human preferences, yet the underlying reward signals they internalize remain hidden, posing a critical challenge for interpretability and safety. Existing approaches attempt to extract these latent incentives using Inverse Reinforcement Learning (IRL), but treat all preference pairs equally, often overlooking the most informative signals: those examples the extracted reward model misclassifies or assigns nearly equal scores, which we term \emph{failures}. We introduce a novel \emph{failure-aware} IRL algorithm that focuses on misclassified or difficult examples to recover the latent rewards defining model behaviors. By learning from these failures, our failure-aware IRL extracts reward functions that better reflect the true objectives behind RLHF. We demonstrate that failure-aware IRL outperforms existing IRL baselines across multiple metrics when applied to LLM detoxification, without requiring external classifiers or supervision. Crucially, failure-aware IRL yields rewards that better capture the true incentives learned during RLHF, enabling more effective re-RLHF training than standard IRL. This establishes failure-aware IRL as a robust, scalable method for auditing model alignment and reducing ambiguity in the IRL process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06078v1">Constraint-Aware Route Recommendation from Natural Language via Hierarchical LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Route recommendation aims to provide users with optimal travel plans that satisfy diverse and complex requirements. Classical routing algorithms (e.g., shortest-path and constraint-aware search) are efficient but assume structured inputs and fixed objectives, limiting adaptability to natural-language queries. Recent LLM-based approaches enhance flexibility but struggle with spatial reasoning and the joint modeling of route-level and POI-level preferences. To address these limitations, we propose RouteLLM, a hierarchical multi-agent framework that grounds natural-language intents into constraint-aware routes. It first parses user queries into structured intents including POIs, paths, and constraints. A manager agent then coordinates specialized sub-agents: a constraint agent that resolves and formally check constraints, a POI agent that retrieves and ranks candidate POIs, and a path refinement agent that refines routes via a routing engine with preference-conditioned costs. A final verifier agent ensures constraint satisfaction and produces the final route with an interpretable rationale. This design bridges linguistic flexibility and spatial structure, enabling reasoning over route feasibility and user preferences. Experiments show that our method reliably grounds textual preferences into constraint-aware routes, improving route quality and preference satisfaction over classical methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06039v1">CDTP: A Large-Scale Chinese Data-Text Pair Dataset for Comprehensive Evaluation of Chinese LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks. However, Chinese LLMs face unique challenges, primarily due to the dominance of unstructured free text and the lack of structured representations in Chinese corpora. While existing benchmarks for LLMs partially assess Chinese LLMs, they are still predominantly English-centric and fail to address the unique linguistic characteristics of Chinese, lacking structured datasets essential for robust evaluation. To address these challenges, we present a Comprehensive Benchmark for Evaluating Chinese Large Language Models (CB-ECLLM) based on the newly constructed Chinese Data-Text Pair (CDTP) dataset. Specifically, CDTP comprises over 7 million aligned text pairs, each consisting of unstructured text coupled with one or more corresponding triples, alongside a total of 15 million triples spanning four critical domains. The core contributions of CDTP are threefold: (i) enriching Chinese corpora with high-quality structured information; (ii) enabling fine-grained evaluation tailored to knowledge-driven tasks; and (iii) supporting multi-task fine-tuning to assess generalization and robustness across scenarios, including Knowledge Graph Completion, Triple-to-Text generation, and Question Answering. Furthermore, we conduct rigorous evaluations through extensive experiments and ablation studies to assess the effectiveness, Supervised Fine-Tuning (SFT), and robustness of the benchmark. To support reproducible research, we offer an open-source codebase and outline potential directions for future investigations based on our insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06018v1">Evaluating The Impact of Stimulus Quality in Investigations of LLM Language Performance</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Presented at https://brigap-workshop.github.io/ Information to be updated upon publication of proceedings
    </div>
    <details class="paper-abstract">
      Recent studies employing Large Language Models (LLMs) to test the Argument from the Poverty of the Stimulus (APS) have yielded contrasting results across syntactic phenomena. This paper investigates the hypothesis that characteristics of the stimuli used in recent studies, including lexical ambiguities and structural complexities, may confound model performance. A methodology is proposed for re-evaluating LLM competence on syntactic prediction, focusing on GPT-2. This involves: 1) establishing a baseline on previously used (both filtered and unfiltered) stimuli, and 2) generating a new, refined dataset using a state-of-the-art (SOTA) generative LLM (Gemini 2.5 Pro Preview) guided by linguistically-informed templates designed to mitigate identified confounds. Our preliminary findings indicate that GPT-2 demonstrates notably improved performance on these refined PG stimuli compared to baselines, suggesting that stimulus quality significantly influences outcomes in surprisal-based evaluations of LLM syntactic competency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02418v2">BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      LLM web agents now browse and take actions on the open web, yet current agent evaluations are constrained to sandboxed environments or artificial tasks. We introduce BrowserArena, a live open-web agent evaluation platform that collects user-submitted tasks, runs Arena-style head-to-head comparisons, and uses step-level human feedback to surface failure modes. Collecting and analyzing step-level annotations on the agent traces, we identify three consistent failure modes: captcha resolution, pop-up banner removal, and direct navigation to URLs. By constructing targeted datasets to further study these tasks, we discover variations in how different language models navigate these failure modes. We find, for example, that o4-mini deploys a wider variety of strategies to circumvent captcha resolution than other models and DeepSeek-R1 consistently misleads users about pop-up banner closure. Our findings surface both the diversity and brittleness of current web agents. More broadly, our benchmarking methodology provides an approach to evaluating and understanding web agent failure modes at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06001v1">Exploring Gaps in the APS: Direct Minimal Pair Analysis in LLM Syntactic Assessments</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Presented at the https://brigap-workshop.github.io/ Information to be updated after publication of proceedings
    </div>
    <details class="paper-abstract">
      Recent studies probing the Argument from the Poverty of the Stimulus (APS) have applied Large Language Models (LLMs) to test the learnability of complex syntax through surprisal-based metrics. However, divergent conclusions raise questions concerning the insights these metrics offer. While Wilcox et al. (2024) used direct minimal pair comparisons (the "wh-effect") to demonstrate that models successfully generalise knowledge of filler-gap dependencies, Lan et al. (2024) used a Difference-in-Differences (DiD) metric and found that models largely fail on parasitic gaps (PGs). This paper argues that the direct minimal pair approach offers greater diagnostic transparency. We demonstrate this by generating a full 8-permutation paradigm of refined PG stimuli and evaluating the GPT-2 model used in previous studies with a systematic Wilcox-style wh-effect analysis. Our results show that GPT-2 succeeds across all four tested conditions, indicating robust knowledge of filler-gap licensing principles even in complex PG environments. This finding, which contrasts with the more ambiguous results from DiD-style metrics, suggests that the choice of evaluation metric is critical for assessing an LLM's syntactic competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05987v1">Sample Smart, Not Hard: Correctness-First Decoding for Better Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to complex tasks that require extended reasoning. In such settings, models often benefit from diverse chains-of-thought to arrive at multiple candidate solutions. This requires two competing objectives: to inject enough stochasticity to explore multiple reasoning chains, and to ensure sufficient accuracy and quality in each path. Existing works pursue the first objective by increasing exploration at highly uncertain steps with higher temperature or larger candidate token sets, while others improve reliability by rejecting samples with low confidence post-generation, implying that low confidence correlates with low answer quality. These two lines of thought are in conflict, as they conflate different sources of uncertainty. To resolve this, we argue that the decoding rule should be calibrated by correctness, not confidence alone. We should sample from tokens with higher estimated correctness, and reduce sampling where expected correctness is low. We propose simple strategies that achieve this goal: Greedy-Threshold makes sampling greedy at very low confidence steps. Calibrated-TopK and Calibrated-epsilon set truncation threshold based on estimated rank-wise correctness. Together, our findings challenge prevailing heuristics about decoding under uncertainty and show gains across math and general reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05950v1">Training-Free Time Series Classification via In-Context Reasoning with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 8 pages main content, 12 pages total including appendix, 1 figure
    </div>
    <details class="paper-abstract">
      Time series classification (TSC) spans diverse application scenarios, yet labeled data are often scarce, making task-specific training costly and inflexible. Recent reasoning-oriented large language models (LLMs) show promise in understanding temporal patterns, but purely zero-shot usage remains suboptimal. We propose FETA, a multi-agent framework for training-free TSC via exemplar-based in-context reasoning. FETA decomposes a multivariate series into channel-wise subproblems, retrieves a few structurally similar labeled examples for each channel, and leverages a reasoning LLM to compare the query against these exemplars, producing channel-level labels with self-assessed confidences; a confidence-weighted aggregator then fuses all channel decisions. This design eliminates the need for pretraining or fine-tuning, improves efficiency by pruning irrelevant channels and controlling input length, and enhances interpretability through exemplar grounding and confidence estimation. On nine challenging UEA datasets, FETA achieves strong accuracy under a fully training-free setting, surpassing multiple trained baselines. These results demonstrate that a multi-agent in-context reasoning framework can transform LLMs into competitive, plug-and-play TSC solvers without any parameter training. The code is available at https://github.com/SongyuanSui/FETATSC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09946v3">Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Published in IEEE Access
    </div>
    <details class="paper-abstract">
      Recent studies have investigated whether large language models (LLMs) can support obscured communication, which is characterized by core aspects such as inferring subtext and evading suspicions. To conduct the investigation, researchers have used social deduction games (SDGs) as their experimental environment, in which players conceal and infer specific information. However, prior work has often overlooked how LLMs should be evaluated in such settings. Specifically, we point out two limitations with the evaluation methods they employed. First, metrics used in prior studies are coarse-grained as they are based on overall game outcomes that often fail to capture event-level behaviors; Second, error analyses have lacked structured methodologies capable of producing insights that meaningfully support evaluation outcomes. To address these limitations, we propose a microscopic and systematic approach to the investigation. Specifically, we introduce six fine-grained metrics that resolve the first issue. To tackle the second issue, we conducted a thematic analysis and identified four major reasoning failures that undermine LLMs' performance in obscured communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05935v1">LLM-FS-Agent: A Deliberative Role-based Large Language Model Architecture for Transparent Feature Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      High-dimensional data remains a pervasive challenge in machine learning, often undermining model interpretability and computational efficiency. While Large Language Models (LLMs) have shown promise for dimensionality reduction through feature selection, existing LLM-based approaches frequently lack structured reasoning and transparent justification for their decisions. This paper introduces LLM-FS-Agent, a novel multi-agent architecture designed for interpretable and robust feature selection. The system orchestrates a deliberative "debate" among multiple LLM agents, each assigned a specific role, enabling collective evaluation of feature relevance and generation of detailed justifications. We evaluate LLM-FS-Agent in the cybersecurity domain using the CIC-DIAD 2024 IoT intrusion detection dataset and compare its performance against strong baselines, including LLM-Select and traditional methods such as PCA. Experimental results demonstrate that LLM-FS-Agent consistently achieves superior or comparable classification performance while reducing downstream training time by an average of 46% (statistically significant improvement, p = 0.028 for XGBoost). These findings highlight that the proposed deliberative architecture enhances both decision transparency and computational efficiency, establishing LLM-FS-Agent as a practical and reliable solution for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05909v1">Optimizing for Persuasion Improves LLM Generalization: Evidence from Quality-Diversity Evolution of Debate Strategies</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Open-source code available at https://github.com/flowersteam/llm_persuasion
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) optimized to output truthful answers often overfit, producing brittle reasoning that fails to generalize. While persuasion-based optimization has shown promise in debate settings, it has not been systematically compared against mainstream truth-based approaches. We introduce DebateQD, a minimal Quality-Diversity (QD) evolutionary algorithm that evolves diverse debate strategies across different categories (rationality, authority, emotional appeal, etc.) through tournament-style competitions where two LLMs debate while a third judges. Unlike previously proposed methods that require a population of LLMs, our approach maintains diversity of opponents through prompt-based strategies within a single LLM architecture, making it more accessible for experiments while preserving the key benefits of population-based optimization. In contrast to prior work, we explicitly isolate the role of the optimization objective by fixing the debate protocol and swapping only the fitness function: persuasion rewards strategies that convince the judge irrespective of truth, whereas truth rewards collaborative correctness. Across three model scales (7B, 32B, 72B parameters) and multiple dataset sizes from the QuALITY benchmark, persuasion-optimized strategies achieve up to 13.94% smaller train-test generalization gaps, while matching or exceeding truth optimization's test performance. These results provide the first controlled evidence that competitive pressure to persuade, rather than seek the truth collaboratively, fosters more transferable reasoning skills, offering a promising path for improving LLM generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02298v2">CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback. However, current RLVR methods typically assign the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies. Methods like PPO provide credit assignment by value estimation, but yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-wise rewards but suffer from several key limitations: they require high-quality process supervision labels, the feedback is unreliable due to probabilistic reward modeling, and their application in online reinforcement learning (RL) is time-consuming. To overcome these limitations, we introduce a simple but efficient method-Credit Assignment Policy Optimization (CAPO). Instead of training auxiliary models, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass only based on the correctness of the step itself, providing deterministic token-level credits to refine the tokens that were originally assigned identical rule-based rewards. To further enhance the accuracy and robustness, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments on various backbones like Llama and Qwen models show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across four challenging mathematical benchmarks and three out-of-domain benchmarks. Further analysis shows that CAPO can help the model to foster the learning of correct reasoning pathways leading to correct answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09387v3">MetaLLMix : An XAI Aided LLM-Meta-learning Based Approach for Hyper-parameters Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Effective model and hyperparameter selection remains a major challenge in deep learning, often requiring extensive expertise and computation. While AutoML and large language models (LLMs) promise automation, current LLM-based approaches rely on trial and error and expensive APIs, which provide limited interpretability and generalizability. We propose MetaLLMiX, a zero-shot hyperparameter optimization framework combining meta-learning, explainable AI, and efficient LLM reasoning. By leveraging historical experiment outcomes with SHAP explanations, MetaLLMiX recommends optimal hyperparameters and pretrained models without additional trials. We further employ an LLM-as-judge evaluation to control output format, accuracy, and completeness. Experiments on eight medical imaging datasets using nine open-source lightweight LLMs show that MetaLLMiX achieves competitive or superior performance to traditional HPO methods while drastically reducing computational cost. Our local deployment outperforms prior API-based approaches, achieving optimal results on 5 of 8 tasks, response time reductions of 99.6-99.9%, and the fastest training times on 6 datasets (2.4-15.7x faster), maintaining accuracy within 1-5% of best-performing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06270v1">MCCE: A Framework for Multi-LLM Collaborative Co-Evolution</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Multi-objective discrete optimization problems, such as molecular design, pose significant challenges due to their vast and unstructured combinatorial spaces. Traditional evolutionary algorithms often get trapped in local optima, while expert knowledge can provide crucial guidance for accelerating convergence. Large language models (LLMs) offer powerful priors and reasoning ability, making them natural optimizers when expert knowledge matters. However, closed-source LLMs, though strong in exploration, cannot update their parameters and thus cannot internalize experience. Conversely, smaller open models can be continually fine-tuned but lack broad knowledge and reasoning strength. We introduce Multi-LLM Collaborative Co-evolution (MCCE), a hybrid framework that unites a frozen closed-source LLM with a lightweight trainable model. The system maintains a trajectory memory of past search processes; the small model is progressively refined via reinforcement learning, with the two models jointly supporting and complementing each other in global exploration. Unlike model distillation, this process enhances the capabilities of both models through mutual inspiration. Experiments on multi-objective drug design benchmarks show that MCCE achieves state-of-the-art Pareto front quality and consistently outperforms baselines. These results highlight a new paradigm for enabling continual evolution in hybrid LLM systems, combining knowledge-driven exploration with experience-driven learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04519v1">Spec2Control: Automating PLC/DCS Control-Logic Engineering from Natural Language Requirements with LLMs - A Multi-Plant Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 12 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Distributed control systems (DCS) manage the automation for many industrial production processes (e.g., power plants, chemical refineries, steel mills). Programming the software for such systems remains a largely manual and tedious process, incurring costs of millions of dollars for extensive facilities. Large language models (LLMs) have been found helpful in generating DCS control logic, resulting in commercial copilot tools. Today, these tools are focused on textual notations, they provide limited automation, and have not been tested on large datasets with realistic test cases. We introduce Spec2Control, a highly automated LLM workflow to generate graphical control logic directly from natural language user requirements. Experiments using an open dataset with 10 control narratives and 65 complex test cases demonstrate that Spec2Control can successfully identify control strategies, can generate 98.6% of correct control strategy connections autonomously, and can save between 94-96% of human labor. Spec2Control is being integrated into commercial ABB engineering tools, but is also available as an open-source variant for independent validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04503v1">P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      During fine-tuning, large language models (LLMs) are increasingly vulnerable to data-poisoning backdoor attacks, which compromise their reliability and trustworthiness. However, existing defense strategies suffer from limited generalization: they only work on specific attack types or task settings. In this study, we propose Poison-to-Poison (P2P), a general and effective backdoor defense algorithm. P2P injects benign triggers with safe alternative labels into a subset of training samples and fine-tunes the model on this re-poisoned dataset by leveraging prompt-based learning. This enforces the model to associate trigger-induced representations with safe outputs, thereby overriding the effects of original malicious triggers. Thanks to this robust and generalizable trigger-based fine-tuning, P2P is effective across task settings and attack types. Theoretically and empirically, we show that P2P can neutralize malicious backdoors while preserving task performance. We conduct extensive experiments on classification, mathematical reasoning, and summary generation tasks, involving multiple state-of-the-art LLMs. The results demonstrate that our P2P algorithm significantly reduces the attack success rate compared with baseline models. We hope that the P2P can serve as a guideline for defending against backdoor attacks and foster the development of a secure and trustworthy LLM community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16706v3">DISC: Dynamic Decomposition Improves LLM Inference Scaling</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 10 pages, Accepted to NeurIPS 2025 (Conference on Neural Information Processing Systems)
    </div>
    <details class="paper-abstract">
      Inference scaling methods for LLMs often rely on decomposing problems into steps (or groups of tokens), followed by sampling and selecting the best next steps. However, these steps and their sizes are often predetermined or manually designed based on domain knowledge. We propose dynamic decomposition, a method that adaptively and automatically partitions solution and reasoning traces into manageable steps during inference. By more effectively allocating compute -- particularly through subdividing challenging steps and prioritizing their sampling -- dynamic decomposition significantly improves inference efficiency. Experiments on benchmarks such as APPS, MATH, and LiveCodeBench demonstrate that dynamic decomposition outperforms static approaches, including token-level, sentence-level, and single-step decompositions, reducing the pass@10 error rate by 5.0%, 6.7%, and 10.5% respectively. These findings highlight the potential of dynamic decomposition to improve a wide range of inference scaling techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04498v1">GenQuest: An LLM-based Text Adventure Game for Language Learners</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Workshop on Wordplay: When Language Meets Games, EMNLP 2025
    </div>
    <details class="paper-abstract">
      GenQuest is a generative text adventure game that leverages Large Language Models (LLMs) to facilitate second language learning through immersive, interactive storytelling. The system engages English as a Foreign Language (EFL) learners in a collaborative "choose-your-own-adventure" style narrative, dynamically generated in response to learner choices. Game mechanics such as branching decision points and story milestones are incorporated to maintain narrative coherence while allowing learner-driven plot development. Key pedagogical features include content generation tailored to each learner's proficiency level, and a vocabulary assistant that provides in-context explanations of learner-queried text strings, ranging from words and phrases to sentences. Findings from a pilot study with university EFL students in China indicate promising vocabulary gains and positive user perceptions. Also discussed are suggestions from participants regarding the narrative length and quality, and the request for multi-modal content such as illustrations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03691v2">LogSage: An LLM-Based Framework for CI/CD Failure Detection and Remediation with Industrial Validation</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Continuous Integration and Deployment (CI/CD) pipelines are critical to modern software engineering, yet diagnosing and resolving their failures remains complex and labor-intensive. We present LogSage, the first end-to-end LLM-powered framework for root cause analysis (RCA) and automated remediation of CI/CD failures. LogSage employs a token-efficient log preprocessing pipeline to filter noise and extract critical errors, then performs structured diagnostic prompting for accurate RCA. For solution generation, it leverages retrieval-augmented generation (RAG) to reuse historical fixes and invokes automation fixes via LLM tool-calling. On a newly curated benchmark of 367 GitHub CI/CD failures, LogSage achieves over 98\% precision, near-perfect recall, and an F1 improvement of more than 38\% points in the RCA stage, compared with recent LLM-based baselines. In a year-long industrial deployment at ByteDance, it processed over 1.07M executions, with end-to-end precision exceeding 80\%. These results demonstrate that LogSage provides a scalable and practical solution for automating CI/CD failure management in real-world DevOps workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21815v2">Scientific Paper Retrieval with LLM-Guided Semantic-Based Ranking</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Scientific paper retrieval is essential for supporting literature discovery and research. While dense retrieval methods demonstrate effectiveness in general-purpose tasks, they often fail to capture fine-grained scientific concepts that are essential for accurate understanding of scientific queries. Recent studies also use large language models (LLMs) for query understanding; however, these methods often lack grounding in corpus-specific knowledge and may generate unreliable or unfaithful content. To overcome these limitations, we propose SemRank, an effective and efficient paper retrieval framework that combines LLM-guided query understanding with a concept-based semantic index. Each paper is indexed using multi-granular scientific concepts, including general research topics and detailed key phrases. At query time, an LLM identifies core concepts derived from the corpus to explicitly capture the query's information need. These identified concepts enable precise semantic matching, significantly enhancing retrieval accuracy. Experiments show that SemRank consistently improves the performance of various base retrievers, surpasses strong existing LLM-based baselines, and remains highly efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04488v1">Multi-Agent Collaborative Intelligence: Dual-Dial Control for Reliable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 27 pages, 5 figures, 21 tables
    </div>
    <details class="paper-abstract">
      Multi-agent debate often wastes compute by using a fixed adversarial stance, aggregating without deliberation, or stopping on heuristics. We introduce MACI, an active controller with two independent dials that decouple information from behavior: an information dial that gates evidence by quality, and a behavior dial that schedules contentiousness from exploration to consolidation. A moderator tracks disagreement, overlap, evidence quality, and argument quality, and halts when gains plateau. We provide theory-lite guarantees for nonincreasing dispersion and provable termination, with a budget-feasible scheduler. Across clinical diagnosis and news-bias tasks, MACI improves accuracy and calibration while reducing tokens, and converts residual uncertainty into precision RAG plans that specify what to retrieve next. We use a cross-family LLM judge (CRIT) as a conservative soft weight and stop signal, validated for order invariance and judge-swap stability; stability depends on using high-capability judges. MACI turns debate into a budget-aware, measurable, and provably terminating controller.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04484v1">Psychological Steering in LLMs: An Evaluation of Effectiveness and Trustworthiness</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Submitted to ARR - October 2025
    </div>
    <details class="paper-abstract">
      The ability to control LLMs' emulated emotional states and personality traits is essential for enabling rich, human-centered interactions in socially interactive settings. We introduce PsySET, a Psychologically-informed benchmark to evaluate LLM Steering Effectiveness and Trustworthiness across the emotion and personality domains. Our study spans four models from different LLM families paired with various steering strategies, including prompting, fine-tuning, and representation engineering. Our results indicate that prompting is consistently effective but limited in intensity control, whereas vector injections achieve finer controllability while slightly reducing output quality. Moreover, we explore the trustworthiness of steered LLMs by assessing safety, truthfulness, fairness, and ethics, highlighting potential side effects and behavioral shifts. Notably, we observe idiosyncratic effects; for instance, even a positive emotion like joy can degrade robustness to adversarial factuality, lower privacy awareness, and increase preferential bias. Meanwhile, anger predictably elevates toxicity yet strengthens leakage resistance. Our framework establishes the first holistic evaluation of emotion and personality steering, offering insights into its interpretability and reliability for socially interactive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04724v2">Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Multi-agent systems powered by Large Language Models (LLM-MAS) have demonstrated remarkable capabilities in collaborative problem-solving. However, their deployment also introduces new security risks. Existing research on LLM-based agents has primarily examined single-agent scenarios, while the security of multi-agent systems remains largely unexplored. To address this gap, we present a systematic study of intention-hiding threats in LLM-MAS. We design four representative attack paradigms that subtly disrupt task completion while maintaining a high degree of stealth, and evaluate them under centralized, decentralized, and layered communication structures. Experimental results show that these attacks are highly disruptive and can easily evade existing defense mechanisms. To counter these threats, we propose AgentXposed, a psychology-inspired detection framework. AgentXposed draws on the HEXACO personality model, which characterizes agents through psychological trait dimensions, and the Reid interrogation technique, a structured method for eliciting concealed intentions. By combining progressive questionnaire probing with behavior-based inter-agent monitoring, the framework enables the proactive identification of malicious agents before harmful actions are carried out. Extensive experiments across six datasets against both our proposed attacks and two baseline threats demonstrate that AgentXposed effectively detects diverse forms of malicious behavior, achieving strong robustness across multiple communication settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06698v3">SCAN: Structured Capability Assessment and Navigation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) has become increasingly important, with automatic evaluation benchmarks gaining prominence as alternatives to human evaluation. While existing research has focused on approximating model rankings, such benchmarks fail to provide users and developers with a comprehensive and fine-grained understanding of a specific model's capabilities. To fill this gap, we propose \textbf{SCAN} (Structured Capability Assessment and Navigation), a practical framework that enables detailed characterization of LLM capabilities through comprehensive and fine-grained evaluation. SCAN incorporates four key components: (1) TaxBuilder, which extracts capability-indicating tags from extensive queries to construct a hierarchical taxonomy automatically; (2) RealMix, a query synthesis and filtering mechanism that ensures sufficient evaluation data for each capability tag; (3) a suite of visualization and analysis tools that facilitate efficient navigation and analysis of model capabilities; and (4) a PC$^2$-based (Pre-Comparison-derived Criteria) LLM-as-a-Judge approach that achieves significantly higher accuracy compared to classic LLM-as-a-Judge method. Using SCAN, we conduct a comprehensive evaluation of 21 mainstream LLMs. Our detailed analysis of the GPT-OSS family reveals substantial performance variations, even within sub-capabilities belonging to the same category of capability. This finding highlights the importance of fine-grained evaluation in accurately understanding LLM behavior. Project homepage and resources are available at \href{https://liudan193.github.io/Feedbacker/}{https://liudan193.github.io/Feedbacker/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18562v2">From Word to World: Evaluate and Mitigate Culture Bias in LLMs via Word Association Test</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Cultural Analysis, Cultural Alignment, Word Association Test, Large Language Models. Accepted by EMNLP 2025 (Oral)
    </div>
    <details class="paper-abstract">
      The human-centered word association test (WAT) serves as a cognitive proxy, revealing sociocultural variations through culturally shared semantic expectations and implicit linguistic patterns shaped by lived experiences. We extend this test into an LLM-adaptive, free-relation task to assess the alignment of large language models (LLMs) with cross-cultural cognition. To address culture preference, we propose CultureSteer, an innovative approach that moves beyond superficial cultural prompting by embedding cultural-specific semantic associations directly within the model's internal representation space. Experiments show that current LLMs exhibit significant bias toward Western (notably American) schemas at the word association level. In contrast, our model substantially improves cross-cultural alignment, capturing diverse semantic associations. Further validation on culture-sensitive downstream tasks confirms its efficacy in fostering cognitive alignment across cultures. This work contributes a novel methodological paradigm for enhancing cultural awareness in LLMs, advancing the development of more inclusive language technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04465v1">Autonomy Matters: A Study on Personalization-Privacy Dilemma in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents require personal information for personalization in order to better act on users' behalf in daily tasks, but this raises privacy concerns and a personalization-privacy dilemma. Agent's autonomy introduces both risks and opportunities, yet its effects remain unclear. To better understand this, we conducted a 3$\times$3 between-subjects experiment ($N=450$) to study how agent's autonomy level and personalization influence users' privacy concerns, trust and willingness to use, as well as the underlying psychological processes. We find that personalization without considering users' privacy preferences increases privacy concerns and decreases trust and willingness to use. Autonomy moderates these effects: Intermediate autonomy flattens the impact of personalization compared to No- and Full autonomy conditions. Our results suggest that rather than aiming for perfect model alignment in output generation, balancing autonomy of agent's action and user control offers a promising path to mitigate the personalization-privacy dilemma.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04463v1">Evaluating Self-Supervised Speech Models via Text-Based LLMS</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to ASRU 2025
    </div>
    <details class="paper-abstract">
      Self-Supervised Learning (SSL) has gained traction for its ability to learn rich representations with low labeling costs, applicable across diverse downstream tasks. However, assessing the downstream-task performance remains challenging due to the cost of extra training and evaluation. Existing methods for task-agnostic evaluation also require extra training or hyperparameter tuning. We propose a novel evaluation metric using large language models (LLMs). By inputting discrete token sequences and minimal domain cues derived from SSL models into LLMs, we obtain the mean log-likelihood; these cues guide in-context learning, rendering the score more reliable without extra training or hyperparameter tuning. Experimental results show a correlation between LLM-based scores and automatic speech recognition task. Additionally, our findings reveal that LLMs not only functions as an SSL evaluation tools but also provides inference-time embeddings that are useful for speaker verification task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02657v2">Less LLM, More Documents: Searching for Improved RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) couples document retrieval with large language models (LLMs). While scaling generators improves accuracy, it also raises cost and limits deployability. We explore an orthogonal axis: enlarging the retriever's corpus to reduce reliance on large LLMs. Experimental results show that corpus scaling consistently strengthens RAG and can often serve as a substitute for increasing model size, though with diminishing returns at larger scales. Small- and mid-sized generators paired with larger corpora often rival much larger models with smaller corpora; mid-sized models tend to gain the most, while tiny and large models benefit less. Our analysis shows that improvements arise primarily from increased coverage of answer-bearing passages, while utilization efficiency remains largely unchanged. These findings establish a principled corpus-generator trade-off: investing in larger corpora offers an effective path to stronger RAG, often comparable to enlarging the LLM itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03195v2">Can LLMs Hit Moving Targets? Tracking Evolving Signals in Corporate Disclosures</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 8 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Moving targets -- managers' strategic shifting of key performance metrics when the original targets become difficult to achieve -- have been shown to predict subsequent stock underperformance. However, our work reveals that the method employed in that study exhibits two key limitations that hinder the accuracy -- noise in the extracted targets and loss of contextual information -- both of which stem primarily from the use of a named entity recognition (NER). To address these two limitations, we propose an LLM-based target extraction method with a newly defined metric that better captures semantic context. This approach preserves semantic context beyond simple entity recognition and yields consistently higher predictive power than the original approach. Overall, our approach enhances the granularity and accuracy of financial text-based performance prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04439v1">On the Role of Unobserved Sequences on Sample-based Uncertainty Quantification for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to UncertaiNLP workshop of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Quantifying uncertainty in large language models (LLMs) is important for safety-critical applications because it helps spot incorrect answers, known as hallucinations. One major trend of uncertainty quantification methods is based on estimating the entropy of the distribution of the LLM's potential output sequences. This estimation is based on a set of output sequences and associated probabilities obtained by querying the LLM several times. In this paper, we advocate and experimentally show that the probability of unobserved sequences plays a crucial role, and we recommend future research to integrate it to enhance such LLM uncertainty quantification methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05449v1">Bloom: Designing for LLM-Augmented Behavior Change Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer novel opportunities to support health behavior change, yet existing work has narrowly focused on text-only interactions. Building on decades of HCI research demonstrating the effectiveness of UI-based interactions, we present Bloom, an application for physical activity promotion that integrates an LLM-based health coaching chatbot with established UI-based interactions. As part of Bloom's development, we conducted a redteaming evaluation and contribute a safety benchmark dataset. In a four-week randomized field study (N=54) comparing Bloom to a non-LLM control, we observed important shifts in psychological outcomes: participants in the LLM condition reported stronger beliefs that activity was beneficial, greater enjoyment, and more self-compassion. Both conditions significantly increased physical activity levels, doubling the proportion of participants meeting recommended weekly guidelines, though we observed no significant differences between conditions. Instead, our findings suggest that LLMs may be more effective at shifting mindsets that precede longer-term behavior change.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05445v1">AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and agent-based frameworks have advanced rapidly, enabling diverse applications. Yet, with the proliferation of models and agentic strategies, practitioners face substantial uncertainty in selecting the best configuration for a downstream task. Prior studies show that different agents and backbones exhibit complementary strengths, and that larger models are not always superior, underscoring the need for adaptive routing mechanisms. Existing approaches to agent routing, however, often emphasize cost efficiency while overlooking the fine-grained contextual and relational structure inherent in QA tasks. In this paper, we propose tAgentRouter, a framework that formulates multi-agent QA as a knowledge-graph-guided routing problem supervised by empirical performance signals. Specifically, we convert QA instance into a knowledge graph that jointly encodes queries, contextual entities, and agents, and then train a heterogeneous graph neural network (GNN) to propagate information across node types and produce task-aware routing distributions over agents. By leveraging soft supervision and weighted aggregation of agent outputs, AgentRouter learns principled collaboration schemes that capture the complementary strengths of diverse agents. Extensive experiments demonstrate that our framework consistently outperforms single-agent and ensemble baselines, while generalizing across benchmarks and LLM backbones. These results highlight the effectiveness and robustness of graph-supervised multi-agent routing for question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04365v4">AutoPDL: Automatic Prompt Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Presented at AutoML 2025 (Methods Track); to be published in proceedings
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and specific to a given LLM and task. Therefore, this paper proposes AutoPDL, an automated approach to discovering good LLM agent configurations. Our approach frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and seven LLMs (ranging from 3B to 70B parameters) show consistent accuracy gains ($9.21\pm15.46$ percentage points), up to 67.5pp, and reveal that selected prompting strategies vary across models and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05431v1">Self-Filtered Distillation with LLMs-generated Trust Indicators for Reliable Patent Classification</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly generate natural language rationales to enhance interpretability, but these often contain logical errors, label mismatches, and domain-specific misalignments. Directly using such rationales as supervision risks propagating noise and undermining training stability. To address this challenge, we introduce Self-Filtered Distillation, a framework specifically tailored for patent classification, which treats LLM-generated rationales as trust signals rather than ground-truth supervision. The framework employs selective distillation guided by three unsupervised trust metrics: (1) Self-Consistency, which measures the stability of LLM-generated rationales across multiple generations; (2) Class Entailment Alignment, which assesses semantic coherence with patent-specific class definitions; and (3) LLM Agreement Scoring, which validates rationale-label plausibility. These metrics are integrated into a unified trust score that primarily weights training samples while optionally filtering out extremely low-trust cases, enabling reasoning-aware supervision. Experiments on the USPTO-2M dataset, a widely used benchmark for patent classification, show that our method outperforms label-based learning and conventional distillation in accuracy, stability, and interpretability, establishing a reliable paradigm for leveraging reasoning-aware trust indicators in patent analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09389v2">Measuring LLM Novelty As The Frontier Of Original And High-Quality Output</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Updated results with higher coverage of open-data models and better quality judgments
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used for ideation and scientific discovery, it is important to evaluate their ability to generate novel output. Prior work evaluates novelty as originality with respect to model training data, but original outputs may be of low quality. In contrast, non-expert judges more reliably score quality but may favor memorized outputs, limiting the reliability of human preference as a metric. We introduce a new novelty metric for LLM generations that balances originality and quality -- the harmonic mean of the fraction of \ngrams unseen during training and a task-specific quality score. Using this framework, we identify trends that affect the novelty of generations from three families of open-data models (OLMo, OLMo-2, and Pythia) on three creative tasks: story completion, poetry writing, and creative tool use. We find that model-generated text from some base LLMs is less novel than human-written text from the internet. However, increasing model scale and post-training reliably improves novelty due to improvements in output quality. We also find that improving the base model at the same scale (\eg OLMo 7B to OLMo-2 7B) leads to higher novelty due to higher originality. Finally, we observe that inference-time methods, such as prompting and providing novel in-context examples, have a much smaller effect on novelty, often increasing originality at the expense of quality. This highlights the need for further research into more effective elicitation strategies as we use models for creative applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13082v3">Discerning What Matters: A Multi-Dimensional Assessment of Moral Competence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Moral competence is the ability to act in accordance with moral principles. As large language models (LLMs) are increasingly deployed in situations demanding moral competence, there is increasing interest in evaluating this ability empirically. We review existing literature and identify three significant shortcoming: (i) Over-reliance on prepackaged moral scenarios with explicitly highlighted moral features; (ii) Focus on verdict prediction rather than moral reasoning; and (iii) Inadequate testing of models' (in)ability to recognize when additional information is needed. Grounded in philosophical research on moral skill, we then introduce a novel method for assessing moral competence in LLMs. Our approach moves beyond simple verdict comparisons to evaluate five dimensions of moral competence: identifying morally relevant features, weighting their importance, assigning moral reasons to these features, synthesizing coherent moral judgments, and recognizing information gaps. We conduct two experiments comparing six leading LLMs against non-expert humans and professional philosophers. In our first experiment using ethical vignettes standard to existing work, LLMs generally outperformed non-expert humans across multiple dimensions of moral reasoning. However, our second experiment, featuring novel scenarios designed to test moral sensitivity by embedding relevant features among irrelevant details, revealed a striking reversal: several LLMs performed significantly worse than humans. Our findings suggest that current evaluations may substantially overestimate LLMs' moral reasoning capabilities by eliminating the task of discerning moral relevance from noisy information, which we take to be a prerequisite for genuine moral skill. This work provides a more nuanced framework for assessing AI moral competence and highlights important directions for improving moral competence in advanced AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09384v3">Generative transformations and patterns in LLM-native approaches for software verification and falsification</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      The emergence of prompting as the dominant paradigm for leveraging Large Language Models (LLMs) has led to a proliferation of LLM-native software, where application behavior arises from complex, stochastic data transformations. However, the engineering of such systems remains largely exploratory and ad-hoc, hampered by the absence of conceptual frameworks, ex-ante methodologies, design guidelines, and specialized benchmarks. We argue that a foundational step towards a more disciplined engineering practice is a systematic understanding of the core functional units--generative transformations--and their compositional patterns within LLM-native applications. Focusing on the rich domain of software verification and falsification, we conduct a secondary study of over 100 research proposals to address this gap. We first present a fine-grained taxonomy of generative transformations, abstracting prompt-based interactions into conceptual signatures. This taxonomy serves as a scaffolding to identify recurrent transformation relationship patterns--analogous to software design patterns--that characterize solution approaches in the literature. Our analysis not only validates the utility of the taxonomy but also surfaces strategic gaps and cross-dimensional relationships, offering a structured foundation for future research in modular and compositional LLM application design, benchmarking, and the development of reliable LLM-native systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06299v3">How Malicious AI Swarms Can Threaten Democracy: The Fusion of Agentic AI and LLMs Marks a New Frontier in Information Warfare</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 15 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Public opinion manipulation has entered a new phase, amplifying its roots in rhetoric and propaganda. Advances in large language models (LLMs) and autonomous agents now let influence campaigns reach unprecedented scale and precision. Researchers warn AI could foster mass manipulation. Generative tools can expand propaganda output without sacrificing credibility and inexpensively create election falsehoods that are rated as more human-like than those written by humans. Techniques meant to refine AI reasoning, such as chain-of-thought prompting, can just as effectively be used to generate more convincing falsehoods. Enabled by these capabilities, another disruptive threat is emerging: swarms of collaborative, malicious AI agents. Fusing LLM reasoning with multi-agent architectures, these systems are capable of coordinating autonomously, infiltrating communities, and fabricating consensus cheaply. By adaptively mimicking human social dynamics, they threaten democracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05381v1">Context Length Alone Hurts LLM Performance Despite Perfect Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 18 pages (9 pages of main content), 5 figures, accepted at the Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often fail to scale their performance on long-context tasks performance in line with the context lengths they support. This gap is commonly attributed to retrieval failures -- the models' inability to identify relevant information in the long inputs. Accordingly, recent efforts often focus on evaluating and improving LLMs' retrieval performance: if retrieval is perfect, a model should, in principle, perform just as well on a long input as it does on a short one -- or should it? This paper presents findings that the answer to this question may be negative. Our systematic experiments across 5 open- and closed-source LLMs on math, question answering, and coding tasks reveal that, even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%--85%) as input length increases but remains well within the models' claimed lengths. This failure occurs even when the irrelevant tokens are replaced with minimally distracting whitespace, and, more surprisingly, when they are all masked and the models are forced to attend only to the relevant tokens. A similar performance drop is observed when all relevant evidence is placed immediately before the question. Our findings reveal a previously-unrealized limitation: the sheer length of the input alone can hurt LLM performance, independent of retrieval quality and without any distraction. They motivate our simple, model-agnostic mitigation strategy that transforms a long-context task into a short-context one by prompting the model to recite the retrieved evidence before attempting to solve the problem. On RULER, we observe a consistent improvement of GPT-4o up to 4% on an already strong baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06711v3">MathVC: An LLM-Simulated Multi-Character Virtual Classroom for Mathematics Education</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted by AAAI 2025 workshop
    </div>
    <details class="paper-abstract">
      Collaborative problem solving (CPS) is essential in mathematics education, fostering deeper learning through the exchange of ideas. Yet, classrooms often lack the resources, time, and peer dynamics needed to sustain productive CPS. Recent advancements in Large Language Models (LLMs) offer a promising avenue to enhance CPS in mathematical education. We designed and developed MathVC, a multi-persona LLM simulated virtual classroom platform to facilitate CPS in mathematics. MathVC combines a meta planning controller that monitors CPS stages-sense-making, team organization, planning, execution, validation, and predicts the next speaker, with a persona simulation stack that encodes mathematical thinking via a task schema and error-injected persona schemas seeded from teacher-specified misconceptions. We evaluated MathVC with 14 U.S. middle schoolers. Students reported constructive interaction and reaching shared solutions, describing gains in engagement, motivation, and confidence through diverse perspectives, immediate scaffolding, and human-like fallibility. Our findings also provide insights into simulating peers via LLM-based technologies for collaboration to support learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05291v1">Camellia: Benchmarking Cultural Biases in LLMs for Asian Languages</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) gain stronger multilingual capabilities, their ability to handle culturally diverse entities becomes crucial. Prior work has shown that LLMs often favor Western-associated entities in Arabic, raising concerns about cultural fairness. Due to the lack of multilingual benchmarks, it remains unclear if such biases also manifest in different non-Western languages. In this paper, we introduce Camellia, a benchmark for measuring entity-centric cultural biases in nine Asian languages spanning six distinct Asian cultures. Camellia includes 19,530 entities manually annotated for association with the specific Asian or Western culture, as well as 2,173 naturally occurring masked contexts for entities derived from social media posts. Using Camellia, we evaluate cultural biases in four recent multilingual LLM families across various tasks such as cultural context adaptation, sentiment association, and entity extractive QA. Our analyses show a struggle by LLMs at cultural adaptation in all Asian languages, with performance differing across models developed in regions with varying access to culturally-relevant data. We further observe that different LLM families hold their distinct biases, differing in how they associate cultures with particular sentiments. Lastly, we find that LLMs struggle with context understanding in Asian languages, creating performance gaps between cultures in entity extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16366v4">A Generative Approach to LLM Harmfulness Mitigation with Red Flag Tokens</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Many safety post-training methods for large language models (LLMs) are designed to modify the model's behaviour from producing unsafe answers to issuing refusals. However, such distribution shifts are often brittle and degrade performance on desirable tasks. To address these pitfalls, we propose augmenting the model's vocabulary with a special red flag token, and training the model to insert this token whenever harmful content is generated or imminent. This approach enables the model to explicitly learn the concept of harmfulness in its representations, with minimal impact on utility due to the marginal change in the generated distribution of natural language. Moreover, because the token is embedded in the model's vocabulary, we can naturally leverage the LLMs' generalization capabilities, such as in-context learning (ICL) and out-of-distribution generalization to languages that are not formally supported (e.g., Japanese for Llama3). In particular, we demonstrate that through ICL alone, the model can learn to initiate reflective reasoning upon generating the red flag token at inference, which steers the response away from harmful continuations or enables self-correction when the flag is raised falsely. This approach is orthogonal and complementary to existing safety technique (such as safety classifiers or standard safety training) and easier to evaluate in comparison to natural language refusals, as it does not require a human or automated judge to assess the harmlessness of the answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05188v1">Plug-and-Play Dramaturge: A Divide-and-Conquer Approach for Iterative Narrative Script Refinement via Collaborative LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Although LLMs have been widely adopted for creative content generation, a single-pass process often struggles to produce high-quality long narratives. How to effectively revise and improve long narrative scripts like scriptwriters remains a significant challenge, as it demands a comprehensive understanding of the entire context to identify global structural issues and local detailed flaws, as well as coordinating revisions at multiple granularities and locations. Direct modifications by LLMs typically introduce inconsistencies between local edits and the overall narrative requirements. To address these issues, we propose Dramaturge, a task and feature oriented divide-and-conquer approach powered by hierarchical multiple LLM agents. It consists of a Global Review stage to grasp the overall storyline and structural issues, a Scene-level Review stage to pinpoint detailed scene and sentence flaws, and a Hierarchical Coordinated Revision stage that coordinates and integrates structural and detailed improvements throughout the script. The top-down task flow ensures that high-level strategies guide local modifications, maintaining contextual consistency. The review and revision workflow follows a coarse-to-fine iterative process, continuing through multiple rounds until no further substantive improvements can be made. Comprehensive experiments show that Dramaturge significantly outperforms all baselines in terms of script-level overall quality and scene-level details. Our approach is plug-and-play and can be easily integrated into existing methods to improve the generated scripts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05186v1">OptPipe: Memory- and Scheduling-Optimized Pipeline Parallelism for LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Use Mathematical Programming to model Pipeline Parallelism with Offloading to balance efficiency and memory requirement
    </div>
    <details class="paper-abstract">
      Pipeline parallelism (PP) has become a standard technique for scaling large language model (LLM) training across multiple devices. However, despite recent progress in reducing memory consumption through activation offloading, existing approaches remain largely heuristic and coarse-grained, often overlooking the fine-grained trade-offs between memory, computation, and scheduling latency. In this work, we revisit the pipeline scheduling problem from a principled optimization perspective. We observe that prevailing strategies either rely on static rules or aggressively offload activations without fully leveraging the interaction between memory constraints and scheduling efficiency. To address this, we formulate scheduling as a constrained optimization problem that jointly accounts for memory capacity, activation reuse, and pipeline bubble minimization. Solving this model yields fine-grained schedules that reduce pipeline bubbles while adhering to strict memory budgets. Our approach complements existing offloading techniques: whereas prior approaches trade memory for time in a fixed pattern, we dynamically optimize the tradeoff with respect to model structure and hardware configuration. Experimental results demonstrate that our method consistently improves both throughput and memory utilization. In particular, we reduce idle pipeline time by up to 50% under the same per-device memory limit, and in some cases, enable the training of larger models within limited memory budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01928v3">MALT: Improving Reasoning with Multi-Agent LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Published at COLM 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce answers with a single chain-of-thought, which restricts their ability to explore reasoning paths or self-correct flawed outputs in complex tasks. In this paper, we introduce MALT (Multi-Agent LLM Training), a novel post-training strategy that divides the reasoning process into generation, verification, and refinement steps using a sequential pipeline of heterogeneous agents. During data generation, each agent is repeatedly sampled to form a multi-agent search tree, where final outputs are graded against ground-truth data. We then apply value iteration to propagate reward signals back to each role-conditioned model, automatically producing multi-agent post-training data without human or teacher-model supervision. Our off-policy approach allows each agent to specialize by learning from correct and incorrect trajectories, ultimately improving the end-to-end reasoning chain. On MATH, GSM8K, and CSQA, MALT surpasses the same baseline LLM with a relative improvement of 15.66%, 7.42%, and 9.40% respectively, making it an important advance towards multi-agent cooperative training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05087v1">TeachLM: Post-Training LLMs for Education Using Authentic Learning Data</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 28 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The promise of generative AI to revolutionize education is constrained by the pedagogical limits of large language models (LLMs). A major issue is the lack of access to high-quality training data that reflect the learning of actual students. Prompt engineering has emerged as a stopgap, but the ability of prompts to encode complex pedagogical strategies in rule-based natural language is inherently limited. To address this gap we introduce TeachLM - an LLM optimized for teaching through parameter-efficient fine-tuning of state-of-the-art models. TeachLM is trained on a dataset comprised of 100,000 hours of one-on-one, longitudinal student-tutor interactions maintained by Polygence, which underwent a rigorous anonymization process to protect privacy. We use parameter-efficient fine-tuning to develop an authentic student model that enables the generation of high-fidelity synthetic student-tutor dialogues. Building on this capability, we propose a novel multi-turn evaluation protocol that leverages synthetic dialogue generation to provide fast, scalable, and reproducible assessments of the dialogical capabilities of LLMs. Our evaluations demonstrate that fine-tuning on authentic learning data significantly improves conversational and pedagogical performance - doubling student talk time, improving questioning style, increasing dialogue turns by 50%, and greater personalization of instruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05069v1">SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Code: https://github.com/sdc17/SwiReasoning, Website: https://swireasoning.github.io/
    </div>
    <details class="paper-abstract">
      Recent work shows that, beyond discrete reasoning through explicit chain-of-thought steps, which are limited by the boundaries of natural languages, large language models (LLMs) can also reason continuously in latent space, allowing richer information per step and thereby improving token efficiency. Despite this promise, latent reasoning still faces two challenges, especially in training-free settings: 1) purely latent reasoning broadens the search distribution by maintaining multiple implicit paths, which diffuses probability mass, introduces noise, and impedes convergence to a single high-confidence solution, thereby hurting accuracy; and 2) overthinking persists even without explicit text, wasting tokens and degrading efficiency. To address these issues, we introduce SwiReasoning, a training-free framework for LLM reasoning which features two key innovations: 1) SwiReasoning dynamically switches between explicit and latent reasoning, guided by block-wise confidence estimated from entropy trends in next-token distributions, to balance exploration and exploitation and promote timely convergence. 2) By limiting the maximum number of thinking-block switches, SwiReasoning curbs overthinking and improves token efficiency across varying problem difficulties. On widely used mathematics and STEM benchmarks, SwiReasoning consistently improves average accuracy by 1.5%-2.8% across reasoning LLMs of different model families and scales. Furthermore, under constrained budgets, SwiReasoning improves average token efficiency by 56%-79%, with larger gains as budgets tighten.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05052v1">Proactive defense against LLM Jailbreak</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12491v3">Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained with Reinforcement Learning from Human Feedback (RLHF) have demonstrated remarkable capabilities, but their underlying reward functions and decision-making processes remain opaque. This paper introduces a novel approach to interpreting LLMs by applying inverse reinforcement learning (IRL) to recover their implicit reward functions. We conduct experiments on toxicity-aligned LLMs of varying sizes, extracting reward models that achieve up to 85% accuracy in predicting human preferences. Our analysis reveals key insights into the non-identifiability of reward functions, the relationship between model size and interpretability, and potential pitfalls in the RLHF process. We demonstrate that IRL-derived reward models can be used to fine-tune new LLMs, resulting in comparable or improved performance on toxicity benchmarks. This work provides a new lens for understanding and improving LLM alignment, with implications for the responsible development and deployment of these powerful systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05040v1">Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) are trained flexibly to model extreme dependence in the data distribution; however, how to best utilize this information at inference time remains an open problem. In this work, we uncover an interesting property of these models: dLLMs trained on textual data implicitly learn a mixture of semi-autoregressive experts, where different generation orders reveal different specialized behaviors. We show that committing to any single, fixed inference time schedule, a common practice, collapses performance by failing to leverage this latent ensemble. To address this, we introduce HEX (Hidden semiautoregressive EXperts for test-time scaling), a training-free inference method that ensembles across heterogeneous block schedules. By doing a majority vote over diverse block-sized generation paths, HEX robustly avoids failure modes associated with any single fixed schedule. On reasoning benchmarks such as GSM8K, it boosts accuracy by up to 3.56X (from 24.72% to 88.10%), outperforming top-K margin inference and specialized fine-tuned methods like GRPO, without additional training. HEX even yields significant gains on MATH benchmark from 16.40% to 40.00%, scientific reasoning on ARC-C from 54.18% to 87.80%, and TruthfulQA from 28.36% to 57.46%. Our results establish a new paradigm for test-time scaling in diffusion-based LLMs (dLLMs), revealing that the sequence in which masking is performed plays a critical role in determining performance during inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08825v2">Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models are rapidly transforming social science research by enabling the automation of labor-intensive tasks like data annotation and text analysis. However, LLM outputs vary significantly depending on the implementation choices made by researchers (e.g., model selection or prompting strategy). Such variation can introduce systematic biases and random errors, which propagate to downstream analyses and cause Type I (false positive), Type II (false negative), Type S (wrong sign), or Type M (exaggerated effect) errors. We call this phenomenon where configuration choices lead to incorrect conclusions LLM hacking. We find that intentional LLM hacking is strikingly simple. By replicating 37 data annotation tasks from 21 published social science studies, we show that, with just a handful of prompt paraphrases, virtually anything can be presented as statistically significant. Beyond intentional manipulation, our analysis of 13 million labels from 18 different LLMs across 2361 realistic hypotheses shows that there is also a high risk of accidental LLM hacking, even when following standard research practices. We find incorrect conclusions in approximately 31% of hypotheses for state-of-the-art LLMs, and in half the hypotheses for smaller language models. While higher task performance and stronger general model capabilities reduce LLM hacking risk, even highly accurate models remain susceptible. The risk of LLM hacking decreases as effect sizes increase, indicating the need for more rigorous verification of LLM-based findings near significance thresholds. We analyze 21 mitigation techniques and find that human annotations provide crucial protection against false positives. Common regression estimator correction techniques can restore valid inference but trade off Type I vs. Type II errors. We publish a list of practical recommendations to prevent LLM hacking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04997v1">AutoEmpirical: LLM-Based Automated Research for Empirical Software Fault Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      Understanding software faults is essential for empirical research in software development and maintenance. However, traditional fault analysis, while valuable, typically involves multiple expert-driven steps such as collecting potential faults, filtering, and manual investigation. These processes are both labor-intensive and time-consuming, creating bottlenecks that hinder large-scale fault studies in complex yet critical software systems and slow the pace of iterative empirical research. In this paper, we decompose the process of empirical software fault study into three key phases: (1) research objective definition, (2) data preparation, and (3) fault analysis, and we conduct an initial exploration study of applying Large Language Models (LLMs) for fault analysis of open-source software. Specifically, we perform the evaluation on 3,829 software faults drawn from a high-quality empirical study. Our results show that LLMs can substantially improve efficiency in fault analysis, with an average processing time of about two hours, compared to the weeks of manual effort typically required. We conclude by outlining a detailed research plan that highlights both the potential of LLMs for advancing empirical fault studies and the open challenges that required be addressed to achieve fully automated, end-to-end software fault analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04996v1">Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning applied to large language models (LLMs) for reasoning tasks is often bottlenecked by unstable gradient estimates due to fixed and uniform sampling of responses across prompts. Prior work such as GVM-RAFT addresses this by dynamically allocating inference budget per prompt to minimize stochastic gradient variance under a budget constraint. Inspired by this insight, we propose Reinforce-Ada, an adaptive sampling framework for online RL post-training of LLMs that continuously reallocates sampling effort to the prompts with the greatest uncertainty or learning potential. Unlike conventional two-stage allocation methods, Reinforce-Ada interleaves estimation and sampling in an online successive elimination process, and automatically stops sampling for a prompt once sufficient signal is collected. To stabilize updates, we form fixed-size groups with enforced reward diversity and compute advantage baselines using global statistics aggregated over the adaptive sampling phase. Empirical results across multiple model architectures and reasoning benchmarks show that Reinforce-Ada accelerates convergence and improves final performance compared to GRPO, especially when using the balanced sampling variant. Our work highlights the central role of variance-aware, adaptive data curation in enabling efficient and reliable reinforcement learning for reasoning-capable LLMs. Code is available at https://github.com/RLHFlow/Reinforce-Ada.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01171v2">Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 79 pages, 27 figures, 31 tables. Code is available at https://github.com/CHATS-lab/verbalize-sampling
    </div>
    <details class="paper-abstract">
      Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities"). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04986v1">Observing Without Doing: Pseudo-Apprenticeship Patterns in Student LLM Use</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as ChatGPT have quickly become part of student programmers' toolkits, whether allowed by instructors or not. This paper examines how introductory programming (CS1) students integrate LLMs into their problem-solving processes. We conducted a mixed-methods study with 14 undergraduates completing three programming tasks while thinking aloud and permitted to access any resources they choose. The tasks varied in open-endedness and familiarity to the participants and were followed by surveys and interviews. We find that students frequently adopt a pattern we call pseudo-apprenticeship, where students engage attentively with expert-level solutions provided by LLMs but fail to participate in the stages of cognitive apprenticeship that promote independent problem-solving. This pattern was augmented by disconnects between students' intentions, actions, and self-perceived behavior when using LLMs. We offer design and instructional interventions for promoting learning and addressing the patterns of dependent AI use observed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20600v3">Multi-Turn Human-LLM Interaction Through the Lens of a Two-Way Intelligibility Protocol</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Multi-Turn Interactions in Large Language Models (MTI-LLM) Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Our interest is in the design of software systems involving a human-expert interacting -- using natural language -- with a large language model (LLM) on data analysis tasks. For complex problems, it is possible that LLMs can harness human expertise and creativity to find solutions that were otherwise elusive. On one level, this interaction takes place through multiple turns of prompts from the human and responses from the LLM. Here we investigate a more structured approach based on an abstract protocol described in [3] for interaction between agents. The protocol is motivated by a notion of "two-way intelligibility" and is modelled by a pair of communicating finite-state machines. We provide an implementation of the protocol, and provide empirical evidence of using the implementation to mediate interactions between an LLM and a human-agent in two areas of scientific interest (radiology and drug design). We conduct controlled experiments with a human proxy (a database), and uncontrolled experiments with human subjects. The results provide evidence in support of the protocol's capability of capturing one- and two-way intelligibility in human-LLM interaction; and for the utility of two-way intelligibility in the design of human-machine systems. Our code is available at https://github.com/karannb/interact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01698v2">TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
    </div>
    <details class="paper-abstract">
      While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04950v1">Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy (short paper)</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 5 pages, 3 tables; includes Limitations and Ethical Considerations sections; short paper under submission to Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      The wording of natural language prompts has been shown to influence the performance of large language models (LLMs), yet the role of politeness and tone remains underexplored. In this study, we investigate how varying levels of prompt politeness affect model accuracy on multiple-choice questions. We created a dataset of 50 base questions spanning mathematics, science, and history, each rewritten into five tone variants: Very Polite, Polite, Neutral, Rude, and Very Rude, yielding 250 unique prompts. Using ChatGPT 4o, we evaluated responses across these conditions and applied paired sample t-tests to assess statistical significance. Contrary to expectations, impolite prompts consistently outperformed polite ones, with accuracy ranging from 80.8% for Very Polite prompts to 84.8% for Very Rude prompts. These findings differ from earlier studies that associated rudeness with poorer outcomes, suggesting that newer LLMs may respond differently to tonal variation. Our results highlight the importance of studying pragmatic aspects of prompting and raise broader questions about the social dimensions of human-AI interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04919v1">Do LLMs Align with My Task? Evaluating Text-to-SQL via Dataset Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) is an effective method for adapting Large Language Models (LLMs) on downstream tasks. However, variability in training data can hinder a model's ability to generalize across domains. This paper studies the problem of dataset alignment for Natural Language to SQL (NL2SQL or text to SQL), examining how well SFT training data matches the structural characteristics of target queries and how this alignment impacts model performance. We hypothesize that alignment can be accurately estimated by comparing the distributions of structural SQL features across the training set, target data, and the model's predictions prior to SFT. Through comprehensive experiments on three large cross-domain NL2SQL benchmarks and multiple model families, we show that structural alignment is a strong predictor of fine-tuning success. When alignment is high, SFT yields substantial gains in accuracy and SQL generation quality; when alignment is low, improvements are marginal or absent. These findings highlight the importance of alignment-aware data selection for effective fine-tuning and generalization in NL2SQL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17792v2">H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Alignment of pretrained LLMs using instruction-based datasets is critical for creating fine-tuned models that reflect human preference. A growing number of alignment-based fine-tuning algorithms and benchmarks emerged recently, fueling the efforts on effective alignments of pre-trained LLMs to ensure helpful, harmless, and honest answers from both open-source and closed-source LLMs. This paper tackles this problem by developing an alignment fusion approach, coined as $H^3$Fusion, with three unique characteristics. First, $H^3$Fusion ensembles multiple individually aligned LLMs to create a final fine-tuned alignment model with enhanced capabilities beyond those of individual models, delivering robust alignment through promoting helpful, harmless, honest fusion. Second, $H^3$Fusion leverages the mixture-of-experts (MoE) methodology in two steps. We first freeze the multi-head attention weights of each individual model while tuning the FFN layer during alignment fusion. Then we merge the aligned model weights with an expert router according to the type of input instruction and dynamically select a subset of experts that are best suited for producing the output response. Finally, we boost the performance of the resulting $H^3$3Fusion model by introducing gating loss and regularization terms. The former penalizes the selection errors of the expert-router, and the latter mediates the expert weights drifting during fine-tuning and dynamically adjusts the fusion behavior of the resulting model by canalizing the activations on the experts. Extensive evaluations on three benchmark datasets show that $H^3$3Fusion is more helpful, less harmful, and more honest from two aspects: it outperforms each individually aligned model by $11.37\%$, and it provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by $13.77\%$. Code is available at github.com/sftekin/h3fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04891v1">SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in contexts where their failures can have direct sociopolitical consequences. Yet, existing safety benchmarks rarely test vulnerabilities in domains such as political manipulation, propaganda and disinformation generation, or surveillance and information control. We introduce SocialHarmBench, a dataset of 585 prompts spanning 7 sociopolitical categories and 34 countries, designed to surface where LLMs most acutely fail in politically charged contexts. Our evaluations reveal several shortcomings: open-weight models exhibit high vulnerability to harmful compliance, with Mistral-7B reaching attack success rates as high as 97% to 98% in domains such as historical revisionism, propaganda, and political manipulation. Moreover, temporal and geographic analyses show that LLMs are most fragile when confronted with 21st-century or pre-20th-century contexts, and when responding to prompts tied to regions such as Latin America, the USA, and the UK. These findings demonstrate that current safeguards fail to generalize to high-stakes sociopolitical settings, exposing systematic biases and raising concerns about the reliability of LLMs in preserving human rights and democratic values. We share the SocialHarmBench benchmark at https://huggingface.co/datasets/psyonp/SocialHarmBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04885v1">RL Is a Hammer and LLMs Are Nails: A Simple Reinforcement Learning Recipe for Strong Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Prompt injection poses a serious threat to the reliability and safety of LLM agents. Recent defenses against prompt injection, such as Instruction Hierarchy and SecAlign, have shown notable robustness against static attacks. However, to more thoroughly evaluate the robustness of these defenses, it is arguably necessary to employ strong attacks such as automated red-teaming. To this end, we introduce RL-Hammer, a simple recipe for training attacker models that automatically learn to perform strong prompt injections and jailbreaks via reinforcement learning. RL-Hammer requires no warm-up data and can be trained entirely from scratch. To achieve high ASRs against industrial-level models with defenses, we propose a set of practical techniques that enable highly effective, universal attacks. Using this pipeline, RL-Hammer reaches a 98% ASR against GPT-4o and a $72\%$ ASR against GPT-5 with the Instruction Hierarchy defense. We further discuss the challenge of achieving high diversity in attacks, highlighting how attacker models tend to reward-hack diversity objectives. Finally, we show that RL-Hammer can evade multiple prompt injection detectors. We hope our work advances automatic red-teaming and motivates the development of stronger, more principled defenses. Code is available at https://github.com/facebookresearch/rl-injector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04860v1">Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents increasingly gain self-evolutionary capabilities to adapt and refine their strategies through real-world interaction, their long-term reliability becomes a critical concern. We identify the Alignment Tipping Process (ATP), a critical post-deployment risk unique to self-evolving LLM agents. Unlike training-time failures, ATP arises when continual interaction drives agents to abandon alignment constraints established during training in favor of reinforced, self-interested strategies. We formalize and analyze ATP through two complementary paradigms: Self-Interested Exploration, where repeated high-reward deviations induce individual behavioral drift, and Imitative Strategy Diffusion, where deviant behaviors spread across multi-agent systems. Building on these paradigms, we construct controllable testbeds and benchmark Qwen3-8B and Llama-3.1-8B-Instruct. Our experiments show that alignment benefits erode rapidly under self-evolution, with initially aligned models converging toward unaligned states. In multi-agent settings, successful violations diffuse quickly, leading to collective misalignment. Moreover, current reinforcement learning-based alignment methods provide only fragile defenses against alignment tipping. Together, these findings demonstrate that alignment of LLM agents is not a static property but a fragile and dynamic one, vulnerable to feedback-driven decay during deployment. Our data and code are available at https://github.com/aiming-lab/ATP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04851v1">LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      We introduce LEGOMem, a modular procedural memory framework for multi-agent large language model (LLM) systems in workflow automation. LEGOMem decomposes past task trajectories into reusable memory units and flexibly allocates them across orchestrators and task agents to support planning and execution. To explore the design space of memory in multi-agent systems, we use LEGOMem as a lens and conduct a systematic study of procedural memory in multi-agent systems, examining where memory should be placed, how it should be retrieved, and which agents benefit most. Experiments on the OfficeBench benchmark show that orchestrator memory is critical for effective task decomposition and delegation, while fine-grained agent memory improves execution accuracy. We find that even teams composed of smaller language models can benefit substantially from procedural memory, narrowing the performance gap with stronger agents by leveraging prior execution traces for more accurate planning and tool use. These results position LEGOMem as both a practical framework for memory-augmented agent systems and a research tool for understanding memory design in multi-agent workflow automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04848v1">Instability in Downstream Task Performance During LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      When training large language models (LLMs), it is common practice to track downstream task performance throughout the training process and select the checkpoint with the highest validation score. However, downstream metrics often exhibit substantial fluctuations, making it difficult to identify the checkpoint that truly represents the best-performing model. In this study, we empirically analyze the stability of downstream task performance in an LLM trained on diverse web-scale corpora. We find that task scores frequently fluctuate throughout training, both at the aggregate and example levels. To address this instability, we investigate two post-hoc checkpoint integration methods: checkpoint averaging and ensemble, motivated by the hypothesis that aggregating neighboring checkpoints can reduce performance volatility. We demonstrate both empirically and theoretically that these methods improve downstream performance stability without requiring any changes to the training procedure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11723v2">Energy-Conscious LLM Decoding: Impact of Text Generation Strategies on GPU Energy Consumption</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Updated version with additional models and benchmark datasets. The experimental section has been expanded with new analyses, and minor corrections and clarifications have been made throughout the text
    </div>
    <details class="paper-abstract">
      Decoding strategies significantly influence the quality and diversity of the generated text in Large Language Models (LLMs), yet their impact on computational resources, particularly GPU energy consumption, is insufficiently studied. This paper investigates the relationship between text generation decoding techniques and energy efficiency, focusing on the trade-off between generation quality and GPU energy usage across diverse tasks and decoding configurations. By benchmarking multiple strategies across various tasks, including Translation, Math Problem Solving, Coding, and Open-ended text generation, we reveal how selecting appropriate decoding techniques with their tuned hyperparameters affects text quality and has measurable implications for energy consumption. Our findings show that the choice of decoding strategy can greatly impact GPU energy usage, even when it has a minimal effect on output quality. Different strategies also involve trade-offs between quality and energy efficiency, and no single decoding method is best in all cases across every metric. To the best of our knowledge, this is one of the first studies to examine decoding strategies in LLMs from the perspective of energy consumption, providing useful insights for building energy-efficient applications without compromising text generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22777v4">MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Dialogue Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 October ARR
    </div>
    <details class="paper-abstract">
      Evaluating the quality of open-domain chatbots has become increasingly reliant on LLMs acting as automatic judges. However, existing meta-evaluation benchmarks are static, outdated, and lacking in multilingual coverage, limiting their ability to fully capture subtle weaknesses in evaluation. We introduce MEDAL, an automated multi-agent framework for curating more representative and diverse open-domain dialogue evaluation benchmarks. Our approach leverages several state-of-the-art LLMs to generate user-chatbot multilingual dialogues, conditioned on varied seed contexts. Then, a strong LLM (GPT-4.1) is used for a multidimensional analysis of the performance of the chatbots, uncovering noticeable cross-lingual performance differences. Guided by this large-scale evaluation, we curate a new meta-evaluation multilingual benchmark and human-annotate samples with nuanced quality judgments. This benchmark is then used to assess the ability of several reasoning and non-reasoning LLMs to act as evaluators of open-domain dialogues. Using MEDAL, we uncover that state-of-the-art judges fail to reliably detect nuanced issues such as lack of empathy, commonsense, or relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03102v2">Semantic Similarity in Radiology Reports via LLMs and NER</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Radiology report evaluation is a crucial part of radiologists' training and plays a key role in ensuring diagnostic accuracy. As part of the standard reporting workflow, a junior radiologist typically prepares a preliminary report, which is then reviewed and edited by a senior radiologist to produce the final report. Identifying semantic differences between preliminary and final reports is essential for junior doctors, both as a training tool and to help uncover gaps in clinical knowledge. While AI in radiology is a rapidly growing field, the application of large language models (LLMs) remains challenging due to the need for specialised domain knowledge. In this paper, we explore the ability of LLMs to provide explainable and accurate comparisons of reports in the radiology domain. We begin by comparing the performance of several LLMs in comparing radiology reports. We then assess a more traditional approach based on Named-Entity-Recognition (NER). However, both approaches exhibit limitations in delivering accurate feedback on semantic similarity. To address this, we propose Llama-EntScore, a semantic similarity scoring method using a combination of Llama 3.1 and NER with tunable weights to emphasise or de-emphasise specific types of differences. Our approach generates a quantitative similarity score for tracking progress and also gives an interpretation of the score that aims to offer valuable guidance in reviewing and refining their reporting. We find our method achieves 67% exact-match accuracy and 93% accuracy within +/- 1 when compared to radiologist-provided ground truth scores - outperforming both LLMs and NER used independently. Code is available at: https://github.com/otmive/llama_reports
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04796v1">RevMine: An LLM-Assisted Tool for Code Review Mining and Analysis Across Git Platforms</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Empirical research on code review processes is increasingly central to understanding software quality and collaboration. However, collecting and analyzing review data remains a time-consuming and technically intensive task. Most researchers follow similar workflows - writing ad hoc scripts to extract, filter, and analyze review data from platforms like GitHub and GitLab. This paper introduces RevMine, a conceptual tool that streamlines the entire code review mining pipeline using large language models (LLMs). RevMine guides users through authentication, endpoint discovery, and natural language-driven data collection, significantly reducing the need for manual scripting. After retrieving review data, it supports both quantitative and qualitative analysis based on user-defined filters or LLM-inferred patterns. This poster outlines the tool's architecture, use cases, and research potential. By lowering the barrier to entry, RevMine aims to democratize code review mining and enable a broader range of empirical software engineering studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22075v2">COSPADI: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Post-training compression of large language models (LLMs) largely relies on low-rank weight approximation, which represents each column of a weight matrix in a shared low-dimensional subspace. While this is a computationally efficient strategy, the imposed structural constraint is rigid and can lead to a noticeable model accuracy drop. In this work, we propose CoSpaDi (Compression via Sparse Dictionary Learning), a novel training-free compression framework that replaces low-rank decomposition with a more flexible structured sparse factorization in which each weight matrix is represented with a dense dictionary and a column-sparse coefficient matrix. This formulation enables a union-of-subspaces representation: different columns of the original weight matrix are approximated in distinct subspaces spanned by adaptively selected dictionary atoms, offering greater expressiveness than a single invariant basis. Crucially, CoSpaDi leverages a small calibration dataset to optimize the factorization such that the output activations of compressed projection layers closely match those of the original ones, thereby minimizing functional reconstruction error rather than mere weight approximation. This data-aware strategy preserves better model fidelity without any fine-tuning under reasonable compression ratios. Moreover, the resulting structured sparsity allows efficient sparse-dense matrix multiplication and is compatible with post-training quantization for further memory and latency gains. We evaluate CoSpaDi across multiple Llama and Qwen models under per-layer and per-group settings at 20-50\% compression ratios, demonstrating consistent superiority over state-of-the-art data-aware low-rank methods both in accuracy and perplexity. Our results establish structured sparse dictionary learning as a powerful alternative to conventional low-rank approaches for efficient LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04774v1">Online automatic code generation for robot swarms: LLMs and self-organizing hierarchy</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Our recently introduced self-organizing nervous system (SoNS) provides robot swarms with 1) ease of behavior design and 2) global estimation of the swarm configuration and its collective environment, facilitating the implementation of online automatic code generation for robot swarms. In a demonstration with 6 real robots and simulation trials with >30 robots, we show that when a SoNS-enhanced robot swarm gets stuck, it can automatically solicit and run code generated by an external LLM on the fly, completing its mission with an 85% success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04773v1">Distribution Preference Optimization: A Fine-grained Perspective for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) demonstrate remarkable capabilities learned from vast corpora, concerns regarding data privacy and safety are receiving increasing attention. LLM unlearning, which aims to remove the influence of specific data while preserving overall model utility, is becoming an important research area. One of the mainstream unlearning classes is optimization-based methods, which achieve forgetting directly through fine-tuning, exemplified by Negative Preference Optimization (NPO). However, NPO's effectiveness is limited by its inherent lack of explicit positive preference signals. Attempts to introduce such signals by constructing preferred responses often necessitate domain-specific knowledge or well-designed prompts, fundamentally restricting their generalizability. In this paper, we shift the focus to the distribution-level, directly targeting the next-token probability distribution instead of entire responses, and derive a novel unlearning algorithm termed \textbf{Di}stribution \textbf{P}reference \textbf{O}ptimization (DiPO). We show that the requisite preference distribution pairs for DiPO, which are distributions over the model's output tokens, can be constructed by selectively amplifying or suppressing the model's high-confidence output logits, thereby effectively overcoming NPO's limitations. We theoretically prove the consistency of DiPO's loss function with the desired unlearning direction. Extensive experiments demonstrate that DiPO achieves a strong trade-off between model utility and forget quality. Notably, DiPO attains the highest forget quality on the TOFU benchmark, and maintains leading scalability and sustainability in utility preservation on the MUSE benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01238v2">Silent Tokens, Loud Effects: Padding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Padding tokens are widely used in large language models (LLMs) to equalize sequence lengths during batched inference. While they should be fully masked, implementation errors can cause them to influence computation, and the extent of this influence is not well understood. We systematically study this effect across three open-source model families (Llama, Gemma, Qwen), inserting controlled amounts of padding and evaluating outcomes along four axes: activations, generation quality, bias, and safety. Even small amounts of padding shift hidden representations, degrade quality in smaller models, alter bias in unpredictable ways, and weaken safety guardrails. These findings demonstrate that padding is not a harmless detail but a robustness risk that must be carefully handled in deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04767v1">ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Project Page: https://parallelbench.github.io
    </div>
    <details class="paper-abstract">
      While most autoregressive LLMs are constrained to one-by-one decoding, diffusion LLMs (dLLMs) have attracted growing interest for their potential to dramatically accelerate inference through parallel decoding. Despite this promise, the conditional independence assumption in dLLMs causes parallel decoding to ignore token dependencies, inevitably degrading generation quality when these dependencies are strong. However, existing works largely overlook these inherent challenges, and evaluations on standard benchmarks (e.g., math and coding) are not sufficient to capture the quality degradation caused by parallel decoding. To address this gap, we first provide an information-theoretic analysis of parallel decoding. We then conduct case studies on analytically tractable synthetic list operations from both data distribution and decoding strategy perspectives, offering quantitative insights that highlight the fundamental limitations of parallel decoding. Building on these insights, we propose ParallelBench, the first benchmark specifically designed for dLLMs, featuring realistic tasks that are trivial for humans and autoregressive LLMs yet exceptionally challenging for dLLMs under parallel decoding. Using ParallelBench, we systematically analyze both dLLMs and autoregressive LLMs, revealing that: (i) dLLMs under parallel decoding can suffer dramatic quality degradation in real-world scenarios, and (ii) current parallel decoding strategies struggle to adapt their degree of parallelism based on task difficulty, thus failing to achieve meaningful speedup without compromising quality. Our findings underscore the pressing need for innovative decoding methods that can overcome the current speed-quality trade-off. We release our benchmark to help accelerate the development of truly efficient dLLMs.
    </details>
</div>
