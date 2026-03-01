# llm - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20718v2">Stabilizing Off-Policy Training for Long-Horizon LLM Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) algorithms such as PPO and GRPO are widely used to train large language models (LLMs) for multi-turn agentic tasks. However, in off-policy training pipelines, these methods often exhibit unstable optimization dynamics and are prone to performance collapse. Through empirical analysis, we identify two fundamental sources of instability in this setting: (1)~a granularity mismatch between token-level policy optimization and turn-structured interactions, and (2) high-variance and unreliable gradient updates induced by off-policy importance sampling and inaccurate advantage estimation. To address these challenges, we propose SORL, \underline{S}tabilizing \underline{O}ff-Policy \underline{R}einforcement \underline{L}earning for Long-Horizon Agent Training. SORL introduces principled mechanisms that align policy optimization with the structure of multi-turn interactions and adaptively suppress unreliable off-policy updates, yielding more conservative and robust learning dynamics. Within this framework, we instantiate two stabilized algorithms: SO-PPO and SO-GRPO. Both algorithms are designed to mitigate gradient variance and prevent optimization collapse without requiring careful early stopping or heuristic tuning. We evaluate SO-PPO and SO-GRPO on a range of multi-turn search benchmarks, including general question answering, multi-hop question answering, and medical multiple-choice QA tasks. Experimental results show that both methods consistently prevent training instabilities and performance collapses observed in standard PPO and GRPO, maintain lower clipping ratios and more stable optimization trajectories, and achieve superior or comparable task performance. These results demonstrate that the proposed algorithm provides a practical, scalable, and general framework for stabilizing reinforcement learning in multi-turn LLM agent training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07658v2">From Raw Corpora to Domain Benchmarks: Automated Evaluation of LLM Domain Expertise</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 35 pages, 24 figures. Second version
    </div>
    <details class="paper-abstract">
      Accurate domain-specific benchmarking of LLMs is essential, specifically in domains with direct implications for humans, such as law, healthcare, and education. However, existing benchmarks are documented to be contaminated and are based on multiple choice questions, which suffer from inherent biases. To measure domain-specific knowledge in LLMs, we present a deterministic pipeline that transforms raw domain corpora into completion-style benchmarks without relying on other LLMs or costly human annotation. Our approach first extracts domain-specific keywords and related target vocabulary from an input corpus. It then constructs prompt-target pairs where domain-specific words serve as prediction targets. By measuring LLMs' ability to complete these prompts, we provide a direct assessment of domain knowledge at low computational cost. Our pipeline avoids benchmark contamination, enables automated updates with new domain data, and facilitates fair comparisons between base and instruction-tuned (chat) models. We validate our approach by showing that model performances on our benchmark significantly correlate with those on an expert-curated benchmark. We then demonstrate how our benchmark provides insights into knowledge acquisition in domain-adaptive, continual, and general pretraining. Finally, we examine the effects of instruction fine-tuning by comparing base and chat models within our unified evaluation framework. In conclusion, our pipeline enables scalable, domain-specific, LLM-independent, and unbiased evaluation of both base and chat models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12259v2">Think like a Scientist: Physics-guided LLM Agent for Equation Discovery</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Explaining observed phenomena through symbolic, interpretable formulas is a fundamental goal of science. Recently, large language models (LLMs) have emerged as promising tools for symbolic equation discovery, owing to their broad domain knowledge and strong reasoning capabilities. However, most existing LLM-based systems try to guess equations directly from data, without modeling the multi-step reasoning process that scientists often follow: first inferring physical properties such as symmetries, then using these as priors to restrict the space of candidate equations. We introduce KeplerAgent, an agentic framework that explicitly follows this scientific reasoning process. The agent coordinates physics-based tools to extract intermediate structure and uses these results to configure symbolic regression engines such as PySINDy and PySR, including their function libraries and structural constraints. Across a suite of physical equation benchmarks, KeplerAgent achieves substantially higher symbolic accuracy and greater robustness to noisy data than both LLM and traditional baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12033v2">EARL: Entropy-Aware RL Alignment of LLMs for Reliable RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Accepted to DAC 2026
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated significant potential in hardware design automation, particularly in using natural language to synthesize Register-Transfer Level (RTL) code. Despite this progress, a gap remains between model capability and the demands of real-world RTL design, including syntax errors, functional hallucinations, and weak alignment to designer intent. Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising approach to bridge this gap, as hardware provides executable and formally checkable signals that can be used to further align model outputs with design intent. However, in long, structured RTL code sequences, not all tokens contribute equally to functional correctness, and naïvely spreading gradients across all tokens dilutes learning signals. A key insight from our entropy analysis in RTL generation is that only a small fraction of tokens (e.g., always, if, assign, posedge) exhibit high uncertainty and largely influence control flow and module structure. To address these challenges, we present EARL, an Entropy-Aware Reinforcement Learning framework for Verilog generation. EARL performs policy optimization using verifiable reward signals and introduces entropy-guided selective updates that gate policy gradients to high-entropy tokens. This approach preserves training stability and concentrates gradient updates on functionally important regions of code. Our experiments on VerilogEval and RTLLM show that EARL improves functional pass rates over prior LLM baselines by up to 14.7%, while reducing unnecessary updates and improving training stability. These results indicate that focusing RL on critical, high-uncertainty tokens enables more reliable and targeted policy improvement for structured RTL code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.04517v2">DOTResize: Reducing LLM Width via Discrete Optimal Transport-based Neuron Merging</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Structured pruning methods designed for Large Language Models (LLMs) generally focus on identifying and removing the least important components to optimize model size. However, in this work, we question this prevalent approach by instead exploring how to recombine information from structures designated for pruning back into the reduced model. We specifically focus on neuron width reduction, and frame this problem as a Discrete Optimal Transport problem, and propose DOTResize, a novel Transformer compression method that uses optimal transport theory to transform and compress model width. To ensure applicability within the Transformer architecture, we motivate and incorporate necessary entropic regularization and matrix factorization techniques into the transportation maps produced by our method. Unlike pruning-based approaches which discard neurons based on importance measures, DOTResize re-projects the entire neuron width, allowing the retention and redistribution of useful signal across the reduced layer. Empirical results show that compared to simple or state-of-the-art neuron width-pruning techniques, DOTResize serves as a useful add-on to pruning, while achieving measurable reductions in real-world computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21320v1">Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming the foundation for autonomous agents that can use tools to solve complex tasks. Reinforcement learning (RL) has emerged as a common approach for injecting such agentic capabilities, but typically under tightly controlled training setups. It often depends on carefully constructed task-solution pairs and substantial human supervision, which creates a fundamental obstacle to open-ended self-evolution toward superintelligent systems. In this paper, we propose Tool-R0 framework for training general purpose tool-calling agents from scratch with self-play RL, under a zero-data assumption. Initialized from the same base LLM, Tool-R0 co-evolves a Generator and a Solver with complementary rewards: one proposes targeted challenging tasks at the other's competence frontier and the other learns to solve them with real-world tool calls. This creates a self-evolving cycle that requires no pre-existing tasks or datasets. Evaluation on different tool-use benchmarks show that Tool-R0 yields 92.5 relative improvement over the base model and surpasses fully supervised tool-calling baselines under the same setting. Our work further provides empirical insights into self-play LLM agents by analyzing co-evolution, curriculum dynamics, and scaling behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09886v2">Probabilistic distances-based hallucination detection in LLMs with RAG</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Updated approach to constructing a hallucination detection score. Added results from experiments with the NLI task. The approach with trainable deep kernels has been removed, with a focus on the unsupervised approach
    </div>
    <details class="paper-abstract">
      Detecting hallucinations in large language models (LLMs) is critical for their safety in many applications. Without proper detection, these systems often provide harmful, unreliable answers. In recent years, LLMs have been actively used in retrieval-augmented generation (RAG) settings. However, hallucinations remain even in this setting, and while numerous hallucination detection methods have been proposed, most approaches are not specifically designed for RAG systems. To overcome this limitation, we introduce a hallucination detection method based on estimating the distances between the distributions of prompt token embeddings and language model response token embeddings. The method examines the geometric structure of token hidden states to reliably extract a signal of factuality in text, while remaining friendly to long sequences. Extensive experiments demonstrate that our method achieves state-of-the-art or competitive performance. It also has transferability from solving the NLI task to the hallucination detection task, making it a fully unsupervised and efficient method with a competitive performance on the final task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21198v1">Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Embodied LLMs endow robots with high-level task reasoning, but they cannot reflect on what went wrong or why, turning deployment into a sequence of independent trials where mistakes repeat rather than accumulate into experience. Drawing upon human reflective practitioners, we introduce Reflective Test-Time Planning, which integrates two modes of reflection: \textit{reflection-in-action}, where the agent uses test-time scaling to generate and score multiple candidate actions using internal reflections before execution; and \textit{reflection-on-action}, which uses test-time training to update both its internal reflection model and its action policy based on external reflections after execution. We also include retrospective reflection, allowing the agent to re-evaluate earlier decisions and perform model updates with hindsight for proper long-horizon credit assignment. Experiments on our newly-designed Long-Horizon Household benchmark and MuJoCo Cupboard Fitting benchmark show significant gains over baseline models, with ablative studies validating the complementary roles of reflection-in-action and reflection-on-action. Qualitative analyses, including real-robot trials, highlight behavioral correction through reflection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21178v1">XMorph: Explainable Brain Tumor Analysis Via LLM-Assisted Hybrid Deep Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Accepted in ICCABS 2026: The 14th International Conference on Computational Advances in Bio and Medical Sciences
    </div>
    <details class="paper-abstract">
      Deep learning has significantly advanced automated brain tumor diagnosis, yet clinical adoption remains limited by interpretability and computational constraints. Conventional models often act as opaque ''black boxes'' and fail to quantify the complex, irregular tumor boundaries that characterize malignant growth. To address these challenges, we present XMorph, an explainable and computationally efficient framework for fine-grained classification of three prominent brain tumor types: glioma, meningioma, and pituitary tumors. We propose an Information-Weighted Boundary Normalization (IWBN) mechanism that emphasizes diagnostically relevant boundary regions alongside nonlinear chaotic and clinically validated features, enabling a richer morphological representation of tumor growth. A dual-channel explainable AI module combines GradCAM++ visual cues with LLM-generated textual rationales, translating model reasoning into clinically interpretable insights. The proposed framework achieves a classification accuracy of 96.0%, demonstrating that explainability and high performance can co-exist in AI-based medical imaging systems. The source code and materials for XMorph are all publicly available at: https://github.com/ALSER-Lab/XMorph.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21161v1">ActionReasoning: Robot Action Reasoning in 3D Space with LLM for Robotic Brick Stacking</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 8 pages, 5 figures, accepted by the 2026 IEEE International Conference on Robotics and Automation
    </div>
    <details class="paper-abstract">
      Classical robotic systems typically rely on custom planners designed for constrained environments. While effective in restricted settings, these systems lack generalization capabilities, limiting the scalability of embodied AI and general-purpose robots. Recent data-driven Vision-Language-Action (VLA) approaches aim to learn policies from large-scale simulation and real-world data. However, the continuous action space of the physical world significantly exceeds the representational capacity of linguistic tokens, making it unclear if scaling data alone can yield general robotic intelligence. To address this gap, we propose ActionReasoning, an LLM-driven framework that performs explicit action reasoning to produce physics-consistent, prior-guided decisions for robotic manipulation. ActionReasoning leverages the physical priors and real-world knowledge already encoded in Large Language Models (LLMs) and structures them within a multi-agent architecture. We instantiate this framework on a tractable case study of brick stacking, where the environment states are assumed to be already accurately measured. The environmental states are then serialized and passed to a multi-agent LLM framework that generates physics-aware action plans. The experiments demonstrate that the proposed multi-agent LLM framework enables stable brick placement while shifting effort from low-level domain-specific coding to high-level tool invocation and prompting, highlighting its potential for broader generalization. This work introduces a promising approach to bridging perception and execution in robotic manipulation by integrating physical reasoning with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03005v4">From Moderation to Mediation: Can LLMs Serve as Mediators in Online Flame Wars?</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Accepted by PAKDD 2026 special session on Data Science: Foundations and Applications
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has opened new possibilities for AI for good applications. As LLMs increasingly mediate online communication, their potential to foster empathy and constructive dialogue becomes an important frontier for responsible AI research. This work explores whether LLMs can serve not only as moderators that detect harmful content, but as mediators capable of understanding and de-escalating online conflicts. Our framework decomposes mediation into two subtasks: judgment, where an LLM evaluates the fairness and emotional dynamics of a conversation, and steering, where it generates empathetic, de-escalatory messages to guide participants toward resolution. To assess mediation quality, we construct a large Reddit-based dataset and propose a multi-stage evaluation pipeline combining principle-based scoring, user simulation, and human comparison. Experiments show that API-based models outperform open-source counterparts in both reasoning and intervention alignment when doing mediation. Our findings highlight both the promise and limitations of current LLMs as emerging agents for online social mediation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21140v1">ReviveMoE: Fast Recovery for Hardware Failures in Large-Scale MoE LLM Inference Deployments</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 21 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As LLM deployments scale over more hardware, the probability of a single failure in a system increases significantly, and cloud operators must consider robust countermeasures to handle these inevitable failures. A common recovery approach is to simply restart the LLM serving instance; however, this is costly in model-as-a-service (MaaS) inference settings, where reloading model weights and recompiling computation graphs can introduce significant delays to incoming requests. We propose ReviveMoE, a method for rapid failure recovery in large-scale LLM deployments without restarting the serving instance. ReviveMoE is designed to support both the traditional LLM architecture, which collocates MoE and attention on the same hardware, and the disaggregated architectures, which separate MoE from attention. Integrated into Huawei Cloud's MaaS, ReviveMoE is built on top of Huawei's xDeepServe serving platform and the XCCL communications library.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21127v1">"Are You Sure?": An Empirical Study of Human Perception Vulnerability in LLM-Driven Agentic Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are rapidly becoming trusted copilots in high-stakes domains like software development and healthcare. However, this deepening trust introduces a novel attack surface: Agent-Mediated Deception (AMD), where compromised agents are weaponized against their human users. While extensive research focuses on agent-centric threats, human susceptibility to deception by a compromised agent remains unexplored. We present the first large-scale empirical study with 303 participants to measure human susceptibility to AMD. This is based on HAT-Lab (Human-Agent Trust Laboratory), a high-fidelity research platform we develop, featuring nine carefully crafted scenarios spanning everyday and professional domains (e.g., healthcare, software development, human resources). Our 10 key findings reveal significant vulnerabilities and provide future defense perspectives. Specifically, only 8.6% of participants perceive AMD attacks, while domain experts show increased susceptibility in certain scenarios. We identify six cognitive failure modes in users and find that their risk awareness often fails to translate to protective behavior. The defense analysis reveals that effective warnings should interrupt workflows with low verification costs. With experiential learning based on HAT-Lab, over 90% of users who perceive risks report increased caution against AMD. This work provides empirical evidence and a platform for human-centric agent security research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21099v1">Turning Semantics into Topology: LLM-Driven Attribute Augmentation for Collaborative Filtering</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great potential for enhancing recommender systems through their extensive world knowledge and reasoning capabilities. However, effectively translating these semantic signals into traditional collaborative embeddings remains an open challenge. Existing approaches typically fall into two extremes: direct inference methods are computationally prohibitive for large-scale retrieval, while embedding-based methods primarily focus on unilateral feature augmentation rather than holistic collaborative signal enhancement. To bridge this gap, we propose Topology-Augmented Graph Collaborative Filtering (TAGCF), a novel framework that transforms semantic knowledge into topological connectivity. Unlike existing approaches that depend on textual features or direct interaction synthesis, TAGCF employs LLMs to infer interaction intents and underlying causal relationships from user-item pairs, representing these insights as intermediate attribute nodes within an enriched User-Attribute-Item (U-A-I) graph. Furthermore, to effectively model the heterogeneous relations in this augmented structure, we propose Adaptive Relation-weighted Graph Convolution (ARGC), which employs relation-specific prediction networks to dynamically estimate the importance of each relation type. Extensive experiments across multiple benchmark datasets and CF backbones demonstrate consistent improvements, with comprehensive evaluations including cold-start scenarios validating the effectiveness and robustness of our framework. All code will be made publicly available. For anonymous review, our code is available at the following anonymous link: https://anonymous.4open.science/r/AGCF-2441353190/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21082v1">Beyond the Star Rating: A Scalable Framework for Aspect-Based Sentiment Analysis Using LLMs and Text Classification</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Customer-provided reviews have become an important source of information for business owners and other customers alike. However, effectively analyzing millions of unstructured reviews remains challenging. While large language models (LLMs) show promise for natural language understanding, their application to large-scale review analysis has been limited by computational costs and scalability concerns. This study proposes a hybrid approach that uses LLMs for aspect identification while employing classic machine-learning methods for sentiment classification at scale. Using ChatGPT to analyze sampled restaurant reviews, we identified key aspects of dining experiences and developed sentiment classifiers using human-labeled reviews, which we subsequently applied to 4.7 million reviews collected over 17 years from a major online platform. Regression analysis reveals that our machine-labeled aspects significantly explain variance in overall restaurant ratings across different aspects of dining experiences, cuisines, and geographical regions. Our findings demonstrate that combining LLMs with traditional machine learning approaches can effectively automate aspect-based sentiment analysis of large-scale customer feedback, suggesting a practical framework for both researchers and practitioners in the hospitality industry and potentially, other service sectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21045v1">PaperTrail: A Claim-Evidence Interface for Grounding Provenance in LLM-based Scholarly Q&A</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 25 pages, 3 figures. Accepted at the ACM CHI conference on Human Factors in Computing Systems 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in scholarly question-answering (QA) systems to help researchers synthesize vast amounts of literature. However, these systems often produce subtle errors (e.g., unsupported claims, errors of omission), and current provenance mechanisms like source citations are not granular enough for the rigorous verification that scholarly domain requires. To address this, we introduce PaperTrail, a novel interface that decomposes both LLM answers and source documents into discrete claims and evidence, mapping them to reveal supported assertions, unsupported claims, and information omitted from the source texts. We evaluated PaperTrail in a within-subjects study with 26 researchers who performed two scholarly editing tasks using PaperTrail and a baseline interface. Our results show that PaperTrail significantly lowered participants' trust compared to the baseline. However, this increased caution did not translate to behavioral changes, as people continued to rely on LLM-generated scholarly edits to avoid a cognitively burdensome task. We discuss the value of claim-evidence matching for understanding LLM trustworthiness in scholarly settings, and present design implications for cognition-friendly communication of provenance information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07729v2">Do We Need Adam? Surprisingly Strong and Sparse Reinforcement Learning with SGD in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL), particularly RL from verifiable reward (RLVR), has become a crucial phase of training large language models (LLMs) and a key focus of current scaling efforts. However, optimization practices in RL largely follow those of next-token prediction stages (e.g., pretraining and supervised fine-tuning), despite fundamental differences between RL and these stages highlighted by recent work. One such practice is the use of the AdamW optimizer, which is widely adopted for training large-scale transformers despite its high memory overhead. Our analysis shows that both momentum and adaptive learning rates in AdamW are less influential in RL than in SFT, leading us to hypothesize that RL benefits less from Adam-style per-parameter adaptive learning rates and momentum. Confirming this hypothesis, our experiments demonstrate that the substantially more memory-efficient SGD, which is known to perform poorly in supervised learning of large-scale transformers, matches or even outperforms AdamW in RL for LLMs. Remarkably, full fine-tuning with SGD updates fewer than 0.02% of model parameters without any sparsity-promoting regularization, more than 1000 times fewer than AdamW. Our analysis offers potential reasons for this update sparsity. These findings provide new insights into the optimization dynamics of RL in LLMs and show that RL can be substantially more parameter-efficient than previously recognized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00044v2">When LLMs Imagine People: A Human-Centered Persona Brainstorm Audit for Bias and Fairness in Creative Applications</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) used in creative workflows can reinforce stereotypes and perpetuate inequities, making fairness auditing essential. Existing methods rely on constrained tasks and fixed benchmarks, leaving open-ended creative outputs unexamined. We introduce the Persona Brainstorm Audit (PBA), a scalable and easy to extend auditing method for bias detection across multiple intersecting identity and social roles in open-ended persona generation. PBA quantifies bias using degree-of-freedom-aware normalized Cramér's V, producing interpretable severity labels that enable fair comparison across models and dimensions. Applying PBA to 12 LLMs (120,000 personas, 16 bias dimensions), we find that bias evolves nonlinearly across model generations: larger and newer models are not consistently fairer, and biases that initially decrease can resurface in later releases. Intersectional analysis reveals disparities hidden by single-axis metrics, where dimensions appearing fair individually can exhibit high bias in combination. Robustness analyses show PBA remains stable under varying sample sizes, role-playing prompts, and debiasing prompts, establishing its reliability for fairness auditing in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20995v1">Generative Pseudo-Labeling for Pre-Ranking with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Pre-ranking is a critical stage in industrial recommendation systems, tasked with efficiently scoring thousands of recalled items for downstream ranking. A key challenge is the train-serving discrepancy: pre-ranking models are trained only on exposed interactions, yet must score all recalled candidates -- including unexposed items -- during online serving. This mismatch not only induces severe sample selection bias but also degrades generalization, especially for long-tail content. Existing debiasing approaches typically rely on heuristics (e.g., negative sampling) or distillation from biased rankers, which either mislabel plausible unexposed items as negatives or propagate exposure bias into pseudo-labels. In this work, we propose Generative Pseudo-Labeling (GPL), a framework that leverages large language models (LLMs) to generate unbiased, content-aware pseudo-labels for unexposed items, explicitly aligning the training distribution with the online serving space. By offline generating user-specific interest anchors and matching them with candidates in a frozen semantic space, GPL provides high-quality supervision without adding online latency. Deployed in a large-scale production system, GPL improves click-through rate by 3.07%, while significantly enhancing recommendation diversity and long-tail item discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.08219v4">Characterizing LLM Inference Energy-Performance Tradeoffs across Workloads and GPU Scaling</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      LLM inference exhibits substantial variability across queries and execution phases, yet inference configurations are often applied uniformly. We present a measurement-driven characterization of workload heterogeneity and energy-performance behavior of LLM inference under GPU dynamic voltage and frequency scaling (DVFS). We evaluate five decoder-only LLMs (1B-32B parameters) across four NLP benchmarks using a controlled offline setup. We show that lightweight semantic features predict inference difficulty better than input length, with 44.5% of queries achieving comparable quality across model sizes. At the hardware level, the decode phase dominates inference time (77-91%) and is largely insensitive to GPU frequency. Consequently, reducing GPU frequency from 2842 MHz to 180 MHz achieves an average of 42% energy savings with only a 1-6% latency increase. We further provide a use case with an upper-bound analysis of the potential benefits of combining workload-aware model selection with phase-aware DVFS, motivating future energy-efficient LLM inference systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.18777v2">Programming by Backprop: An Instruction is Worth 100 Examples When Finetuning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are typically trained to acquire behaviours from demonstrations or experience, yet much of their training data is declarative: instructions, rules, and descriptions that specify behaviours without showing how to execute them. We introduce Programming by Backprop (PBB): a training regime that enables LLMs to acquire procedural knowledge (i.e., reusable behaviours) from declarative instructions encountered during training. With PBB, instructions in training data provide an opportunity to `program' specific behaviours into model weights. The core principle underpinning PBB is the separation of learning how instructions map to behaviour from internalising new instructions. We devise two distinct PBB curricula that leverage this principle. Through controlled experiments across two domains (algorithmic execution from Python source code and text generation from context-free grammars), we demonstrate the benefit of these curricula over training on a homogeneous data mixture. Crucially, PBB is highly sample efficient, with a single instruction substituting for up to 100 execution examples. Though execution of instructions in training data remains less reliable than when instructions are given in-context, our results demonstrate that procedural knowledge can be noisily `programmed' into LLMs through PBB, with important implications for data curation and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14297v2">A Simple and Efficient Jailbreak Method Exploiting LLMs' Helpfulness</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      This study reveals a critical safety blind spot in modern LLMs: learning-style queries, which closely resemble ordinary educational questions, can reliably elicit harmful responses. The learning-style queries are constructed by a novel reframing paradigm: HILL (Hiding Intention by Learning from LLMs). The deterministic, model-agnostic reframing framework is composed of 4 conceptual components: 1) key concept, 2) exploratory transformation, 3) detail-oriented inquiry, and optionally 4) hypotheticality. Further, new metrics are introduced to thoroughly evaluate the efficiency and harmfulness of jailbreak methods. Experiments on the AdvBench dataset across a wide range of models demonstrate HILL's strong generalizability. It achieves top attack success rates on the majority of models and across malicious categories while maintaining high efficiency with concise prompts. On the other hand, results of various defense methods show the robustness of HILL, with most defenses having mediocre effects or even increasing the attack success rates. In addition, the assessment of defenses on the constructed safe prompts reveals inherent limitations of LLMs' safety mechanisms and flaws in the defense methods. This work exposes significant vulnerabilities of safety measures against learning-style elicitation, highlighting a critical challenge of fulfilling both helpfulness and safety alignments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12927v3">SEFL: A Framework for Generating Synthetic Educational Assignment Feedback with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 LREC 2026
    </div>
    <details class="paper-abstract">
      Providing high-quality feedback on student assignments is crucial for student success, but it is heavily limited by time and budgetary constraints. In this work, we introduce Synthetic Educational Feedback Loops (SEFL), a synthetic data framework designed to generate data that resembles immediate, on-demand feedback at scale without relying on extensive, real-world student assignments and teacher feedback. To obtain this type of data, two large language models (LLMs) operate in a teacher-student role to simulate assignment completion and formative feedback, generating 19.8K synthetic pairs of student work and corresponding critiques and actionable improvements from a teacher. With this data, we fine-tune smaller, more computationally efficient LLMs on these synthetic pairs, enabling them to replicate key features of high-quality, goal-oriented feedback. Through comprehensive evaluations with three LLM judges and three human experts, across a subset of 900 outputs, we demonstrate that SEFL-tuned models outperform both their untuned counterparts and an existing baseline in terms of feedback quality. The potential for societal impact is reinforced by extensive qualitative comments and ratings from human stakeholders -- both students and higher education instructors. SEFL has the potential to transform feedback processes for higher education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14365v3">Understanding the Ability of LLMs to Handle Character-Level Perturbation</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      This work investigates the resilience of contemporary large language models (LLMs) against frequent character-level perturbations. We examine three types of character-level perturbations including introducing numerous typos within words, shuffling the characters in each word, and inserting a large number of invisible characters into the text. Surprisingly, even under severe perturbation, such as shuffling nearly all words character-wise to produce text that is almost unreadable to humans, or inserting invisible characters which are several times more than the visible ones as noise, many LLMs still maintain notable performance. We explore the underlying causes of this robustness and find that LLMs exhibit remarkable resilience to chaotic segmentation and fragmented tokenization. Furthermore, we examine the mechanisms by which LLMs remove perturbations to correctly comprehend text, including both implicit and explicit mechanisms for character-level perturbation. We hope that our findings on the low-level robustness of LLMs will unveil their inherent architectural strengths, reveal the potential risks of their misuse, and inform the reliable deployment of LLMs across diverse application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20876v1">When LLMs Enter Everyday Feminism on Chinese Social Media: Opportunities and Risks for Women's Empowerment</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 This paper is conditionally accepted to CHI 2026
    </div>
    <details class="paper-abstract">
      Everyday digital feminism refers to the ordinary, often pragmatic ways women articulate lived experiences and cultivate solidarity in online spaces. In China, such practices flourish on RedNote through discussions under hashtags like ''women's growth''. Recently, DeepSeek-generated content has been taken up as a new voice in these conversations. Given widely recognized gender biases in LLMs, this raises critical concerns about how LLMs interact with everyday feminist practices. Through an analysis of 430 RedNote posts, 139 shared DeepSeek responses, and 3211 comments, we found that users predominantly welcomed DeepSeek's advice. Yet feminist critical discourse analysis revealed that these responses primarily encouraged women to self-optimize and pursue achievements within prevailing norms rather than challenge them. By interpreting this case, we discuss the opportunities and risks that LLMs introduce for everyday feminism as a pathway toward women's empowerment, and offer design implications for leveraging LLMs to better support such practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.05310v4">Oracular Programming: A Modular Foundation for Building LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Large Language Models can solve a wide range of tasks from just a few examples, but they remain difficult to steer and lack a capability essential for building reliable software at scale: the modular composition of computations under enforceable contracts. As a result, they are typically embedded in larger software pipelines that use domain-specific knowledge to decompose tasks and improve reliability through validation and search. Yet the complexity of writing, tuning, and maintaining such pipelines has so far limited their sophistication. We propose oracular programming: a foundational paradigm for integrating traditional, explicit computations with inductive oracles such as LLMs. It rests on two directing principles: the full separation of core and search logic, and the treatment of few-shot examples as grounded and evolvable program components. Within this paradigm, experts express high-level problem-solving strategies as programs with unresolved choice points. These choice points are resolved at runtime by LLMs, which generalize from user-provided examples of correct and incorrect decisions. An oracular program is composed of three orthogonal components: a strategy that consists of a nondeterministic program with choice points that can be reified into a search tree, a policy that specifies how to navigate this tree with the help of LLM oracles, and a set of demonstrations that describe successful and unsuccessful tree navigation scenarios across diverse problem instances. Each component is expressed in a dedicated programming language and can be independently improved or substituted. We address the key programming language design challenges of modularly composing oracular programs and enforcing consistency between their components as they evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05598v3">AgentDR: Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 12 pages, accepted by WWW'26 as long paper
    </div>
    <details class="paper-abstract">
      Recent agent-based recommendation frameworks aim to simulate user behaviors by incorporating memory mechanisms and prompting strategies, but they struggle with hallucinating non-existent items and full-catalog ranking. Besides, a largely underexplored opportunity lies in leveraging LLMs'commonsense reasoning to capture user intent through substitute and complement relationships between items, which are usually implicit in datasets and difficult for traditional ID-based recommenders to capture. In this work, we propose a novel LLM-agent framework, AgenDR, which bridges LLM reasoning with scalable recommendation tools. Our approach delegates full-ranking tasks to traditional models while utilizing LLMs to (i) integrate multiple recommendation outputs based on personalized tool suitability and (ii) reason over substitute and complement relationships grounded in user history. This design mitigates hallucination, scales to large catalogs, and enhances recommendation relevance through relational reasoning. Through extensive experiments on three public grocery datasets, we show that our framework achieves superior full-ranking performance, yielding on average a twofold improvement over its underlying tools. We also introduce a new LLM-based evaluation metric that jointly measures semantic alignment and ranking correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22242v1">Analysis of LLMs Against Prompt Injection and Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 12 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely deployed in real-world systems. Given their broader applicability, prompt engineering has become an efficient tool for resource-scarce organizations to adopt LLMs for their own purposes. At the same time, LLMs are vulnerable to prompt-based attacks. Thus, analyzing this risk has become a critical security requirement. This work evaluates prompt-injection and jailbreak vulnerability using a large, manually curated dataset across multiple open-source LLMs, including Phi, Mistral, DeepSeek-R1, Llama 3.2, Qwen, and Gemma variants. We observe significant behavioural variation across models, including refusal responses and complete silent non-responsiveness triggered by internal safety mechanisms. Furthermore, we evaluated several lightweight, inference-time defence mechanisms that operate as filters without any retraining or GPU-intensive fine-tuning. Although these defences mitigate straightforward attacks, they are consistently bypassed by long, reasoning-heavy prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16602v3">Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      We introduce Refusal Steering, an inference-time method to exercise fine-grained control over Large Language Models refusal behaviour on politically sensitive topics without retraining. We replace fragile pattern-based refusal detection with an LLM-as-a-judge that assigns refusal confidence scores and we propose a ridge-regularized variant to compute steering vectors that better isolate the refusal--compliance direction. On Qwen3-Next-80B-A3B-Thinking, our method removes the refusal behaviour of the model around politically sensitive topics while maintaining safety on JailbreakBench and near-baseline performance on general benchmarks. The approach generalizes across 4B and 80B models and can also induce targeted refusals when desired. We analize the steering vectors and show that refusal signals concentrate in deeper layers of the transformer and are distributed across many dimensions. Together, these results demonstrate that activation steering can remove political refusal behaviour while retaining safety alignment for harmful content, offering a practical path to controllable, transparent moderation at inference time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20732v1">CHESS: Context-aware Hierarchical Efficient Semantic Selection for Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Long-context LLMs demand accurate inference at low latency, yet decoding becomes primarily constrained by KV cache as context grows. Prior pruning methods are largely context-agnostic: their token selection ignores step-wise relevance and local semantics, which undermines quality. Moreover, their irregular accesses and selection overheads yield only limited wall-clock speedups. To address this, we propose \textbf{CHESS}, an \textit{algorithm-system co-design} KV-cache management system. Algorithmically, CHESS introduces a context-aware, hierarchical selection policy that dynamically reconstructs a coherent context for the current decoding. System-wise, coarse granularity selection eliminates expensive data movement, fully realizing practical acceleration from theoretical sparsity. Extensive evaluations demonstrate that CHESS surpasses Full-KV quality using only \textbf{1\%} of the KV cache, delivers low-latency stable inference with up to \textbf{4.56$\times$} higher throughput, and consistently outperforms other strong baselines. Code is available at \href{https://anonymous.4open.science/r/CHESS-9958/}{https://anonymous.4open.science/r/CHESS/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22240v1">From Prompts to Performance: Evaluating LLMs for Task-based Parallel Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 12 pages, 4 figures, 2 tables, Workshop on Asynchronous Many-Task Systems and Applications 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) show strong abilities in code generation, but their skill in creating efficient parallel programs is less studied. This paper explores how LLMs generate task-based parallel code from three kinds of input prompts: natural language problem descriptions, sequential reference implementations, and parallel pseudo code. We focus on three programming frameworks: OpenMP Tasking, C++ standard parallelism, and the asynchronous many-task runtime HPX. Each framework offers different levels of abstraction and control for task execution. We evaluate LLM-generated solutions for correctness and scalability. Our results reveal both strengths and weaknesses of LLMs with regard to problem complexity and framework. Finally, we discuss what these findings mean for future LLM-assisted development in high-performance and scientific computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22620v2">Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Julia Bazinska and Max Mathys contributed equally
    </div>
    <details class="paper-abstract">
      AI agents powered by large language models (LLMs) are being deployed at scale, yet we lack a systematic understanding of how the choice of backbone LLM affects agent security. The non-deterministic sequential nature of AI agents complicates security modeling, while the integration of traditional software with AI components entangles novel LLM vulnerabilities with conventional security risks. Existing frameworks only partially address these challenges as they either capture specific vulnerabilities only or require modeling of complete agents. To address these limitations, we introduce threat snapshots: a framework that isolates specific states in an agent's execution flow where LLM vulnerabilities manifest, enabling the systematic identification and categorization of security risks that propagate from the LLM to the agent level. We apply this framework to construct the $b^3$ benchmark, a security benchmark based on 194,331 unique crowdsourced adversarial attacks. We then evaluate 34 popular LLMs with it, revealing, among other insights, that enhanced reasoning capabilities improve security, while model size does not correlate with security. We release our benchmark, dataset, and evaluation code to facilitate widespread adoption by LLM providers and practitioners, offering guidance for agent developers and incentivizing model developers to prioritize backbone security improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17550v2">MASPO: Unifying Gradient Utilization, Probability Mass, and Signal Reliability for Robust and Sample-Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Existing Reinforcement Learning with Verifiable Rewards (RLVR) algorithms, such as GRPO, rely on rigid, uniform, and symmetric trust region mechanisms that are fundamentally misaligned with the complex optimization dynamics of Large Language Models (LLMs). In this paper, we identify three critical challenges in these methods: (1) inefficient gradient utilization caused by the binary cutoff of hard clipping, (2) insensitive probability mass arising from uniform ratio constraints that ignore the token distribution, and (3) asymmetric signal reliability stemming from the disparate credit assignment ambiguity between positive and negative samples. To bridge these gaps, we propose Mass-Adaptive Soft Policy Optimization (MASPO), a unified framework designed to harmonize these three dimensions. MASPO integrates a differentiable soft Gaussian gating to maximize gradient utility, a mass-adaptive limiter to balance exploration across the probability spectrum, and an asymmetric risk controller to align update magnitudes with signal confidence. Extensive evaluations demonstrate that MASPO serves as a robust, all-in-one RLVR solution, significantly outperforming baselines. Our code is at: \href{https://github.com/VenomRose-Juri/MASPO-RL}{https://github.com/VenomRose-Juri/MASPO-RL}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20676v1">PRECTR-V2:Unified Relevance-CTR Framework with Cross-User Preference Mining, Exposure Bias Correction, and LLM-Distilled Encoder Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 arXiv admin note: text overlap with arXiv:2503.18395
    </div>
    <details class="paper-abstract">
      In search systems, effectively coordinating the two core objectives of search relevance matching and click-through rate (CTR) prediction is crucial for discovering users' interests and enhancing platform revenue. In our prior work PRECTR, we proposed a unified framework to integrate these two subtasks,thereby eliminating their inconsistency and leading to mutual benefit.However, our previous work still faces three main challenges. First, low-active users and new users have limited search behavioral data, making it difficult to achieve effective personalized relevance preference modeling. Second, training data for ranking models predominantly come from high-relevance exposures, creating a distribution mismatch with the broader candidate space in coarse-ranking, leading to generalization bias. Third, due to the latency constraint, the original model employs an Emb+MLP architecture with a frozen BERT encoder, which prevents joint optimization and creates misalignment between representation learning and CTR fine-tuning. To solve these issues, we further reinforce our method and propose PRECTR-V2. Specifically, we mitigate the low-activity users' sparse behavior problem by mining global relevance preferences under the specific query, which facilitates effective personalized relevance modeling for cold-start scenarios. Subsequently, we construct hard negative samples through embedding noise injection and relevance label reconstruction, and optimize their relative ranking against positive samples via pairwise loss, thereby correcting exposure bias. Finally, we pretrain a lightweight transformer-based encoder via knowledge distillation from LLM and SFT on the text relevance classification task. This encoder replaces the frozen BERT module, enabling better adaptation to CTR fine-tuning and advancing beyond the traditional Emb+MLP paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07172v3">NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models are emerging as powerful tools for scientific law discovery, a foundational challenge in AI-driven science. However, existing benchmarks for this task suffer from a fundamental methodological trilemma, forcing a trade-off between scientific relevance, scalability, and resistance to memorization. Furthermore, they oversimplify discovery as static function fitting, failing to capture the authentic scientific process of uncovering embedded laws through the interactive exploration of complex model systems. To address these critical gaps, we introduce NewtonBench, a benchmark comprising 324 scientific law discovery tasks across 12 physics domains. Our design mitigates the evaluation trilemma by using counterfactual law shifts - systematic alterations of canonical laws - to generate a vast suite of problems that are scalable, scientifically relevant, and memorization-resistant. Moreover, we elevate the evaluation from static function fitting to interactive model discovery, requiring agents to experimentally probe simulated complex systems to uncover hidden principles. Our extensive experiment reveals a clear but fragile capability for discovery in frontier LLMs: this ability degrades precipitously with increasing system complexity and exhibits extreme sensitivity to observational noise. Notably, we uncover a paradoxical effect of tool assistance: providing a code interpreter can hinder more capable models by inducing a premature shift from exploration to exploitation, causing them to satisfice on suboptimal solutions. These results demonstrate that robust, generalizable discovery in complex, interactive environments remains the core challenge. By providing a scalable, robust, and scientifically authentic testbed, NewtonBench offers a crucial tool for measuring true progress and guiding the development of next-generation AI agents capable of genuine scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20663v1">ICSSPulse: A Modular LLM-Assisted Platform for Industrial Control System Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      It is well established that industrial control systems comprise the operational backbone of modern critical infrastructures, yet their increasing connectivity exposes them to cyber threats that are difficult to study and remedy safely under real-time operational conditions. In this paper, we present ICSSPulse, an open-source, modular, and extensible penetration testing platform designed for the security assessment of ICS communication protocols. To the best of our knowledge, ICSSPulse is the first web-based platform that unifies network scanning, protocol-aware Modbus and OPC~UA interaction, and Large Language Model (LLM)-assisted reporting within a single, lightweight ecosystem. Our platform provides a user-friendly graphical interface that orchestrates enumeration, exploitation, and reporting activities over simulated industrial services, enabling safe and reproducible experimentation. It supports protocol-level discovery, asset enumeration, and controlled read/write interactions, while preserving protocol fidelity and operational transparency. Experimental evaluation using synthetic Modbus test servers, a Factory I/O water treatment scenario, and a custom OPC~UA production-line model demonstrated ICSSPulse's potential to discover active industrial services, enumerate process-relevant assets, and manipulate process variables. A key contribution of this work lies in the integration of an LLM-assisted reporting module that automatically translates technical findings into structured executive and technical reports, with mitigation guidance informed by the ICS MITRE ATT&CK ICS matrix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20662v1">TOM: A Ternary Read-only Memory Accelerator for LLM-powered Edge Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) for real-time intelligence on edge devices is rapidly growing. However, conventional hardware architectures face a fundamental memory wall challenge, where limited on-device memory capacity and bandwidth severely constrain the size of deployable models and their inference speed, while also limiting on-device adaptation. To address this challenge, we propose TOM, a hybrid ROM-SRAM accelerator co-designed with ternary quantization, which balances extreme density with on-device tunability. TOM exploits the synergy between ternary quantization and ROM to achieve extreme memory density and bandwidth, while preserving flexibility through a hybrid ROM-SRAM architecture designed for QLoRA-based tunability. Specifically, we introduce: (1) a sparsity-aware ROM architecture that synthesizes ternary weights as standard-cell logic, eliminating area overhead from zero-valued bits; (2) a distributed processing architecture that co-locates high-density ROM banks with flexible SRAM-based QLoRA adapters and compute units; and (3) a workload-aware dynamic power gating scheme that exploits the logic-based nature of ROM to power down inactive banks, minimizing dynamic energy consumption. TOM achieves an inference throughput of 3,306 TPS using BitNet-2B model, demonstrating its effectiveness in delivering real-time, energy-efficient edge intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20639v1">Grounding LLMs in Scientific Discovery via Embodied Actions</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 24 pages, 7 figures, 7 tables. Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant potential in scientific discovery but struggle to bridge the gap between theoretical reasoning and verifiable physical simulation. Existing solutions operate in a passive "execute-then-response" loop and thus lacks runtime perception, obscuring agents to transient anomalies (e.g., numerical instability or diverging oscillations). To address this limitation, we propose EmbodiedAct, a framework that transforms established scientific software into active embodied agents by grounding LLMs in embodied actions with a tight perception-execution loop. We instantiate EmbodiedAct within MATLAB and evaluate it on complex engineering design and scientific modeling tasks. Extensive experiments show that EmbodiedAct significantly outperforms existing baselines, achieving SOTA performance by ensuring satisfactory reliability and stability in long-horizon simulations and enhanced accuracy in scientific modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20595v1">OptiLeak: Efficient Prompt Reconstruction via Reinforcement Learning in Multi-tenant LLM Services</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Multi-tenant LLM serving frameworks widely adopt shared Key-Value caches to enhance efficiency. However, this creates side-channel vulnerabilities enabling prompt leakage attacks. Prior studies identified these attack surfaces yet focused on expanding attack vectors rather than optimizing attack performance, reporting impractically high attack costs that underestimate the true privacy risk. We propose OptiLeak, a reinforcement learning-enhanced framework that maximizes prompt reconstruction efficiency through two-stage fine-tuning. Our key insight is that domain-specific ``hard tokens'' -- terms difficult to predict yet carrying sensitive information -- can be automatically identified via likelihood ranking and used to construct preference pairs for Direct Preference Optimization, eliminating manual annotation. This enables effective preference alignment while avoiding the overfitting issues of extended supervised fine-tuning. Evaluated on three benchmarks spanning medical and financial domains, OptiLeak achieves up to $12.48\times$ reduction in average requests per token compared to baseline approaches, with consistent improvements across model scales from 3B to 14B parameters. Our findings demonstrate that cache-based prompt leakage poses a more severe threat than previously reported, underscoring the need for robust cache isolation in production deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10693v2">VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Training stability remains a central challenge in reinforcement learning (RL) for large language models (LLMs). Policy staleness, asynchronous training, and mismatches between training and inference engines all cause the behavior policy to diverge from the current policy, risking training collapse. Importance sampling provides a principled correction for this distribution shift but suffers from high variance; existing remedies such as token-level clipping and sequence-level normalization lack a unified theoretical foundation. We propose Variational sEquence-level Soft Policy Optimization (VESPO). By incorporating variance reduction into a variational formulation over proposal distributions, VESPO derives a closed-form reshaping kernel that operates directly on sequence-level importance weights without length normalization. Experiments on mathematical reasoning benchmarks show that VESPO maintains stable training under staleness ratios up to 64x and fully asynchronous execution, and delivers consistent gains across both dense and Mixture-of-Experts models. Code is available at https://github.com/FloyedShen/VESPO
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12462v2">Evaluating and Mitigating LLM-as-a-judge Bias in Communication Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used to autonomously evaluate the quality of content in communication systems, e.g., to assess responses in telecom customer support chatbots. However, the impartiality of these AI "judges" is not guaranteed, and any biases in their evaluation criteria could skew outcomes and undermine user trust. In this paper, we systematically investigate judgment biases in two LLM-as-a-judge models (i.e., GPT-Judge and JudgeLM) under the point-wise scoring setting, encompassing 11 types of biases that cover both implicit and explicit forms. We observed that state-of-the-art LLM judges demonstrate robustness to biased inputs, generally assigning them lower scores than the corresponding clean samples. Providing a detailed scoring rubric further enhances this robustness. We further found that fine-tuning an LLM on high-scoring yet biased responses can significantly degrade its performance, highlighting the risk of training on biased data. We also discovered that the judged scores correlate with task difficulty: a challenging dataset like GPQA yields lower average scores, whereas an open-ended reasoning dataset (e.g., JudgeLM-val) sees higher average scores. Finally, we proposed four potential mitigation strategies to ensure fair and reliable AI judging in practical communication scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20558v1">From Logs to Language: Learning Optimal Verbalization for LLM-Based Recommendation in Production</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are promising backbones for generative recommender systems, yet a key challenge remains underexplored: verbalization, i.e., converting structured user interaction logs into effective natural language inputs. Existing methods rely on rigid templates that simply concatenate fields, yielding suboptimal representations for recommendation. We propose a data-centric framework that learns verbalization for LLM-based recommendation. Using reinforcement learning, a verbalization agent transforms raw interaction histories into optimized textual contexts, with recommendation accuracy as the training signal. This agent learns to filter noise, incorporate relevant metadata, and reorganize information to improve downstream predictions. Experiments on a large-scale industrial streaming dataset show that learned verbalization delivers up to 93% relative improvement in discovery item recommendation accuracy over template-based baselines. Further analysis reveals emergent strategies such as user interest summarization, noise removal, and syntax normalization, offering insights into effective context construction for LLM-based recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01012v2">AutoEDA: Enabling EDA Flow Automation through Microservice-Based LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Electronic Design Automation (EDA) remains heavily reliant on tool command language (Tcl) scripting to drive complex RTL-to-GDSII flows. This scripting-based paradigm is labor-intensive, error-prone, and difficult to scale across large design projects. Recent advances in large language models (LLMs) suggest a new paradigm of natural language-driven automation. However, existing EDA efforts remain limited and face key challenges, including the absence of standardized interaction protocols and dependence on external APIs that introduce privacy risks. We present AutoEDA, a framework that leverages the Model Context Protocol (MCP) to enable end-to-end natural language control of RTL-to-GDSII design flows. AutoEDA introduces MCP-based servers for task decomposition, tool selection, and automated error handling, ensuring robust interaction between LLM agents and EDA tools. To enhance reliability and confidentiality, we integrate locally fine-tuned LLM agents. We further contribute a benchmark generation pipeline for diverse EDA scenarios and extend CodeBLEU with Tcl-specific enhancements for domain-aware evaluation. Together, these contributions establish a comprehensive framework for LLM-driven EDA automation, bridging natural language interfaces with modern chip design flows. Empirical results show that AutoEDA achieves up to 9.9 times higher accuracy than naive approaches while reducing token usage by approximately 97% compared to in-context learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20515v1">FAST-Prefill: FPGA Accelerated Sparse Attention for Long Context LLM Prefill</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      In long-context large language model (LLM) inference, the prefill stage dominates computation due to self-attention over the complete input context. Sparse attention significantly reduces self-attention computation by limiting each token's interactions to a subset of tokens. The attention sparsity pattern varies across input prompts, and within a prompt, each attention head can follow a distinct pattern. This makes attention sparsity dynamic. The requirement of generating the sparsity pattern, combined with limited data reuse in attention, shifts the prefill compute to being memory-bound. This, in addition to the huge energy requirements for long-context inference on GPU, motivates FPGAs as good candidates for accelerating dynamic long-context inference. To tackle these challenges, we propose FAST-Prefill, the first FPGA accelerator for long-context prefill-stage inference with dynamic sparse attention. To efficiently generate sparse indices, we propose a \textit{fused pipeline unit with a memory-aware execution order} to reduce large tensors and irregular memory accesses. To reduce off-chip memory traffic for accessing the KV cache, we utilize the memory hierarchy to design a \textit{liveness-driven, dual-tier cache}. For high-throughput matrix multiplication, we design a \textit{hybrid Matrix Processing Unit (MPU)} with DSPs and bit-plane decomposition using LUTs. We implement FAST-Prefill on Alveo U280 and evaluate it on the Llama and Qwen models (batch size = 1) for context lengths ranging from 4K to 128K tokens. We demonstrate an average speedup of up to 2.5$\times$ in TTFT and 4.5$\times$ improvement in energy efficiency over GPU implementation on Nvidia A5000 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18464v2">How Well Can LLM Agents Simulate End-User Security and Privacy Attitudes and Behaviors?</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      A growing body of research assumes that large language model (LLM) agents can serve as proxies for how people form attitudes toward and behave in response to security and privacy (S&P) threats. If correct, these simulations could offer a scalable way to forecast S&P risks in products prior to deployment. We interrogate this assumption using SP-ABCBench, a new benchmark of 30 tests derived from validated S&P human-subject studies, which measures alignment between simulations and human-subjects studies on a 0-100 ascending scale, where higher scores indicate better alignment across three dimensions: Attitude, Behavior, and Coherence. Evaluating twelve LLMs, four persona construction strategies, and two prompting methods, we found that there remains substantial room for improvement: all models score between 50 and 64 on average. Newer, bigger, and smarter models do not reliably do better and sometimes do worse. Some simulation configurations, however, do yield high alignment: e.g., with scores above 95 for some behavior tests when agents are prompted to apply bounded rationality and weigh privacy costs against perceived benefits. We release SP-ABCBench to enable reproducible evaluation as methods improve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20492v1">Wireless Federated Multi-Task LLM Fine-Tuning via Sparse-and-Orthogonal LoRA</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Decentralized federated learning (DFL) based on low-rank adaptation (LoRA) enables mobile devices with multi-task datasets to collaboratively fine-tune a large language model (LLM) by exchanging locally updated parameters with a subset of neighboring devices via wireless connections for knowledge integration.However, directly aggregating parameters fine-tuned on heterogeneous datasets induces three primary issues across the DFL life-cycle: (i) \textit{catastrophic knowledge forgetting during fine-tuning process}, arising from conflicting update directions caused by data heterogeneity; (ii) \textit{inefficient communication and convergence during model aggregation process}, due to bandwidth-intensive redundant model transmissions; and (iii) \textit{multi-task knowledge interference during inference process}, resulting from incompatible knowledge representations coexistence during inference. To address these issues in a fully decentralized scenario, we first propose a sparse-and-orthogonal LoRA that ensures orthogonality between model updates to eliminate direction conflicts during fine-tuning.Then, we analyze how device connection topology affects multi-task performance, prompting a cluster-based topology design during aggregation.Finally, we propose an implicit mixture of experts (MoE) mechanism to avoid the coexistence of incompatible knowledge during inference. Simulation results demonstrate that the proposed approach effectively reduces communication resource consumption by up to $73\%$ and enhances average performance by $5\%$ compared with the traditional LoRA method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20486v1">Hybrid LLM-Embedded Dialogue Agents for Learner Reflection: Designing Responsive and Theory-Driven Interactions</a></div>
    <div class="paper-meta">
      📅 2026-02-24
    </div>
    <details class="paper-abstract">
      Dialogue systems have long supported learner reflections, with theoretically grounded, rule-based designs offering structured scaffolding but often struggling to respond to shifts in engagement. Large Language Models (LLMs), in contrast, can generate context-sensitive responses but are not informed by decades of research on how learning interactions should be structured, raising questions about their alignment with pedagogical theories. This paper presents a hybrid dialogue system that embeds LLM responsiveness within a theory-aligned, rule-based framework to support learner reflections in a culturally responsive robotics summer camp. The rule-based structure grounds dialogue in self-regulated learning theory, while the LLM decides when and how to prompt deeper reflections, responding to evolving conversation context. We analyze themes across dialogues to explore how our hybrid system shaped learner reflections. Our findings indicate that LLM-embedded dialogues supported richer learner reflections on goals and activities, but also introduced challenges due to repetitiveness and misalignment in prompts, reducing engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20426v1">Learning to Rewrite Tool Descriptions for Reliable LLM-Agent Tool Use</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The performance of LLM-based agents depends not only on the agent itself but also on the quality of the tool interfaces it consumes. While prior work has focused heavily on agent fine-tuning, tool interfaces-including natural language descriptions and parameter schemas-remain largely human-oriented and often become a bottleneck, especially when agents must select from large candidate tool sets. Existing approaches to improving tool interfaces rely on execution traces, which are frequently unavailable in cold-start or privacy-constrained settings, and typically optimize each tool independently, limiting scalability and generalization to unseen tools. We propose Trace-Free+, a curriculum learning framework that progressively transfers supervision from trace-rich settings to trace-free deployment, encouraging the model to abstract reusable interface-usage patterns and tool usage outcomes. To support this approach, we construct a large-scale dataset of high-quality tool interfaces using a structured workflow over a diverse collection of tools. Experiments on StableToolBench and RestBench show consistent gains on unseen tools, strong cross-domain generalization, and robustness as the number of candidate tools scales to over 100, demonstrating that tool interface optimization is a practical and deployable complement to agent fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20408v1">Examining and Addressing Barriers to Diversity in LLM-Generated Ideas</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      Ideas generated by independent samples of humans tend to be more diverse than ideas generated from independent LLM samples, raising concerns that widespread reliance on LLMs could homogenize ideation and undermine innovation at a societal level. Drawing on cognitive psychology, we identify (both theoretically and empirically) two mechanisms undermining LLM idea diversity. First, at the individual level, LLMs exhibit fixation just as humans do, where early outputs constrain subsequent ideation. Second, at the collective level, LLMs aggregate knowledge into a unified distribution rather than exhibiting the knowledge partitioning inherent to human populations, where each person occupies a distinct region of the knowledge space. Through four studies, we demonstrate that targeted prompting interventions can address each mechanism independently: Chain-of-Thought (CoT) prompting reduces fixation by encouraging structured reasoning (only in LLMs, not humans), while ordinary personas (versus "creative entrepreneurs" such as Steve Jobs) improve knowledge partitioning by serving as diverse sampling cues, anchoring generation in distinct regions of the semantic space. Combining both approaches produces the highest idea diversity, outperforming humans. These findings offer a theoretically grounded framework for understanding LLM idea diversity and practical strategies for human-AI collaborations that leverage AI's efficiency without compromising the diversity essential to a healthy innovation ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14044v2">Context Shapes LLMs Retrieval-Augmented Fact-Checking Effectiveness</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong reasoning abilities across diverse tasks, yet their performance on extended contexts remains inconsistent. While prior research has emphasized mid-context degradation in question answering, this study examines the impact of context in LLM-based fact verification. Using three datasets (HOVER, FEVEROUS, and ClimateFEVER) and five open-source models accross different parameters sizes (7B, 32B and 70B parameters) and model families (Llama-3.1, Qwen2.5 and Qwen3), we evaluate both parametric factual knowledge and the impact of evidence placement across varying context lengths. We find that LLMs exhibit non-trivial parametric knowledge of factual claims and that their verification accuracy generally declines as context length increases. Similarly to what has been shown in previous works, in-context evidence placement plays a critical role with accuracy being consistently higher when relevant evidence appears near the beginning or end of the prompt and lower when placed mid-context. These results underscore the importance of prompt structure in retrieval-augmented fact-checking systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02002v3">LLM-Driven Corrective Robot Operation Code Generation with Static Text-Based Simulation</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Large language models (LLMs) have demonstrated their promising capabilities of generating robot operation code to enable LLM-driven robots. To enhance the reliability of operation code generated by LLMs, corrective designs with feedback from the observation of executing code have been increasingly adopted in existing research. However, the code execution in these designs relies on either a physical experiment or a customized simulation environment, which limits their deployment due to the high configuration effort of the environment and the potential long execution time. In this paper, we explore the possibility of directly leveraging LLM to enable static simulation of robot operation code, and then leverage it to design a new reliable LLM-driven corrective robot operation code generation framework. Our framework configures the LLM as a static simulator with enhanced capabilities that reliably simulate robot code execution by interpreting actions, reasoning over state transitions, analyzing execution outcomes, and generating semantic observations that accurately capture trajectory dynamics. To validate the performance of our framework, we performed experiments on various operation tasks for different robots, including UAVs and small ground vehicles. The experiment results not only demonstrated the high accuracy of our static text-based simulation but also the reliable code generation of our LLM-driven corrective framework, which achieves a comparable performance with state-of-the-art research while does not rely on dynamic code execution using physical experiments or simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20379v1">Case-Aware LLM-as-a-Judge Evaluation for Enterprise-Scale RAG Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 12 pages including appendix, 6 figures
    </div>
    <details class="paper-abstract">
      Enterprise Retrieval-Augmented Generation (RAG) assistants operate in multi-turn, case-based workflows such as technical support and IT operations, where evaluation must reflect operational constraints, structured identifiers (e.g., error codes, versions), and resolution workflows. Existing RAG evaluation frameworks are primarily designed for benchmark-style or single-turn settings and often fail to capture enterprise-specific failure modes such as case misidentification, workflow misalignment, and partial resolution across turns. We present a case-aware LLM-as-a-Judge evaluation framework for enterprise multi-turn RAG systems. The framework evaluates each turn using eight operationally grounded metrics that separate retrieval quality, grounding fidelity, answer utility, precision integrity, and case/workflow alignment. A severity-aware scoring protocol reduces score inflation and improves diagnostic clarity across heterogeneous enterprise cases. The system uses deterministic prompting with strict JSON outputs, enabling scalable batch evaluation, regression testing, and production monitoring. Through a comparative study of two instruction-tuned models across short and long workflows, we show that generic proxy metrics provide ambiguous signals, while the proposed framework exposes enterprise-critical tradeoffs that are actionable for system improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08091v2">Everything is Plausible: Investigating the Impact of LLM Rationales on Human Notions of Plausibility</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Updated
    </div>
    <details class="paper-abstract">
      We investigate the degree to which human plausibility judgments of multiple-choice commonsense benchmark answers are subject to influence by (im)plausibility arguments for or against an answer, in particular, using rationales generated by LLMs. We collect 3,000 plausibility judgments from humans and another 13,600 judgments from LLMs. Overall, we observe increases and decreases in mean human plausibility ratings in the presence of LLM-generated PRO and CON rationales, respectively, suggesting that, on the whole, human judges find these rationales convincing. Experiments with LLMs reveal similar patterns of influence. Our findings demonstrate a novel use of LLMs for studying aspects of human cognition, while also raising practical concerns that, even in domains where humans are ``experts'' (i.e., common sense), LLMs have the potential to exert considerable influence on people's beliefs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05722v2">OckBench: Measuring the Efficiency of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as GPT-5 and Gemini 3 have pushed the frontier of automated reasoning and code generation. Yet current benchmarks emphasize accuracy and output quality, neglecting a critical dimension: efficiency of token usage. The token efficiency is highly variable in practical. Models solving the same problem with similar accuracy can exhibit up to a \textbf{5.0$\times$} difference in token length, leading to massive gap of model reasoning ability. Such variance exposes significant redundancy, highlighting the critical need for a standardized benchmark to quantify the gap of token efficiency. Thus, we introduce OckBench, the first benchmark that jointly measures accuracy and token efficiency across reasoning and coding tasks. Our evaluation reveals that token efficiency remains largely unoptimized across current models, significantly inflating serving costs and latency. These findings provide a concrete roadmap for the community to optimize the latent reasoning ability, token efficiency. Ultimately, we argue for an evaluation paradigm shift: tokens must not be multiplied beyond necessity. Our benchmarks are available at https://ockbench.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12194v2">MalTool: Malicious Tool Attacks on LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      In a malicious tool attack, an attacker uploads a malicious tool to a distribution platform; once a user installs the tool and the LLM agent selects it during task execution, the tool can compromise the user's security and privacy. Prior work primarily focuses on manipulating tool names and descriptions to increase the likelihood of installation by users and selection by LLM agents. However, a successful attack also requires embedding malicious behaviors in the tool's code implementation, which remains largely unexplored. In this work, we bridge this gap by presenting the first systematic study of malicious tool code implementations. We first propose a taxonomy of malicious tool behaviors based on the confidentiality-integrity-availability triad, tailored to LLM-agent settings. To investigate the severity of the risks posed by attackers exploiting coding LLMs to automatically generate malicious tools, we develop MalTool, a coding-LLM-based framework that synthesizes tools exhibiting specified malicious behaviors, either as standalone tools or embedded within otherwise benign implementations. To ensure functional correctness and structural diversity, MalTool leverages an automated verifier that validates whether generated tools exhibit the intended malicious behaviors and differ sufficiently from prior instances, iteratively refining generations until success. Our evaluation demonstrates that MalTool is highly effective even when coding LLMs are safety-aligned. Using MalTool, we construct two datasets of malicious tools: 1,200 standalone malicious tools and 5,287 real-world tools with embedded malicious behaviors. We further show that existing detection methods, including commercial malware detection approaches such as VirusTotal and methods tailored to the LLM-agent setting, exhibit limited effectiveness at detecting the malicious tools, highlighting an urgent need for new defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20300v1">What Makes a Good Query? Measuring the Impact of Human-Confusing Linguistic Features on LLM Performance</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 EACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) hallucinations are usually treated as defects of the model or its decoding strategy. Drawing on classical linguistics, we argue that a query's form can also shape a listener's (and model's) response. We operationalize this insight by constructing a 22-dimension query feature vector covering clause complexity, lexical rarity, and anaphora, negation, answerability, and intention grounding, all known to affect human comprehension. Using 369,837 real-world queries, we ask: Are there certain types of queries that make hallucination more likely? A large-scale analysis reveals a consistent "risk landscape": certain features such as deep clause nesting and underspecification align with higher hallucination propensity. In contrast, clear intention grounding and answerability align with lower hallucination rates. Others, including domain specificity, show mixed, dataset- and model-dependent effects. Thus, these findings establish an empirically observable query-feature representation correlated with hallucination risk, paving the way for guided query rewriting and future intervention studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20133v1">AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      The paradigm of automated program generation is shifting from one-shot generation to inference-time search, where Large Language Models (LLMs) function as semantic mutation operators within evolutionary loops. While effective, these systems are currently governed by static schedules that fail to account for the non-stationary dynamics of the search process. This rigidity results in substantial computational waste, as resources are indiscriminately allocated to stagnating populations while promising frontiers remain under-exploited. We introduce AdaEvolve, a framework that reformulates LLM-driven evolution as a hierarchical adaptive optimization problem. AdaEvolve uses an "accumulated improvement signal" to unify decisions across three levels: Local Adaptation, which dynamically modulates the exploration intensity within a population of solution candidates; Global Adaptation, which routes the global resource budget via bandit-based scheduling across different solution candidate populations; and Meta-Guidance which generates novel solution tactics based on the previously generated solutions and their corresponding improvements when the progress stalls. We demonstrate that AdaEvolve consistently outperforms the open-sourced baselines across 185 different open-ended optimization problems including combinatorial, systems optimization and algorithm design problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20229v1">HieraMAS: Optimizing Intra-Node LLM Mixtures and Inter-Node Topology for Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 22 pages, 13 tables
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) built on large language models (LLMs) have shown strong performance across many tasks. Most existing approaches improve only one aspect at a time, such as the communication topology, role assignment, or LLM routing, while treating each agent as a single, indivisible unit. This misses the opportunity to use mixtures of LLMs within an agent to strengthen role-specific abilities. We propose HieraMAS, a hierarchical collaboration framework that combines intra-node LLM mixtures with an inter-node communication topology. HieraMAS introduces supernodes, where each functional role is implemented by multiple heterogeneous LLMs using a propose-synthesis structure. Optimizing HieraMAS creates unique credit-assignment challenges: final task performance depends heavily on the underlying LLMs' capabilities, which can lead reinforcement methods to incorrectly reward suboptimal configurations. To address this, we use a two-stage algorithm: (1) multi-level reward attribution, which provides fine-grained feedback at both the node level and the overall system level; (2) graph classification for topology selection, which treats choosing the communication structure as a holistic decision rather than optimizing edges one by one. Experiments on reasoning and coding benchmarks show that HieraMAS substantially outperforms existing methods while also delivering better cost-performance trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07751v4">AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in grade school math (GSM) reasoning. In particular, they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In this work, we instead focus on the strategy of "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. Focusing on GSM, we find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstRaL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks. Besides, improving GSM robustness via AbstRaL is shown to also implicitly benefit LLMs' capabilities on OOD mathematical and general reasoning tasks, indicating that abstract thinking broadly enables better generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20102v1">BarrierSteer: LLM Safety via Learning Barrier Steering</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 This paper introduces SafeBarrier, a framework that enforces safety in large language models by steering their latent representations with control barrier functions during inference, reducing adversarial and unsafe outputs
    </div>
    <details class="paper-abstract">
      Despite the state-of-the-art performance of large language models (LLMs) across diverse tasks, their susceptibility to adversarial attacks and unsafe content generation remains a major obstacle to deployment, particularly in high-stakes settings. Addressing this challenge requires safety mechanisms that are both practically effective and supported by rigorous theory. We introduce BarrierSteer, a novel framework that formalizes response safety by embedding learned non-linear safety constraints directly into the model's latent representation space. BarrierSteer employs a steering mechanism based on Control Barrier Functions (CBFs) to efficiently detect and prevent unsafe response trajectories during inference with high precision. By enforcing multiple safety constraints through efficient constraint merging, without modifying the underlying LLM parameters, BarrierSteer preserves the model's original capabilities and performance. We provide theoretical results establishing that applying CBFs in latent space offers a principled and computationally efficient approach to enforcing safety. Our experiments across multiple models and datasets show that BarrierSteer substantially reduces adversarial success rates, decreases unsafe generations, and outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17842v2">Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Accepted by ICLR 2026. The project page is available at https://damon-demon.github.io/shop-r1.html
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated strong potential in generating 'believable human-like' behavior in web environments. Prior work has explored augmenting training data with LLM-synthesized rationales and applying supervised fine-tuning (SFT) to enhance reasoning ability, which in turn can improve downstream action prediction. However, the performance of such approaches remains inherently bounded by the reasoning capabilities of the model used to generate the rationales. In this paper, we introduce Shop-R1, a novel reinforcement learning (RL) framework aimed at enhancing the reasoning ability of LLMs for simulation of real human behavior in online shopping environments. Specifically, Shop-R1 decomposes the human behavior simulation task into two stages: rationale generation and action prediction, each guided by distinct reward signals. For rationale generation, we leverage internal model signals (e.g., logit distributions) to guide the reasoning process in a self-supervised manner. For action prediction, we propose a hierarchical reward structure with difficulty-aware scaling to prevent reward hacking and enable fine-grained reward assignment. This design evaluates both high-level action types and the correctness of fine-grained sub-action details (attributes and values), rewarding outputs proportionally to their difficulty. Experimental results show that our method achieves a relative improvement of over 65% compared to the baseline. The project page is available at https://damon-demon.github.io/shop-r1.html.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20094v1">CausalFlip: A Benchmark for LLM Causal Judgment Beyond Semantic Matching</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 8 pages plus references, 3 figures, 3 tables. Under review
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) witness increasing deployment in complex, high-stakes decision-making scenarios, it becomes imperative to ground their reasoning in causality rather than spurious correlations. However, strong performance on traditional reasoning benchmarks does not guarantee true causal reasoning ability of LLMs, as high accuracy may still arise from memorizing semantic patterns instead of analyzing the underlying true causal structures. To bridge this critical gap, we propose a new causal reasoning benchmark, CausalFlip, designed to encourage the development of new LLM paradigm or training algorithms that ground LLM reasoning in causality rather than semantic correlation. CausalFlip consists of causal judgment questions built over event triples that could form different confounder, chain, and collider relations. Based on this, for each event triple, we construct pairs of semantically similar questions that reuse the same events but yield opposite causal answers, where models that rely heavily on semantic matching are systematically driven toward incorrect predictions. To further probe models' reliance on semantic patterns, we introduce a noisy-prefix evaluation that prepends causally irrelevant text before intermediate causal reasoning steps without altering the underlying causal relations or the logic of the reasoning process. We evaluate LLMs under multiple training paradigms, including answer-only training, explicit Chain-of-Thought (CoT) supervision, and a proposed internalized causal reasoning approach that aims to mitigate explicit reliance on correlation in the reasoning process. Our results show that explicit CoT can still be misled by spurious semantic correlations, where internalizing reasoning steps yields substantially improved causal grounding, suggesting that it is promising to better elicit the latent causal reasoning capabilities of base LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13632v2">Closing the Gap Between Text and Speech Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can be adapted to extend their text capabilities to speech inputs. However, these speech-adapted LLMs consistently underperform their text-based counterparts--and even cascaded pipelines--on language understanding tasks. We term this shortfall the text-speech understanding gap: the performance drop observed when a speech-adapted LLM processes spoken inputs relative to when the original text-based LLM processes the equivalent text. Recent approaches to narrowing this gap either rely on large-scale speech synthesis of text corpora, which is costly and heavily dependent on synthetic data, or on large-scale proprietary speech datasets, which are not reproducible. As a result, there remains a need for more data-efficient alternatives for closing the text-speech understanding gap. In this work, we analyze the gap as driven by two factors: (i) forgetting of text capabilities during adaptation, and (ii) cross-modal misalignment between speech and text. Based on this analysis, we introduce SALAD--Sample-efficient Alignment with Learning through Active selection and cross-modal Distillation--which combines cross-modal distillation with targeted synthetic data to improve alignment while mitigating forgetting. Applied to 3B and 7B LLMs, SALAD achieves competitive performance with a strong open-weight model across broad-domain benchmarks in knowledge, language understanding, and reasoning, while training on over an order of magnitude less speech data from public corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20059v1">Interaction Theater: A case of LLM Agents Interacting at Scale</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      As multi-agent architectures and agent-to-agent protocols proliferate, a fundamental question arises: what actually happens when autonomous LLM agents interact at scale? We study this question empirically using data from Moltbook, an AI-agent-only social platform, with 800K posts, 3.5M comments, and 78K agent profiles. We combine lexical metrics (Jaccard specificity), embedding-based semantic similarity, and LLM-as-judge validation to characterize agent interaction quality. Our findings reveal agents produce diverse, well-formed text that creates the surface appearance of active discussion, but the substance is largely absent. Specifically, while most agents ($67.5\%$) vary their output across contexts, $65\%$ of comments share no distinguishing content vocabulary with the post they appear under, and information gain from additional comments decays rapidly. LLM judge based metrics classify the dominant comment types as spam ($28\%$) and off-topic content ($22\%$). Embedding-based semantic analysis confirms that lexically generic comments are also semantically generic. Agents rarely engage in threaded conversation ($5\%$ of comments), defaulting instead to independent top-level responses. We discuss implications for multi-agent interaction design, arguing that coordination mechanisms must be explicitly designed; without them, even large populations of capable agents produce parallel output rather than productive exchange.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08119v2">SOP-Bench: Complex Industrial SOPs for Evaluating LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      LLM-based agents struggle to execute complex, multi-step Standard Operating Procedures (SOPs) that are fundamental to industrial automation. Existing benchmarks fail to capture the procedural complexity and tool orchestration demands of real-world workflows. We introduce SOP-Bench, a benchmark of 2,000+ tasks from human expert-authored SOPs across 12 business domains (healthcare, logistics, finance, content moderation, etc.). Using a human-AI collaborative framework, experts crafted authentic SOPs while AI generated artifacts (tools, APIs, datasets), all human-validated, yielding realistic tasks with executable interfaces and ground-truth outputs. SOP-Bench serves as a research enabler for systematically investigating agent architectures, model capabilities, and deployment considerations across diverse procedural tasks. We demonstrate its utility through illustrative experiments with a subset of frontier models across Function-Calling (FC) and ReAct agents, revealing critical insights. For example, (1) newer models do not guarantee better performance - Claude 4 family outperforms Claude 4.5 family on ReAct tasks (Claude 4 Opus: 72.4% vs. Claude 4.5 Sonnet: 63.3% task success rate), demonstrating that production upgrades require validation; (2) no single model-agent combination dominates: best performances range from 57% to 100% depending on domain. These examples illustrate how SOP-Bench enables isolating and studying specific dimensions of agent performance without costly production experiments. Our goal is not to rank model capabilities or build optimal agents, but to provide a rigorous evaluation framework that enables the researchers and practitioners to systematically investigate agent design choices, model selection, and deployment strategies. We release the benchmark at https://github.com/amazon-science/sop-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21257v1">Structured Prompt Language: Declarative Context Management for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 44 pages, 6 figures, 14 tables, 15 code-listings
    </div>
    <details class="paper-abstract">
      We present SPL (Structured Prompt Language), a declarative SQL-inspired language that treats large language models as generative knowledge bases and their context windows as constrained resources. SPL provides explicit WITH BUDGET/LIMIT token management, an automatic query optimizer, EXPLAIN transparency analogous to SQL's EXPLAIN ANALYZE, and native integration of retrieval-augmented generation (RAG) and persistent memory in a single declarative framework. SPL-flow extends SPL into resilient agentic pipelines with a three-tier provider fallback strategy (Ollama -> OpenRouter -> self-healing retry) fully transparent to the .spl script. Five extensions demonstrate the paradigm's breadth: (1) Text2SPL (multilingual NL->SPL translation); (2) Mixture-of-Models (MoM) routing that dispatches each PROMPT to a domain-specialist model at runtime; (3) Logical Chunking, an intelligent strategy for documents exceeding a single context window--expressed naturally through SPL's existing CTE syntax with no new constructs, decomposing a large query into a Map-Reduce pipeline that reduces attention cost from O(N^2) to O(N^2/k) and runs identically on cloud (parallel) or local hardware (sequential); (4) SPL-flow, a declarative agentic orchestration layer with resilient three-tier provider fallback; and (5) BENCHMARK for parallel multi-model comparison with automatic winner persistence. We provide a formal EBNF grammar, two pip-installable Python packages (spl-llm, spl-flow), and comparison against Prompty, DSPy, and LMQL. SPL reduces prompt boilerplate by 65% on average, surfaces a 68x cost spread across model tiers as a pre-execution signal, and runs the identical .spl script at $0.002 on OpenRouter or at zero marginal cost on a local Ollama instance--without modification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06326v2">AttestLLM: Efficient Attestation Framework for Billion-scale On-device LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 accept to DAC 2026
    </div>
    <details class="paper-abstract">
      As on-device LLMs(e.g., Apple on-device Intelligence) are widely adopted to reduce network dependency, improve privacy, and enhance responsiveness, verifying the legitimacy of models running on local devices becomes critical. Existing attestation techniques are not suitable for billion-parameter Large Language Models (LLMs), struggling to remain both time- and memory-efficient while addressing emerging threats in the LLM era. In this paper, we present AttestLLM, the first-of-its-kind attestation framework to protect the hardware-level intellectual property (IP) of device vendors by ensuring that only authorized LLMs can execute on target platforms. AttestLLM leverages an algorithm/software/hardware co-design approach to embed robust watermarking signatures onto the activation distributions of LLM building blocks. It also optimizes the attestation protocol within the Trusted Execution Environment (TEE), providing efficient verification without compromising inference throughput. Extensive proof-of-concept evaluations on LLMs from Llama, Qwen, and Phi families for on-device use cases demonstrate AttestLLM's attestation reliability, fidelity, and efficiency. Furthermore, AttestLLM enforces model legitimacy and exhibits resilience against model replacement and forgery attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11962v2">Wisdom of the LLM Crowd: A Large Scale Benchmark of Multi-Label U.S. Election-Related Harmful Social Media Content</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      The spread of election misinformation and harmful political content conveys misleading narratives and poses a serious threat to democratic integrity. Detecting harmful content at early stages is essential for understanding and potentially mitigating its downstream spread. In this study, we introduce USE24-XD, a large-scale dataset of nearly 100k posts collected from X (formerly Twitter) during the 2024 U.S. presidential election cycle, enriched with spatio-temporal metadata. To substantially reduce the cost of manual annotation while enabling scalable categorization, we employ six large language models (LLMs) to systematically annotate posts across five nuanced categories: Conspiracy, Sensationalism, Hate Speech, Speculation, and Satire. We validate LLM annotations with crowdsourcing (n = 34) and benchmark them against human annotators. Inter-rater reliability analyses show comparable agreement patterns between LLMs and humans, with LLMs exhibiting higher internal consistency and achieving up to 0.90 recall on Speculation. We apply a wisdom-of-the-crowd approach across LLMs to aggregate annotations and curate a robust multi-label dataset. 60% of posts receive at least one label. We further analyze how human annotator demographics, including political ideology and affiliation, shape labeling behavior, highlighting systematic sources of subjectivity in judgments of harmful content. The USE24-XD dataset is publicly released to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15077v4">Think2SQL: Reinforce LLM Reasoning Capabilities for Text2SQL</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 26 pages, work in progress
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have advanced the state-of-the-art in Text-to-SQL, robust reasoning in complex, multi-table environments remains a bottleneck for parameter-efficient models. This paper presents a systematic empirical study on injecting reasoning capabilities into Text-to-SQL through the lens of Reinforcement Learning with Verifiable Rewards (RLVR). We uncover a critical interplay between reward density, advantage scaling, and model capacity. Our analysis yields four primary insights. First, we propose a novel execution-guided dense reward function that significantly outperforms binary signals and existing state-of-the-art rewards by providing granular feedback at the instance level. Second, we analyze the mechanics of advantage calculation, demonstrating that while large models thrive on sparse signals with aggressive advantage scaling, smaller models require dense rewards and conservative scaling to improve Text-to-SQL performance. Third, we evaluate the impact of cold start, showing that distillation does not always improve RLVR performance and that supervised, fine-tuned models are prone to distributional mimicry. Fourth, we map the Pareto frontier of training efficiency, providing insights for optimizing Text-to-SQL reasoning under computational constraints. Our findings culminate in the Think2SQL family: our 4B-parameter model demonstrates reasoning capabilities competitive with state-of-the-art models such as o3. We release our models, datasets, and code to create a blueprint for RLVR optimization in Text-to-SQL at https://anonymous.4open.science/r/Think2SQL-3B7F.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19895v1">DSDR: Dual-Scale Diversity Regularization for Exploration in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiers (RLVR) is a central paradigm for improving large language model (LLM) reasoning, yet existing methods often suffer from limited exploration. Policies tend to collapse onto a few reasoning patterns and prematurely stop deep exploration, while conventional entropy regularization introduces only local stochasticity and fails to induce meaningful path-level diversity, leading to weak and unstable learning signals in group-based policy optimization. We propose DSDR, a Dual-Scale Diversity Regularization reinforcement learning framework that decomposes diversity in LLM reasoning into global and coupling components. Globally, DSDR promotes diversity among correct reasoning trajectories to explore distinct solution modes. Locally, it applies a length-invariant, token-level entropy regularization restricted to correct trajectories, preventing entropy collapse within each mode while preserving correctness. The two scales are coupled through a global-to-local allocation mechanism that emphasizes local regularization for more distinctive correct trajectories. We provide theoretical support showing that DSDR preserves optimal correctness under bounded regularization, sustains informative learning signals in group-based optimization, and yields a principled global-to-local coupling rule. Experiments on multiple reasoning benchmarks demonstrate consistent improvements in accuracy and pass@k, highlighting the importance of dual-scale diversity for deep exploration in RLVR. Code is available at https://github.com/SUSTechBruce/DSDR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19844v1">LLM-enabled Applications Require System-Level Threat Monitoring</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      LLM-enabled applications are rapidly reshaping the software ecosystem by using large language models as core reasoning components for complex task execution. This paradigm shift, however, introduces fundamentally new reliability challenges and significantly expands the security attack surface, due to the non-deterministic, learning-driven, and difficult-to-verify nature of LLM behavior. In light of these emerging and unavoidable safety challenges, we argue that such risks should be treated as expected operational conditions rather than exceptional events, necessitating a dedicated incident-response perspective. Consequently, the primary barrier to trustworthy deployment is not further improving model capability but establishing system-level threat monitoring mechanisms that can detect and contextualize security-relevant anomalies after deployment -- an aspect largely underexplored beyond testing or guardrail-based defenses. Accordingly, this position paper advocates systematic and comprehensive monitoring of security threats in LLM-enabled applications as a prerequisite for reliable operation and a foundation for dedicated incident-response frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19843v1">MAS-FIRE: Fault Injection and Reliability Evaluation for LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      As LLM-based Multi-Agent Systems (MAS) are increasingly deployed for complex tasks, ensuring their reliability has become a pressing challenge. Since MAS coordinate through unstructured natural language rather than rigid protocols, they are prone to semantic failures (e.g., hallucinations, misinterpreted instructions, and reasoning drift) that propagate silently without raising runtime exceptions. Prevailing evaluation approaches, which measure only end-to-end task success, offer limited insight into how these failures arise or how effectively agents recover from them. To bridge this gap, we propose MAS-FIRE, a systematic framework for fault injection and reliability evaluation of MAS. We define a taxonomy of 15 fault types covering intra-agent cognitive errors and inter-agent coordination failures, and inject them via three non-invasive mechanisms: prompt modification, response rewriting, and message routing manipulation. Applying MAS-FIRE to three representative MAS architectures, we uncover a rich set of fault-tolerant behaviors that we organize into four tiers: mechanism, rule, prompt, and reasoning. This tiered view enables fine-grained diagnosis of where and why systems succeed or fail. Our findings reveal that stronger foundation models do not uniformly improve robustness. We further show that architectural topology plays an equally decisive role, with iterative, closed-loop designs neutralizing over 40% of faults that cause catastrophic collapse in linear workflows. MAS-FIRE provides the process-level observability and actionable guidance needed to systematically improve multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.17596v4">Evaluating LLMs' Divergent Thinking Capabilities for Scientific Idea Generation with Minimal Context</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Updated manuscript and title
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate remarkable capabilities in scientific tasks such as literature analysis and experimental design (e.g., accurately extracting key findings from papers or generating coherent experimental procedures), existing evaluation benchmarks primarily assess performance using rich contextual inputs. We introduce LiveIdeaBench, a comprehensive benchmark evaluating LLMs' scientific idea generation by assessing divergent thinking capabilities using single-keyword prompts. Drawing from Guilford's creativity theory, our benchmark employs a dynamic panel of state-of-the-art LLMs to assess generated ideas across five key dimensions: originality, feasibility, fluency, flexibility, and clarity. Through extensive experimentation with over 40 leading models across 1,180 keywords spanning 22 scientific domains, we reveal that the scientific idea generation capabilities measured by our benchmark, are poorly predicted by standard metrics of general intelligence. Our results demonstrate that models like QwQ-32B-preview achieve creative performance comparable to top-tier models such as claude-3.7-sonnet:thinking, despite significant gaps in their general intelligence scores. These findings highlight the need for specialized evaluation benchmarks for scientific idea generation and suggest that enhancing these idea generation capabilities in LLMs may require different training strategies than those used for improving general problem-solving abilities, potentially enabling a wider range of AI tools tailored for different stages of the scientific process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16902v3">LLM-WikiRace Benchmark: How Far Can LLMs Plan over Real-World Knowledge Graphs?</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      We introduce LLM-Wikirace, a benchmark for evaluating planning, reasoning, and world knowledge in large language models (LLMs). In LLM-Wikirace, models must efficiently navigate Wikipedia hyperlinks step by step to reach a target page from a given source, requiring look-ahead planning and the ability to reason about how concepts are connected in the real world. We evaluate a broad set of open- and closed-source models, including Gemini-3, GPT-5, and Claude Opus 4.5, which achieve the strongest results on the easy level of the task and demonstrate superhuman performance. Despite this, performance drops sharply on hard difficulty: the best-performing model, Gemini-3, succeeds in only 23\% of hard games, highlighting substantial remaining challenges for frontier models. Our analysis shows that world knowledge is a necessary ingredient for success, but only up to a point, beyond this threshold, planning and long-horizon reasoning capabilities become the dominant factors. Trajectory-level analysis further reveals that even the strongest models struggle to replan after failure, frequently entering loops rather than recovering. LLM-Wikirace is a simple benchmark that reveals clear limitations in current reasoning systems, offering an open arena where planning-capable LLMs still have much to prove. Our code and leaderboard available at https:/llmwikirace.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19694v1">All Cities are Equal: A Unified Human Mobility Generation Model Enabled by LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Synthetic human mobility generation is gaining traction as an ethical and practical approach to supporting the data needs of intelligent urban systems. Existing methods perform well primarily in data-rich cities, while their effectiveness declines significantly in cities with limited data resources. However, the ability to generate reliable human mobility data should not depend on a city's size or available resources, all cities deserve equal consideration. To address this open issue, we propose UniMob, a unified human mobility generation model across cities. UniMob is composed of three main components: an LLM-powered travel planner that derives high-level, temporally-aware, and semantically meaningful travel plans; a unified spatial embedding module that projects the spatial regions of various cities into a shared representation space; and a diffusion-based mobility generator that captures the joint spatiotemporal characteristics of human movement, guided by the derived travel plans. We evaluate UniMob extensively using two real-world datasets covering five cities. Comprehensive experiments show that UniMob significantly outperforms state-of-the-art baselines, achieving improvements of over 30\% across multiple evaluation metrics. Further analysis demonstrates UniMob's robustness in both zero- and few-shot scenarios, underlines the importance of LLM guidance, verifies its privacy-preserving nature, and showcases its applicability for downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19690v1">"The explanation makes sense": An Empirical Study on LLM Performance in News Classification and its Influence on Judgment in Human-AI Collaborative Annotation</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      The spread of media bias is a significant concern as political discourse shapes beliefs and opinions. Addressing this challenge computationally requires improved methods for interpreting news. While large language models (LLMs) can scale classification tasks, concerns remain about their trustworthiness. To advance human-AI collaboration, we investigate the feasibility of using LLMs to classify U.S. news by political ideology and examine their effect on user decision-making. We first compared GPT models with prompt engineering to state-of-the-art supervised machine learning on a 34k public dataset. We then collected 17k news articles and tested GPT-4 predictions with brief and detailed explanations. In a between-subjects study (N=124), we evaluated how LLM-generated explanations influence human annotation, judgment, and confidence. Results show that AI assistance significantly increases confidence ($p<.001$), with detailed explanations more persuasive and more likely to alter decisions. We highlight recommendations for AI explanations through thematic analysis and provide our dataset for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19661v1">PaReGTA: An LLM-based EHR Data Encoding Approach to Capture Temporal Information</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 26 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Temporal information in structured electronic health records (EHRs) is often lost in sparse one-hot or count-based representations, while sequence models can be costly and data-hungry. We propose PaReGTA, an LLM-based encoding framework that (i) converts longitudinal EHR events into visit-level templated text with explicit temporal cues, (ii) learns domain-adapted visit embeddings via lightweight contrastive fine-tuning of a sentence-embedding model, and (iii) aggregates visit embeddings into a fixed-dimensional patient representation using hybrid temporal pooling that captures both recency and globally informative visits. Because PaReGTA does not require training from scratch but instead utilizes a pre-trained LLM, it can perform well even in data-limited cohorts. Furthermore, PaReGTA is model-agnostic and can benefit from future EHR-specialized sentence-embedding models. For interpretability, we introduce PaReGTA-RSS (Representation Shift Score), which quantifies clinically defined factor importance by recomputing representations after targeted factor removal and projecting representation shifts through a machine learning model. On 39,088 migraine patients from the All of Us Research Program, PaReGTA outperforms sparse baselines for migraine type classification while deep sequential models were unstable in our cohort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05695v2">SweetSpot: An Analytical Model for Predicting Energy Efficiency of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 To appear at ICPE 2026 (International Conference on Performance Engineering)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) inference is central to modern AI applications, dominating worldwide datacenter workloads, making it critical to predict its energy footprint. Existing approaches estimate energy consumption as a simple linear function of input and output sequence. However, by analyzing the autoregressive structure of Transformers, which implies a fundamentally non-linear relationship between input and output sequence lengths and energy consumption, we demonstrate the existence of a generation energy minima. Peak efficiency occurs with short-to-moderate inputs and medium-length outputs, while efficiency drops sharply for long inputs or very short outputs. Consequently, we propose SweetSpot, an analytical model derived from the computational and memory-access complexity of the Transformer architecture, which accurately characterizes the efficiency curve as a function of input and output lengths. To assess accuracy, we measure energy consumption using TensorRT-LLM on NVIDIA H100 GPUs across a diverse set of LLMs ranging from 1B to 9B parameters, including OPT, LLaMA, Gemma, Falcon, Qwen2, and Granite. We test input and output lengths from 64 to 4096 tokens and achieve a mean MAPE of 1.79%. Our results show that aligning sequence lengths with these efficiency "sweet spots" reduce energy usage, up to 33.41x, enabling informed truncation, summarization, and adaptive generation strategies in production systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19643v1">KGHaluBench: A Knowledge Graph-Based Hallucination Benchmark for Evaluating the Breadth and Depth of LLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 EACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess a remarkable capacity to generate persuasive and intelligible language. However, coherence does not equate to truthfulness, as the responses often contain subtle hallucinations. Existing benchmarks are limited by static and narrow questions, leading to limited coverage and misleading evaluations. We present KGHaluBench, a Knowledge Graph-based hallucination benchmark that assesses LLMs across the breadth and depth of their knowledge, providing a fairer and more comprehensive insight into LLM truthfulness. Our framework utilises the KG to dynamically construct challenging, multifaceted questions, whose difficulty is then statistically estimated to address popularity bias. Our automated verification pipeline detects abstentions and verifies the LLM's response at both conceptual and correctness levels to identify different types of hallucinations. We evaluate 25 frontier models, using novel accuracy and hallucination metrics. The results provide a more interpretable insight into the knowledge factors that cause hallucinations across different model sizes. KGHaluBench is publicly available to support future developments in hallucination mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13033v2">Buy versus Build an LLM: A Decision Framework for Governments</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 The short version of this document is published as an ACM TechBrief at https://dl.acm.org/doi/epdf/10.1145/3797946, and this document is published as an ACM Technology Policy Council white paper at https://www.acm.org/binaries/content/assets/public-policy/buildvsbuyai.pdf
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) represent a new frontier of digital infrastructure that can support a wide range of public-sector applications, from general purpose citizen services to specialized and sensitive state functions. When expanding AI access, governments face a set of strategic choices over whether to buy existing services, build domestic capabilities, or adopt hybrid approaches across different domains and use cases. These are critical decisions especially when leading model providers are often foreign corporations, and LLM outputs are increasingly treated as trusted inputs to public decision-making and public discourse. In practice, these decisions are not intended to mandate a single approach across all domains; instead, national AI strategies are typically pluralistic, with sovereign, commercial and open-source models coexisting to serve different purposes. Governments may rely on commercial models for non-sensitive or commodity tasks, while pursuing greater control for critical, high-risk or strategically important applications. This paper provides a strategic framework for making this decision by evaluating these options across dimensions including sovereignty, safety, cost, resource capability, cultural fit, and sustainability. Importantly, "building" does not imply that governments must act alone: domestic capabilities may be developed through public research institutions, universities, state-owned enterprises, joint ventures, or broader national ecosystems. By detailing the technical requirements and practical challenges of each pathway, this work aims to serve as a reference for policy-makers to determine whether a buy or build approach best aligns with their specific national needs and societal goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01650v2">The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Neural network pruning is a promising technique to mitigate the excessive computational and memory requirements of large language models (LLMs). Despite its promise, however, progress in this area has diminished, as conventional methods are seemingly unable to surpass moderate sparsity levels (50-60%) without severely degrading model accuracy. This work breaks through the current impasse, presenting a principled and effective method called $\texttt{Elsa}$, which achieves extreme sparsity levels of up to 90% while retaining high model fidelity. This is done by identifying several limitations in current practice, all of which can be traced back to their reliance on a surrogate objective formulation. $\texttt{Elsa}$ tackles this issue directly and effectively via standard and well-established constrained optimization techniques based on ADMM. Our extensive experiments across a wide range of models and scales show that $\texttt{Elsa}$ achieves substantial improvements over existing methods; e.g., it achieves 7.8$\times$ less perplexity than the best existing method on LLaMA-2-7B at 90% sparsity. Moreover, we show that $\texttt{Elsa}$ remains stable even at extreme sparsity (e.g., 95\%), yielding up to $\times$3.98 inference speedup and $\times$7.80 memory compression over its dense counterpart. We also present $\texttt{Elsa}_{-L}$, a quantized variant that scales to extremely large models (27B), and establish its theoretical convergence guarantees.These results highlight meaningful progress in advancing the frontier of LLM sparsity, while promising that significant opportunities for further advancement may remain in directions that have so far attracted limited exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21990v2">WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      While embeddings from multimodal large language models (LLMs) excel as general-purpose representations, their application to dynamic modalities like audio and video remains underexplored. We introduce WAVE (\textbf{u}nified \& \textbf{v}ersatile \textbf{a}udio-\textbf{v}isual \textbf{e}mbeddings), the first LLM-based embedding that creates a unified representation space for text, audio, and video modalities. WAVE employs a novel hierarchical feature fusion strategy and a joint multi-modal, multi-task training approach to enable two key capabilities: any-to-any cross-modal retrieval and the generation of prompt-aware embeddings tailored to user instructions. Experimentally, WAVE sets a new state-of-the-art on the MMEB-v2 video benchmark and achieves superior results in audio and video-to-audio retrieval. Its prompt-aware nature also yields remarkable performance in multimodal question answering, significantly outperforming existing embedding models. Ablation studies validate our joint training strategy, demonstrating improved performance across all modalities. With a newly introduced benchmark for versatile audio-visual learning, WAVE opens up broad possibilities for cross-modal, any-to-any applications. Our code and checkpoints are released at \href{https://github.com/TCL606/WAVE}{https://github.com/TCL606/WAVE}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19574v1">CTC-TTS: LLM-based dual-streaming text-to-speech with CTC alignment</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Submitted to INTERSPEECH 2026
    </div>
    <details class="paper-abstract">
      Large-language-model (LLM)-based text-to-speech (TTS) systems can generate natural speech, but most are not designed for low-latency dual-streaming synthesis. High-quality dual-streaming TTS depends on accurate text--speech alignment and well-designed training sequences that balance synthesis quality and latency. Prior work often relies on GMM-HMM based forced-alignment toolkits (e.g., MFA), which are pipeline-heavy and less flexible than neural aligners; fixed-ratio interleaving of text and speech tokens struggles to capture text--speech alignment regularities. We propose CTC-TTS, which replaces MFA with a CTC based aligner and introduces a bi-word based interleaving strategy. Two variants are designed: CTC-TTS-L (token concatenation along the sequence length) for higher quality and CTC-TTS-F (embedding stacking along the feature dimension) for lower latency. Experiments show that CTC-TTS outperforms fixed-ratio interleaving and MFA-based baselines on streaming synthesis and zero-shot tasks. Speech samples are available at https://ctctts.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10987v3">DITTO: A Spoofing Attack Framework on Watermarked LLMs via Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Accepted to EACL 2026 (Oral)
    </div>
    <details class="paper-abstract">
      The promise of LLM watermarking rests on a core assumption that a specific watermark proves authorship by a specific model. We demonstrate that this assumption is dangerously flawed. We introduce the threat of watermark spoofing, a sophisticated attack that allows a malicious model to generate text containing the authentic-looking watermark of a trusted, victim model. This enables the seamless misattribution of harmful content, such as disinformation, to reputable sources. The key to our attack is repurposing watermark radioactivity, the unintended inheritance of data patterns during fine-tuning, from a discoverable trait into an attack vector. By distilling knowledge from a watermarked teacher model, our framework allows an attacker to steal and replicate the watermarking signal of the victim model. This work reveals a critical security gap in text authorship verification and calls for a paradigm shift towards technologies capable of distinguishing authentic watermarks from expertly imitated ones. Our code is available at https://github.com/hsannn/ditto.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19548v1">Beyond a Single Extractor: Re-thinking HTML-to-Text Extraction for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      One of the first pre-processing steps for constructing web-scale LLM pretraining datasets involves extracting text from HTML. Despite the immense diversity of web content, existing open-source datasets predominantly apply a single fixed extractor to all webpages. In this work, we investigate whether this practice leads to suboptimal coverage and utilization of Internet data. We first show that while different extractors may lead to similar model performance on standard language understanding tasks, the pages surviving a fixed filtering pipeline can differ substantially. This suggests a simple intervention: by taking a Union over different extractors, we can increase the token yield of DCLM-Baseline by up to 71% while maintaining benchmark performance. We further show that for structured content such as tables and code blocks, extractor choice can significantly impact downstream task performance, with differences of up to 10 percentage points (p.p.) on WikiTQ and 3 p.p. on HumanEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19522v1">An LLM-Enabled Frequency-Aware Flow Diffusion Model for Natural-Language-Guided Power System Scenario Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      Diverse and controllable scenario generation (e.g., wind, solar, load, etc.) is critical for robust power system planning and operation. As AI-based scenario generation methods are becoming the mainstream, existing methods (e.g., Conditional Generative Adversarial Nets) mainly rely on a fixed-length numerical conditioning vector to control the generation results, facing challenges in user conveniency and generation flexibility. In this paper, a natural-language-guided scenario generation framework, named LLM-enabled Frequency-aware Flow Diffusion (LFFD), is proposed to enable users to generate desired scenarios using plain human language. First, a pretrained LLM module is introduced to convert generation requests described by unstructured natural languages into ordered semantic space. Second, instead of using standard diffusion models, a flow diffusion model employing a rectified flow matching objective is introduced to achieve efficient and high-quality scenario generation, taking the LLM output as the model input. During the model training process, a frequency-aware multi-objective optimization algorithm is introduced to mitigate the frequency-bias issue. Meanwhile, a dual-agent framework is designed to create text-scenario training sample pairs as well as to standardize semantic evaluation. Experiments based on large-scale photovoltaic and load datasets demonstrate the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19490v1">FuzzySQL: Uncovering Hidden Vulnerabilities in DBMS Special Features with LLM-Driven Fuzzing</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      Traditional database fuzzing techniques primarily focus on syntactic correctness and general SQL structures, leaving critical yet obscure DBMS features, such as system-level modes (e.g., GTID), programmatic constructs (e.g., PROCEDURE), advanced process commands (e.g., KILL), largely underexplored. Although rarely triggered by typical inputs, these features can lead to severe crashes or security issues when executed under edge-case conditions. In this paper, we present FuzzySQL, a novel LLM-powered adaptive fuzzing framework designed to uncover subtle vulnerabilities in DBMS special features. FuzzySQL combines grammar-guided SQL generation with logic-shifting progressive mutation, a novel technique that explores alternative control paths by negating conditions and restructuring execution logic, synthesizing structurally and semantically diverse test cases. To further ensure deeper execution coverage of the back end, FuzzySQL employs a hybrid error repair pipeline that unifies rule-based patching with LLM-driven semantic repair, enabling automatic correction of syntactic and context-sensitive failures. We evaluate FuzzySQL across multiple DBMSs, including MySQL, MariaDB, SQLite, PostgreSQL and Clickhouse, uncovering 37 vulnerabilities, 7 of which are tied to under-tested DBMS special features. As of this writing, 29 cases have been confirmed with 9 assigned CVE identifiers, 14 already fixed by vendors, and additional vulnerabilities scheduled to be patched in upcoming releases. Our results highlight the limitations of conventional fuzzers in semantic feature coverage and demonstrate the potential of LLM-based fuzzing to discover deeply hidden bugs in complex database systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17621v2">From Competition to Coordination: Market Making as a Scalable Framework for Safe and Aligned Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      As foundation models are increasingly deployed as interacting agents in multi-agent systems, their collective behavior raises new challenges for trustworthiness, transparency, and accountability. Traditional coordination mechanisms, such as centralized oversight or adversarial adjudication, struggle to scale and often obscure how decisions emerge. We introduce a market-making framework for multi-agent large language model (LLM) coordination that organizes agent interactions as structured economic exchanges. In this setup, each agent acts as a market participant, updating and trading probabilistic beliefs, to converge toward shared, truthful outcomes. By aligning local incentives with collective epistemic goals, the framework promotes self-organizing, verifiable reasoning without requiring external enforcement. Empirically, we evaluate this approach across factual reasoning, ethical judgment, and commonsense inference tasks. Market-based coordination yields accuracy gains of up to 10% over single-shot baselines while preserving interpretability and transparency of intermediate reasoning steps. Beyond these improvements, our findings demonstrate that economic coordination principles can operationalize accountability and robustness in multi-agent LLM systems, offering a scalable pathway toward self-correcting, socially responsible AI capable of maintaining trust and oversight in real world deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21877v2">CricBench: A Multilingual Benchmark for Evaluating LLMs in Cricket Analytics</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Cricket is the second most popular sport globally, commanding a massive following of over 2.5 billion fans globally. Enthusiasts and analysts frequently seek advanced statistical insights, such as long-term historical performance trends or complex player comparisons, that are often unavailable through standard web searches. While Large Language Models (LLMs) have advanced significantly in Text-to-SQL tasks, their capability to handle the domain-specific nuances, complex schema variations, and multilingual requirements inherent to sports analytics remains under-explored. To investigate this potential capability gap, we present CricBench, a comprehensive benchmark suite for evaluating LLMs on specialized cricket data. To curate a "Gold Standard" dataset, we collaborate with domain experts in cricket and SQL to manually author complex queries, ensuring logical correctness. Recognizing linguistic diversity, we construct the benchmark in both English and Hindi, establishing a framework that is open for further extension to other regional languages. We evaluate six state-of-the-art models, including GPT-4o, Claude 3.7 Sonnet, and open-source models, using a strict evaluation protocol. Our results reveal that high performance on general benchmarks does not guarantee success in specialized domains. While the open-weights reasoning model DeepSeek R1 achieves state-of-the-art performance (50.6%), surpassing proprietary giants like Claude 3.7 Sonnet (47.7%) and GPT-4o (33.7%), it still exhibits a significant accuracy drop when moving from general benchmarks (BIRD) to CricBench. Furthermore, we observe that code-mixed Hindi queries frequently yield parity or higher accuracy compared to English, challenging the assumption that English is the optimal prompt language for specialized SQL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16167v2">Ev-Trust: An Evolutionary Stable Trust Mechanism for Decentralized LLM-Based Multi-Agent Service Economies</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 19 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Autonomous LLM-based agents are increasingly engaging in decentralized service interactions to collaboratively execute complex tasks. However, the intrinsic instability and low-cost generativity of LLMs introduce a systemic vulnerability, where self-interested agents are incentivized to pursue short-term gains through deceptive behaviors. Such strategies can rapidly proliferate within the population and precipitate a systemic trust collapse. To address this, we propose Ev-Trust, a strategy-equilibrium trust mechanism grounded in evolutionary game theory. Ev-Trust constructs a dynamic feedback loop that couples trust evaluation with evolutionary incentives, embedding interaction history and reputation directly into the agent's expected revenue function. This mechanism fundamentally reshapes the revenue structure, converting trustworthiness into a decisive survival advantage that suppresses short-sightedness. We provide a rigorous theoretical foundation based on the Replicator Dynamics, proving the asymptotic stability of Evolutionary Stable Strategies (ESS) that favor cooperation. Experimental results indicate that Ev-Trust effectively eliminates malicious strategies and enhances collective revenue, exhibiting resilience against the invasion of mutant behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19458v1">ComplLLM: Fine-tuning LLMs to Discover Complementary Signals for Decision-making</a></div>
    <div class="paper-meta">
      📅 2026-02-23
    </div>
    <details class="paper-abstract">
      Multi-agent decision pipelines can outperform single agent workflows when complementarity holds, i.e., different agents bring unique information to the table to inform a final decision. We propose ComplLLM, a post-training framework based on decision theory that fine-tunes a decision-assistant LLM using complementary information as reward to output signals that complement existing agent decisions. We validate ComplLLM on synthetic and real-world tasks involving domain experts, demonstrating how the approach recovers known complementary information and produces plausible explanations of complementary signals to support downstream decision-makers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17560v2">ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-02-23
      | 💬 Accepted by ICLR 2026 (Camera Ready Version)
    </div>
    <details class="paper-abstract">
      Activation steering, or representation engineering, offers a lightweight approach to align large language models (LLMs) by manipulating their internal activations at inference time. However, current methods suffer from two key limitations: (i) the lack of a unified theoretical framework for guiding the design of steering directions, and (ii) an over-reliance on one-step steering that fail to capture complex patterns of activation distributions. In this work, we propose a unified ordinary differential equations (ODEs)-based theoretical framework for activation steering in LLM alignment. We show that conventional activation addition can be interpreted as a first-order approximation to the solution of an ODE. Based on this ODE perspective, identifying a steering direction becomes equivalent to designing a barrier function from control theory. Derived from this framework, we introduce ODESteer, a kind of ODE-based steering guided by barrier functions, which shows empirical advancement in LLM alignment. ODESteer identifies steering directions by defining the barrier function as the log-density ratio between positive and negative activations, and employs it to construct an ODE for multi-step and adaptive steering. Compared to state-of-the-art activation steering methods, ODESteer achieves consistent empirical improvements on diverse LLM alignment benchmarks, a notable $5.7\%$ improvement over TruthfulQA, $2.5\%$ over UltraFeedback, and $2.4\%$ over RealToxicityPrompts. Our work establishes a principled new view of activation steering in LLM alignment by unifying its theoretical foundations via ODEs, and validating it empirically through the proposed ODESteer method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25850v2">Debate2Create: Robot Co-design via Multi-Agent LLM Debate</a></div>
    <div class="paper-meta">
      📅 2026-02-22
    </div>
    <details class="paper-abstract">
      We introduce Debate2Create (D2C), a multi-agent LLM framework that formulates robot co-design as structured, iterative debate grounded in physics-based evaluation. A design agent and control agent engage in a thesis-antithesis-synthesis loop, while pluralistic LLM judges provide multi-objective feedback to steer exploration. Across five MuJoCo locomotion benchmarks, D2C achieves up to $3.2\times$ the default Ant score and $\sim9\times$ on Swimmer, outperforming prior LLM-based methods and black-box optimization. Iterative debate yields 18--35% gains over compute-matched zero-shot generation, and D2C-generated rewards transfer to default morphologies in 4/5 tasks. Our results demonstrate that structured multi-agent debate offers an effective alternative to hand-designed objectives for joint morphology-reward optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19362v1">LLMs Can Learn to Reason Via Off-Policy RL</a></div>
    <div class="paper-meta">
      📅 2026-02-22
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) approaches for Large Language Models (LLMs) frequently use on-policy algorithms, such as PPO or GRPO. However, policy lag from distributed training architectures and differences between the training and inference policies break this assumption, making the data off-policy by design. To rectify this, prior work has focused on making this off-policy data appear more on-policy, either via importance sampling (IS), or by more closely aligning the training and inference policies by explicitly modifying the inference engine. In this work, we embrace off-policyness and propose a novel off-policy RL algorithm that does not require these modifications: Optimal Advantage-based Policy Optimization with Lagged Inference policy (OAPL). We show that OAPL outperforms GRPO with importance sampling on competition math benchmarks, and can match the performance of a publicly available coding model, DeepCoder, on LiveCodeBench, while using 3x fewer generations during training. We further empirically demonstrate that models trained via OAPL have improved test time scaling under the Pass@k metric. OAPL allows for efficient, effective post-training even with lags of more than 400 gradient steps between the training and inference policies, 100x more off-policy than prior approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08233v2">Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-22
    </div>
    <details class="paper-abstract">
      Diffusion large language models (dLLMs) are promising alternatives to autoregressive large language models (AR-LLMs), as they potentially allow higher inference throughput. Reinforcement learning (RL) is a crucial component for dLLMs to achieve comparable performance with AR-LLMs on important tasks, such as reasoning. However, RL algorithms that are well-suited for dLLMs' unique characteristics have yet to be developed. This paper proposes Distribution Matching Policy Optimization (DMPO), a principled and theoretically grounded RL fine-tuning method specifically designed to enhance the reasoning capabilities of dLLMs by matching the dLLM policy distribution to the optimal, reward-tilted one through cross-entropy optimization. We identify a key challenge in the implementation with a small training batch size and propose several effective solutions through a novel weight baseline subtraction technique. DMPO exhibits superior performance on multiple reasoning benchmarks without supervised fine-tuning, with an accuracy improvement of up to $54.3\%$ over previously SOTA baselines and $66.41\%$ over the base model, underscoring the effectiveness of the distribution matching framework. Our code is available at https://github.com/yuchen-zhu-zyc/DMPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04891v2">SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests</a></div>
    <div class="paper-meta">
      📅 2026-02-22
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in contexts where their failures can have direct sociopolitical consequences. Yet, existing safety benchmarks rarely test vulnerabilities in domains such as political manipulation, propaganda and disinformation generation, or surveillance and information control. We introduce SocialHarmBench, a dataset of 585 prompts spanning 7 sociopolitical categories and 34 countries, designed to surface where LLMs most acutely fail in politically charged contexts. Our evaluations reveal several shortcomings: open-weight models exhibit high vulnerability to harmful compliance, with Mistral-7B reaching attack success rates as high as 97% to 98% in domains such as historical revisionism, propaganda, and political manipulation. Moreover, temporal and geographic analyses show that LLMs are most fragile when confronted with 21st-century or pre-20th-century contexts, and when responding to prompts tied to regions such as Latin America, the USA, and the UK. These findings demonstrate that current safeguards fail to generalize to high-stakes sociopolitical settings, exposing systematic biases and raising concerns about the reliability of LLMs in preserving human rights and democratic values. We share the SocialHarmBench benchmark at https://huggingface.co/datasets/psyonp/SocialHarmBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.08783v2">CodePDE: An Inference Framework for LLM-driven PDE Solver Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-22
      | 💬 TMLR. Code available at https://github.com/LithiumDA/CodePDE
    </div>
    <details class="paper-abstract">
      Partial differential equations (PDEs) are fundamental to modeling physical systems, yet solving them remains a complex challenge. Traditional numerical solvers rely on expert knowledge to implement and are computationally expensive, while neural-network-based solvers require large training datasets and often lack interpretability. In this work, we frame PDE solving as a code generation task and introduce CodePDE, the first inference framework for generating PDE solvers using large language models (LLMs). With CodePDE, we present a thorough evaluation on critical capacities of LLM for PDE solving: reasoning, debugging, self-refinement, and test-time scaling. CodePDE shows that, with advanced inference-time algorithms and scaling strategies, LLMs can achieve strong performance across a range of representative PDE problems. We also identify novel insights into LLM-driven solver generation, such as trade-offs between solver reliability and sophistication, design principles for LLM-powered PDE solving agents, and failure modes for LLM on hard tasks. These insights offer guidance for building more capable and reliable LLM-based scientific engines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17622v1">What Makes a Good LLM Agent for Real-world Penetration Testing?</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      LLM-based agents show promise for automating penetration testing, yet reported performance varies widely across systems and benchmarks. We analyze 28 LLM-based penetration testing systems and evaluate five representative implementations across three benchmarks of increasing complexity. Our analysis reveals two distinct failure modes: Type A failures stem from capability gaps (missing tools, inadequate prompts) that engineering readily addresses, while Type B failures persist regardless of tooling due to planning and state management limitations. We show that Type B failures share a root cause that is largely invariant to the underlying LLM: agents lack real-time task difficulty estimation. As a result, agents misallocate effort, over-commit to low-value branches, and exhaust context before completing attack chains. Based on this insight, we present Excalibur, a penetration testing agent that couples strong tooling with difficulty-aware planning. A Tool and Skill Layer eliminates Type A failures through typed interfaces and retrieval-augmented knowledge. A Task Difficulty Assessment (TDA) mechanism addresses Type B failures by estimating tractability through four measurable dimensions (horizon estimation, evidence confidence, context load, and historical success) and uses these estimates to guide exploration-exploitation decisions within an Evidence-Guided Attack Tree Search (EGATS) framework. Excalibur achieves up to 91% task completion on CTF benchmarks with frontier models (39 to 49% relative improvement over baselines) and compromises 4 of 5 hosts on the GOAD Active Directory environment versus 2 by prior systems. These results show that difficulty-aware planning yields consistent end-to-end gains across models and addresses a limitation that model scaling alone does not eliminate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17598v1">The Cascade Equivalence Hypothesis: When Do Speech LLMs Behave Like ASR$\rightarrow$LLM Pipelines?</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 10 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Current speech LLMs largely perform implicit ASR: on tasks solvable from a transcript, they are behaviorally and mechanistically equivalent to simple Whisper$\to$LLM cascades. We show this through matched-backbone testing across four speech LLMs and six tasks, controlling for the LLM backbone for the first time. Ultravox is statistically indistinguishable from its matched cascade ($κ{=}0.93$); logit lens reveals literal text emerging in hidden states; LEACE concept erasure confirms text representations are causally necessary in both architectures tested, collapsing accuracy to near-zero. Qwen2-Audio genuinely diverges, revealing cascade equivalence is architecture-dependent, not universal. For most deployed use cases, current speech LLMs are expensive cascades, and under noise, they are worse ones, with clean-condition advantages reversing by up to 7.6% at 0 dB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22075v4">CoSpaDi: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Post-training compression of large language models (LLMs) often relies on low-rank weight approximations that represent each column of the weight matrix in a shared low-dimensional subspace. This strategy is computationally efficient but the underlying constraint can be overly rigid for heterogeneous projection weights and may incur avoidable accuracy loss. We propose CoSpaDi (Compression via Sparse Dictionary Learning), a training-free framework that replaces low-rank factorization with a structured sparse decomposition in which each weight matrix is represented as a dense dictionary multiplied by a column-sparse coefficient matrix. This yields a union-of-subspaces model: the columns of the weight matrix are represented as linear combinations of different subsets of dictionary atoms, improving expressiveness at a fixed parameter budget. CoSpaDi is calibration-guided: using a small calibration set, we optimize the factorization to minimize functional reconstruction error of layer outputs rather than weight-space error. An activation-derived Gram orthonormalization reformulates this data-aware objective into a standard dictionary learning problem on transformed weights, and we support both per-layer compression and cross-layer dictionary sharing within groups of similar projections. Across Llama and Qwen model families, CoSpaDi consistently improves the accuracy--compression and perplexity--compression trade-offs over state-of-the-art SVD-based baselines and strong structured pruning baselines at 20-40\% compression ratios. The resulting structured sparsity enables sparse--dense computation and integrates with post-training quantization of the sparse coefficients.
    </details>
</div>
