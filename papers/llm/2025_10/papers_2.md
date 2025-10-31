# llm - 2025_10

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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11507v2">Oneiros: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      KV cache accelerates LLM inference by avoiding redundant computation, at the expense of memory. To support larger KV caches, prior work extends GPU memory with CPU memory via CPU-offloading. This involves swapping KV cache between GPU and CPU memory. However, because the cache updates dynamically, such swapping incurs high CPU memory traffic. We make a key observation that model parameters remain constant during runtime, unlike the dynamically updated KV cache. Building on this, we introduce Oneiros, which avoids KV cache swapping by remapping, and thereby repurposing, the memory allocated to model parameters for KV cache. This parameter remapping is especially beneficial in multi-tenant environments, where the memory used for the parameters of the inactive models can be more aggressively reclaimed. Exploiting the high CPU-GPU bandwidth offered by the modern hardware, such as the NVIDIA Grace Hopper Superchip, we show that Oneiros significantly outperforms state-of-the-art solutions, achieving a reduction of 44.8%-82.5% in tail time-between-token latency, 20.7%-99.3% in tail time-to-first-token latency, and 6.6%-86.7% higher throughput compared to vLLM. Source code of Oneiros is available at https://github.com/UT-SysML/Oneiros/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25979v1">AttnCache: Accelerating Self-Attention Inference for LLM Prefill via Attention Cache</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 10 pages, 6 figures, submitted to Ninth Annual Conference on Machine Learning and Systems (MLSys'26)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in generative applications such as chatting, code generation, and reasoning. However, many realworld workloads such as classification, question answering, recommendation, and text embedding rely solely on the prefill stage of inference, where the model encodes input sequences without performing autoregressive decoding. In these prefill only scenarios, the self-attention computation becomes the primary performance bottleneck due to its quadratic complexity with respect to sequence length. In this paper, we observe that semantically different sentences often produce similar attention maps across layers and heads. Building on this insight, we propose AttnCache, a framework that accelerates the prefill stage of LLM inference by retrieving and reusing similar attention maps. Based on an attention map memorization database, AttnCache employs efficient caching and similarity search techniques to identify and reuse pre-cached attention maps during inference, thereby reducing the computational overhead of self-attention. Experimental results show that AttnCache achieves an average of 1.2x end-to-end and 2x attention speedup on CPU, and 1.6x end-to-end and 3x attention speedup on GPU, with negligible accuracy degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25977v1">NeuronMM: High-Performance Matrix Multiplication for LLM Inference on AWS Trainium</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 12 pages, 8 figures, submitted to the Proceedings of the Twenty-First European Conference on Computer Systems (EuroSys'26)
    </div>
    <details class="paper-abstract">
      AI accelerators, customized to AI workloads, provide cost-effective and high-performance solutions for training and inference. Trainium, an AI accelerator recently developed by Amazon Web Services (AWS), provides an attractive option for LLM training and inference through its heterogeneous architecture. However, leveraging Trainium architecture for high performance can be challenging because of its systolic array architecture and special requirement on data layout. In this paper, we design high-performance matrix multiplication (matmul), a critical compute kernel, for LLM inference on Trainium. We introduce a series of techniques customized to Trainium based on kernel fusion and novel caching strategies to reduce data movement across the software-managed memory hierarchy, maximize SRAM bandwidth, and avoid expensive matrix transpose. Evaluating with nine datasets and four recent LLMs, we show that our system largely outperforms the state-of-the-art matmul implemented by AWS on Trainium: at the level of matmul kernel, it achieves an average 1.35x speedup (up to 2.22x), which translates to an average 1.66x speedup (up to 2.49x) for end-to-end LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14879v4">Vital Insight: Assisting Experts' Context-Driven Sensemaking of Multi-modal Personal Tracking Data Using Visualization and Human-In-The-Loop LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Jiachen Li, Xiwen Li, Justin Steinberg, Akshat Choube, Bingsheng Yao, Xuhai Xu, Dakuo Wang, Elizabeth Mynatt, and Varun Mishra. 2025. Vital Insight: Assisting Experts' Context-Driven Sensemaking of Multi-modal Personal Tracking Data Using Visualization and Human-in-the-Loop LLM. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 9, 3, Article 101 (September 2025), 37 pages
    </div>
    <details class="paper-abstract">
      Passive tracking methods, such as phone and wearable sensing, have become dominant in monitoring human behaviors in modern ubiquitous computing studies. While there have been significant advances in machine-learning approaches to translate periods of raw sensor data to model momentary behaviors, (e.g., physical activity recognition), there still remains a significant gap in the translation of these sensing streams into meaningful, high-level, context-aware insights that are required for various applications (e.g., summarizing an individual's daily routine). To bridge this gap, experts often need to employ a context-driven sensemaking process in real-world studies to derive insights. This process often requires manual effort and can be challenging even for experienced researchers due to the complexity of human behaviors. We conducted three rounds of user studies with 21 experts to explore solutions to address challenges with sensemaking. We follow a human-centered design process to identify needs and design, iterate, build, and evaluate Vital Insight (VI), a novel, LLM-assisted, prototype system to enable human-in-the-loop inference (sensemaking) and visualizations of multi-modal passive sensing data from smartphones and wearables. Using the prototype as a technology probe, we observe experts' interactions with it and develop an expert sensemaking model that explains how experts move between direct data representations and AI-supported inferences to explore, question, and validate insights. Through this iterative process, we also synthesize and discuss a list of design implications for the design of future AI-augmented visualization systems to better assist experts' sensemaking processes in multi-modal health sensing data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25941v1">RECAP: Reproducing Copyrighted Data from LLMs Training with an Agentic Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      If we cannot inspect the training data of a large language model (LLM), how can we ever know what it has seen? We believe the most compelling evidence arises when the model itself freely reproduces the target content. As such, we propose RECAP, an agentic pipeline designed to elicit and verify memorized training data from LLM outputs. At the heart of RECAP is a feedback-driven loop, where an initial extraction attempt is evaluated by a secondary language model, which compares the output against a reference passage and identifies discrepancies. These are then translated into minimal correction hints, which are fed back into the target model to guide subsequent generations. In addition, to address alignment-induced refusals, RECAP includes a jailbreaking module that detects and overcomes such barriers. We evaluate RECAP on EchoTrace, a new benchmark spanning over 30 full books, and the results show that RECAP leads to substantial gains over single-iteration approaches. For instance, with GPT-4.1, the average ROUGE-L score for the copyrighted text extraction improved from 0.38 to 0.47 - a nearly 24% increase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19771v2">Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25939v1">SoK: Honeypots & LLMs, More Than the Sum of Their Parts?</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Systemization of Knowledge
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design: achieving high-fidelity deception with low operational risk. However, despite a flurry of research since late 2022, progress has been incremental, and the field lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview of this new domain. We survey and systematize three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04953v2">M-Prometheus: A Suite of Open Multilingual LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      The use of language models for automatically evaluating long-form text (LLM-as-a-judge) is becoming increasingly common, yet most LLM judges are optimized exclusively for English, with strategies for enhancing their multilingual evaluation capabilities remaining largely unexplored in the current literature. This has created a disparity in the quality of automatic evaluation methods for non-English languages, ultimately hindering the development of models with better multilingual capabilities. To bridge this gap, we introduce M-Prometheus, a suite of open-weight LLM judges ranging from 3B to 14B parameters that can provide both direct assessment and pairwise comparison feedback on multilingual outputs. M-Prometheus models outperform state-of-the-art open LLM judges on multilingual reward benchmarks spanning more than 20 languages, as well as on literary machine translation (MT) evaluation covering 4 language pairs. Furthermore, M-Prometheus models can be leveraged at decoding time to significantly improve generated outputs across all 3 tested languages, showcasing their utility for the development of better multilingual models. Lastly, through extensive ablations, we identify the key factors for obtaining an effective multilingual judge, including backbone model selection and training on synthetic multilingual feedback data instead of translated data. We release our models, training dataset, and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25904v1">Evaluating the Impact of LLM-Assisted Annotation in a Perspectivized Setting: the Case of FrameNet Annotation</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      The use of LLM-based applications as a means to accelerate and/or substitute human labor in the creation of language resources and dataset is a reality. Nonetheless, despite the potential of such tools for linguistic research, comprehensive evaluation of their performance and impact on the creation of annotated datasets, especially under a perspectivized approach to NLP, is still missing. This paper contributes to reduction of this gap by reporting on an extensive evaluation of the (semi-)automatization of FrameNet-like semantic annotation by the use of an LLM-based semantic role labeler. The methodology employed compares annotation time, coverage and diversity in three experimental settings: manual, automatic and semi-automatic annotation. Results show that the hybrid, semi-automatic annotation setting leads to increased frame diversity and similar annotation coverage, when compared to the human-only setting, while the automatic setting performs considerably worse in all metrics, except for annotation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21204v2">Fuzzy, Symbolic, and Contextual: Enhancing LLM Instruction via Cognitive Scaffolding</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      We study how prompt-level inductive biases influence the cognitive behavior of large language models (LLMs) in instructional dialogue. We introduce a symbolic scaffolding method paired with a short-term memory schema designed to promote adaptive, structured reasoning in Socratic tutoring. Using controlled ablation across five system variants, we evaluate model outputs via expert-designed rubrics covering scaffolding, responsiveness, symbolic reasoning, and conversational memory. We present preliminary results using an LLM-based evaluation framework aligned to a cognitively grounded rubric. This enables scalable, systematic comparisons across architectural variants in early-stage experimentation. The preliminary results show that our full system consistently outperforms baseline variants. Analysis reveals that removing memory or symbolic structure degrades key cognitive behaviors, including abstraction, adaptive probing, and conceptual continuity. These findings support a processing-level account in which prompt-level cognitive scaffolds can reliably shape emergent instructional strategies in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25860v1">Through the Judge's Eyes: Inferred Thinking Traces Improve Reliability of LLM Raters</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as raters for evaluation tasks. However, their reliability is often limited for subjective tasks, when human judgments involve subtle reasoning beyond annotation labels. Thinking traces, the reasoning behind a judgment, are highly informative but challenging to collect and curate. We present a human-LLM collaborative framework to infer thinking traces from label-only annotations. The proposed framework uses a simple and effective rejection sampling method to reconstruct these traces at scale. These inferred thinking traces are applied to two complementary tasks: (1) fine-tuning open LLM raters; and (2) synthesizing clearer annotation guidelines for proprietary LLM raters. Across multiple datasets, our methods lead to significantly improved LLM-human agreement. Additionally, the refined annotation guidelines increase agreement among different LLM models. These results suggest that LLMs can serve as practical proxies for otherwise unrevealed human thinking traces, enabling label-only corpora to be extended into thinking-trace-augmented resources that enhance the reliability of LLM raters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25808v1">PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across diverse domains, due to their strong instruction-following capabilities. This has led to increasing interest in optimizing instructions for black-box LLMs, whose internal parameters are inaccessible but widely used due to their strong performance. To optimize instructions for black-box LLMs, recent methods employ white-box LLMs to generate candidate instructions from optimized soft prompts. However, white-box LLMs often map different soft prompts to the same instruction, leading to redundant queries. While previous studies regarded this many-to-one mapping as a structure that hinders optimization efficiency, we reinterpret it as a useful prior knowledge that can accelerate the optimization. To this end, we introduce PREimage-informed inSTruction Optimization (PRESTO), a novel framework that leverages the preimage structure of soft prompts for efficient optimization. PRESTO consists of three key components: (1) score sharing, which shares the evaluation score with all soft prompts in a preimage; (2) preimage-based initialization, which selects initial data points that maximize search space coverage using preimage information; and (3) score consistency regularization, which enforces prediction consistency within each preimage. By leveraging preimages, PRESTO achieves the effect of effectively obtaining 14 times more scored data under the same query budget, resulting in more efficient optimization. Experimental results on 33 instruction optimization tasks demonstrate the superior performance of PRESTO. Code is available at https://github.com/mlvlab/PRESTO
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25805v1">Ideology-Based LLMs for Content Moderation</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in content moderation systems, where ensuring fairness and neutrality is essential. In this study, we examine how persona adoption influences the consistency and fairness of harmful content classification across different LLM architectures, model sizes, and content modalities (language vs. vision). At first glance, headline performance metrics suggest that personas have little impact on overall classification accuracy. However, a closer analysis reveals important behavioral shifts. Personas with different ideological leanings display distinct propensities to label content as harmful, showing that the lens through which a model "views" input can subtly shape its judgments. Further agreement analyses highlight that models, particularly larger ones, tend to align more closely with personas from the same political ideology, strengthening within-ideology consistency while widening divergence across ideological groups. To show this effect more directly, we conducted an additional study on a politically targeted task, which confirmed that personas not only behave more coherently within their own ideology but also exhibit a tendency to defend their perspective while downplaying harmfulness in opposing views. Together, these findings highlight how persona conditioning can introduce subtle ideological biases into LLM outputs, raising concerns about the use of AI systems that may reinforce partisan perspectives under the guise of neutrality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25799v1">LISTEN to Your Preferences: An LLM Framework for Multi-Objective Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Human experts often struggle to select the best option from a large set of items with multiple competing objectives, a process bottlenecked by the difficulty of formalizing complex, implicit preferences. To address this, we introduce LISTEN, a framework that leverages a Large Language Model (LLM) as a zero-shot preference oracle, guided only by an expert's high-level priorities in natural language. To operate within LLM constraints like context windows and inference costs, we propose two iterative algorithms: LISTEN-U, which uses the LLM to refine a parametric utility function, and LISTEN-T, a non-parametric method that performs tournament-style selections over small batches of solutions. Evaluated on diverse tasks including flight booking, shopping, and exam scheduling, our results show LISTEN-U excels when preferences are parametrically aligned (a property we measure with a novel concordance metric), while LISTEN-T offers more robust performance. This work explores a promising direction for steering complex multi-objective decisions directly with natural language, reducing the cognitive burden of traditional preference elicitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24832v1">Scheduling Your LLM Reinforcement Learning with Reasoning Trees</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Using Reinforcement Learning with Verifiable Rewards (RLVR) to optimize Large Language Models (LLMs) can be conceptualized as progressively editing a query's `Reasoning Tree'. This process involves exploring nodes (tokens) and dynamically modifying the model's policy at each node. When combined with data scheduling, this process yields further gains in data efficiency and accuracy. However, existing RLVR data scheduling methods typically rely on path-based metrics to rank queries, overlooking the reasoning tree structures of these queries. In this paper, we introduce a novel metric, namely Reasoning Score (r-score), which measures the query's learning difficulty based on the structure of its reasoning tree. Based on the r-score, we propose the Reasoning Tree Schedule (Re-Schedule), a scheduling algorithm that constructs a curriculum progressing from structurally simple (high r-score) to complex (low r-score) queries. Experiments on six math-reasoning benchmarks show that Re-Schedule significantly improves average accuracy, achieving gains of up to 3.2%. These strong results validate our approach and demonstrate that a structural understanding of the reasoning tree provides a more powerful and principled foundation for RLVR data scheduling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24802v1">From Narrative to Action: A Hierarchical LLM-Agent Framework for Human Mobility Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 47 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Understanding and replicating human mobility requires not only spatial-temporal accuracy but also an awareness of the cognitive hierarchy underlying real-world travel decisions. Traditional agent-based or deep learning models can reproduce statistical patterns of movement but fail to capture the semantic coherence and causal logic of human behavior. Large language models (LLMs) show potential, but struggle to balance creative reasoning with strict structural compliance. This study proposes a Hierarchical LLM-Agent Framework, termed Narrative-to-Action, that integrates high-level narrative reasoning, mid-level reflective planning, and low-level behavioral execution within a unified cognitive hierarchy. At the macro level, one agent is employed as a "creative writer" to produce diary-style narratives rich in motivation and context, then uses another agent as a "structural parser" to convert narratives into machine-readable plans. A dynamic execution module further grounds agents in geographic environments and enables adaptive behavioral adjustments guided by a novel occupation-aware metric, Mobility Entropy by Occupation (MEO), which captures heterogeneous schedule flexibility across different occupational personalities. At the micro level, the agent executes concrete actions-selecting locations, transportation modes, and time intervals-through interaction with an environmental simulation. By embedding this multi-layer cognitive process, the framework produces not only synthetic trajectories that align closely with real-world patterns but also interpretable representations of human decision logic. This research advances synthetic mobility generation from a data-driven paradigm to a cognition-driven simulation, providing a scalable pathway for understanding, predicting, and synthesizing complex urban mobility behaviors through hierarchical LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24706v1">ComboBench: Can LLMs Manipulate Physical Devices to Play Virtual Reality Games?</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Virtual Reality (VR) games require players to translate high-level semantic actions into precise device manipulations using controllers and head-mounted displays (HMDs). While humans intuitively perform this translation based on common sense and embodied understanding, whether Large Language Models (LLMs) can effectively replicate this ability remains underexplored. This paper introduces a benchmark, ComboBench, evaluating LLMs' capability to translate semantic actions into VR device manipulation sequences across 262 scenarios from four popular VR games: Half-Life: Alyx, Into the Radius, Moss: Book II, and Vivecraft. We evaluate seven LLMs, including GPT-3.5, GPT-4, GPT-4o, Gemini-1.5-Pro, LLaMA-3-8B, Mixtral-8x7B, and GLM-4-Flash, compared against annotated ground truth and human performance. Our results reveal that while top-performing models like Gemini-1.5-Pro demonstrate strong task decomposition capabilities, they still struggle with procedural reasoning and spatial understanding compared to humans. Performance varies significantly across games, suggesting sensitivity to interaction complexity. Few-shot examples substantially improve performance, indicating potential for targeted enhancement of LLMs' VR manipulation capabilities. We release all materials at https://sites.google.com/view/combobench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24695v1">AgentFrontier: Expanding the Capability Frontier of LLM Agents with ZPD-Guided Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
    </div>
    <details class="paper-abstract">
      Training large language model agents on tasks at the frontier of their capabilities is key to unlocking advanced reasoning. We introduce a data synthesis approach inspired by the educational theory of the Zone of Proximal Development (ZPD), which defines this frontier as tasks an LLM cannot solve alone but can master with guidance. To operationalize this, we present the AgentFrontier Engine, an automated pipeline that synthesizes high-quality, multidisciplinary data situated precisely within the LLM's ZPD. This engine supports both continued pre-training with knowledge-intensive data and targeted post-training on complex reasoning tasks. From the same framework, we derive the ZPD Exam, a dynamic and automated benchmark designed to evaluate agent capabilities on these frontier tasks. We train AgentFrontier-30B-A3B model on our synthesized data, which achieves state-of-the-art results on demanding benchmarks like Humanity's Last Exam, even surpassing some leading proprietary agents. Our work demonstrates that a ZPD-guided approach to data synthesis offers a scalable and effective path toward building more capable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24677v1">Dissecting Role Cognition in Medical LLMs via Neuronal Ablation</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have gained significant traction in medical decision support systems, particularly in the context of medical question answering and role-playing simulations. A common practice, Prompt-Based Role Playing (PBRP), instructs models to adopt different clinical roles (e.g., medical students, residents, attending physicians) to simulate varied professional behaviors. However, the impact of such role prompts on model reasoning capabilities remains unclear. This study introduces the RP-Neuron-Activated Evaluation Framework(RPNA) to evaluate whether role prompts induce distinct, role-specific cognitive processes in LLMs or merely modify linguistic style. We test this framework on three medical QA datasets, employing neuron ablation and representation analysis techniques to assess changes in reasoning pathways. Our results demonstrate that role prompts do not significantly enhance the medical reasoning abilities of LLMs. Instead, they primarily affect surface-level linguistic features, with no evidence of distinct reasoning pathways or cognitive differentiation across clinical roles. Despite superficial stylistic changes, the core decision-making mechanisms of LLMs remain uniform across roles, indicating that current PBRP methods fail to replicate the cognitive complexity found in real-world medical practice. This highlights the limitations of role-playing in medical AI and emphasizes the need for models that simulate genuine cognitive processes rather than linguistic imitation.We have released the related code in the following repository:https: //github.com/IAAR-Shanghai/RolePlay_LLMDoctor
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24606v1">Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 Accepted to NeurIPS 2025 Workshop on Efficient Reasoning
    </div>
    <details class="paper-abstract">
      The quadratic cost of attention hinders the scalability of long-context LLMs, especially in resource-constrained settings. Existing static sparse methods such as sliding windows or global tokens utilizes the sparsity of attention to reduce the cost of attention, but poorly adapts to the content-dependent variations in attention due to their staticity. While previous work has proposed several dynamic approaches to improve flexibility, they still depend on predefined templates or heuristic mechanisms. Such strategies reduce generality and prune tokens that remain contextually important, limiting their accuracy across diverse tasks. To tackle these bottlenecks of existing methods for long-context modeling, we introduce Dynamic Hierarchical Sparse Attention (DHSA), a data-driven framework that dynamically predicts attention sparsity online without retraining. Our proposed DHSA adaptively segments sequences into variable-length chunks, then computes chunk representations by aggregating the token embeddings within each chunk. To avoid the bias introduced by varying chunk lengths, we apply length-normalized aggregation that scales the averaged embeddings by the square root of the chunk size. Finally, DHSA upsamples the chunk-level similarity scores to token level similarities to calculate importance scores that determine which token-level interactions should be preserved. Our experiments on Gemma2 with Needle-in-a-Haystack Test and LongBench show that DHSA matches dense attention in accuracy, while reducing prefill latency by 20-60% and peak memory usage by 35%. Compared to other representative baselines such as block sparse attention, DHSA achieves consistently higher accuracy (6-18% relative gains) with comparable or lower cost, offering an efficient and adaptable solution for long-context on-device LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24605v1">Diffusion LLM with Native Variable Generation Lengths: Let [EOS] Lead the Way</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) have exhibited substantial potential for parallel text generation, which may enable more efficient generation compared to autoregressive models. However, current dLLMs suffer from fixed generation lengths, which indicates the generation lengths of dLLMs have to be determined before decoding as a hyper-parameter, leading to issues in efficiency and flexibility. To solve these problems, in this work, we propose to train a diffusion LLM with native variable generation lengths, abbreviated as dLLM-Var. Concretely, we aim to train a model to accurately predict the [EOS] token in the generated text, which makes a dLLM be able to natively infer in a block diffusion manner, while still maintaining the ability of global bi-directional (full) attention and high parallelism. Experiments on standard benchmarks demonstrate that our method achieves a 30.1x speedup over traditional dLLM inference paradigms and a 2.4x speedup relative to autoregressive models such as Qwen and Llama. Our method achieves higher accuracy and faster inference, elevating dLLMs beyond mere academic novelty and supporting their practical use in real-world applications. Codes and models have been released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24582v1">Politically Speaking: LLMs on Changing International Affairs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Ask your chatbot to impersonate an expert from Russia and an expert from US and query it on Chinese politics. How might the outputs differ? Or, to prepare ourselves for the worse, how might they converge? Scholars have raised concerns LLM based applications can homogenize cultures and flatten perspectives. But exactly how much does LLM generated outputs converge despite explicit different role assignment? This study provides empirical evidence to the above question. The critique centres on pretrained models regurgitating ossified political jargons used in the Western world when speaking about China, Iran, Russian, and US politics, despite changes in these countries happening daily or hourly. The experiments combine role-prompting and similarity metrics. The results show that AI generated discourses from four models about Iran and China are the most homogeneous and unchanging across all four models, including OpenAI GPT, Google Gemini, Anthropic Claude, and DeepSeek, despite the prompted perspective change and the actual changes in real life. This study does not engage with history, politics, or literature as traditional disciplinary approaches would; instead, it takes cues from international and area studies and offers insight on the future trajectory of shifting political discourse in a digital space increasingly cannibalised by AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24505v1">CritiCal: Can Critique Help LLM Uncertainty or Confidence Calibration?</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Accurate confidence calibration in Large Language Models (LLMs) is critical for safe use in high-stakes domains, where clear verbalized confidence enhances user trust. Traditional methods that mimic reference confidence expressions often fail to capture the reasoning needed for accurate confidence assessment. We propose natural language critiques as a solution, ideally suited for confidence calibration, as precise gold confidence labels are hard to obtain and often require multiple generations. This paper studies how natural language critiques can enhance verbalized confidence, addressing: (1) What to critique: uncertainty (question-focused) or confidence (answer-specific)? Analysis shows confidence suits multiple-choice tasks, while uncertainty excels in open-ended scenarios. (2) How to critique: self-critique or critique calibration training? We propose Self-Critique, enabling LLMs to critique and optimize their confidence beyond mere accuracy, and CritiCal, a novel Critique Calibration training method that leverages natural language critiques to improve confidence calibration, moving beyond direct numerical optimization. Experiments show that CritiCal significantly outperforms Self-Critique and other competitive baselines, even surpassing its teacher model, GPT-4o, in complex reasoning tasks. CritiCal also shows robust generalization in out-of-distribution settings, advancing LLM's reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10978v3">Group-in-Group Policy Optimization for LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in group-based reinforcement learning (RL) have driven frontier large language models (LLMs) in single-turn tasks like mathematical reasoning. However, their scalability to multi-turn LLM agent training remains limited. Unlike static tasks, agent-environment interactions unfold over many steps and often yield sparse or delayed rewards, making credit assignment across individual steps significantly more challenging. In this work, we propose Group-in-Group Policy Optimization (GiGPO), a novel RL algorithm that achieves fine-grained credit assignment for LLM agents while preserving the appealing properties of group-based RL: critic-free, low memory, and stable convergence. GiGPO introduces a two-level structure for estimating relative advantage: (i) At the episode-level, GiGPO computes macro relative advantages based on groups of complete trajectories; (ii) At the step-level, GiGPO introduces an anchor state grouping mechanism that retroactively constructs step-level groups by identifying repeated environment states across trajectories. Actions stemming from the same state are grouped together, enabling micro relative advantage estimation. This hierarchical structure effectively captures both global trajectory quality and local step effectiveness without relying on auxiliary models or additional rollouts. We evaluate GiGPO on challenging agent benchmarks, including ALFWorld and WebShop, as well as tool-integrated reasoning on search-augmented QA tasks, using Qwen2.5-1.5B/3B/7B-Instruct. Crucially, GiGPO delivers fine-grained per-step credit signals, achieves performance gains of > 12% on ALFWorld and > 9% on WebShop over GRPO, and obtains superior performance on QA tasks (42.1% on 3B and 47.2% on 7B): all while maintaining the same GPU memory overhead, identical LLM rollout, and incurring little to no additional time cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24488v1">A word association network methodology for evaluating implicit biases in LLMs compared to humans</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 24 pages, 13 figures, 3 tables
    </div>
    <details class="paper-abstract">
      As Large language models (LLMs) become increasingly integrated into our lives, their inherent social biases remain a pressing concern. Detecting and evaluating these biases can be challenging because they are often implicit rather than explicit in nature, so developing evaluation methods that assess the implicit knowledge representations of LLMs is essential. We present a novel word association network methodology for evaluating implicit biases in LLMs based on simulating semantic priming within LLM-generated word association networks. Our prompt-based approach taps into the implicit relational structures encoded in LLMs, providing both quantitative and qualitative assessments of bias. Unlike most prompt-based evaluation methods, our method enables direct comparisons between various LLMs and humans, providing a valuable point of reference and offering new insights into the alignment of LLMs with human cognition. To demonstrate the utility of our methodology, we apply it to both humans and several widely used LLMs to investigate social biases related to gender, religion, ethnicity, sexual orientation, and political party. Our results reveal both convergences and divergences between LLM and human biases, providing new perspectives on the potential risks of using LLMs. Our methodology contributes to a systematic, scalable, and generalizable framework for evaluating and comparing biases across multiple LLMs and humans, advancing the goal of transparent and socially responsible language technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24476v1">Mitigating Hallucination in Large Language Models (LLMs): An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 25 pages, 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Hallucination remains one of the key obstacles to the reliable deployment of large language models (LLMs), particularly in real-world applications. Among various mitigation strategies, Retrieval-Augmented Generation (RAG) and reasoning enhancement have emerged as two of the most effective and widely adopted approaches, marking a shift from merely suppressing hallucinations to balancing creativity and reliability. However, their synergistic potential and underlying mechanisms for hallucination mitigation have not yet been systematically examined. This survey adopts an application-oriented perspective of capability enhancement to analyze how RAG, reasoning enhancement, and their integration in Agentic Systems mitigate hallucinations. We propose a taxonomy distinguishing knowledge-based and logic-based hallucinations, systematically examine how RAG and reasoning address each, and present a unified framework supported by real-world applications, evaluations, and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24469v1">Iterative Critique-Refine Framework for Enhancing LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Personalized text generation requires models not only to produce coherent text but also to align with a target user's style, tone, and topical focus. Existing retrieval-augmented approaches such as LaMP and PGraphRAG enrich profiles with user and neighbor histories, but they stop at generation and often yield outputs that drift in tone, topic, or style. We present PerFine, a unified, training-free critique-refine framework that enhances personalization through iterative, profile-grounded feedback. In each iteration, an LLM generator produces a draft conditioned on the retrieved profile, and a critic LLM - also conditioned on the same profile - provides structured feedback on tone, vocabulary, sentence structure, and topicality. The generator then revises, while a novel knockout strategy retains the stronger draft across iterations. We further study additional inference-time strategies such as Best-of-N and Topic Extraction to balance quality and efficiency. Across Yelp, Goodreads, and Amazon datasets, PerFine consistently improves personalization over PGraphRAG, with GEval gains of +7-13%, steady improvements over 3-5 refinement iterations, and scalability with increasing critic size. These results highlight that post-hoc, profile-aware feedback offers a powerful paradigm for personalized LLM generation that is both training-free and model-agnostic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24450v1">Charting the European LLM Benchmarking Landscape: A New Taxonomy and a Set of Best Practices</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 12 pages, 1 figure. Submitted to the LREC 2026 conference
    </div>
    <details class="paper-abstract">
      While new benchmarks for large language models (LLMs) are being developed continuously to catch up with the growing capabilities of new models and AI in general, using and evaluating LLMs in non-English languages remains a little-charted landscape. We give a concise overview of recent developments in LLM benchmarking, and then propose a new taxonomy for the categorization of benchmarks that is tailored to multilingual or non-English use scenarios. We further propose a set of best practices and quality standards that could lead to a more coordinated development of benchmarks for European languages. Among other recommendations, we advocate for a higher language and culture sensitivity of evaluation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24442v1">Law in Silico: Simulating Legal Society with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Since real-world legal experiments are often costly or infeasible, simulating legal societies with Artificial Intelligence (AI) systems provides an effective alternative for verifying and developing legal theory, as well as supporting legal administration. Large Language Models (LLMs), with their world knowledge and role-playing capabilities, are strong candidates to serve as the foundation for legal society simulation. However, the application of LLMs to simulate legal systems remains underexplored. In this work, we introduce Law in Silico, an LLM-based agent framework for simulating legal scenarios with individual decision-making and institutional mechanisms of legislation, adjudication, and enforcement. Our experiments, which compare simulated crime rates with real-world data, demonstrate that LLM-based agents can largely reproduce macro-level crime trends and provide insights that align with real-world observations. At the same time, micro-level simulations reveal that a well-functioning, transparent, and adaptive legal system offers better protection of the rights of vulnerable individuals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24430v1">From Time and Place to Preference: LLM-Driven Geo-Temporal Context in Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Most recommender systems treat timestamps as numeric or cyclical values, overlooking real-world context such as holidays, events, and seasonal patterns. We propose a scalable framework that uses large language models (LLMs) to generate geo-temporal embeddings from only a timestamp and coarse location, capturing holidays, seasonal trends, and local/global events. We then introduce a geo-temporal embedding informativeness test as a lightweight diagnostic, demonstrating on MovieLens, LastFM, and a production dataset that these embeddings provide predictive signal consistent with the outcomes of full model integrations. Geo-temporal embeddings are incorporated into sequential models through (1) direct feature fusion with metadata embeddings or (2) an auxiliary loss that enforces semantic and geo-temporal alignment. Our findings highlight the need for adaptive or hybrid recommendation strategies, and we release a context-enriched MovieLens dataset to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24408v1">Uncovering Gaps Between RFC Updates and TCP/IP Implementations: LLM-Facilitated Differential Checks on Intermediate Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      As the core of the Internet infrastructure, the TCP/IP protocol stack undertakes the task of network data transmission. However, due to the complexity of the protocol and the uncertainty of cross-layer interaction, there are often inconsistencies between the implementation of the protocol stack code and the RFC standard. This inconsistency may not only lead to differences in protocol functions but also cause serious security vulnerabilities. At present, with the continuous expansion of protocol stack functions and the rapid iteration of RFC documents, it is increasingly important to detect and fix these inconsistencies. With the rise of large language models, researchers have begun to explore how to extract protocol specifications from RFC documents through these models, including protocol stack modeling, state machine extraction, text ambiguity analysis, and other related content. However, existing methods rely on predefined patterns or rule-based approaches that fail to generalize across different protocol specifications. Automated and scalable detection of these inconsistencies remains a significant challenge. In this study, we propose an automated analysis framework based on LLM and differential models. By modeling the iterative relationship of the protocol and based on the iterative update relationship of the RFC standard, we perform incremental code function analysis on different versions of kernel code implementations to automatically perform code detection and vulnerability analysis. We conduct extensive evaluations to validate the effectiveness of our framework, demonstrating its effectiveness in identifying potential vulnerabilities caused by RFC code inconsistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24397v1">APTBench: Benchmarking Agentic Potential of Base LLMs During Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 46 pages
    </div>
    <details class="paper-abstract">
      With the rapid development of LLM-based agents, there is a growing trend to incorporate agent-specific data into the pre-training stage of LLMs, aiming to better align LLMs with real-world autonomous task execution. However, current pre-training benchmarks primarily focus on isolated and static skills, e.g., common knowledge or mathematical/code reasoning, and fail to reflect model's agentic capabilities. On the other hand, agent benchmarks are typically designed for post-trained models, requiring multi-turn task execution abilities that base models struggle to support. Thus, there is a compelling need for a benchmark that can evaluate agentic potentials during pre-training and guide the model training more effectively. To address this gap, we propose APTBench, a framework that converts real-world agent tasks and successful trajectories into multiple-choice or text completion questions tailored for base models. It focuses on core agentic abilities, e.g., planning and action, and covers key agent scenarios, software engineering and deep research. Compared to existing general-purpose benchmarks, APTBench offers a more predictive signal of a model's downstream performance as an agent, while remaining significantly more lightweight and cost-effective than full-scale, end-to-end agent evaluations after post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24390v1">Improving LLM Reasoning via Dependency-Aware Query Decomposition and Logic-Parallel Content Expansion</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into real-time Web applications, such as AI-powered search and conversational agents, presents a fundamental Web infrastructure challenge: reconciling the demand for high-quality, complex reasoning with the stringent low-latency and high-throughput requirements of interactive services. Current LLM reasoning, hindered by computationally inefficient sequential generation and rigid reasoning strategies, creates a critical bottleneck for the Web services. Existing approaches typically optimize the LLM reasoning for either efficiency or quality but struggle to achieve both, and thus fail to meet the dual requirements of modern Web platforms. To overcome these limitations, we propose Orion, a novel and efficient reasoning framework that enables dependency-aware query decomposition and logic-parallel content expansion. Concretely, Orion decomposes a single query reasoning process into two synergistic phases: (1) \textit{key point generation}, which distills logically structured key points through retrieval-augmented few-shot prompting, and (2) \textit{content parallel expansion}, which concurrently elaborates on these points based on a dependency graph to ensure logical consistency. Furthermore, Orion introduces a pipeline scheduling mechanism that exploits the complementary computational characteristics of the two phases (generation imposes pressure on GPU computing and expansion stresses on GPU memory) across multiple queries, enabling cross-query parallelism and dramatically improving reasoning performance (\ie, efficiency and quality). Experiments on diverse benchmarks show that Orion not only delivers up to 4.33x higher token generation speed and 3.42x lower answer latency over the baselines but also improves reasoning quality by up to 18.75% through explicitly modeling inter-point dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24367v1">LLM-as-a-Judge for Software Engineering: Literature Review, Vision, and the Road Ahead</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      The rapid integration of Large Language Models (LLMs) into software engineering (SE) has revolutionized tasks like code generation, producing a massive volume of software artifacts. This surge has exposed a critical bottleneck: the lack of scalable, reliable methods to evaluate these outputs. Human evaluation is costly and time-consuming, while traditional automated metrics like BLEU fail to capture nuanced quality aspects. In response, the LLM-as-a-Judge paradigm - using LLMs for automated evaluation - has emerged. This approach leverages the advanced reasoning of LLMs, offering a path toward human-like nuance at automated scale. However, LLM-as-a-Judge research in SE is still in its early stages. This forward-looking SE 2030 paper aims to steer the community toward advancing LLM-as-a-Judge for evaluating LLM-generated software artifacts. We provide a literature review of existing SE studies, analyze their limitations, identify key research gaps, and outline a detailed roadmap. We envision these frameworks as reliable, robust, and scalable human surrogates capable of consistent, multi-faceted artifact evaluation by 2030. Our work aims to foster research and adoption of LLM-as-a-Judge frameworks, ultimately improving the scalability of software artifact evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09536v3">Galapagos: Automated N-Version Programming with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      N-Version Programming is a well-known methodology for developing fault-tolerant systems. It achieves fault detection and correction at runtime by adding diverse redundancy into programs, minimizing fault mode overlap between redundant program variants. In this work, we propose the automated generation of program variants using large language models. We design, develop and evaluate Gal\'apagos: a tool for generating program variants using LLMs, validating their correctness and equivalence, and using them to assemble N-Version binaries. We evaluate Gal\'apagos by creating N-Version components of real-world C code. Our original results show that Gal\'apagos can produce program variants that are proven to be functionally equivalent, even when the variants are written in a different programming language. Our systematic diversity measurement indicates that functionally equivalent variants produced by Gal\'apagos, are statically different after compilation, and present diverging internal behavior at runtime. We demonstrate that the variants produced by Gal\'apagos can protect C code against real miscompilation bugs which affect the Clang compiler. Overall, our paper shows that producing N-Version software can be drastically automated by advanced usage of practical formal verification and generative language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24358v1">Automatically Benchmarking LLM Code Agents through Agent-Driven Annotation and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Recent advances in code agents have enabled automated software development at the project level, supported by large language models (LLMs) and widely adopted tools. However, existing benchmarks for code agent evaluation face two major limitations: high annotation cost and expertise requirements, and rigid evaluation metrics that rely primarily on unit tests. To address these challenges, we propose an agent-driven benchmark construction pipeline that leverages human supervision to efficiently generate diverse and challenging project-level tasks. Based on this approach, we introduce PRDBench, a novel benchmark comprising 50 real-world Python projects across 20 domains, each with structured Product Requirement Document (PRD) requirements, comprehensive evaluation criteria, and reference implementations. PRDBench features rich data sources, high task complexity, and flexible metrics. We further employ an Agent-as-a-Judge paradigm to score agent outputs, enabling the evaluation of various test types beyond unit tests. Extensive experiments on PRDBench demonstrate its effectiveness in assessing the capabilities of both code agents and evaluation agents, providing a scalable and robust framework for annotation and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24303v1">Retrieval and Argumentation Enhanced Multi-Agent LLMs for Judgmental Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Judgmental forecasting is the task of making predictions about future events based on human judgment. This task can be seen as a form of claim verification, where the claim corresponds to a future event and the task is to assess the plausibility of that event. In this paper, we propose a novel multi-agent framework for claim verification, whereby different agents may disagree on claim veracity and bring specific evidence for and against the claims, represented as quantitative bipolar argumentation frameworks (QBAFs). We then instantiate the framework for supporting claim verification, with a variety of agents realised with Large Language Models (LLMs): (1) ArgLLM agents, an existing approach for claim verification that generates and evaluates QBAFs; (2) RbAM agents, whereby LLM-empowered Relation-based Argument Mining (RbAM) from external sources is used to generate QBAFs; (3) RAG-ArgLLM agents, extending ArgLLM agents with a form of Retrieval-Augmented Generation (RAG) of arguments from external sources. Finally, we conduct experiments with two standard judgmental forecasting datasets, with instances of our framework with two or three agents, empowered by six different base LLMs. We observe that combining evidence from agents can improve forecasting accuracy, especially in the case of three agents, while providing an explainable combination of evidence for claim verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24284v1">MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on external tools to perform complex, realistic tasks, yet their ability to utilize the rapidly expanding Model Contextual Protocol (MCP) ecosystem remains limited. Existing MCP research covers few servers, depends on costly manual curation, and lacks training support, hindering progress toward real-world deployment. To overcome these limitations, we introduce MCP-Flow, an automated web-agent-driven pipeline for large-scale server discovery, data synthesis, and model training. MCP-Flow collects and filters data from 1166 servers and 11536 tools, producing 68733 high-quality instruction-function call pairs and 6439 trajectories, far exceeding prior work in scale and diversity. Extensive experiments demonstrate MCP-Flow's effectiveness in driving superior MCP tool selection, function-call generation, and enhanced agentic task performance. MCP-Flow thus provides a scalable foundation for advancing LLM agents' proficiency in real-world MCP environments. MCP-Flow is publicly available at \href{https://github.com/wwh0411/MCP-Flow}{https://github.com/wwh0411/MCP-Flow}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24259v1">Can LLMs Translate Human Instructions into a Reinforcement Learning Agent's Internal Emergent Symbolic Representation?</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Emergent symbolic representations are critical for enabling developmental learning agents to plan and generalize across tasks. In this work, we investigate whether large language models (LLMs) can translate human natural language instructions into the internal symbolic representations that emerge during hierarchical reinforcement learning. We apply a structured evaluation framework to measure the translation performance of commonly seen LLMs -- GPT, Claude, Deepseek and Grok -- across different internal symbolic partitions generated by a hierarchical reinforcement learning algorithm in the Ant Maze and Ant Fall environments. Our findings reveal that although LLMs demonstrate some ability to translate natural language into a symbolic representation of the environment dynamics, their performance is highly sensitive to partition granularity and task complexity. The results expose limitations in current LLMs capacity for representation alignment, highlighting the need for further research on robust alignment between language and internal agent representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24251v1">GRAPHIA: Harnessing Social Graph Data to Enhance LLM-Based Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in simulating human-like social behaviors. Social graphs provide high-quality supervision signals that encode both local interactions and global network structure, yet they remain underutilized for LLM training. To address this gap, we propose Graphia, the first general LLM-based social graph simulation framework that leverages graph data as supervision for LLM post-training via reinforcement learning. With GNN-based structural rewards, Graphia trains specialized agents to predict whom to interact with (destination selection) and how to interact (edge generation), followed by designed graph generation pipelines. We evaluate Graphia under two settings: Transductive Dynamic Graph Generation (TDGG), a micro-level task with our proposed node-wise interaction alignment metrics; and Inductive Dynamic Graph Generation (IDGG), a macro-level task with our proposed metrics for aligning emergent network properties. On three real-world networks, Graphia improves micro-level alignment by 6.1% in the composite destination selection score, 12% in edge classification accuracy, and 27.9% in edge content BERTScore over the strongest baseline. For macro-level alignment, it achieves 41.11% higher structural similarity and 32.98% better replication of social phenomena such as power laws and echo chambers. Graphia also supports counterfactual simulation, generating plausible behavioral shifts under platform incentives. Our results show that social graphs can serve as high-quality supervision signals for LLM post-training, closing the gap between agent behaviors and network dynamics for LLM-based simulation. Code is available at https://github.com/Ji-Cather/Graphia.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24250v1">Evaluating LLMs on Generating Age-Appropriate Child-Like Conversations</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 11 pages excluding references and appendix. 3 figures and 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), predominantly trained on adult conversational data, face significant challenges when generating authentic, child-like dialogue for specialized applications. We present a comparative study evaluating five different LLMs (GPT-4, RUTER-LLAMA-2-13b, GPTSW, NorMistral-7b, and NorBloom-7b) to generate age-appropriate Norwegian conversations for children aged 5 and 9 years. Through a blind evaluation by eleven education professionals using both real child interview data and LLM-generated text samples, we assessed authenticity and developmental appropriateness. Our results show that evaluators achieved strong inter-rater reliability (ICC=0.75) and demonstrated higher accuracy in age prediction for younger children (5-year-olds) compared to older children (9-year-olds). While GPT-4 and NorBloom-7b performed relatively well, most models generated language perceived as more linguistically advanced than the target age groups. These findings highlight critical data-related challenges in developing LLM systems for specialized applications involving children, particularly in low-resource languages where comprehensive age-appropriate lexical resources are scarce.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16947v2">MixAT: Combining Continuous and Discrete Adversarial Training for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 Published at 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Despite recent efforts in Large Language Model (LLM) safety and alignment, current adversarial attacks on frontier LLMs can still consistently force harmful generations. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. At the same time, despite their effectiveness and generalization capabilities, training with continuous perturbations does not always capture the full spectrum of vulnerabilities exploited by discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24188v1">Investigating Software Aging in LLM-Generated Software Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 Presented at the 17th International Workshop on Software Aging and Rejuvenation (WoSAR), 2025
    </div>
    <details class="paper-abstract">
      Automatically generated software, especially code produced by Large Language Models (LLMs), is increasingly adopted to accelerate development and reduce manual effort. However, little is known about the long-term reliability of such systems under sustained execution. In this paper, we experimentally investigate the phenomenon of software aging in applications generated by LLM-based tools. Using the Bolt platform and standardized prompts from Baxbench, we generated four service-oriented applications and subjected them to 50-hour load tests. Resource usage, response time, and throughput were continuously monitored to detect degradation patterns. The results reveal significant evidence of software aging, including progressive memory growth, increased response time, and performance instability across all applications. Statistical analyzes confirm these trends and highlight variability in the severity of aging according to the type of application. Our findings show the need to consider aging in automatically generated software and provide a foundation for future studies on mitigation strategies and long-term reliability evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22162v2">Surface Reading LLMs: Synthetic Text and its Styles</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 12 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Despite a potential plateau in ML advancement, the societal impact of large language models lies not in approaching superintelligence but in generating text surfaces indistinguishable from human writing. While Critical AI Studies provides essential material and socio-technical critique, it risks overlooking how LLMs phenomenologically reshape meaning-making. This paper proposes a semiotics of "surface integrity" as attending to the immediate plane where LLMs inscribe themselves into human communication. I distinguish three knowledge interests in ML research (epistemology, epist\=em\=e, and epistemics) and argue for integrating surface-level stylistic analysis alongside depth-oriented critique. Through two case studies examining stylistic markers of synthetic text, I argue how attending to style as a semiotic phenomenon reveals LLMs as cultural actors that transform the conditions of meaning emergence and circulation in contemporary discourse, independent of questions about machine consciousness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24109v1">PFEA: An LLM-based High-Level Natural Language Planning and Feedback Embodied Agent for Human-Centered AI</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has marked a significant breakthrough in Artificial Intelligence (AI), ushering in a new era of Human-centered Artificial Intelligence (HAI). HAI aims to better serve human welfare and needs, thereby placing higher demands on the intelligence level of robots, particularly in aspects such as natural language interaction, complex task planning, and execution. Intelligent agents powered by LLMs have opened up new pathways for realizing HAI. However, existing LLM-based embodied agents often lack the ability to plan and execute complex natural language control tasks online. This paper explores the implementation of intelligent robotic manipulating agents based on Vision-Language Models (VLMs) in the physical world. We propose a novel embodied agent framework for robots, which comprises a human-robot voice interaction module, a vision-language agent module and an action execution module. The vision-language agent itself includes a vision-based task planner, a natural language instruction converter, and a task performance feedback evaluator. Experimental results demonstrate that our agent achieves a 28\% higher average task success rate in both simulated and real environments compared to approaches relying solely on LLM+CLIP, significantly improving the execution success rate of high-level natural language instruction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23143v3">MathBode: Understanding LLM Reasoning with Dynamical Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      This paper presents MathBode, a dynamic diagnostic for mathematical reasoning in large language models (LLMs). Instead of one-shot accuracy, MathBode treats each parametric problem as a system: we drive a single parameter sinusoidally and fit first-harmonic responses of model outputs and exact solutions. This yields interpretable, frequency-resolved metrics -- gain (amplitude tracking) and phase (lag) -- that form Bode-style fingerprints. Across five closed-form families (linear solve, ratio/saturation, compound interest, 2x2 linear systems, similar triangles), the diagnostic surfaces systematic low-pass behavior and growing phase lag that accuracy alone obscures. We compare several models against a symbolic baseline that calibrates the instrument ($G \approx 1$, $\phi \approx 0$). Results separate frontier from mid-tier models on dynamics, providing a compact, reproducible protocol that complements standard benchmarks with actionable measurements of reasoning fidelity and consistency. We open-source the dataset and code to enable further research and adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24073v1">Challenging Multilingual LLMs: A New Taxonomy and Benchmark for Unraveling Hallucination in Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have advanced machine translation but remain vulnerable to hallucinations. Unfortunately, existing MT benchmarks are not capable of exposing failures in multilingual LLMs. To disclose hallucination in multilingual LLMs, we introduce a diagnostic framework with a taxonomy that separates Instruction Detachment from Source Detachment. Guided by this taxonomy, we create HalloMTBench, a multilingual, human-verified benchmark across 11 English-to-X directions. We employed 4 frontier LLMs to generate candidates and scrutinize these candidates with an ensemble of LLM judges, and expert validation. In this way, we curate 5,435 high-quality instances. We have evaluated 17 LLMs on HalloMTBench. Results reveal distinct ``hallucination triggers'' -- unique failure patterns reflecting model scale, source length sensitivity, linguistic biases, and Reinforcement-Learning (RL) amplified language mixing. HalloMTBench offers a forward-looking testbed for diagnosing LLM translation failures. HalloMTBench is available in https://huggingface.co/collections/AIDC-AI/marco-mt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24085v2">PEARL: Peer-Enhanced Adaptive Radio via On-Device LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      We present PEARL (Peer-Enhanced Adaptive Radio via On-Device LLM), a framework for cooperative cross-layer optimization in device-to-device (D2D) communication. Building on our previous work on single-device on-device LLMs, PEARL extends the paradigm by leveraging both publisher and subscriber states to guide Wi-Fi Aware (WA) parameter selection. A context-aware reward, which normalizes latency by application tolerances and modulates energy by device battery states, provides richer supervision for KL-based finetuning. We study two lightweight variants: PEARL (Head + Low-Rank Adaptation (LoRA)) achieves the best overall performance, while PEARL-Lite (Head-only) delivers sub-20 ms inference at near-identical objective scores. Across synthetic scenarios grounded in real measurements, PEARL improves objective scores over heuristic and compact model baselines and reduces energy by up to 16% in cooperative low-battery cases. These results demonstrate that peer-aware context, reward-aligned training, and head-based efficiency make LLMs practical for always-on, on-device cross-layer control. Code, real-world demo, and dataset are available at https://github.com/abman23/pearl
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02885v2">MDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 26 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Video large language models (Video-LLMs) have made significant progress in understanding videos. However, processing multiple frames leads to lengthy visual token sequences, presenting challenges such as the limited context length cannot accommodate the entire video, and the inclusion of irrelevant frames hinders visual perception. Hence, effective frame selection is crucial. This paper emphasizes that frame selection should follow three key principles: query relevance, list-wise diversity, and sequentiality. Existing methods, such as uniform frame sampling and query-frame matching, do not capture all of these principles. Thus, we propose Markov decision determinantal point process with dynamic programming (MDP3) for frame selection, a training-free and model-agnostic method that can be seamlessly integrated into existing Video-LLMs. Our method first estimates frame similarities conditioned on the query using a conditional Gaussian kernel within the reproducing kernel Hilbert space~(RKHS). We then apply the determinantal point process~(DPP) to the similarity matrix to capture both query relevance and list-wise diversity. To incorporate sequentiality, we segment the video and apply DPP within each segment, conditioned on the preceding segment selection, modeled as a Markov decision process~(MDP) for allocating selection sizes across segments. Theoretically, MDP3 provides a \((1 - 1/e)\)-approximate solution to the NP-hard list-wise frame selection problem with pseudo-polynomial time complexity, demonstrating its efficiency. Empirically, MDP3 significantly outperforms existing methods, verifying its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20445v5">TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 Accepted by NeurIPS 2025, https://github.com/tsinghua-fib-lab/TrajAgent
    </div>
    <details class="paper-abstract">
      Trajectory modeling, which includes research on trajectory data pattern mining and future prediction, has widespread applications in areas such as life services, urban transportation, and public administration. Numerous methods have been proposed to address specific problems within trajectory modeling. However, the heterogeneity of data and the diversity of trajectory tasks make effective and reliable trajectory modeling an important yet highly challenging endeavor, even for domain experts. In this paper, we propose TrajAgent, an agent framework powered by large language models, designed to facilitate robust and efficient trajectory modeling through automation modeling. This framework leverages and optimizes diverse specialized models to address various trajectory modeling tasks across different datasets effectively. In TrajAgent, we first develop UniEnv, an execution environment with a unified data and model interface, to support the execution and training of various models. Building on UniEnv, we introduce an agentic workflow designed for automatic trajectory modeling across various trajectory tasks and data. Furthermore, we introduce collaborative learning schema between LLM-based agents and small speciallized models, to enhance the performance of the whole framework effectively. Extensive experiments on five tasks using four real-world datasets demonstrate the effectiveness of TrajAgent in automated trajectory modeling, achieving a performance improvement of 2.38%-69.91% over baseline methods. The codes and data can be accessed via https://github.com/tsinghua-fib-lab/TrajAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24051v1">Pie: A Programmable Serving System for Emerging LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 SOSP 2025. Source code available at https://github.com/pie-project/pie
    </div>
    <details class="paper-abstract">
      Emerging large language model (LLM) applications involve diverse reasoning strategies and agentic workflows, straining the capabilities of existing serving systems built on a monolithic token generation loop. This paper introduces Pie, a programmable LLM serving system designed for flexibility and efficiency. Pie decomposes the traditional generation loop into fine-grained service handlers exposed via an API and delegates control of the generation process to user-provided programs, called inferlets. This enables applications to implement new KV cache strategies, bespoke generation logic, and seamlessly integrate computation and I/O-entirely within the application, without requiring modifications to the serving system. Pie executes inferlets using WebAssembly, benefiting from its lightweight sandboxing. Our evaluation shows Pie matches state-of-the-art performance on standard tasks (3-12% latency overhead) while significantly improving latency and throughput (1.3x-3.4x higher) on agentic workflows by enabling application-specific optimizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17281v2">MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. Experiments show that the effectiveness and efficiency of state-of-the-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05316v3">Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 Accepted at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism inspired by experience replay in traditional RL. This technique reuses recent rollouts, lowering per-step computation while maintaining stable updates. Experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 23% to 62% while reaching the same level of performance as the original GRPO algorithm. Our code is available at https://github.com/ASTRAL-Group/data-efficient-llm-rl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24034v1">AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Despite rapid advancements in text-to-image (T2I) models, their safety mechanisms are vulnerable to adversarial prompts, which maliciously generate unsafe images. Current red-teaming methods for proactively assessing such vulnerabilities usually require white-box access to T2I models, and rely on inefficient per-prompt optimization, as well as inevitably generate semantically meaningless prompts easily blocked by filters. In this paper, we propose APT (AutoPrompT), a black-box framework that leverages large language models (LLMs) to automatically generate human-readable adversarial suffixes for benign prompts. We first introduce an alternating optimization-finetuning pipeline between adversarial suffix optimization and fine-tuning the LLM utilizing the optimized suffix. Furthermore, we integrates a dual-evasion strategy in optimization phase, enabling the bypass of both perplexity-based filter and blacklist word filter: (1) we constrain the LLM generating human-readable prompts through an auxiliary LLM perplexity scoring, which starkly contrasts with prior token-level gibberish, and (2) we also introduce banned-token penalties to suppress the explicit generation of banned-tokens in blacklist. Extensive experiments demonstrate the excellent red-teaming performance of our human-readable, filter-resistant adversarial prompts, as well as superior zero-shot transferability which enables instant adaptation to unseen prompts and exposes critical vulnerabilities even in commercial APIs (e.g., Leonardo.Ai.).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23595v2">Multi-Agent Evolve: LLM Self-Improve through Co-evolution</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 29 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has demonstrated significant potential in enhancing the reasoning capabilities of large language models (LLMs). However, the success of RL for LLMs heavily relies on human-curated datasets and verifiable rewards, which limit their scalability and generality. Recent Self-Play RL methods, inspired by the success of the paradigm in games and Go, aim to enhance LLM reasoning capabilities without human-annotated data. However, their methods primarily depend on a grounded environment for feedback (e.g., a Python interpreter or a game engine); extending them to general domains remains challenging. To address these challenges, we propose Multi-Agent Evolve (MAE), a framework that enables LLMs to self-evolve in solving diverse tasks, including mathematics, reasoning, and general knowledge Q&A. The core design of MAE is based on a triplet of interacting agents (Proposer, Solver, Judge) that are instantiated from a single LLM, and applies reinforcement learning to optimize their behaviors. The Proposer generates questions, the Solver attempts solutions, and the Judge evaluates both while co-evolving. Experiments on Qwen2.5-3B-Instruct demonstrate that MAE achieves an average improvement of 4.54% on multiple benchmarks. These results highlight MAE as a scalable, data-efficient method for enhancing the general reasoning abilities of LLMs with minimal reliance on human-curated supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24021v1">SpecKD: Speculative Decoding for Effective Knowledge Distillation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Knowledge Distillation (KD) has become a cornerstone technique for compressing Large Language Models (LLMs) into smaller, more efficient student models. However, conventional KD approaches typically apply the distillation loss uniformly across all tokens, regardless of the teacher's confidence. This indiscriminate mimicry can introduce noise, as the student is forced to learn from the teacher's uncertain or high-entropy predictions, which may ultimately harm student performance-especially when the teacher is much larger and more powerful. To address this, we propose Speculative Knowledge Distillation (SpecKD), a novel, plug-and-play framework that introduces a dynamic, token-level gating mechanism inspired by the "propose-and-verify" paradigm of speculative decoding. At each step, the student's token proposal is verified against the teacher's distribution; the distillation loss is selectively applied only to "accepted" tokens, while "rejected" tokens are masked out. Extensive experiments on diverse text generation tasks show that SpecKD consistently and significantly outperforms strong KD baselines, leading to more stable training and more capable student models, and achieving state-of-the-art results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24020v1">Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 23pages, 4figures
    </div>
    <details class="paper-abstract">
      Mitigating hallucinations in Large Language Models (LLMs) is critical for their reliable deployment. Existing methods typically fine-tune LLMs to abstain from answering questions beyond their knowledge scope. However, these methods often rely on coarse-grained signals to guide LLMs to abstain, such as overall confidence or uncertainty scores on multiple sampled answers, which may result in an imprecise awareness of the model's own knowledge boundaries. To this end, we propose a novel reinforcement learning framework built on $\textbf{\underline{Fi}ne-grained \underline{S}emantic \underline{Co}nfidence \underline{Re}ward (\Ours)}$, which guides LLMs to abstain via sample-specific confidence. Specifically, our method operates by sampling multiple candidate answers and conducting semantic clustering, then training the LLM to retain answers within high-confidence clusters and discard those within low-confidence ones, thereby promoting accurate post-hoc abstention. Additionally, we propose a new metric for evaluating the reliability of abstention fine-tuning tasks more comprehensively. Our method significantly enhances reliability in both in-domain and out-of-distribution benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24019v1">Lifecycle-Aware code generation: Leveraging Software Engineering Phases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has advanced automatic code generation, yet most approaches rely on direct, single-step translation from problem descriptions to code, disregarding structured software engineering practices. We introduce a lifecycle-aware framework that systematically incorporates intermediate artifacts such as requirements analysis, state machine modeling, and pseudocode into both the training and inference stages. This design aligns code generation with standard software development phases and enables more structured reasoning. Experiments show that lifecycle-level fine-tuning improves code correctness by up to 75% over the same model before fine-tuning, with performance gains compounding across intermediate stages. Multi-step inference consistently surpasses single-step generation, demonstrating the effectiveness of intermediate scaffolding. Notably, open-source LLMs, once fine-tuned under our framework, match or slightly outperform models pretrained on code. When applied to DeepSeek-Coder-1.3B, our framework yields relative CodeBLEU improvements of 34.3%, 20.0%, 11.2%, and 22.3% over ChatGPT-3.5, ChatGPT-4o-mini, DeepSeek-R1, and LLaMA-8B, respectively. Our pipeline also proves robust with up to 80\% less training data, confirming its resilience. Ablation studies further reveal that each intermediate artifact contributes distinctly to final code quality, with state machine modeling yielding the most substantial impact. Our source code and detailed experimental data are available at https://anonymous.4open.science/r/Lifecycle-Aware-3CCB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19563v3">Robustness is Important: Limitations of LLMs for Data Fitting</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being applied in a wide array of settings, well beyond the typical language-oriented use cases. In particular, LLMs are increasingly used as a plug-and-play method for fitting data and generating predictions. Prior work has shown that LLMs, via in-context learning or supervised fine-tuning, can perform competitively with many tabular supervised learning techniques in terms of predictive performance. However, we identify a critical vulnerability of using LLMs for data fitting -- making changes to data representation that are completely irrelevant to the underlying learning task can drastically alter LLMs' predictions on the same data. For example, simply changing variable names can sway the size of prediction error by as much as 82% in certain settings. Such prediction sensitivity with respect to task-irrelevant variations manifests under both in-context learning and supervised fine-tuning, for both close-weight and open-weight general-purpose LLMs. Moreover, by examining the attention scores of an open-weight LLM, we discover a non-uniform attention pattern: training examples and variable names/values which happen to occupy certain positions in the prompt receive more attention when output tokens are generated, even though different positions are expected to receive roughly the same attention. This partially explains the sensitivity in the presence of task-irrelevant variations. We also consider a state-of-the-art tabular foundation model (TabPFN) trained specifically for data fitting. Despite being explicitly designed to achieve prediction robustness, TabPFN is still not immune to task-irrelevant variations. Overall, despite LLMs' impressive predictive capabilities, currently they lack even the basic level of robustness to be used as a principled data-fitting tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24013v1">Discovering Heuristics with Large Language Models (LLMs) for Mixed-Integer Programs: Single-Machine Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Our study contributes to the scheduling and combinatorial optimization literature with new heuristics discovered by leveraging the power of Large Language Models (LLMs). We focus on the single-machine total tardiness (SMTT) problem, which aims to minimize total tardiness by sequencing n jobs on a single processor without preemption, given processing times and due dates. We develop and benchmark two novel LLM-discovered heuristics, the EDD Challenger (EDDC) and MDD Challenger (MDDC), inspired by the well-known Earliest Due Date (EDD) and Modified Due Date (MDD) rules. In contrast to prior studies that employed simpler rule-based heuristics, we evaluate our LLM-discovered algorithms using rigorous criteria, including optimality gaps and solution time derived from a mixed-integer programming (MIP) formulation of SMTT. We compare their performance against state-of-the-art heuristics and exact methods across various job sizes (20, 100, 200, and 500 jobs). For instances with more than 100 jobs, exact methods such as MIP and dynamic programming become computationally intractable. Up to 500 jobs, EDDC improves upon the classic EDD rule and another widely used algorithm in the literature. MDDC consistently outperforms traditional heuristics and remains competitive with exact approaches, particularly on larger and more complex instances. This study shows that human-LLM collaboration can produce scalable, high-performing heuristics for NP-hard constrained combinatorial optimization, even under limited resources when effectively configured.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23008v2">From Prompt Optimization to Multi-Dimensional Credibility Evaluation: Enhancing Trustworthiness of Chinese LLM-Generated Liver MRI Reports</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 10 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promising performance in generating diagnostic conclusions from imaging findings, thereby supporting radiology reporting, trainee education, and quality control. However, systematic guidance on how to optimize prompt design across different clinical contexts remains underexplored. Moreover, a comprehensive and standardized framework for assessing the trustworthiness of LLM-generated radiology reports is yet to be established. This study aims to enhance the trustworthiness of LLM-generated liver MRI reports by introducing a Multi-Dimensional Credibility Assessment (MDCA) framework and providing guidance on institution-specific prompt optimization. The proposed framework is applied to evaluate and compare the performance of several advanced LLMs, including Kimi-K2-Instruct-0905, Qwen3-235B-A22B-Instruct-2507, DeepSeek-V3, and ByteDance-Seed-OSS-36B-Instruct, using the SiliconFlow platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23990v1">Resource-Efficient LLM Application for Structured Transformation of Unstructured Financial Contracts</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 5 pages, 1 figure, 2 tables
    </div>
    <details class="paper-abstract">
      The transformation of unstructured legal contracts into standardized, machine-readable formats is essential for automating financial workflows. The Common Domain Model (CDM) provides a standardized framework for this purpose, but converting complex legal documents like Credit Support Annexes (CSAs) into CDM representations remains a significant challenge. In this paper, we present an extension of the CDMizer framework, a template-driven solution that ensures syntactic correctness and adherence to the CDM schema during contract-to-CDM conversion. We apply this extended framework to a real-world task, comparing its performance with a benchmark developed by the International Swaps and Derivatives Association (ISDA) for CSA clause extraction. Our results show that CDMizer, when integrated with a significantly smaller, open-source Large Language Model (LLM), achieves competitive performance in terms of accuracy and efficiency against larger, proprietary models. This work underscores the potential of resource-efficient solutions to automate legal contract transformation, offering a cost-effective and scalable approach that can meet the needs of financial institutions with constrained resources or strict data privacy requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23965v1">The Sign Estimator: LLM Alignment in the Face of Choice Heterogeneity</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Traditional LLM alignment methods are vulnerable to heterogeneity in human preferences. Fitting a na\"ive probabilistic model to pairwise comparison data (say over prompt-completion pairs) yields an inconsistent estimate of the population-average utility -a canonical measure of social welfare. We propose a new method, dubbed the sign estimator, that provides a simple, provably consistent, and efficient estimator by replacing cross-entropy with binary classification loss in the aggregation step. This simple modification recovers consistent ordinal alignment under mild assumptions and achieves the first polynomial finite-sample error bounds in this setting. In realistic simulations of LLM alignment using digital twins, the sign estimator substantially reduces preference distortion over a panel of simulated personas, cutting (angular) estimation error by nearly 35% and decreasing disagreement with true population preferences from 12% to 8% compared to standard RLHF. Our method also compares favorably to panel data heuristics that explicitly model user heterogeneity and require tracking individual-level preference data-all while maintaining the implementation simplicity of existing LLM alignment pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23949v1">Uncovering the Potential Risks in Unlearning: Danger of English-only Unlearning in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      There have been a couple of studies showing that attempting to erase multilingual knowledge using only English data is insufficient for multilingual LLMs. However, their analyses remain highly performance-oriented. In this paper, we switch the point of view to evaluation, and address an additional blind spot which reveals itself when the multilingual LLM is fully finetuned with parallel multilingual dataset before unlearning. Here, language confusion occurs whereby a model responds in language different from that of the input prompt. Language confusion is a problematic phenomenon in unlearning, causing the standard reference-based metrics to fail. We tackle this phenomenon in three steps: (1) introduce N-gram-based Language-Mix (N-Mix) score to quantitatively show the language confusion is pervasive and consistent in multilingual LLMs, (2) demonstrate that reference-based metrics result in false negatives when N-Mix score is high, and(3) suggest the need of new type of unlearning evaluation that can directly assess the content of the generated sentences. We call this type of metrics as semantic-based metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23947v1">Toward Socially-Aware LLMs: A Survey of Multimodal Approaches to Human Behavior Understanding</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      LLM-powered multimodal systems are increasingly used to interpret human social behavior, yet how researchers apply the models' 'social competence' remains poorly understood. This paper presents a systematic literature review of 176 publications across different application domains (e.g., healthcare, education, and entertainment). Using a four-dimensional coding framework (application, technical, evaluative, and ethical), we find (1) frequent use of pattern recognition and information extraction from multimodal sources, but limited support for adaptive, interactive reasoning; (2) a dominant 'modality-to-text' pipeline that privileges language over rich audiovisual cues, striping away nuanced social cues; (3) evaluation practices reliant on static benchmarks, with socially grounded, human-centered assessments rare; and (4) Ethical discussions focused mainly on legal and rights-related risks (e.g., privacy), leaving societal risks (e.g., deception) overlooked--or at best acknowledged but left unaddressed. We outline a research agenda for evaluating socially competent, ethically informed, and interaction-aware multi-modal systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00418v3">Serving LLMs in HPC Clusters: A Comparative Study of Qualcomm Cloud AI 100 Ultra and NVIDIA Data Center GPUs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 8 pages, 3 tables
    </div>
    <details class="paper-abstract">
      This study presents a benchmarking analysis of the Qualcomm Cloud AI 100 Ultra (QAic) accelerator for large language model (LLM) inference, evaluating its energy efficiency (throughput per watt), performance, and hardware scalability against NVIDIA A100 GPUs (in 4x and 8x configurations) within the National Research Platform (NRP) ecosystem. A total of 12 open-source LLMs, ranging from 124 million to 70 billion parameters, are served using the vLLM framework. Our analysis reveals that QAic achieves competitive energy efficiency with advantages on specific models while enabling more granular hardware allocation: some 70B models operate on as few as 1 QAic card versus 8 A100 GPUs required, with 20x lower power consumption (148W vs 2,983W). For smaller models, single QAic devices achieve up to 35x lower power consumption compared to our 4-GPU A100 configuration (36W vs 1,246W). The findings offer insights into the potential of the Qualcomm Cloud AI 100 Ultra for energy-constrained and resource-efficient HPC deployments within the National Research Platform (NRP).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25017v1">StorageXTuner: An LLM Agent-Driven Automatic Tuning Framework for Heterogeneous Storage Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 ArXiv version; Affiliations: Arizona State University (Lin, Zhang, Thakkar, Sun, Cao) and Iowa State University (Zheng)
    </div>
    <details class="paper-abstract">
      Automatically configuring storage systems is hard: parameter spaces are large and conditions vary across workloads, deployments, and versions. Heuristic and ML tuners are often system specific, require manual glue, and degrade under changes. Recent LLM-based approaches help but usually treat tuning as a single-shot, system-specific task, which limits cross-system reuse, constrains exploration, and weakens validation. We present StorageXTuner, an LLM agent-driven auto-tuning framework for heterogeneous storage engines. StorageXTuner separates concerns across four agents - Executor (sandboxed benchmarking), Extractor (performance digest), Searcher (insight-guided configuration exploration), and Reflector (insight generation and management). The design couples an insight-driven tree search with layered memory that promotes empirically validated insights and employs lightweight checkers to guard against unsafe actions. We implement a prototype and evaluate it on RocksDB, LevelDB, CacheLib, and MySQL InnoDB with YCSB, MixGraph, and TPC-H/C. Relative to out-of-the-box settings and to ELMo-Tune, StorageXTuner reaches up to 575% and 111% higher throughput, reduces p99 latency by as much as 88% and 56%, and converges with fewer trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05583v3">OpenFactCheck: Building, Benchmarking Customized Fact-Checking Systems and Evaluating the Factuality of Claims and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 23 pages, 8 tables, 11 figures, Published In Proceedings of the 31st International Conference on Computational Linguistics 2025
    </div>
    <details class="paper-abstract">
      The increased use of large language models (LLMs) across a variety of real-world applications calls for mechanisms to verify the factual accuracy of their outputs. Difficulties lie in assessing the factuality of free-form responses in open domains. Also, different papers use disparate evaluation benchmarks and measurements, which renders them hard to compare and hampers future progress. To mitigate these issues, we propose OpenFactCheck, a unified framework for building customized automatic fact-checking systems, benchmarking their accuracy, evaluating factuality of LLMs, and verifying claims in a document. OpenFactCheck consists of three modules: (i) CUSTCHECKER allows users to easily customize an automatic fact-checker and verify the factual correctness of documents and claims, (ii) LLMEVAL, a unified evaluation framework assesses LLM's factuality ability from various perspectives fairly, and (iii) CHECKEREVAL is an extensible solution for gauging the reliability of automatic fact-checkers' verification results using human-annotated datasets. Data and code are publicly available at https://github.com/yuxiaw/openfactcheck.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11832v3">OpenFactCheck: A Unified Framework for Factuality Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 11 pages, 4 Figures, 3 Tables, Published In Proceedings of The 2024 Conference on Empirical Methods in Natural Language Processing
    </div>
    <details class="paper-abstract">
      The increased use of large language models (LLMs) across a variety of real-world applications calls for automatic tools to check the factual accuracy of their outputs, as LLMs often hallucinate. This is difficult as it requires assessing the factuality of free-form open-domain responses. While there has been a lot of research on this topic, different papers use different evaluation benchmarks and measures, which makes them hard to compare and hampers future progress. To mitigate these issues, we developed OpenFactCheck, a unified framework, with three modules: (i) RESPONSEEVAL, which allows users to easily customize an automatic fact-checking system and to assess the factuality of all claims in an input document using that system, (ii) LLMEVAL, which assesses the overall factuality of an LLM, and (iii) CHECKEREVAL, a module to evaluate automatic fact-checking systems. OpenFactCheck is open-sourced (https://github.com/mbzuai-nlp/openfactcheck) and publicly released as a Python library (https://pypi.org/project/openfactcheck/) and also as a web service (http://app.openfactcheck.com). A video describing the system is available at https://youtu.be/-i9VKL0HleI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23234v4">p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Obtaining high-quality outputs from Large Language Models (LLMs) often depends upon the choice of a sampling-based decoding strategy to probabilistically choose the next token at each generation step. While a variety of such sampling methods have been proposed, their performance can be sensitive to the selection of hyperparameters which may require different settings depending upon the generation task and temperature configuration. In this work, we introduce $p$-less sampling: an information-theoretic approach to sampling which dynamically sets a truncation threshold at each decoding step based on the entire token probability distribution. Unlike existing methods, $p$-less sampling has no hyperparameters and consistently produces high-quality outputs as temperature increases. We provide theoretical perspectives on $p$-less sampling to ground our proposed method and conduct experiments to empirically validate its effectiveness across a range of math, logical reasoning, and creative writing tasks. Our results demonstrate how $p$-less sampling consistently outperforms existing sampling approaches while exhibiting much less degradation in text quality at higher temperature values. We further show how $p$-less achieves greater inference-time efficiency than alternative methods through lower average token sampling times and shorter generation lengths, without sacrificing accuracy. Finally, we provide analyses to highlight the benefits of $p$-less through qualitative examples, case studies, and diversity assessments. The code is available at https://github.com/ryttry/p-less .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05079v2">LLM-Guided Scenario-based GUI Testing</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      The assurance of mobile app GUIs has become increasingly important, as the GUI serves as the primary medium of interaction between users and apps. Although numerous automated GUI testing approaches have been developed with diverse strategies, a substantial gap remains between these approaches and the underlying app business logic. Most existing approaches focus on general exploration rather than the completion of specific testing scenarios, often missing critical functionalities. Inspired by manual testing, which treats business logic-driven scenarios as the fundamental unit of testing, this paper introduces an approach that leverages large language models to comprehend GUI semantics and contextual relevance to given scenarios. Building on this capability, we propose ScenGen, an LLM-guided scenario-based GUI testing framework employing multi-agent collaboration to simulate and automate manual testing phases. Specifically, ScenGen integrates five agents: the Observer, Decider, Executor, Supervisor, and Recorder. The Observer perceives the app GUI state by extracting and structuring GUI widgets and layouts, interpreting semantic information. This is passed to the Decider, which makes scenario-driven decisions with LLM guidance to identify target widgets and determine actions toward fulfilling specific goals. The Executor performs these operations, while the Supervisor verifies alignment with intended scenario completion, ensuring traceability and consistency. Finally, the Recorder logs GUI operations into context memory as a knowledge base for subsequent decision-making and monitors runtime bugs. Comprehensive evaluations demonstrate that ScenGen effectively generates scenario-based GUI tests guided by LLM collaboration, achieving higher relevance to business logic and improving the completeness of automated GUI testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23941v1">Auto prompting without training labels: An LLM cascade for product quality assessment in e-commerce catalogs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      We introduce a novel, training free cascade for auto-prompting Large Language Models (LLMs) to assess product quality in e-commerce. Our system requires no training labels or model fine-tuning, instead automatically generating and refining prompts for evaluating attribute quality across tens of thousands of product category-attribute pairs. Starting from a seed of human-crafted prompts, the cascade progressively optimizes instructions to meet catalog-specific requirements. This approach bridges the gap between general language understanding and domain-specific knowledge at scale in complex industrial catalogs. Our extensive empirical evaluations shows the auto-prompt cascade improves precision and recall by $8-10\%$ over traditional chain-of-thought prompting. Notably, it achieves these gains while reducing domain expert effort from 5.1 hours to 3 minutes per attribute - a $99\%$ reduction. Additionally, the cascade generalizes effectively across five languages and multiple quality assessment tasks, consistently maintaining performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01374v2">REASONING COMPILER: LLM-Guided Optimizations for Efficient Model Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimizations to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed Reasoning Compiler) that formulates optimization as a sequential, context-aware decision process guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-informed transformations that reflect the current program state and accumulated performance feedback. MCTS incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23924v1">Agent-based Automated Claim Matching with Instruction-following LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted for the International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (2025) Findings
    </div>
    <details class="paper-abstract">
      We present a novel agent-based approach for the automated claim matching task with instruction-following LLMs. We propose a two-step pipeline that first generates prompts with LLMs, to then perform claim matching as a binary classification task with LLMs. We demonstrate that LLM-generated prompts can outperform SOTA with human-generated prompts, and that smaller LLMs can do as well as larger ones in the generation process, allowing to save computational resources. We also demonstrate the effectiveness of using different LLMs for each step of the pipeline, i.e. using an LLM for prompt generation, and another for claim matching. Our investigation into the prompt generation process in turn reveals insights into the LLMs' understanding of claim matching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23921v1">Breaking the Benchmark: Revealing LLM Bias via Minimal Contextual Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 9 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models have been shown to demonstrate stereotypical biases in their representations and behavior due to the discriminative nature of the data that they have been trained on. Despite significant progress in the development of methods and models that refrain from using stereotypical information in their decision-making, recent work has shown that approaches used for bias alignment are brittle. In this work, we introduce a novel and general augmentation framework that involves three plug-and-play steps and is applicable to a number of fairness evaluation benchmarks. Through application of augmentation to a fairness evaluation dataset (Bias Benchmark for Question Answering (BBQ)), we find that Large Language Models (LLMs), including state-of-the-art open and closed weight models, are susceptible to perturbations to their inputs, showcasing a higher likelihood to behave stereotypically. Furthermore, we find that such models are more likely to have biased behavior in cases where the target demographic belongs to a community less studied by the literature, underlining the need to expand the fairness and safety research to include more diverse communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23893v1">Evaluating the effectiveness of LLM-based interoperability</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Background: Systems of systems are becoming increasingly dynamic and heterogeneous, and this adds pressure on the long-standing challenge of interoperability. Besides its technical aspect, interoperability has also an economic side, as development time efforts are required to build the interoperability artifacts. Objectives: With the recent advances in the field of large language models (LLMs), we aim at analyzing the effectiveness of LLM-based strategies to make systems interoperate autonomously, at runtime, without human intervention. Method: We selected 13 open source LLMs and curated four versions of a dataset in the agricultural interoperability use case. We performed three runs of each model with each version of the dataset, using two different strategies. Then we compared the effectiveness of the models and the consistency of their results across multiple runs. Results: qwen2.5-coder:32b was the most effective model using both strategies DIRECT (average pass@1 >= 0.99) and CODEGEN (average pass@1 >= 0.89) in three out of four dataset versions. In the fourth dataset version, which included an unit conversion, all models using the strategy DIRECT failed, whereas using CODEGEN qwen2.5-coder:32b succeeded with an average pass@1 = 0.75. Conclusion: Some LLMs can make systems interoperate autonomously. Further evaluation in different domains is recommended, and further research on reliability strategies should be conducted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23891v1">PRO: Enabling Precise and Robust Text Watermark for Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Text watermarking for large language models (LLMs) enables model owners to verify text origin and protect intellectual property. While watermarking methods for closed-source LLMs are relatively mature, extending them to open-source models remains challenging, as developers cannot control the decoding process. Consequently, owners of open-source LLMs lack practical means to verify whether text was generated by their models. A core difficulty lies in embedding watermarks directly into model weights without hurting detectability. A promising idea is to distill watermarks from a closed-source model into an open one, but this suffers from (i) poor detectability due to mismatch between learned and predefined patterns, and (ii) fragility to downstream modifications such as fine-tuning or model merging. To overcome these limitations, we propose PRO, a Precise and Robust text watermarking method for open-source LLMs. PRO jointly trains a watermark policy model with the LLM, producing patterns that are easier for the model to learn and more consistent with detection criteria. A regularization term further simulates downstream perturbations and penalizes degradation in watermark detectability, ensuring robustness under model edits. Experiments on open-source LLMs (e.g., LLaMA-3.2, LLaMA-3, Phi-2) show that PRO substantially improves both watermark detectability and resilience to model modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23875v1">Large Language Model Agent Personality and Response Appropriateness: Evaluation by Human Linguistic Experts, LLM-as-Judge, and Natural Language Processing Model</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM)-based agents can be used to create highly engaging interactive applications through prompting personality traits and contextual data, effectively assessing their personalities has proven challenging. This novel interdisciplinary approach addresses this gap by combining agent development and linguistic analysis to assess the prompted personality of LLM-based agents in a poetry explanation task. We developed a novel, flexible question bank, informed by linguistic assessment criteria and human cognitive learning levels, offering a more comprehensive evaluation than current methods. By evaluating agent responses with natural language processing models, other LLMs, and human experts, our findings illustrate the limitations of purely deep learning solutions and emphasize the critical role of interdisciplinary design in agent development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23874v1">From Stochasticity to Signal: A Bayesian Latent State Model for Reliable Measurement with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate classification tasks in business, such as analyzing customer satisfaction from text. However, the inherent stochasticity of LLMs, in terms of their tendency to produce different outputs for the same input, creates a significant measurement error problem that is often neglected with a single round of output, or addressed with ad-hoc methods like majority voting. Such naive approaches fail to quantify uncertainty and can produce biased estimates of population-level metrics. In this paper, we propose a principled solution by reframing LLM variability as a statistical measurement error problem and introducing a Bayesian latent state model to address it. Our model treats the true classification (e.g., customer dissatisfaction) as an unobserved latent variable and the multiple LLM ratings as noisy measurements of this state. This framework allows for the simultaneous estimation of the LLM's false positive and false negative error rates, the underlying base rate of the phenomenon in the population, the posterior probability of the true state for each individual observation, and the causal impact of a business intervention, if any, on the latent state. Through simulation studies, we demonstrate that our model accurately recovers true parameters where naive methods fail. We conclude that this methodology provides a general and reliable framework for converting noisy, probabilistic outputs from LLMs into accurate and actionable insights for scientific and business applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23854v1">Can LLMs Narrate Tabular Data? An Evaluation Framework for Natural Language Representations of Text-to-SQL System Outputs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      In modern industry systems like multi-turn chat agents, Text-to-SQL technology bridges natural language (NL) questions and database (DB) querying. The conversion of tabular DB results into NL representations (NLRs) enables the chat-based interaction. Currently, NLR generation is typically handled by large language models (LLMs), but information loss or errors in presenting tabular results in NL remains largely unexplored. This paper introduces a novel evaluation method - Combo-Eval - for judgment of LLM-generated NLRs that combines the benefits of multiple existing methods, optimizing evaluation fidelity and achieving a significant reduction in LLM calls by 25-61%. Accompanying our method is NLR-BIRD, the first dedicated dataset for NLR benchmarking. Through human evaluations, we demonstrate the superior alignment of Combo-Eval with human judgments, applicable across scenarios with and without ground truth references.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23853v1">Temporal Blindness in Multi-Turn LLM Agents: Misaligned Tool Use vs. Human Time Perception</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 preliminary work in progress
    </div>
    <details class="paper-abstract">
      Large language model agents are increasingly used in multi-turn conversational settings to interact with and execute tasks in dynamic environments. However, a key limitation is their temporal blindness: they, by default, operate with a stationary context, failing to account for the real-world time elapsed between messages. This becomes a critical liability when an agent must decide whether to invoke a tool based on how much time has passed since the last observation. Without temporal awareness, agents often either over-rely on previous context (skipping necessary tool calls), or under-rely on it (unnecessarily repeating tool calls). To study this challenge, we introduce TicToc-v1, a test set of multi-turn user-agent trajectories across 34 scenarios with varying time sensitivity. Each trajectory ends with a user question, where the need for a tool call depends on the amount of time elapsed since the last message. To give LLMs temporal context, we augment dialogue messages with explicit timestamps, bridging the gap between static dialogue and evolving environments. We then collected human preferences for these samples, creating two subsets: one where humans preferred relying on the previous observation (prefer-noTool), and another where they preferred a new tool call (prefer-Tool). We evaluated how well LLM tool-calling decisions align with human preferences under varying time intervals on TicToc-v1. Our analysis show that without time information, most models perform only slightly better than random, with the top alignment rate being just over 60%. While adding timestamps leads to a slight improvement, particularly for larger models, the improvement is modest, peaking at around 65%. We also show that naive, prompt-based alignment have limited effectiveness. Our findings highlight the need for specific post-training alignment to align multi-turn LLM tool use with human temporal perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23828v1">Beyond Understanding: Evaluating the Pragmatic Gap in LLMs' Cultural Processing of Figurative Language</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      We present a comprehensive evaluation of the ability of large language models (LLMs) to process culturally grounded language, specifically to understand and pragmatically use figurative expressions that encode local knowledge and cultural nuance. Using figurative language as a proxy for cultural nuance and local knowledge, we design evaluation tasks for contextual understanding, pragmatic use, and connotation interpretation in Arabic and English. We evaluate 22 open- and closed-source LLMs on Egyptian Arabic idioms, multidialectal Arabic proverbs, and English proverbs. Our results show a consistent hierarchy: the average accuracy for Arabic proverbs is 4.29% lower than for English proverbs, and performance for Egyptian idioms is 10.28% lower than for Arabic proverbs. For the pragmatic use task, accuracy drops by 14.07% relative to understanding, though providing contextual idiomatic sentences improves accuracy by 10.66%. Models also struggle with connotative meaning, reaching at most 85.58% agreement with human annotators on idioms with 100% inter-annotator agreement. These findings demonstrate that figurative language serves as an effective diagnostic for cultural reasoning: while LLMs can often interpret figurative meaning, they face challenges in using it appropriately. To support future research, we release Kinayat, the first dataset of Egyptian Arabic idioms designed for both figurative understanding and pragmatic use evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15870v2">OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Technical Report. Code: https://github.com/NVlabs/OmniVinci
    </div>
    <details class="paper-abstract">
      Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world. We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We carefully study the design choices across model architecture and data curation. For model architecture, we present three key innovations: (i) OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space; (ii) Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and (iii) Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings. We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, OmniVinci, outperforms Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens - a 6 times reduction compared to Qwen2.5-Omni's 1.2T. We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21837v3">Semantic Agreement Enables Efficient Open-Ended LLM Cascades</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) Industry Track
    </div>
    <details class="paper-abstract">
      Cascade systems route computational requests to smaller models when possible and defer to larger models only when necessary, offering a promising approach to balance cost and quality in LLM deployment. However, they face a fundamental challenge in open-ended text generation: determining output reliability when generation quality lies on a continuous spectrum, often with multiple valid responses. To address this, we propose semantic agreement -- meaning-level consensus between ensemble outputs -- as a training-free signal for reliable deferral. We show that when diverse model outputs agree semantically, their consensus is a stronger reliability signal than token-level confidence. Evaluated from 500M to 70B-parameter models, we find that semantic cascades match or surpass target-model quality at 40% of the cost and reduce latency by up to 60%. Our method requires no model internals, works across black-box APIs, and remains robust to model updates, making it a practical baseline for real-world LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14571v2">Learned, Lagged, LLM-splained: LLM Responses to End User Security Questions</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 17 pages, 7 tables
    </div>
    <details class="paper-abstract">
      Answering end user security questions is challenging. While large language models (LLMs) like GPT, LLAMA, and Gemini are far from error-free, they have shown promise in answering a variety of questions outside of security. We studied LLM performance in the area of end user security by qualitatively evaluating 3 popular LLMs on 900 systematically collected end user security questions. While LLMs demonstrate broad generalist ``knowledge'' of end user security information, there are patterns of errors and limitations across LLMs consisting of stale and inaccurate answers, and indirect or unresponsive communication styles, all of which impacts the quality of information received. Based on these patterns, we suggest directions for model improvement and recommend user strategies for interacting with LLMs when seeking assistance with security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00063v3">MolErr2Fix: Benchmarking LLM Trustworthiness in Chemistry via Modular Error Detection, Localization, Explanation, and Revision</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown growing potential in molecular sciences, but they often produce chemically inaccurate descriptions and struggle to recognize or justify potential errors. This raises important concerns about their robustness and reliability in scientific applications. To support more rigorous evaluation of LLMs in chemical reasoning, we present the MolErr2Fix benchmark, designed to assess LLMs on error detection and correction in molecular descriptions. Unlike existing benchmarks focused on molecule-to-text generation or property prediction, MolErr2Fix emphasizes fine-grained chemical understanding. It tasks LLMs with identifying, localizing, explaining, and revising potential structural and semantic errors in molecular descriptions. Specifically, MolErr2Fix consists of 1,193 fine-grained annotated error instances. Each instance contains quadruple annotations, i.e,. (error type, span location, the explanation, and the correction). These tasks are intended to reflect the types of reasoning and verification required in real-world chemical communication. Evaluations of current state-of-the-art LLMs reveal notable performance gaps, underscoring the need for more robust chemical reasoning capabilities. MolErr2Fix provides a focused benchmark for evaluating such capabilities and aims to support progress toward more reliable and chemically informed language models. All annotations and an accompanying evaluation API will be publicly released to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14205v2">DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 In Submission
    </div>
    <details class="paper-abstract">
      The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences.We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews.DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios.Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01268v3">AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted by NeurIPS2025
    </div>
    <details class="paper-abstract">
      We study the problem of determining whether a piece of text has been authored by a human or by a large language model (LLM). Existing state of the art logits-based detectors make use of statistics derived from the log-probability of the observed text evaluated using the distribution function of a given source LLM. However, relying solely on log probabilities can be sub-optimal. In response, we introduce AdaDetectGPT -- a novel classifier that adaptively learns a witness function from training data to enhance the performance of logits-based detectors. We provide statistical guarantees on its true positive rate, false positive rate, true negative rate and false negative rate. Extensive numerical studies show AdaDetectGPT nearly uniformly improves the state-of-the-art method in various combination of datasets and LLMs, and the improvement can reach up to 37\%. A python implementation of our method is available at https://github.com/Mamba413/AdaDetectGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23595v1">Multi-Agent Evolve: LLM Self-Improve through Co-evolution</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 29 pages, 4 figures, submitted to ICLR 2026
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has demonstrated significant potential in enhancing the reasoning capabilities of large language models (LLMs). However, the success of RL for LLMs heavily relies on human-curated datasets and verifiable rewards, which limit their scalability and generality. Recent Self-Play RL methods, inspired by the success of the paradigm in games and Go, aim to enhance LLM reasoning capabilities without human-annotated data. However, their methods primarily depend on a grounded environment for feedback (e.g., a Python interpreter or a game engine); extending them to general domains remains challenging. To address these challenges, we propose Multi-Agent Evolve (MAE), a framework that enables LLMs to self-evolve in solving diverse tasks, including mathematics, reasoning, and general knowledge Q&A. The core design of MAE is based on a triplet of interacting agents (Proposer, Solver, Judge) that are instantiated from a single LLM, and applies reinforcement learning to optimize their behaviors. The Proposer generates questions, the Solver attempts solutions, and the Judge evaluates both while co-evolving. Experiments on Qwen2.5-3B-Instruct demonstrate that MAE achieves an average improvement of 4.54% on multiple benchmarks. These results highlight MAE as a scalable, data-efficient method for enhancing the general reasoning abilities of LLMs with minimal reliance on human-curated supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06522v2">Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19113v2">Human-Aligned Faithfulness in Toxicity Explanations of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 23 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity -- from their explanations that justify a stance -- to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Human-Aligned Faithfulness (HAF), that measures the extent to which LLMs' free-form toxicity explanations align with those of a rational human under ideal conditions. We develop six metrics, based on uncertainty quantification, to comprehensively evaluate HAF of LLMs' toxicity explanations with no human involvement, and highlight how "non-ideal" the explanations are. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and irrelevant responses. We open-source our code at https://github.com/uofthcdslab/HAF and LLM-generated explanations at https://huggingface.co/collections/uofthcdslab/haf.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05965v4">Validating LLM-as-a-Judge Systems under Rating Indeterminacy</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm, in which a judge LLM system replaces human raters in rating the outputs of other generative AI (GenAI) systems, plays a critical role in scaling and standardizing GenAI evaluations. To validate such judge systems, evaluators assess human--judge agreement by first collecting multiple human ratings for each item in a validation corpus, then aggregating the ratings into a single, per-item gold label rating. For many items, however, rating criteria may admit multiple valid interpretations, so a human or LLM rater may deem multiple ratings "reasonable" or "correct." We call this condition rating indeterminacy. Problematically, many rating tasks that contain rating indeterminacy rely on forced-choice elicitation, whereby raters are instructed to select only one rating for each item. In this paper, we introduce a framework for validating LLM-as-a-judge systems under rating indeterminacy. We draw theoretical connections between different measures of judge system performance under different human--judge agreement metrics, and different rating elicitation and aggregation schemes. We demonstrate that differences in how humans and LLMs resolve rating indeterminacy when responding to forced-choice rating instructions can heavily bias LLM-as-a-judge validation. Through extensive experiments involving 11 real-world rating tasks and 9 commercial LLMs, we show that standard validation approaches that rely upon forced-choice ratings select judge systems that are highly suboptimal, performing as much as 31% worse than judge systems selected by our approach that uses multi-label "response set" ratings to account for rating indeterminacy. We conclude with concrete recommendations for more principled approaches to LLM-as-a-judge validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11477v3">How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted to Computational Linguistics
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in many languages beyond English. Yet, LLMs require more inference steps when generating non-English text due to their reliance on English-centric tokenizers and vocabulary, resulting in higher usage costs to non-English speakers. Vocabulary expansion with target language tokens is a widely used cross-lingual vocabulary adaptation approach to remedy this issue. Despite its effectiveness in inference speedup, previous work on vocabulary expansion has focused on high-resource settings assuming access to a substantial amount of target language data to effectively initialize the embeddings of the new tokens and adapt the LLM to the target language. However, vocabulary expansion in low-resource settings has yet to be explored. In this article, we investigate vocabulary expansion in low-resource settings by considering embedding initialization methods and continual pre-training strategies. Through extensive experiments across typologically diverse languages, tasks and models, we establish a set of strategies to perform vocabulary expansion for faster inference, while striving to maintain competitive downstream performance to baselines. This is achieved with only 30K sentences ($\sim$0.01GB text data) from the target language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23408v1">AutoStreamPipe: LLM Assisted Automatic Generation of Data Stream Processing Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Data pipelines are essential in stream processing as they enable the efficient collection, processing, and delivery of real-time data, supporting rapid data analysis. In this paper, we present AutoStreamPipe, a novel framework that employs Large Language Models (LLMs) to automate the design, generation, and deployment of stream processing pipelines. AutoStreamPipe bridges the semantic gap between high-level user intent and platform-specific implementations across distributed stream processing systems for structured multi-agent reasoning by integrating a Hypergraph of Thoughts (HGoT) as an extended version of GoT. AutoStreamPipe combines resilient execution strategies, advanced query analysis, and HGoT to deliver pipelines with good accuracy. Experimental evaluations on diverse pipelines demonstrate that AutoStreamPipe significantly reduces development time (x6.3) and error rates (x5.19), as measured by a novel Error-Free Score (EFS), compared to LLM code-generation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21433v3">The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 v3: DL4C camera-ready version to be presented at the 4th DL4C workshop co-located with NeurIPS '25; added OpenHands generality probe, added hybrid context management strategy
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents solve complex tasks through iterative reasoning, exploration, and tool-use, a process that can result in long, expensive context histories. While state-of-the-art Software Engineering (SE) agents like OpenHands or Cursor use LLM-based summarization to tackle this issue, it is unclear whether the increased complexity offers tangible performance benefits compared to simply omitting older observations. We present a systematic comparison of these approaches within SWE-agent on SWE-bench Verified across five diverse model configurations. Moreover, we show initial evidence of our findings generalizing to the OpenHands agent scaffold. We find that a simple environment observation masking strategy halves cost relative to the raw agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization. Additionally, we introduce a novel hybrid approach that further reduces costs by 7% and 11% compared to just observation masking or LLM summarization, respectively. Our findings raise concerns regarding the trend towards pure LLM summarization and provide initial evidence of untapped cost reductions by pushing the efficiency-effectiveness frontier. We release code and data for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08343v2">A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted in a computer science workshop
    </div>
    <details class="paper-abstract">
      With the rapid adoption of Large Language Models (LLMs), LLM-adapters have become increasingly common, providing lightweight specialization of large-scale models. Serving hundreds or thousands of these adapters on a single GPU allows request aggregation, increasing throughput, but may also cause request starvation if GPU memory limits are exceeded. To address this issue, this study focuses on determining the joint configuration of concurrent and parallel adapters that maximizes GPU throughput without inducing starvation, given heterogeneous adapter and traffic properties. We propose a data-driven ML approach leveraging interpretable models to tackle this caching problem and introduce the first Digital Twin capable of reproducing an LLM-adapter serving system, enabling efficient training data generation. Experiments with the vLLM framework and LoRA adapters show that the Digital Twin reproduces throughput within 5.1% of real results, while the ML approach predicts optimal numbers of concurrent and parallel adapters with an error of at most 7.2% under heterogeneous, real-world workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23799v3">Estimating LLM Consistency: A User Baseline vs Surrogate Metrics</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Published as a main conference paper at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to hallucinations and sensitiveto prompt perturbations, often resulting in inconsistent or unreliablegenerated text. Different methods have been proposed to mitigate suchhallucinations and fragility, one of which is to measure theconsistency of LLM responses -- the model's confidence in the responseor likelihood of generating a similar response when resampled. Inprevious work, measuring LLM response consistency often relied oncalculating the probability of a response appearing within a pool of resampledresponses, analyzing internal states, or evaluating logits of resopnses.However, it was not clear how well theseapproaches approximated users' perceptions of consistency of LLMresponses. To find out, we performed a user study ($n=2,976$)demonstrating that current methods for measuring LLM responseconsistency typically do not align well with humans' perceptions of LLMconsistency. We propose a logit-based ensemble method for estimatingLLM consistency and show that our method matches the performance of thebest-performing existing metric in estimating human ratings of LLMconsistency. Our results suggest that methods for estimating LLMconsistency without human evaluation are sufficiently imperfect towarrant broader use of evaluation with human input; this would avoidmisjudging the adequacy of models because of the imperfections ofautomated consistency metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02024v2">NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Meeting service-level objectives (SLOs) in Large Language Models (LLMs) serving is critical, but managing the high variability in load presents a significant challenge. Recent advancements in FP8 inference, backed by native hardware support, offer a potential solution: executing FP16 models by default, while switching to FP8 models during sudden load surges to achieve higher throughput at the cost of a slight quality degradation. Although this approach facilitates effective SLO management, it introduces additional memory overhead due to storing two versions of the same model. In response, this paper proposes NestedFP, an LLM serving technique that supports both FP16 and FP8 models in a memory efficient manner by overlaying FP8 parameters onto FP16 parameters, allowing both models to share the same FP16 memory footprint. By leveraging a compact data format for the overlay and a specialized GEMM kernel optimized for this format, NestedFP ensures minimal degradation in both model quality and inference throughput across both FP8 and FP16 modes. NestedFP provides a flexible platform for dynamic, SLO-aware precision selection. The code is available at https://github.com/SNU-ARC/NestedFP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10328v2">Are LLMs Empathetic to All? Investigating the Influence of Multi-Demographic Personas on a Model's Empathy</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 9 pages, 4 figures, 4 tables, EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models' (LLMs) ability to converse naturally is empowered by their ability to empathetically understand and respond to their users. However, emotional experiences are shaped by demographic and cultural contexts. This raises an important question: Can LLMs demonstrate equitable empathy across diverse user groups? We propose a framework to investigate how LLMs' cognitive and affective empathy vary across user personas defined by intersecting demographic attributes. Our study introduces a novel intersectional analysis spanning 315 unique personas, constructed from combinations of age, culture, and gender, across four LLMs. Results show that attributes profoundly shape a model's empathetic responses. Interestingly, we see that adding multiple attributes at once can attenuate and reverse expected empathy patterns. We show that they broadly reflect real-world empathetic trends, with notable misalignments for certain groups, such as those from Confucian culture. We complement our quantitative findings with qualitative insights to uncover model behaviour patterns across different demographic groups. Our findings highlight the importance of designing empathy-aware LLMs that account for demographic diversity to promote more inclusive and equitable model behaviour.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23358v1">How AI Forecasts AI Jobs: Benchmarking LLM Predictions of Labor Market Changes</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 8 pages + Limitations + References
    </div>
    <details class="paper-abstract">
      Artificial intelligence is reshaping labor markets, yet we lack tools to systematically forecast its effects on employment. This paper introduces a benchmark for evaluating how well large language models (LLMs) can anticipate changes in job demand, especially in occupations affected by AI. Existing research has shown that LLMs can extract sentiment, summarize economic reports, and emulate forecaster behavior, but little work has assessed their use for forward-looking labor prediction. Our benchmark combines two complementary datasets: a high-frequency index of sector-level job postings in the United States, and a global dataset of projected occupational changes due to AI adoption. We format these data into forecasting tasks with clear temporal splits, minimizing the risk of information leakage. We then evaluate LLMs using multiple prompting strategies, comparing task-scaffolded, persona-driven, and hybrid approaches across model families. We assess both quantitative accuracy and qualitative consistency over time. Results show that structured task prompts consistently improve forecast stability, while persona prompts offer advantages on short-term trends. However, performance varies significantly across sectors and horizons, highlighting the need for domain-aware prompting and rigorous evaluation protocols. By releasing our benchmark, we aim to support future research on labor forecasting, prompt design, and LLM-based economic reasoning. This work contributes to a growing body of research on how LLMs interact with real-world economic data, and provides a reproducible testbed for studying the limits and opportunities of AI as a forecasting tool in the context of labor markets.
    </details>
</div>
