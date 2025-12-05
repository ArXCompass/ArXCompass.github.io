# embodied ai - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05060v1">4DLangVGGT: 4D Language-Visual Geometry Grounded Transformer</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 Code: https://github.com/hustvl/4DLangVGGT, Webpage: https://hustvl.github.io/4DLangVGGT
    </div>
    <details class="paper-abstract">
      Constructing 4D language fields is crucial for embodied AI, augmented/virtual reality, and 4D scene understanding, as they provide enriched semantic representations of dynamic environments and enable open-vocabulary querying in complex scenarios. However, existing approaches to 4D semantic field construction primarily rely on scene-specific Gaussian splatting, which requires per-scene optimization, exhibits limited generalization, and is difficult to scale to real-world applications. To address these limitations, we propose 4DLangVGGT, the first Transformer-based feed-forward unified framework for 4D language grounding, that jointly integrates geometric perception and language alignment within a single architecture. 4DLangVGGT has two key components: the 4D Visual Geometry Transformer, StreamVGGT, which captures spatio-temporal geometric representations of dynamic scenes; and the Semantic Bridging Decoder (SBD), which projects geometry-aware features into a language-aligned semantic space, thereby enhancing semantic interpretability while preserving structural fidelity. Unlike prior methods that depend on costly per-scene optimization, 4DLangVGGT can be jointly trained across multiple dynamic scenes and directly applied during inference, achieving both deployment efficiency and strong generalization. This design significantly improves the practicality of large-scale deployment and establishes a new paradigm for open-vocabulary 4D scene understanding. Experiments on HyperNeRF and Neu3D datasets demonstrate that our approach not only generalizes effectively but also achieves state-of-the-art performance, achieving up to 2% gains under per-scene training and 1% improvements under multi-scene training. Our code released in https://github.com/hustvl/4DLangVGGT
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07161v2">LLMscape</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 Accepted to NeurIPS 2025, Creative AI Track (updated to include poster)
    </div>
    <details class="paper-abstract">
      LLMscape is an interactive installation that investigates how humans and AI construct meaning under shared conditions of uncertainty. Within a mutable, projection-mapped landscape, human participants reshape the world and engage with multiple AI agents, each developing incomplete and provisional accounts of their environment. Exhibited in Shanghai and continually evolving, the work positions AI not as deterministic tools but as embodied co-witnesses to an unstable world, examining the parallels between human and artificial meaning-making and inviting reflection on our shared epistemic limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04686v1">Towards Cross-View Point Correspondence in Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Cross-view correspondence is a fundamental capability for spatial understanding and embodied AI. However, it is still far from being realized in Vision-Language Models (VLMs), especially in achieving precise point-level correspondence, which is crucial for precise affordance interaction. So we propose the Cross-View Point Correspondence (CVPC) task and CrossPoint-Bench, a comprehensive benchmark with hierarchical design, inspired by the human cognitive process of "perceive", "reason", and "correspond". Our evaluation shows the state-of-the-art models (e.g., Gemini-2.5-Pro) still fall far behind humans, with a gap of over 54.65% in overall accuracy, exposing a challenge in transitioning from coarse-grained judgement to fine-grained coordinate prediction. To address this problem, we construct CrossPoint-378K, a dataset with 378K question-answering pairs across 900 scenes, focused on actionable affordance regions that better reflect real-world manipulation and interaction scenarios. Furthermore, we propose CroPond that trained on the CrossPoint-378K dataset. Our CroPond achieves state-of-the-art performance on CrossPoint-Bench, surpassing Gemini-2.5-Pro by 39.7% accuracy, which offers a foundation for advancing future work on cross-view correspondence. The benchmark, dataset, and model are publicly available at https://github.com/WangYipu2002/CrossPoint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01424v3">ViRectify: A Challenging Benchmark for Video Reasoning Correction with Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-04
      | 💬 22 pages, 11 figures
    </div>
    <details class="paper-abstract">
      As multimodal large language models (MLLMs) frequently exhibit errors in complex video reasoning scenarios, correcting these errors is critical for uncovering their weaknesses and improving performance. However, existing benchmarks lack systematic evaluation of MLLMs' ability to identify and correct these video reasoning errors. To bridge this gap, we propose ViRectify, a comprehensive benchmark to evaluate their fine-grained correction capability. Through an AI-assisted annotation pipeline with human verification, we construct a dataset of over 30K instances spanning dynamic perception, scientific reasoning, and embodied decision-making domains. In ViRectify, we challenge MLLMs to perform step-wise error identification and generate rationales with key video evidence grounding. In addition, we further propose the trajectory evidence-driven correction framework, comprising step-wise error trajectory and reward modeling on visual evidence-grounded correction. It encourages the model to explicitly concentrate on error propagation and key timestamps for correction. Extensive evaluation across 16 advanced MLLMs demonstrates that our ViRectify serves as a challenging testbed, where GPT-5 achieves only 31.94% correction accuracy. Our framework enables a Qwen2.5-VL-7B to consistently outperform the variants of 72B on ViRectify, showing the effectiveness of our approach. Further analysis uncovers systematic asymmetries in error correction across models, and our dataset is also a valuable data resource to perform reflection learning. We believe ViRectify provides a new direction for comprehensively evaluating the advanced MLLMs in video reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04537v1">X-Humanoid: Robotize Human Videos to Generate Humanoid Videos at Scale</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      The advancement of embodied AI has unlocked significant potential for intelligent humanoid robots. However, progress in both Vision-Language-Action (VLA) models and world models is severely hampered by the scarcity of large-scale, diverse training data. A promising solution is to "robotize" web-scale human videos, which has been proven effective for policy training. However, these solutions mainly "overlay" robot arms to egocentric videos, which cannot handle complex full-body motions and scene occlusions in third-person videos, making them unsuitable for robotizing humans. To bridge this gap, we introduce X-Humanoid, a generative video editing approach that adapts the powerful Wan 2.2 model into a video-to-video structure and finetunes it for the human-to-humanoid translation task. This finetuning requires paired human-humanoid videos, so we designed a scalable data creation pipeline, turning community assets into 17+ hours of paired synthetic videos using Unreal Engine. We then apply our trained model to 60 hours of the Ego-Exo4D videos, generating and releasing a new large-scale dataset of over 3.6 million "robotized" humanoid video frames. Quantitative analysis and user studies confirm our method's superiority over existing baselines: 69% of users rated it best for motion consistency, and 62.1% for embodiment correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04515v1">EgoLCD: Egocentric Video Generation with Long Context Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-12-04
    </div>
    <details class="paper-abstract">
      Generating long, coherent egocentric videos is difficult, as hand-object interactions and procedural tasks require reliable long-term memory. Existing autoregressive models suffer from content drift, where object identity and scene semantics degrade over time. To address this challenge, we introduce EgoLCD, an end-to-end framework for egocentric long-context video generation that treats long video synthesis as a problem of efficient and stable memory management. EgoLCD combines a Long-Term Sparse KV Cache for stable global context with an attention-based short-term memory, extended by LoRA for local adaptation. A Memory Regulation Loss enforces consistent memory usage, and Structured Narrative Prompting provides explicit temporal guidance. Extensive experiments on the EgoVid-5M benchmark demonstrate that EgoLCD achieves state-of-the-art performance in both perceptual quality and temporal consistency, effectively mitigating generative forgetting and representing a significant step toward building scalable world models for embodied AI. Code: https://github.com/AIGeeksGroup/EgoLCD. Website: https://aigeeksgroup.github.io/EgoLCD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01204v2">TabletopGen: Instance-Level Interactive 3D Tabletop Scene Generation from Text or Single Image</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 Project page: https://d-robotics-ai-lab.github.io/TabletopGen.project/
    </div>
    <details class="paper-abstract">
      Generating high-fidelity, physically interactive 3D simulated tabletop scenes is essential for embodied AI--especially for robotic manipulation policy learning and data synthesis. However, current text- or image-driven 3D scene generation methods mainly focus on large-scale scenes, struggling to capture the high-density layouts and complex spatial relations that characterize tabletop scenes. To address these challenges, we propose TabletopGen, a training-free, fully automatic framework that generates diverse, instance-level interactive 3D tabletop scenes. TabletopGen accepts a reference image as input, which can be synthesized by a text-to-image model to enhance scene diversity. We then perform instance segmentation and completion on the reference to obtain per-instance images. Each instance is reconstructed into a 3D model followed by canonical coordinate alignment. The aligned 3D models then undergo pose and scale estimation before being assembled into a collision-free, simulation-ready tabletop scene. A key component of our framework is a novel pose and scale alignment approach that decouples the complex spatial reasoning into two stages: a Differentiable Rotation Optimizer for precise rotation recovery and a Top-view Spatial Alignment mechanism for robust translation and scale estimation, enabling accurate 3D reconstruction from 2D reference. Extensive experiments and user studies show that TabletopGen achieves state-of-the-art performance, markedly surpassing existing methods in visual fidelity, layout accuracy, and physical plausibility, capable of generating realistic tabletop scenes with rich stylistic and spatial diversity. Our code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03438v1">Multimodal Reinforcement Learning with Agentic Verifier for AI Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-03
    </div>
    <details class="paper-abstract">
      Agentic reasoning models trained with multimodal reinforcement learning (MMRL) have become increasingly capable, yet they are almost universally optimized using sparse, outcome-based rewards computed based on the final answers. Richer rewards computed from the reasoning tokens can improve learning significantly by providing more fine-grained guidance. However, it is challenging to compute more informative rewards in MMRL beyond those based on outcomes since different samples may require different scoring functions and teacher models may provide noisy reward signals too. In this paper, we introduce the Argos (Agentic Reward for Grounded & Objective Scoring), a principled reward agent to train multimodal reasoning models for agentic tasks. For each sample, Argos selects from a pool of teacher-model derived and rule-based scoring functions to simultaneously evaluate: (i) final response accuracy, (ii) spatiotemporal localization of referred entities and actions, and (iii) the quality of the reasoning process. We find that by leveraging our agentic verifier across both SFT data curation and RL training, our model achieves state-of-the-art results across multiple agentic tasks such as spatial reasoning, visual hallucination as well as robotics and embodied AI benchmarks. Critically, we demonstrate that just relying on SFT post-training on highly curated reasoning data is insufficient, as agents invariably collapse to ungrounded solutions during RL without our online verification. We also show that our agentic verifier can help to reduce reward-hacking in MMRL. Finally, we also provide a theoretical justification for the effectiveness of Argos through the concept of pareto-optimality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03418v1">YOLOA: Real-Time Affordance Detection via LLM Adapter</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 13 pages, 9 figures, conference
    </div>
    <details class="paper-abstract">
      Affordance detection aims to jointly address the fundamental "what-where-how" challenge in embodied AI by understanding "what" an object is, "where" the object is located, and "how" it can be used. However, most affordance learning methods focus solely on "how" objects can be used while neglecting the "what" and "where" aspects. Other affordance detection methods treat object detection and affordance learning as two independent tasks, lacking effective interaction and real-time capability. To overcome these limitations, we introduce YOLO Affordance (YOLOA), a real-time affordance detection model that jointly handles these two tasks via a large language model (LLM) adapter. Specifically, YOLOA employs a lightweight detector consisting of object detection and affordance learning branches refined through the LLM Adapter. During training, the LLM Adapter interacts with object and affordance preliminary predictions to refine both branches by generating more accurate class priors, box offsets, and affordance gates. Experiments on our relabeled ADG-Det and IIT-Heat benchmarks demonstrate that YOLOA achieves state-of-the-art accuracy (52.8 / 73.1 mAP on ADG-Det / IIT-Heat) while maintaining real-time performance (up to 89.77 FPS, and up to 846.24 FPS for the lightweight variant). This indicates that YOLOA achieves an excellent trade-off between accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01424v2">\textit{ViRectify}: A Challenging Benchmark for Video Reasoning Correction with Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 22 pages, 11 figures
    </div>
    <details class="paper-abstract">
      As multimodal large language models (MLLMs) frequently exhibit errors in complex video reasoning scenarios, correcting these errors is critical for uncovering their weaknesses and improving performance. However, existing benchmarks lack systematic evaluation of MLLMs' ability to identify and correct these video reasoning errors. To bridge this gap, we propose \textit{ViRectify}, a comprehensive benchmark to evaluate their fine-grained correction capability. Through an AI-assisted annotation pipeline with human verification, we construct a dataset of over 30\textit{K} instances spanning dynamic perception, scientific reasoning, and embodied decision-making domains. In \textit{ViRectify}, we challenge MLLMs to perform step-wise error identification and generate rationales with key video evidence grounding. In addition, we further propose the trajectory evidence-driven correction framework, comprising step-wise error trajectory and reward modeling on visual evidence-grounded correction. It encourages the model to explicitly concentrate on error propagation and key timestamps for correction. Extensive evaluation across 16 advanced MLLMs demonstrates that our \textit{ViRectify} serves as a challenging testbed, where GPT-5 achieves only 31.94\% correction accuracy. Our framework enables a Qwen2.5-VL-7B to consistently outperform the variants of 72B on \textit{ViRectify}, showing the effectiveness of our approach. Further analysis uncovers systematic asymmetries in error correction across models, and our dataset is also a valuable data resource to perform reflection learning. We believe \textit{ViRectify} provides a new direction for comprehensively evaluating the advanced MLLMs in video reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04308v1">ResponsibleRobotBench: Benchmarking Responsible Robot Manipulation using Multi-modal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-03
      | 💬 https://sites.google.com/view/responsible-robotbench
    </div>
    <details class="paper-abstract">
      Recent advances in large multimodal models have enabled new opportunities in embodied AI, particularly in robotic manipulation. These models have shown strong potential in generalization and reasoning, but achieving reliable and responsible robotic behavior in real-world settings remains an open challenge. In high-stakes environments, robotic agents must go beyond basic task execution to perform risk-aware reasoning, moral decision-making, and physically grounded planning. We introduce ResponsibleRobotBench, a systematic benchmark designed to evaluate and accelerate progress in responsible robotic manipulation from simulation to real world. This benchmark consists of 23 multi-stage tasks spanning diverse risk types, including electrical, chemical, and human-related hazards, and varying levels of physical and planning complexity. These tasks require agents to detect and mitigate risks, reason about safety, plan sequences of actions, and engage human assistance when necessary. Our benchmark includes a general-purpose evaluation framework that supports multimodal model-based agents with various action representation modalities. The framework integrates visual perception, context learning, prompt construction, hazard detection, reasoning and planning, and physical execution. It also provides a rich multimodal dataset, supports reproducible experiments, and includes standardized metrics such as success rate, safety rate, and safe success rate. Through extensive experimental setups, ResponsibleRobotBench enables analysis across risk categories, task types, and agent configurations. By emphasizing physical reliability, generalization, and safety in decision-making, this benchmark provides a foundation for advancing the development of trustworthy, real-world responsible dexterous robotic systems. https://sites.google.com/view/responsible-robotbench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02982v1">U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Preprint; 19 pages, 7 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Modeling dynamic 3D environments from LiDAR sequences is central to building reliable 4D worlds for autonomous driving and embodied AI. Existing generative frameworks, however, often treat all spatial regions uniformly, overlooking the varying uncertainty across real-world scenes. This uniform generation leads to artifacts in complex or ambiguous regions, limiting realism and temporal stability. In this work, we present U4D, an uncertainty-aware framework for 4D LiDAR world modeling. Our approach first estimates spatial uncertainty maps from a pretrained segmentation model to localize semantically challenging regions. It then performs generation in a "hard-to-easy" manner through two sequential stages: (1) uncertainty-region modeling, which reconstructs high-entropy regions with fine geometric fidelity, and (2) uncertainty-conditioned completion, which synthesizes the remaining areas under learned structural priors. To further ensure temporal coherence, U4D incorporates a mixture of spatio-temporal (MoST) block that adaptively fuses spatial and temporal representations during diffusion. Extensive experiments show that U4D produces geometrically faithful and temporally consistent LiDAR sequences, advancing the reliability of 4D world modeling for autonomous perception and simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18960v2">AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 18 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in embodied AI tasks. However, existing VLA models, often built upon Vision-Language Models (VLMs), typically process dense visual inputs independently at each timestep. This approach implicitly models the task as a Markov Decision Process (MDP). However, this history-agnostic design is suboptimal for effective visual token processing in dynamic sequential decision-making, as it fails to leverage the context of history. To address this limitation, we reformulate the problem from a Partially Observable Markov Decision Process (POMDP) perspective and propose a novel framework named AVA-VLA. Inspired by the POMDP that the action generation should be conditioned on the belief state. AVA-VLA introduces Active Visual Attention (AVA) to dynamically modulate visual processing. It achieves this by leveraging the recurrent state, which is a neural approximation of the agent's belief state derived from the previous decision step. Specifically, the AVA module uses the recurrent state to compute the soft weights to actively process task-relevant visual tokens based on its historical context. Comprehensive evaluations demonstrate that AVA-VLA achieves state-of-the-art performance across popular robotic benchmarks, including LIBERO and CALVIN. Furthermore, real-world deployments on a dual-arm robot platform validate the framework's practical applicability and robust sim-to-real transferability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02569v1">Reframing Human-Robot Interaction Through Extended Reality: Unlocking Safer, Smarter, and More Empathic Interactions with Virtual Robots and Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 This paper is under review
    </div>
    <details class="paper-abstract">
      This perspective reframes human-robot interaction (HRI) through extended reality (XR), arguing that virtual robots powered by large foundation models (FMs) can serve as cognitively grounded, empathic agents. Unlike physical robots, XR-native agents are unbound by hardware constraints and can be instantiated, adapted, and scaled on demand, while still affording embodiment and co-presence. We synthesize work across XR, HRI, and cognitive AI to show how such agents can support safety-critical scenarios, socially and cognitively empathic interaction across domains, and outreaching physical capabilities with XR and AI integration. We then discuss how multimodal large FMs (e.g., large language model, large vision model, and vision-language model) enable context-aware reasoning, affect-sensitive situations, and long-term adaptation, positioning virtual robots as cognitive and empathic mediators rather than mere simulation assets. At the same time, we highlight challenges and potential risks, including overtrust, cultural and representational bias, privacy concerns around biometric sensing, and data governance and transparency. The paper concludes by outlining a research agenda for human-centered, ethically grounded XR agents - emphasizing multi-layered evaluation frameworks, multi-user ecosystems, mixed virtual-physical embodiment, and societal and ethical design practices to envision XR-based virtual agents powered by FMs as reshaping future HRI into a more efficient and adaptive paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01629v2">SPARK: Sim-ready Part-level Articulated Reconstruction with VLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Project page: https://heyumeng.com/SPARK/index.html. 17 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Articulated 3D objects are critical for embodied AI, robotics, and interactive scene understanding, yet creating simulation-ready assets remains labor-intensive and requires expert modeling of part hierarchies and motion structures. We introduce SPARK, a framework for reconstructing physically consistent, kinematic part-level articulated objects from a single RGB image. Given an input image, we first leverage VLMs to extract coarse URDF parameters and generate part-level reference images. We then integrate the part-image guidance and the inferred structure graph into a generative diffusion transformer to synthesize consistent part and complete shapes of articulated objects. To further refine the URDF parameters, we incorporate differentiable forward kinematics and differentiable rendering to optimize joint types, axes, and origins under VLM-generated open-state supervision. Extensive experiments show that SPARK produces high-quality, simulation-ready articulated assets across diverse categories, enabling downstream applications such as robotic manipulation and interaction modeling. Project page: https://heyumeng.com/SPARK/index.html.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02458v1">Vision to Geometry: 3D Spatial Memory for Sequential Embodied MLLM Reasoning and Exploration</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Existing research on indoor embodied tasks typically requires agents to actively explore unknown environments and reason about the scene to achieve a specific goal. However, when deployed in real life, agents often face sequential tasks, where each new sub-task follows the completion of the previous one, and certain sub-tasks may be infeasible, such as searching for a non-existent object. Compared with the single-task setting, the core challenge lies in reusing spatial knowledge accumulated from previous explorations to support subsequent reasoning and exploration. In this work, we investigate this underexplored yet practically significant embodied AI challenge. To evaluate this challenge, we introduce SEER-Bench, a new Sequential Embodied Exploration and Reasoning Benchmark encompassing encompassing two classic embodied tasks: Embodied Question Answering (EQA) and Embodied Multi-modal Navigation (EMN). Building on SEER-Bench, we propose 3DSPMR, a 3D SPatial Memory Reasoning approach that exploits relational, visual, and geometric cues from explored regions to augment Multi-Modal Large Language Models (MLLMs) for reasoning and exploration in sequential embodied tasks. To the best of our knowledge, this is the first work to explicitly incorporate geometric information into MLLM-based spatial understanding and reasoning. Extensive experiments verify that 3DSPMR achieves substantial performance gains on both sequential EQA and EMN tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17467v2">PersonaAgent with GraphRAG: Community-Aware Knowledge Graphs for Personalized LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      We propose a novel framework for persona-based language model system, motivated by the need for personalized AI agents that adapt to individual user preferences. In our approach, the agent embodies the user's "persona" (e.g. user profile or taste) and is powered by a large language model (LLM). To enable the agent to leverage rich contextual information, we introduce a Knowledge-Graph-enhanced Retrieval-Augmented Generation (Graph RAG) mechanism that constructs an LLM-derived graph index of relevant documents and summarizes communities of related information. Our framework generates personalized prompts by combining: (1) a summary of the user's historical behaviors and preferences extracted from the knowledge graph, and (2) relevant global interaction patterns identified through graph-based community detection. This dynamic prompt engineering approach allows the agent to maintain consistent persona-aligned behaviors while benefiting from collective knowledge. On the LaMP benchmark, our method improves news categorization F1 by 11.1%, movie tagging F1 by 56.1%, and reduces product rating MAE by 10.4% over prior methods. Our code is available at https://anonymous.4open.science/r/PersonaAgentwGraphRAG-DE6F
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02329v1">Towards autonomous normative multi-agent systems for Human-AI software engineering teams</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      This paper envisions a transformative paradigm in software engineering, where Artificial Intelligence, embodied in fully autonomous agents, becomes the primary driver of the core software development activities. We introduce a new class of software engineering agents, empowered by Large Language Models and equipped with beliefs, desires, intentions, and memory to enable human-like reasoning. These agents collaborate with humans and other agents to design, implement, test, and deploy software systems with a level of speed, reliability, and adaptability far beyond the current software development processes. Their coordination and collaboration are governed by norms expressed as deontic modalities - commitments, obligations, prohibitions and permissions - that regulate interactions and ensure regulatory compliance. These innovations establish a scalable, transparent and trustworthy framework for future Human-AI software engineering teams.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02020v1">EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Accepted by AAAI 2026. Project Page: https://efficientflow.github.io/
    </div>
    <details class="paper-abstract">
      Generative modeling has recently shown remarkable promise for visuomotor policy learning, enabling flexible and expressive control across diverse embodied AI tasks. However, existing generative policies often struggle with data inefficiency, requiring large-scale demonstrations, and sampling inefficiency, incurring slow action generation during inference. We introduce EfficientFlow, a unified framework for efficient embodied AI with flow-based policy learning. To enhance data efficiency, we bring equivariance into flow matching. We theoretically prove that when using an isotropic Gaussian prior and an equivariant velocity prediction network, the resulting action distribution remains equivariant, leading to improved generalization and substantially reduced data demands. To accelerate sampling, we propose a novel acceleration regularization strategy. As direct computation of acceleration is intractable for marginal flow trajectories, we derive a novel surrogate loss that enables stable and scalable training using only conditional trajectories. Across a wide range of robotic manipulation benchmarks, the proposed algorithm achieves competitive or superior performance under limited data while offering dramatically faster inference. These results highlight EfficientFlow as a powerful and efficient paradigm for high-performance embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01889v1">KM-ViPE: Online Tightly Coupled Vision-Language-Geometry Fusion for Open-Vocabulary Semantic SLAM</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      We present KM-ViPE (Knowledge Mapping Video Pose Engine), a real-time open-vocabulary SLAM framework for uncalibrated monocular cameras in dynamic environments. Unlike systems requiring depth sensors and offline calibration, KM-ViPE operates directly on raw RGB streams, making it ideal for ego-centric applications and harvesting internet-scale video data for training. KM-ViPE tightly couples DINO visual features with geometric constraints through a high-level features based adaptive robust kernel that handles both moving objects and movable static objects (e.g., moving furniture in ego-centric views). The system performs simultaneous online localization and open-vocabulary semantic mapping by fusing geometric and deep visual features aligned with language embeddings. Our results are competitive with state-of-the-art approaches, while existing solutions either operate offline, need depth data and/or odometry estimation, or lack dynamic scene robustness. KM-ViPE benefits from internet-scale training and uniquely combines online operation, uncalibrated monocular input, and robust handling of dynamic scenes, which makes it a good fit for autonomous robotics and AR/VR applications and advances practical spatial intelligence capabilities for embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01629v1">SPARK: Sim-ready Part-level Articulated Reconstruction with VLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Articulated 3D objects are critical for embodied AI, robotics, and interactive scene understanding, yet creating simulation-ready assets remains labor-intensive and requires expert modeling of part hierarchies and motion structures. We introduce SPARK, a framework for reconstructing physically consistent, kinematic part-level articulated objects from a single RGB image. Given an input image, we first leverage VLMs to extract coarse URDF parameters and generate part-level reference images. We then integrate the part-image guidance and the inferred structure graph into a generative diffusion transformer to synthesize consistent part and complete shapes of articulated objects. To further refine the URDF parameters, we incorporate differentiable forward kinematics and differentiable rendering to optimize joint types, axes, and origins under VLM-generated open-state supervision. Extensive experiments show that SPARK produces high-quality, simulation-ready articulated assets across diverse categories, enabling downstream applications such as robotic manipulation and interaction modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01424v1">\textit{ViRectify}: A Challenging Benchmark for Video Reasoning Correction with Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 22 pages, 11 figures
    </div>
    <details class="paper-abstract">
      As multimodal large language models (MLLMs) frequently exhibit errors in complex video reasoning scenarios, correcting these errors is critical for uncovering their weaknesses and improving performance. However, existing benchmarks lack systematic evaluation of MLLMs' ability to identify and correct these video reasoning errors. To bridge this gap, we propose \textit{ViRectify}, a comprehensive benchmark to evaluate their fine-grained correction capability. Through an AI-assisted annotation pipeline with human verification, we construct a dataset of over 30\textit{K} instances spanning dynamic perception, scientific reasoning, and embodied decision-making domains. In \textit{ViRectify}, we challenge MLLMs to perform step-wise error identification and generate rationales with key video evidence grounding. In addition, we further propose the trajectory evidence-driven correction framework, comprising step-wise error trajectory and reward modeling on visual evidence-grounded correction. It encourages the model to explicitly concentrate on error propagation and key timestamps for correction. Extensive evaluation across 16 advanced MLLMs demonstrates that our \textit{ViRectify} serves as a challenging testbed, where GPT-5 achieves only 31.94\% correction accuracy. Our framework enables a Qwen2.5-VL-7B to consistently outperform the variants of 72B on \textit{ViRectify}, showing the effectiveness of our approach. Further analysis uncovers systematic asymmetries in error correction across models, and our dataset is also a valuable data resource to perform reflection learning. We believe \textit{ViRectify} provides a new direction for comprehensively evaluating the advanced MLLMs in video reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01223v1">S$^2$-MLLM: Boosting Spatial Reasoning Capability of MLLMs for 3D Visual Grounding with Structural Guidance</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 18 pages, 9 figures
    </div>
    <details class="paper-abstract">
      3D Visual Grounding (3DVG) focuses on locating objects in 3D scenes based on natural language descriptions, serving as a fundamental task for embodied AI and robotics. Recent advances in Multi-modal Large Language Models (MLLMs) have motivated research into extending them to 3DVG. However, MLLMs primarily process 2D visual inputs and struggle with understanding 3D spatial structure of scenes solely from these limited perspectives. Existing methods mainly utilize viewpoint-dependent rendering of reconstructed point clouds to provide explicit structural guidance for MLLMs in 3DVG tasks, leading to inefficiency and limited spatial reasoning. To address this issue, we propose S$^2$-MLLM, an efficient framework that enhances spatial reasoning in MLLMs through implicit spatial reasoning. We introduce a spatial guidance strategy that leverages the structure awareness of feed-forward 3D reconstruction. By acquiring 3D structural understanding during training, our model can implicitly reason about 3D scenes without relying on inefficient point cloud reconstruction. Moreover, we propose a structure-enhanced module (SE), which first employs intra-view and inter-view attention mechanisms to capture dependencies within views and correspondences across views. The module further integrates multi-level position encoding to associate visual representations with spatial positions and viewpoint information, enabling more accurate structural understanding. Extensive experiments demonstrate that S$^2$-MLLM unifies superior performance, generalization, and efficiency, achieving significant performance over existing methods across the ScanRefer, Nr3D, and Sr3D datasets. Code will be available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01204v1">TabletopGen: Instance-Level Interactive 3D Tabletop Scene Generation from Text or Single Image</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Project page: https://d-robotics-ai-lab.github.io/TabletopGen.project/
    </div>
    <details class="paper-abstract">
      Generating high-fidelity, physically interactive 3D simulated tabletop scenes is essential for embodied AI--especially for robotic manipulation policy learning and data synthesis. However, current text- or image-driven 3D scene generation methods mainly focus on large-scale scenes, struggling to capture the high-density layouts and complex spatial relations that characterize tabletop scenes. To address these challenges, we propose TabletopGen, a training-free, fully automatic framework that generates diverse, instance-level interactive 3D tabletop scenes. TabletopGen accepts a reference image as input, which can be synthesized by a text-to-image model to enhance scene diversity. We then perform instance segmentation and completion on the reference to obtain per-instance images. Each instance is reconstructed into a 3D model followed by canonical coordinate alignment. The aligned 3D models then undergo pose and scale estimation before being assembled into a collision-free, simulation-ready tabletop scene. A key component of our framework is a novel pose and scale alignment approach that decouples the complex spatial reasoning into two stages: a Differentiable Rotation Optimizer for precise rotation recovery and a Top-view Spatial Alignment mechanism for robust translation and scale estimation, enabling accurate 3D reconstruction from 2D reference. Extensive experiments and user studies show that TabletopGen achieves state-of-the-art performance, markedly surpassing existing methods in visual fidelity, layout accuracy, and physical plausibility, capable of generating realistic tabletop scenes with rich stylistic and spatial diversity. Our code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02280v1">Bridging the Gap: Toward Cognitive Autonomy in Artificial Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Artificial intelligence has advanced rapidly across perception, language, reasoning, and multimodal domains. Yet despite these achievements, modern AI systems remain fundamentally limited in their ability to self-monitor, self-correct, and regulate their behavior autonomously in dynamic contexts. This paper identifies and analyzes seven core deficiencies that constrain contemporary AI models: the absence of intrinsic self-monitoring, lack of meta-cognitive awareness, fixed and non-adaptive learning mechanisms, inability to restructure goals, lack of representational maintenance, insufficient embodied feedback, and the absence of intrinsic agency. Alongside identifying these limitations, we also outline a forward-looking perspective on how AI may evolve beyond them through architectures that mirror neurocognitive principles. We argue that these structural limitations prevent current architectures, including deep learning and transformer-based systems, from achieving robust generalization, lifelong adaptability, and real-world autonomy. Drawing on a comparative analysis of artificial systems and biological cognition [7], and integrating insights from AI research, cognitive science, and neuroscience, we outline how these capabilities are absent in current models and why scaling alone cannot resolve them. We conclude by advocating for a paradigmatic shift toward cognitively grounded AI (cognitive autonomy) capable of self-directed adaptation, dynamic representation management, and intentional, goal-oriented behavior, paired with reformative oversight mechanisms [8] that ensure autonomous systems remain interpretable, governable, and aligned with human values.
    </details>
</div>
