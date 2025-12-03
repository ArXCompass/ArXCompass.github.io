# embodied ai - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

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
