# embodied ai - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18733v1">Advances and Innovations in the Multi-Agent Robotic System (MARS) Challenge</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 MARS Challenge @ NeurIPS 2025 Workshop on Space in Vision, Language, and Embodied AI. Challenge page: https://mars-eai.github.io/MARS-Challenge-Webpage/
    </div>
    <details class="paper-abstract">
      Recent advancements in multimodal large language models and vision-languageaction models have significantly driven progress in Embodied AI. As the field transitions toward more complex task scenarios, multi-agent system frameworks are becoming essential for achieving scalable, efficient, and collaborative solutions. This shift is fueled by three primary factors: increasing agent capabilities, enhancing system efficiency through task delegation, and enabling advanced human-agent interactions. To address the challenges posed by multi-agent collaboration, we propose the Multi-Agent Robotic System (MARS) Challenge, held at the NeurIPS 2025 Workshop on SpaVLE. The competition focuses on two critical areas: planning and control, where participants explore multi-agent embodied planning using vision-language models (VLMs) to coordinate tasks and policy execution to perform robotic manipulation in dynamic environments. By evaluating solutions submitted by participants, the challenge provides valuable insights into the design and coordination of embodied multi-agent systems, contributing to the future development of advanced collaborative AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18323v1">TC-IDM: Grounding Video Generation for Executable Zero-shot Robot Motion</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The vision-language-action (VLA) paradigm has enabled powerful robotic control by leveraging vision-language models, but its reliance on large-scale, high-quality robot data limits its generalization. Generative world models offer a promising alternative for general-purpose embodied AI, yet a critical gap remains between their pixel-level plans and physically executable actions. To this end, we propose the Tool-Centric Inverse Dynamics Model (TC-IDM). By focusing on the tool's imagined trajectory as synthesized by the world model, TC-IDM establishes a robust intermediate representation that bridges the gap between visual planning and physical control. TC-IDM extracts the tool's point cloud trajectories via segmentation and 3D motion estimation from generated videos. Considering diverse tool attributes, our architecture employs decoupled action heads to project these planned trajectories into 6-DoF end-effector motions and corresponding control signals. This plan-and-translate paradigm not only supports a wide range of end-effectors but also significantly improves viewpoint invariance. Furthermore, it exhibits strong generalization capabilities across long-horizon and out-of-distribution tasks, including interacting with deformable objects. In real-world evaluations, the world model with TC-IDM achieves an average success rate of 61.11 percent, with 77.7 percent on simple tasks and 38.46 percent on zero-shot deformable object tasks. It substantially outperforms end-to-end VLA-style baselines and other inverse dynamics models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17657v1">SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Contrastive Language-Image Pre-training (CLIP) has accomplished extraordinary success for semantic understanding but inherently struggles to perceive geometric structure. Existing methods attempt to bridge this gap by querying CLIP with textual prompts, a process that is often indirect and inefficient. This paper introduces a fundamentally different approach using a dual-pathway decoder. We present SPACE-CLIP, an architecture that unlocks and interprets latent geometric knowledge directly from a frozen CLIP vision encoder, completely bypassing the text encoder and its associated textual prompts. A semantic pathway interprets high-level features, dynamically conditioned on global context using feature-wise linear modulation (FiLM). In addition, a structural pathway extracts fine-grained spatial details from early layers. These complementary streams are hierarchically fused, enabling a robust synthesis of semantic context and precise geometry. Extensive experiments on the KITTI benchmark show that SPACE-CLIP dramatically outperforms previous CLIP-based methods. Our ablation studies validate that the synergistic fusion of our dual pathways is critical to this success. SPACE-CLIP offers a new, efficient, and architecturally elegant blueprint for repurposing large-scale vision models. The proposed method is not just a standalone depth estimator, but a readily integrable spatial perception module for the next generation of embodied AI systems, such as vision-language-action (VLA) models. Our model is available at https://github.com/taewan2002/space-clip
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10046v3">SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Conference: NeurIPS 2025 (main)
    </div>
    <details class="paper-abstract">
      Recent advances in foundation models have shown promising results in developing generalist robotics that can perform diverse tasks in open-ended scenarios given multimodal inputs. However, current work has been mainly focused on indoor, household scenarios. In this work, we present SimWorld-Robotics~(SWR), a simulation platform for embodied AI in large-scale, photorealistic urban environments. Built on Unreal Engine 5, SWR procedurally generates unlimited photorealistic urban scenes populated with dynamic elements such as pedestrians and traffic systems, surpassing prior urban simulations in realism, complexity, and scalability. It also supports multi-robot control and communication. With these key features, we build two challenging robot benchmarks: (1) a multimodal instruction-following task, where a robot must follow vision-language navigation instructions to reach a destination in the presence of pedestrians and traffic; and (2) a multi-agent search task, where two robots must communicate to cooperatively locate and meet each other. Unlike existing benchmarks, these two new benchmarks comprehensively evaluate a wide range of critical robot capacities in realistic scenarios, including (1) multimodal instructions grounding, (2) 3D spatial reasoning in large environments, (3) safe, long-range navigation with people and traffic, (4) multi-robot collaboration, and (5) grounded communication. Our experimental results demonstrate that state-of-the-art models, including vision-language models (VLMs), struggle with our tasks, lacking robust perception, reasoning, and planning abilities necessary for urban environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10046v2">SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Conference: NeurIPS 2025 (main)
    </div>
    <details class="paper-abstract">
      Recent advances in foundation models have shown promising results in developing generalist robotics that can perform diverse tasks in open-ended scenarios given multimodal inputs. However, current work has been mainly focused on indoor, household scenarios. In this work, we present SimWorld-Robotics~(SWR), a simulation platform for embodied AI in large-scale, photorealistic urban environments. Built on Unreal Engine 5, SWR procedurally generates unlimited photorealistic urban scenes populated with dynamic elements such as pedestrians and traffic systems, surpassing prior urban simulations in realism, complexity, and scalability. It also supports multi-robot control and communication. With these key features, we build two challenging robot benchmarks: (1) a multimodal instruction-following task, where a robot must follow vision-language navigation instructions to reach a destination in the presence of pedestrians and traffic; and (2) a multi-agent search task, where two robots must communicate to cooperatively locate and meet each other. Unlike existing benchmarks, these two new benchmarks comprehensively evaluate a wide range of critical robot capacities in realistic scenarios, including (1) multimodal instructions grounding, (2) 3D spatial reasoning in large environments, (3) safe, long-range navigation with people and traffic, (4) multi-robot collaboration, and (5) grounded communication. Our experimental results demonstrate that state-of-the-art models, including vision-language models (VLMs), struggle with our tasks, lacking robust perception, reasoning, and planning abilities necessary for urban environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09049v3">VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Accepted by NeurIPS 2025 Track on Datasets and Benchmarks. Project page: https://faceong.github.io/VIKI-R/
    </div>
    <details class="paper-abstract">
      Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09954v2">The Spatial Blindspot of Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Work done as part of the EleutherAI SOAR Program
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) have advanced rapidly, but their ability to capture spatial relationships remains a blindspot. Current VLMs are typically built with contrastive language-image pretraining (CLIP) style image encoders. The training recipe often flattens images into 1D patch sequences, discarding the 2D structure necessary for spatial reasoning. We argue that this lack of spatial awareness is a missing dimension in VLM design and a bottleneck for applications requiring spatial grounding, such as robotics and embodied AI. To address this, we investigate (i) image encoders trained with alternative objectives and (ii) 2D positional encodings. Our experiments show that these architectural choices can lead to improved spatial reasoning on several benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15282v1">Rethinking Video Generation Model for the Embodied World</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Github: https://github.com/DAGroup-PKU/ReVidgen/ Project website: https://dagroup-pku.github.io/ReVidgen.github.io/
    </div>
    <details class="paper-abstract">
      Video generation models have significantly advanced embodied intelligence, unlocking new possibilities for generating diverse robot data that capture perception, reasoning, and action in the physical world. However, synthesizing high-quality videos that accurately reflect real-world robotic interactions remains challenging, and the lack of a standardized benchmark limits fair comparisons and progress. To address this gap, we introduce a comprehensive robotics benchmark, RBench, designed to evaluate robot-oriented video generation across five task domains and four distinct embodiments. It assesses both task-level correctness and visual fidelity through reproducible sub-metrics, including structural consistency, physical plausibility, and action completeness. Evaluation of 25 representative models highlights significant deficiencies in generating physically realistic robot behaviors. Furthermore, the benchmark achieves a Spearman correlation coefficient of 0.96 with human evaluations, validating its effectiveness. While RBench provides the necessary lens to identify these deficiencies, achieving physical realism requires moving beyond evaluation to address the critical shortage of high-quality training data. Driven by these insights, we introduce a refined four-stage data pipeline, resulting in RoVid-X, the largest open-source robotic dataset for video generation with 4 million annotated video clips, covering thousands of tasks and enriched with comprehensive physical property annotations. Collectively, this synergistic ecosystem of evaluation and data establishes a robust foundation for rigorous assessment and scalable training of video models, accelerating the evolution of embodied AI toward general intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14140v1">CREATE: Cross-Layer Resilience Characterization and Optimization for Efficient yet Reliable Embodied AI Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 18 pages, 21 figures. Accepted by ASPLOS 2026
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (AI) has recently attracted significant attention as it bridges AI with the physical world. Modern embodied AI systems often combine a Large Language Model (LLM)-based planner for high-level task planning and a reinforcement learning (RL)-based controller for low-level action generation, enabling embodied agents to tackle complex tasks in real-world environments. However, deploying embodied agents remains challenging due to their high computation requirements, especially for battery-powered local devices. Although techniques like lowering operating voltage can improve energy efficiency, they can introduce bit errors and result in task failures. In this work, we propose CREATE, a general design principle that leverages heterogeneous resilience at different layers for synergistic energy-reliability co-optimization. For the first time, we conduct a comprehensive error injection study on modern embodied AI systems and observe an inherent but heterogeneous fault tolerance. Building upon these insights, we develop an anomaly detection and clearance mechanism at the circuit level to eliminate outlier errors. At the model level, we propose a weight-rotation-enhanced planning algorithm to improve the fault tolerance of the LLM-based planner. Furthermore, we introduce an application-level technique, autonomy-adaptive voltage scaling, to dynamically adjust the operating voltage of the controllers. The voltage scaling circuit is co-designed to enable online voltage adjustment. Extensive experiments demonstrate that without compromising task quality, CREATE achieves 40.6% computational energy savings on average over nominal-voltage baselines and 35.0% over prior-art techniques. This further leads to 29.5% to 37.3% chip-level energy savings and approximately a 15% to 30% improvement in battery life.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13945v1">Efficient Coordination with the System-Level Shared State: An Embodied-AI Native Modular Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      As Embodied AI systems move from research prototypes to real world deployments, they tend to evolve rapidly while remaining reliable under workload changes and partial failures. In practice, many deployments are only partially decoupled: middleware moves messages, but shared context and feedback semantics are implicit, causing interface drift, cross-module interference, and brittle recovery at scale. We present ANCHOR, a modular framework that makes decoupling and robustness explicit system-level primitives. ANCHOR separates (i) Canonical Records, an evolvable contract for the standardized shared state, from (ii) a communication bus for many-to-many dissemination and feedback-oriented coordination, forming an inspectable end-to-end loop. We validate closed-loop feasibility on a de-identified workflow instantiation, characterize latency distributions under varying payload sizes and publish rates, and demonstrate automatic stream resumption after hard crashes and restarts even with shared-memory loss. Overall, ANCHOR turns ad-hoc integration glue into explicit contracts, enabling controlled degradation under load and self-healing recovery for scalable deployment of closed-loop AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.09680v3">Object-Centric Latent Action Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted by AAAI 2026 (Oral). Source code: https://github.com/dunnolab/object-centric-lapo
    </div>
    <details class="paper-abstract">
      Leveraging vast amounts of unlabeled internet video data for embodied AI is currently bottlenecked by the lack of action labels and the presence of action-correlated visual distractors. Although recent latent action policy optimization (LAPO) has shown promise in inferring proxy action labels from visual observations, its performance degrades significantly when distractors are present. To address this limitation, we propose a novel object-centric latent action learning framework that centers on objects rather than pixels. We leverage self-supervised object-centric pretraining to disentangle the movement of the agent and distracting background dynamics. This allows LAPO to focus on task-relevant interactions, resulting in more robust proxy-action labels, enabling better imitation learning and efficient adaptation of the agent with just a few action-labeled trajectories. We evaluated our method in eight visually complex tasks across the Distracting Control Suite (DCS) and Distracting MetaWorld (DMW). Our results show that object-centric pretraining mitigates the negative effects of distractors by 50%, as measured by downstream task performance: average return (DCS) and success rate (DMW).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13556v1">LogicEnvGen: Task-Logic Driven Generation of Diverse Simulated Environments for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 19 pages, 15 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Simulated environments play an essential role in embodied AI, functionally analogous to test cases in software engineering. However, existing environment generation methods often emphasize visual realism (e.g., object diversity and layout coherence), overlooking a crucial aspect: logical diversity from the testing perspective. This limits the comprehensive evaluation of agent adaptability and planning robustness in distinct simulated environments. To bridge this gap, we propose LogicEnvGen, a novel method driven by Large Language Models (LLMs) that adopts a top-down paradigm to generate logically diverse simulated environments as test cases for agents. Given an agent task, LogicEnvGen first analyzes its execution logic to construct decision-tree-structured behavior plans and then synthesizes a set of logical trajectories. Subsequently, it adopts a heuristic algorithm to refine the trajectory set, reducing redundant simulation. For each logical trajectory, which represents a potential task situation, LogicEnvGen correspondingly instantiates a concrete environment. Notably, it employs constraint solving for physical plausibility. Furthermore, we introduce LogicEnvEval, a novel benchmark comprising four quantitative metrics for environment evaluation. Experimental results verify the lack of logical diversity in baselines and demonstrate that LogicEnvGen achieves 1.04-2.61x greater diversity, significantly improving the performance in revealing agent faults by 4.00%-68.00%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14352v1">RoboBrain 2.5: Depth in Sight, Time in Mind</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 37 pages, 13 figures, Technical Report
    </div>
    <details class="paper-abstract">
      We introduce RoboBrain 2.5, a next-generation embodied AI foundation model that advances general perception, spatial reasoning, and temporal modeling through extensive training on high-quality spatiotemporal supervision. Building upon its predecessor, RoboBrain 2.5 introduces two major capability upgrades. Specifically, it unlocks Precise 3D Spatial Reasoning by shifting from 2D pixel-relative grounding to depth-aware coordinate prediction and absolute metric constraint comprehension, generating complete 3D manipulation traces as ordered keypoint sequences under physical constraints. Complementing this spatial precision, the model establishes Dense Temporal Value Estimation that provides dense, step-aware progress prediction and execution state understanding across varying viewpoints, producing stable feedback signals for downstream learning. Together, these upgrades extend the framework toward more physically grounded and execution-aware embodied intelligence for complex, fine-grained manipulation. The code and checkpoints are available at project website: https://superrobobrain.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14339v1">CityCube: Benchmarking Cross-view Spatial Reasoning on Vision-Language Models in Urban Environments</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Cross-view spatial reasoning is essential for embodied AI, underpinning spatial understanding, mental simulation and planning in complex environments. Existing benchmarks primarily emphasize indoor or street settings, overlooking the unique challenges of open-ended urban spaces characterized by rich semantics, complex geometries, and view variations. To address this, we introduce CityCube, a systematic benchmark designed to probe cross-view reasoning capabilities of current VLMs in urban settings. CityCube integrates four viewpoint dynamics to mimic camera movements and spans a wide spectrum of perspectives from multiple platforms, e.g., vehicles, drones and satellites. For a comprehensive assessment, it features 5,022 meticulously annotated multi-view QA pairs categorized into five cognitive dimensions and three spatial relation expressions. A comprehensive evaluation of 33 VLMs reveals a significant performance disparity with humans: even large-scale models struggle to exceed 54.1% accuracy, remaining 34.2% below human performance. By contrast, small-scale fine-tuned VLMs achieve over 60.0% accuracy, highlighting the necessity of our benchmark. Further analyses indicate the task correlations and fundamental cognitive disparity between VLMs and human-like reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.14093v6">A Survey on Vision-Language-Action Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Project page: https://github.com/yueen-ma/Awesome-VLA
    </div>
    <details class="paper-abstract">
      Embodied AI is widely recognized as a cornerstone of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models -- referred to as vision-language-action models (VLAs) -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. The recent proliferation of VLAs necessitates a comprehensive survey to capture the rapidly evolving landscape. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing VLA-based control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges facing VLAs and outline promising future directions in embodied AI. A curated repository associated with this survey is available at: https://github.com/yueen-ma/Awesome-VLA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11421v1">The Great March 100: 100 Detail-oriented Tasks for Evaluating Embodied AI Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Recently, with the rapid development of robot learning and imitation learning, numerous datasets and methods have emerged. However, these datasets and their task designs often lack systematic consideration and principles. This raises important questions: Do the current datasets and task designs truly advance the capabilities of robotic agents? Do evaluations on a few common tasks accurately reflect the differentiated performance of various methods proposed by different teams and evaluated on different tasks? To address these issues, we introduce the Great March 100 (\textbf{GM-100}) as the first step towards a robot learning Olympics. GM-100 consists of 100 carefully designed tasks that cover a wide range of interactions and long-tail behaviors, aiming to provide a diverse and challenging set of tasks to comprehensively evaluate the capabilities of robotic agents and promote diversity and complexity in robot dataset task designs. These tasks are developed through systematic analysis and expansion of existing task designs, combined with insights from human-object interaction primitives and object affordances. We collect a large amount of trajectory data on different robotic platforms and evaluate several baseline models. Experimental results demonstrate that the GM-100 tasks are 1) feasible to execute and 2) sufficiently challenging to effectively differentiate the performance of current VLA models. Our data and code are available at https://rhos.ai/research/gm-100.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05810v2">SceneFoundry: Generating Interactive Infinite 3D Worlds</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The ability to automatically generate large-scale, interactive, and physically realistic 3D environments is crucial for advancing robotic learning and embodied intelligence. However, existing generative approaches often fail to capture the functional complexity of real-world interiors, particularly those containing articulated objects with movable parts essential for manipulation and navigation. This paper presents SceneFoundry, a language-guided diffusion framework that generates apartment-scale 3D worlds with functionally articulated furniture and semantically diverse layouts for robotic training. From natural language prompts, an LLM module controls floor layout generation, while diffusion-based posterior sampling efficiently populates the scene with articulated assets from large-scale 3D repositories. To ensure physical usability, SceneFoundry employs differentiable guidance functions to regulate object quantity, prevent articulation collisions, and maintain sufficient walkable space for robotic navigation. Extensive experiments demonstrate that our framework generates structurally valid, semantically coherent, and functionally interactive environments across diverse scene types and conditions, enabling scalable embodied AI research. project page: https://anc891203.github.io/SceneFoundry-Demo/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08355v2">Semantic Misalignment in Vision-Language Models under Perceptual Degradation</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 10 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are increasingly deployed in autonomous driving and embodied AI systems, where reliable perception is critical for safe semantic reasoning and decision-making. While recent VLMs demonstrate strong performance on multimodal benchmarks, their robustness to realistic perception degradation remains poorly understood. In this work, we systematically study semantic misalignment in VLMs under controlled degradation of upstream visual perception, using semantic segmentation on the Cityscapes dataset as a representative perception module. We introduce perception-realistic corruptions that induce only moderate drops in conventional segmentation metrics, yet observe severe failures in downstream VLM behavior, including hallucinated object mentions, omission of safety-critical entities, and inconsistent safety judgments. To quantify these effects, we propose a set of language-level misalignment metrics that capture hallucination, critical omission, and safety misinterpretation, and analyze their relationship with segmentation quality across multiple contrastive and generative VLMs. Our results reveal a clear disconnect between pixel-level robustness and multimodal semantic reliability, highlighting a critical limitation of current VLM-based systems and motivating the need for evaluation frameworks that explicitly account for perception uncertainty in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09954v1">The Spatial Blindspot of Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) have advanced rapidly, but their ability to capture spatial relationships remains a blindspot. Current VLMs are typically built with contrastive language-image pretraining (CLIP) style image encoders. The training recipe often flattens images into 1D patch sequences, discarding the 2D structure necessary for spatial reasoning. We argue that this lack of spatial awareness is a missing dimension in VLM design and a bottleneck for applications requiring spatial grounding, such as robotics and embodied AI. To address this, we investigate (i) image encoders trained with alternative objectives and (ii) 2D positional encodings. Our experiments show that these architectural choices can lead to improved spatial reasoning on several benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09697v1">Efficient Camera-Controlled Video Generation of Static Scenes via Sparse Diffusion and 3D Rendering</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Project page: https://ayushtewari.com/projects/srender/
    </div>
    <details class="paper-abstract">
      Modern video generative models based on diffusion models can produce very realistic clips, but they are computationally inefficient, often requiring minutes of GPU time for just a few seconds of video. This inefficiency poses a critical barrier to deploying generative video in applications that require real-time interactions, such as embodied AI and VR/AR. This paper explores a new strategy for camera-conditioned video generation of static scenes: using diffusion-based generative models to generate a sparse set of keyframes, and then synthesizing the full video through 3D reconstruction and rendering. By lifting keyframes into a 3D representation and rendering intermediate views, our approach amortizes the generation cost across hundreds of frames while enforcing geometric consistency. We further introduce a model that predicts the optimal number of keyframes for a given camera trajectory, allowing the system to adaptively allocate computation. Our final method, SRENDER, uses very sparse keyframes for simple trajectories and denser ones for complex camera motion. This results in video generation that is more than 40 times faster than the diffusion-based baseline in generating 20 seconds of video, while maintaining high visual fidelity and temporal stability, offering a practical path toward efficient and controllable video synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.19789v4">What Can RL Bring to VLA Generalization? An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Vision-Language Action (VLA) models have shown significant potential for embodied AI. However, their predominant training via supervised fine-tuning (SFT) limits generalization due to susceptibility to compounding errors under distribution shifts. Reinforcement learning (RL) offers a path to overcome these limitations by optimizing for task objectives via trial-and-error, yet a systematic understanding of its specific generalization benefits for VLAs compared to SFT is lacking. To address this, our study introduces a comprehensive benchmark for evaluating VLA generalization and systematically investigates the impact of RL fine-tuning across diverse visual, semantic, and execution dimensions. Our extensive experiments reveal that RL fine-tuning, particularly with PPO, significantly enhances generalization in semantic understanding and execution robustness over SFT, while maintaining comparable visual robustness. We identify PPO as a more effective RL algorithm for VLAs than LLM-derived methods like DPO and GRPO. We also develop a simple recipe for efficient PPO training on VLAs, and demonstrate its practical utility for improving VLA generalization. The project page is at https://rlvla.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08355v1">Semantic Misalignment in Vision-Language Models under Perceptual Degradation</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are increasingly deployed in autonomous driving and embodied AI systems, where reliable perception is critical for safe semantic reasoning and decision-making. While recent VLMs demonstrate strong performance on multimodal benchmarks, their robustness to realistic perception degradation remains poorly understood. In this work, we systematically study semantic misalignment in VLMs under controlled degradation of upstream visual perception, using semantic segmentation on the Cityscapes dataset as a representative perception module. We introduce perception-realistic corruptions that induce only moderate drops in conventional segmentation metrics, yet observe severe failures in downstream VLM behavior, including hallucinated object mentions, omission of safety-critical entities, and inconsistent safety judgments. To quantify these effects, we propose a set of language-level misalignment metrics that capture hallucination, critical omission, and safety misinterpretation, and analyze their relationship with segmentation quality across multiple contrastive and generative VLMs. Our results reveal a clear disconnect between pixel-level robustness and multimodal semantic reliability, highlighting a critical limitation of current VLM-based systems and motivating the need for evaluation frameworks that explicitly account for perception uncertainty in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07553v1">VirtualEnv: A Platform for Embodied AI Research</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to improve in reasoning and decision-making, there is a growing need for realistic and interactive environments where their abilities can be rigorously evaluated. We present VirtualEnv, a next-generation simulation platform built on Unreal Engine 5 that enables fine-grained benchmarking of LLMs in embodied and interactive scenarios. VirtualEnv supports rich agent-environment interactions, including object manipulation, navigation, and adaptive multi-agent collaboration, as well as game-inspired mechanics like escape rooms and procedurally generated environments. We provide a user-friendly API built on top of Unreal Engine, allowing researchers to deploy and control LLM-driven agents using natural language instructions. We integrate large-scale LLMs and vision-language models (VLMs), such as GPT-based models, to generate novel environments and structured tasks from multimodal inputs. Our experiments benchmark the performance of several popular LLMs across tasks of increasing complexity, analyzing differences in adaptability, planning, and multi-agent coordination. We also describe our methodology for procedural task generation, task validation, and real-time environment control. VirtualEnv is released as an open-source platform, we aim to advance research at the intersection of AI and gaming, enable standardized evaluation of LLMs in embodied AI settings, and pave the way for future developments in immersive simulations and interactive entertainment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01705v2">Explicit World Models for Reliable Human-Robot Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      This paper addresses the topic of robustness under sensing noise, ambiguous instructions, and human-robot interaction. We take a radically different tack to the issue of reliable embodied AI: instead of focusing on formal verification methods aimed at achieving model predictability and robustness, we emphasise the dynamic, ambiguous and subjective nature of human-robot interactions that requires embodied AI systems to perceive, interpret, and respond to human intentions in a manner that is consistent, comprehensible and aligned with human expectations. We argue that when embodied agents operate in human environments that are inherently social, multimodal, and fluid, reliability is contextually determined and only has meaning in relation to the goals and expectations of humans involved in the interaction. This calls for a fundamentally different approach to achieving reliable embodied AI that is centred on building and updating an accessible "explicit world model" representing the common ground between human and AI, that is used to align robot behaviours with human expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08876v1">The Semantic Lifecycle in Embodied AI: Acquisition, Representation and Storage via Foundation Models</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Semantic information in embodied AI is inherently multi-source and multi-stage, making it challenging to fully leverage for achieving stable perception-to-action loops in real-world environments. Early studies have combined manual engineering with deep neural networks, achieving notable progress in specific semantic-related embodied tasks. However, as embodied agents encounter increasingly complex environments and open-ended tasks, the demand for more generalizable and robust semantic processing capabilities has become imperative. Recent advances in foundation models (FMs) address this challenge through their cross-domain generalization abilities and rich semantic priors, reshaping the landscape of embodied AI research. In this survey, we propose the Semantic Lifecycle as a unified framework to characterize the evolution of semantic knowledge within embodied AI driven by foundation models. Departing from traditional paradigms that treat semantic processing as isolated modules or disjoint tasks, our framework offers a holistic perspective that captures the continuous flow and maintenance of semantic knowledge. Guided by this embodied semantic lifecycle, we further analyze and compare recent advances across three key stages: acquisition, representation, and storage. Finally, we summarize existing challenges and outline promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06573v1">QMAVIS: Long Video-Audio Understanding using Fusion of Large Multimodal Models</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Large Multimodal Models (LMMs) for video-audio understanding have traditionally been evaluated only on shorter videos of a few minutes long. In this paper, we introduce QMAVIS (Q Team-Multimodal Audio Video Intelligent Sensemaking), a novel long video-audio understanding pipeline built through a late fusion of LMMs, Large Language Models, and speech recognition models. QMAVIS addresses the gap in long-form video analytics, particularly for longer videos of a few minutes to beyond an hour long, opening up new potential applica- tions in sensemaking, video content analysis, embodied AI, etc. Quantitative experiments using QMAVIS demonstrated a 38.75% improvement over state-of-the-art video-audio LMMs like Vide- oLlaMA2 and InternVL2 on the VideoMME (with subtitles) dataset, which comprises long videos with audio information. Evaluations on other challenging video understanding datasets like PerceptionTest and EgoSchema saw up to 2% improvement, indicating competitive performance. Qualitative experiments also showed that QMAVIS is able to extract the nuances of different scenes in a long video audio content while understanding the overarching narrative. Ablation studies were also conducted to ascertain the impact of each component in the fusion pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05991v1">Open-Vocabulary 3D Instruction Ambiguity Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      In safety-critical domains, linguistic ambiguity can have severe consequences; a vague command like "Pass me the vial" in a surgical setting could lead to catastrophic errors. Yet, most embodied AI research overlooks this, assuming instructions are clear and focusing on execution rather than confirmation. To address this critical safety gap, we are the first to define Open-Vocabulary 3D Instruction Ambiguity Detection, a fundamental new task where a model must determine if a command has a single, unambiguous meaning within a given 3D scene. To support this research, we build Ambi3D, the large-scale benchmark for this task, featuring over 700 diverse 3D scenes and around 22k instructions. Our analysis reveals a surprising limitation: state-of-the-art 3D Large Language Models (LLMs) struggle to reliably determine if an instruction is ambiguous. To address this challenge, we propose AmbiVer, a two-stage framework that collects explicit visual evidence from multiple views and uses it to guide an vision-language model (VLM) in judging instruction ambiguity. Extensive experiments demonstrate the challenge of our task and the effectiveness of AmbiVer, paving the way for safer and more trustworthy embodied AI. Code and dataset available at https://jiayuding031020.github.io/ambi3d/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05810v1">SceneFoundry: Generating Interactive Infinite 3D Worlds</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The ability to automatically generate large-scale, interactive, and physically realistic 3D environments is crucial for advancing robotic learning and embodied intelligence. However, existing generative approaches often fail to capture the functional complexity of real-world interiors, particularly those containing articulated objects with movable parts essential for manipulation and navigation. This paper presents SceneFoundry, a language-guided diffusion framework that generates apartment-scale 3D worlds with functionally articulated furniture and semantically diverse layouts for robotic training. From natural language prompts, an LLM module controls floor layout generation, while diffusion-based posterior sampling efficiently populates the scene with articulated assets from large-scale 3D repositories. To ensure physical usability, SceneFoundry employs differentiable guidance functions to regulate object quantity, prevent articulation collisions, and maintain sufficient walkable space for robotic navigation. Extensive experiments demonstrate that our framework generates structurally valid, semantically coherent, and functionally interactive environments across diverse scene types and conditions, enabling scalable embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07855v1">An Empirical Study on Knowledge Transfer under Domain and Label Shifts in 3D LiDAR Point Clouds</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      For 3D perception systems to be practical in real-world applications -- from autonomous driving to embodied AI -- models must adapt to continuously evolving object definitions and sensor domains. Yet, research on continual and transfer learning in 3D point cloud perception remains underexplored compared to 2D vision -- particularly under simultaneous domain and label shifts. To address this gap, we propose the RObust Autonomous driving under Dataset shifts (ROAD) benchmark, a comprehensive evaluation suite for LiDAR-based object classification that explicitly accounts for domain shifts as well as three key forms of label evolution: class split, class expansion, and class insertion. Using large-scale datasets (Waymo, NuScenes, Argoverse2), we evaluate zero-shot transfer, linear probe, and CL, and analyze the impact of backbone architectures, training objectives, and CL methods. Our findings reveal limitations of existing approaches under realistic shifts and establish strong baselines for future research in robust 3D perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03470v2">Toward Maturity-Based Certification of Embodied AI: Quantifying Trustworthiness Through Measurement Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      We propose a maturity-based framework for certifying embodied AI systems through explicit measurement mechanisms. We argue that certifiable embodied AI requires structured assessment frameworks, quantitative scoring mechanisms, and methods for navigating multi-objective trade-offs inherent in trustworthiness evaluation. We demonstrate this approach using uncertainty quantification as an exemplar measurement mechanism and illustrate feasibility through an Uncrewed Aircraft System (UAS) detection case study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04137v1">Wow, wo, val! A Comprehensive Embodied World Model Evaluation Turing Test</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      As world models gain momentum in Embodied AI, an increasing number of works explore using video foundation models as predictive world models for downstream embodied tasks like 3D prediction or interactive generation. However, before exploring these downstream tasks, video foundation models still have two critical questions unanswered: (1) whether their generative generalization is sufficient to maintain perceptual fidelity in the eyes of human observers, and (2) whether they are robust enough to serve as a universal prior for real-world embodied agents. To provide a standardized framework for answering these questions, we introduce the Embodied Turing Test benchmark: WoW-World-Eval (Wow,wo,val). Building upon 609 robot manipulation data, Wow-wo-val examines five core abilities, including perception, planning, prediction, generalization, and execution. We propose a comprehensive evaluation protocol with 22 metrics to assess the models' generation ability, which achieves a high Pearson Correlation between the overall score and human preference (>0.93) and establishes a reliable foundation for the Human Turing Test. On Wow-wo-val, models achieve only 17.27 on long-horizon planning and at best 68.02 on physical consistency, indicating limited spatiotemporal consistency and physical reasoning. For the Inverse Dynamic Model Turing Test, we first use an IDM to evaluate the video foundation models' execution accuracy in the real world. However, most models collapse to $\approx$ 0% success, while WoW maintains a 40.74% success rate. These findings point to a noticeable gap between the generated videos and the real world, highlighting the urgency and necessity of benchmarking World Model in Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04266v1">State Backdoor: Towards Stealthy Real-world Poisoning Attack on Vision-Language-Action Model in State Space</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models are widely deployed in safety-critical embodied AI applications such as robotics. However, their complex multimodal interactions also expose new security vulnerabilities. In this paper, we investigate a backdoor threat in VLA models, where malicious inputs cause targeted misbehavior while preserving performance on clean data. Existing backdoor methods predominantly rely on inserting visible triggers into visual modality, which suffer from poor robustness and low insusceptibility in real-world settings due to environmental variability. To overcome these limitations, we introduce the State Backdoor, a novel and practical backdoor attack that leverages the robot arm's initial state as the trigger. To optimize trigger for insusceptibility and effectiveness, we design a Preference-guided Genetic Algorithm (PGA) that efficiently searches the state space for minimal yet potent triggers. Extensive experiments on five representative VLA models and five real-world tasks show that our method achieves over 90% attack success rate without affecting benign task performance, revealing an underexplored vulnerability in embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03136v1">Limited Linguistic Diversity in Embodied AI Datasets</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Language plays a critical role in Vision-Language-Action (VLA) models, yet the linguistic characteristics of the datasets used to train and evaluate these systems remain poorly documented. In this work, we present a systematic dataset audit of several widely used VLA corpora, aiming to characterize what kinds of instructions these datasets actually contain and how much linguistic variety they provide. We quantify instruction language along complementary dimensions-including lexical variety, duplication and overlap, semantic similarity, and syntactic complexity. Our analysis shows that many datasets rely on highly repetitive, template-like commands with limited structural variation, yielding a narrow distribution of instruction forms. We position these findings as descriptive documentation of the language signal available in current VLA training and evaluation data, intended to support more detailed dataset reporting, more principled dataset selection, and targeted curation or augmentation strategies that broaden language coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16853v2">ISCS: Parameter-Guided Feature Pruning for Resource-Constrained Embodied Perception</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Significant revision: The focus has been pivoted from learned image compression to embodied perception tasks. Experimental results and downstream applications have been updated to demonstrate the method's efficiency in split computing
    </div>
    <details class="paper-abstract">
      Prior studies in embodied AI consistently show that robust perception is critical for human-robot interaction, yet deploying high-fidelity visual models on resource-constrained agents remains challenging due to limited on-device computation power and transmission latency. Exploiting the redundancy in latent representations could improve system efficiency, yet existing approaches often rely on costly dataset-specific ablation tests or heavy entropy models unsuitable for real-time edge-robot collaboration. We propose a generalizable, dataset-agnostic method to identify and selectively transmit structure-critical channels in pretrained encoders. Instead of brute-force empirical evaluations, our approach leverages intrinsic parameter statistics-weight variances and biases-to estimate channel importance. This analysis reveals a consistent organizational structure, termed the Invariant Salient Channel Space (ISCS), where Salient-Core channels capture dominant structures while Salient-Auxiliary channels encode fine visual details. Building on ISCS, we introduce a deterministic static pruning strategy that enables lightweight split-computing. Experiments across different datasets demonstrate that our method achieves a deterministic, ultra-low latency pipeline by bypassing heavy entropy modeling. Our method reduces end-to-end latency, providing a critical speed-accuracy trade-off for resource-constrained human-aware embodied systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03470v1">Toward Maturity-Based Certification of Embodied AI: Quantifying Trustworthiness Through Measurement Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 5 pages, Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      We propose a maturity-based framework for certifying embodied AI systems through explicit measurement mechanisms. We argue that certifiable embodied AI requires structured assessment frameworks, quantitative scoring mechanisms, and methods for navigating multi-objective trade-offs inherent in trustworthiness evaluation. We demonstrate this approach using uncertainty quantification as an exemplar measurement mechanism and illustrate feasibility through an Uncrewed Aircraft System (UAS) detection case study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01705v1">Explicit World Models for Reliable Human-Robot Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      This paper addresses the topic of robustness under sensing noise, ambiguous instructions, and human-robot interaction. We take a radically different tack to the issue of reliable embodied AI: instead of focusing on formal verification methods aimed at achieving model predictability and robustness, we emphasise the dynamic, ambiguous and subjective nature of human-robot interactions that requires embodied AI systems to perceive, interpret, and respond to human intentions in a manner that is consistent, comprehensible and aligned with human expectations. We argue that when embodied agents operate in human environments that are inherently social, multimodal, and fluid, reliability is contextually determined and only has meaning in relation to the goals and expectations of humans involved in the interaction. This calls for a fundamentally different approach to achieving reliable embodied AI that is centred on building and updating an accessible "explicit world model" representing the common ground between human and AI, that is used to align robot behaviours with human expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10071v3">Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Post-challenge bug fix
    </div>
    <details class="paper-abstract">
      The 2025 BEHAVIOR Challenge is designed to rigorously track progress toward solving long-horizon tasks by physical agents in simulated environments. BEHAVIOR-1K focuses on everyday household tasks that people most want robots to assist with and these tasks introduce long-horizon mobile manipulation challenges in realistic settings, bridging the gap between current research and real-world, human-centric applications. This report presents our solution to the 2025 BEHAVIOR Challenge in a very close 2nd place and substantially outperforms the rest of the submissions. Building on $π_{0.5}$, we focus on systematically building our solution by studying the effects of training techniques and data. Through careful ablation studies, we reveal the scaling benefits in both the pre-training and post-training phases, leading to a validation Q-score of 0.345, significantly surpassing previous state-of-the-art performance. We summarize our practical lessons and design recommendations that we hope will provide actionable insights for the broader embodied AI community when adapting powerful foundation models to complex embodied scenarios. Project page: https://github.com/mli0603/openpi-comet
    </details>
</div>
