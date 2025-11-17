# embodied ai - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00108v2">Pelican-VL 1.0: A Foundation Brain Model for Embodied Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      This report presents Pelican-VL 1.0, a new family of open-source embodied brain models with parameter scales ranging from 7 billion to 72 billion. Our explicit mission is clearly stated as: To embed powerful intelligence into various embodiments. Pelican-VL 1.0 is currently the largest-scale open-source embodied multimodal brain model. Its core advantage lies in the in-depth integration of data power and intelligent adaptive learning mechanisms. Specifically, metaloop distilled a high-quality dataset from a raw dataset containing 4+ billion tokens. Pelican-VL 1.0 is trained on a large-scale cluster of 1000+ A800 GPUs, consuming over 50k+ A800 GPU-hours per checkpoint. This translates to a 20.3% performance uplift from its base model and outperforms 100B-level open-source counterparts by 10.6%, placing it on par with leading proprietary systems on well-known embodied benchmarks. We establish a novel framework, DPPO (Deliberate Practice Policy Optimization), inspired by human metacognition to train Pelican-VL 1.0. We operationalize this as a metaloop that teaches the AI to practice deliberately, which is a RL-Refine-Diagnose-SFT loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18544v2">SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), as the foundational architecture for next-generation interactive AI applications, not only power intelligent dialogue systems but also drive the evolution of embodied intelligence on edge devices, including humanoid robots, smart vehicles, and other scenarios. The applications running on these edge devices impose differentiated Service Level Objectives (SLO) requirements on LLM services, specifically manifested as distinct constraints on Time to First Token (TTFT) and Time Per Output Token (TPOT) as well as end-to-end latency. Notably, edge devices typically handle real-time tasks that are extremely sensitive to latency, such as machine control and navigation planning. However, existing scheduling service systems still prioritize maximizing output token throughput as the sole optimization objective, failing to adequately address the diversity of SLO requirements. This ultimately results in persistently high violation rates for end-to-end latency or TPOT related SLOs. This paper proposes SLICE, an innovative scheduling solution designed for edge computing scenarios with differentiated SLO requirements. By combining a utility-maximizing request scheduling algorithm with a dynamic iterative control mechanism for generation rates, SLICE significantly improves LLM inference service SLO attainment. Experimental results demonstrate that compared to state-of-the-art solutions Orca and FastServe, SLICE achieves up to 35x higher SLO attainment and 3.4x advantage in task completion time than the other two solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.09497v1">Fundamentals of Physical AI</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 This paper is already published in Journal of Intelligent System of Systems Lifecycle Management
    </div>
    <details class="paper-abstract">
      This work will elaborate the fundamental principles of physical artificial intelligence (Physical AI) from a scientific and systemic perspective. The aim is to create a theoretical foundation that describes the physical embodiment, sensory perception, ability to act, learning processes, and context sensitivity of intelligent systems within a coherent framework. While classical AI approaches rely on symbolic processing and data driven models, Physical AI understands intelligence as an emergent phenomenon of real interaction between body, environment, and experience. The six fundamentals presented here are embodiment, sensory perception, motor action, learning, autonomy, and context sensitivity, and form the conceptual basis for designing and evaluating physically intelligent systems. Theoretically, it is shown that these six principles do not represent loose functional modules but rather act as a closed control loop in which energy, information, control, and context are in constant interaction. This circular interaction enables a system to generate meaning not from databases, but from physical experience, a paradigm shift that understands intelligence as an physical embodied process. Physical AI understands learning not as parameter adjustment, but as a change in the structural coupling between agents and the environment. To illustrate this, the theoretical model is explained using a practical scenario: An adaptive assistant robot supports patients in a rehabilitation clinic. This example illustrates that physical intelligence does not arise from abstract calculation, but from immediate, embodied experience. It shows how the six fundamentals interact in a real system: embodiment as a prerequisite, perception as input, movement as expression, learning as adaptation, autonomy as regulation, and context as orientation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07090v3">Green AI: A systematic review and meta-analysis of its definitions, lifecycle models, hardware and measurement attempts</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Across the Artificial Intelligence (AI) lifecycle - from hardware to development, deployment, and reuse - burdens span energy, carbon, water, and embodied impacts. Cloud provider tools improve transparency but remain heterogeneous and often omit water and value chain effects, limiting comparability and reproducibility. Addressing these multi dimensional burdens requires a lifecycle approach linking phase explicit mapping with system levers (hardware, placement, energy mix, cooling, scheduling) and calibrated measurement across facility, system, device, and workload levels. This article (i) establishes a unified, operational definition of Green AI distinct from Sustainable AI; (ii) formalizes a five phase lifecycle mapped to Life Cycle Assessment (LCA) stages, making energy, carbon, water, and embodied impacts first class; (iii) specifies governance via Plan Do Check Act (PDCA) cycles with decision gateways; (iv) systematizes hardware and system level strategies across the edge cloud continuum to reduce embodied burdens; and (v) defines a calibrated measurement framework combining estimator models with direct metering to enable reproducible, provider agnostic comparisons. Combining definition, lifecycle processes, hardware strategies, and calibrated measurement, this article offers actionable, evidence based guidance for researchers, practitioners, and policymakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2508.15201v2">Survey of Vision-Language-Action Models for Embodied Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 in Chinese language
    </div>
    <details class="paper-abstract">
      Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07090v4">Green AI: A systematic review and meta-analysis of its definitions, lifecycle models, hardware and measurement attempts</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Across the Artificial Intelligence (AI) lifecycle - from hardware to development, deployment, and reuse - burdens span energy, carbon, water, and embodied impacts. Cloud provider tools improve transparency but remain heterogeneous and often omit water and value chain effects, limiting comparability and reproducibility. Addressing these multi dimensional burdens requires a lifecycle approach linking phase explicit mapping with system levers (hardware, placement, energy mix, cooling, scheduling) and calibrated measurement across facility, system, device, and workload levels. This article (i) establishes a unified, operational definition of Green AI distinct from Sustainable AI; (ii) formalizes a five phase lifecycle mapped to Life Cycle Assessment (LCA) stages, making energy, carbon, water, and embodied impacts first class; (iii) specifies governance via Plan Do Check Act (PDCA) cycles with decision gateways; (iv) systematizes hardware and system level strategies across the edge cloud continuum to reduce embodied burdens; and (v) defines a calibrated measurement framework combining estimator models with direct metering to enable reproducible, provider agnostic comparisons. Combining definition, lifecycle processes, hardware strategies, and calibrated measurement, this article offers actionable, evidence based guidance for researchers, practitioners, and policymakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09964v1">EnvTrace: Simulation-Based Semantic Evaluation of LLM Code via Execution Trace Alignment -- Demonstrated at Synchrotron Beamlines</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) for instrument control requires methods that go beyond standard, stateless algorithmic benchmarks, since the behavior of physical systems cannot be fully captured by unit tests alone. Here we introduce EnvTrace, a simulation-based method that evaluates execution traces to assess semantic code equivalence. EnvTrace is demonstrated with a beamline control-logic digital twin to facilitate the evaluation of instrument control code, with the digital twin itself also enabling the pre-execution validation of live experiments. Over 30 LLMs were evaluated using trace alignment to generate a multi-faceted score for functional correctness across key behavioral dimensions, showing that many top-tier models can approach human-level performance in rapid control-code generation. This is a first step toward a broader vision where LLMs and digital twins work symbiotically: LLMs providing intuitive control and agentic orchestration, and digital twins offering safe and high-fidelity environments, paving the way towards autonomous embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07090v2">Green AI: A systematic review and meta-analysis of its definitions, lifecycle models, hardware and measurement attempts</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Across the Artificial Intelligence (AI) lifecycle - from hardware to development, deployment, and reuse - burdens span energy, carbon, water, and embodied impacts. Cloud provider tools improve transparency but remain heterogeneous and often omit water and value chain effects, limiting comparability and reproducibility. Addressing these multi dimensional burdens requires a lifecycle approach linking phase explicit mapping with system levers (hardware, placement, energy mix, cooling, scheduling) and calibrated measurement across facility, system, device, and workload levels. This article (i) establishes a unified, operational definition of Green AI distinct from Sustainable AI; (ii) formalizes a five phase lifecycle mapped to Life Cycle Assessment (LCA) stages, making energy, carbon, water, and embodied impacts first class; (iii) specifies governance via Plan Do Check Act (PDCA) cycles with decision gateways; (iv) systematizes hardware and system level strategies across the edge cloud continuum to reduce embodied burdens; and (v) defines a calibrated measurement framework combining estimator models with direct metering to enable reproducible, provider agnostic comparisons. Combining definition, lifecycle processes, hardware strategies, and calibrated measurement, this article offers actionable, evidence based guidance for researchers, practitioners, and policymakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09497v1">Fundamentals of Physical AI</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 This paper is already published in Journal of Intelligent System of Systems Lifecycle Management
    </div>
    <details class="paper-abstract">
      This work will elaborate the fundamental principles of physical artificial intelligence (Physical AI) from a scientific and systemic perspective. The aim is to create a theoretical foundation that describes the physical embodiment, sensory perception, ability to act, learning processes, and context sensitivity of intelligent systems within a coherent framework. While classical AI approaches rely on symbolic processing and data driven models, Physical AI understands intelligence as an emergent phenomenon of real interaction between body, environment, and experience. The six fundamentals presented here are embodiment, sensory perception, motor action, learning, autonomy, and context sensitivity, and form the conceptual basis for designing and evaluating physically intelligent systems. Theoretically, it is shown that these six principles do not represent loose functional modules but rather act as a closed control loop in which energy, information, control, and context are in constant interaction. This circular interaction enables a system to generate meaning not from databases, but from physical experience, a paradigm shift that understands intelligence as an physical embodied process. Physical AI understands learning not as parameter adjustment, but as a change in the structural coupling between agents and the environment. To illustrate this, the theoretical model is explained using a practical scenario: An adaptive assistant robot supports patients in a rehabilitation clinic. This example illustrates that physical intelligence does not arise from abstract calculation, but from immediate, embodied experience. It shows how the six fundamentals interact in a real system: embodiment as a prerequisite, perception as input, movement as expression, learning as adaptation, autonomy as regulation, and context as orientation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15201v2">Survey of Vision-Language-Action Models for Embodied Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 in Chinese language
    </div>
    <details class="paper-abstract">
      Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05294v4">Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Foundation models, including large language models (LLMs) and vision-language models (VLMs), have recently enabled novel approaches to robot autonomy and human-robot interfaces. In parallel, vision-language-action models (VLAs) or large behavior models (LBMs) are increasing the dexterity and capabilities of robotic systems. This survey paper reviews works that advance agentic applications and architectures, including initial efforts with GPT-style interfaces and more complex systems where AI agents function as coordinators, planners, perception actors, or generalist interfaces. Such agentic architectures allow robots to reason over natural language instructions, invoke APIs, plan task sequences, or assist in operations and diagnostics. In addition to peer-reviewed research, due to the fast-evolving nature of the field, we highlight and include community-driven projects, ROS packages, and industrial frameworks that show emerging trends. We propose a taxonomy for classifying model integration approaches and present a comparative analysis of the role that agents play in different solutions in today's literature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08007v2">EAGLE: Episodic Appearance- and Geometry-aware Memory for Unified 2D-3D Visual Query Localization in Egocentric Vision</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 13 Pages, accepted by AAAI-2026
    </div>
    <details class="paper-abstract">
      Egocentric visual query localization is vital for embodied AI and VR/AR, yet remains challenging due to camera motion, viewpoint changes, and appearance variations. We present EAGLE, a novel framework that leverages episodic appearance- and geometry-aware memory to achieve unified 2D-3D visual query localization in egocentric vision. Inspired by avian memory consolidation, EAGLE synergistically integrates segmentation guided by an appearance-aware meta-learning memory (AMM), with tracking driven by a geometry-aware localization memory (GLM). This memory consolidation mechanism, through structured appearance and geometry memory banks, stores high-confidence retrieval samples, effectively supporting both long- and short-term modeling of target appearance variations. This enables precise contour delineation with robust spatial discrimination, leading to significantly improved retrieval accuracy. Furthermore, by integrating the VQL-2D output with a visual geometry grounded Transformer (VGGT), we achieve a efficient unification of 2D and 3D tasks, enabling rapid and accurate back-projection into 3D space. Our method achieves state-ofthe-art performance on the Ego4D-VQ benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07412v1">TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit embodied agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create photorealistic and dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains unclear. We introduce TwinOR, a framework for constructing photorealistic, dynamic digital twins of ORs for embodied AI research. The system reconstructs static geometry from pre-scan videos and continuously models human and equipment motion through multi-view perception of OR activities. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter level accuracy while preserving dynamic interaction across surgical workflows, enabling realistic renderings and a virtual playground for embodied AI systems. In our experiments, TwinOR simulates stereo and monocular sensor streams for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 on TwinOR-synthesized data achieve performance within their reported accuracy on real indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for perception and localization challenges. By establishing a real-to-sim pipeline for constructing dynamic, photorealistic digital twins of OR environments, TwinOR enables the safe, scalable, and data-efficient development and benchmarking of embodied AI, ultimately accelerating the deployment of embodied AI from sim-to-real.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07161v1">LLMscape</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted to NeurIPS 2025, Creative AI Track
    </div>
    <details class="paper-abstract">
      LLMscape is an interactive installation that investigates how humans and AI construct meaning under shared conditions of uncertainty. Within a mutable, projection-mapped landscape, human participants reshape the world and engage with multiple AI agents, each developing incomplete and provisional accounts of their environment. Exhibited in Shanghai and continually evolving, the work positions AI not as deterministic tools but as embodied co-witnesses to an unstable world, examining the parallels between human and artificial meaning-making and inviting reflection on our shared epistemic limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.01386v4">CATransformers: Carbon Aware Transformers Through Joint Model-Hardware Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Machine learning solutions are rapidly adopted to enable a variety of key use cases, from conversational AI assistants to scientific discovery. This growing adoption is expected to increase the associated lifecycle carbon footprint, including both \emph{operational carbon} from training and inference and \emph{embodied carbon} from AI hardware manufacturing. We introduce \ourframework -- the first carbon-aware co-optimization framework for Transformer-based models and hardware accelerators. By integrating both operational and embodied carbon into early-stage design space exploration, \ourframework enables sustainability-driven model architecture and hardware accelerator co-design that reveals fundamentally different trade-offs than latency- or energy-centric approaches. Evaluated across a range of Transformer models, \ourframework consistently demonstrates the potential to reduce total carbon emissions -- by up to 30\% -- while maintaining accuracy and latency. We further highlight its extensibility through a focused case study on multi-modal models. Our results emphasize the need for holistic optimization methods that prioritize carbon efficiency without compromising model capability and execution time performance. The source code of \ourframework is available at {\small{\href{https://github.com/facebookresearch/CATransformers}{\texttt{https://github.com/facebookresearch/CATransformers}}}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08896v4">Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on low-level grasp stability metrics, neglecting affordance-aware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pre-trained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably human-like in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms state-of-the-art baselines across seen objects, unseen instances, and even entirely novel categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11773v4">AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      A major challenge in developing robust and generalizable Human Activity Recognition (HAR) systems for smart homes is the lack of large and diverse labeled datasets. Variations in home layouts, sensor configurations, and individual behaviors further exacerbate this issue. To address this, we leverage the idea of embodied AI agents -- virtual agents that perceive and act within simulated environments guided by internal world models. We introduce AgentSense, a virtual data generation pipeline in which agents live out daily routines in simulated smart homes, with behavior guided by Large Language Models (LLMs). The LLM generates diverse synthetic personas and realistic routines grounded in the environment, which are then decomposed into fine-grained actions. These actions are executed in an extended version of the VirtualHome simulator, which we augment with virtual ambient sensors that record the agents' activities. Our approach produces rich, privacy-preserving sensor data that reflects real-world diversity. We evaluate AgentSense on five real HAR datasets. Models pretrained on the generated data consistently outperform baselines, especially in low-resource settings. Furthermore, combining the generated virtual sensor data with a small amount of real data achieves performance comparable to training on full real-world datasets. These results highlight the potential of using LLM-guided embodied agents for scalable and cost-effective sensor data generation in HAR. Our code is publicly available at https://github.com/ZikangLeng/AgentSense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07412v1">TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit embodied agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create photorealistic and dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains unclear. We introduce TwinOR, a framework for constructing photorealistic, dynamic digital twins of ORs for embodied AI research. The system reconstructs static geometry from pre-scan videos and continuously models human and equipment motion through multi-view perception of OR activities. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter level accuracy while preserving dynamic interaction across surgical workflows, enabling realistic renderings and a virtual playground for embodied AI systems. In our experiments, TwinOR simulates stereo and monocular sensor streams for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 on TwinOR-synthesized data achieve performance within their reported accuracy on real indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for perception and localization challenges. By establishing a real-to-sim pipeline for constructing dynamic, photorealistic digital twins of OR environments, TwinOR enables the safe, scalable, and data-efficient development and benchmarking of embodied AI, ultimately accelerating the deployment of embodied AI from sim-to-real.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07161v1">LLMscape</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted to NeurIPS 2025, Creative AI Track
    </div>
    <details class="paper-abstract">
      LLMscape is an interactive installation that investigates how humans and AI construct meaning under shared conditions of uncertainty. Within a mutable, projection-mapped landscape, human participants reshape the world and engage with multiple AI agents, each developing incomplete and provisional accounts of their environment. Exhibited in Shanghai and continually evolving, the work positions AI not as deterministic tools but as embodied co-witnesses to an unstable world, examining the parallels between human and artificial meaning-making and inviting reflection on our shared epistemic limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07682v1">Designing and Evaluating Malinowski's Lens: An AI-Native Educational Game for Ethnographic Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 21 pages, 8 figures. Full preprint version; shorter version in preparation
    </div>
    <details class="paper-abstract">
      This study introduces 'Malinowski's Lens', the first AI-native educational game for anthropology that transforms Bronislaw Malinowski's 'Argonauts of the Western Pacific' (1922) into an interactive learning experience. The system combines Retrieval-Augmented Generation with DALL-E 3 text-to-image generation, creating consistent VGA-style visuals as players embody Malinowski during his Trobriand Islands fieldwork (1915-1918). To address ethical concerns, indigenous peoples appear as silhouettes while Malinowski is detailed, prompting reflection on anthropological representation. Two validation studies confirmed effectiveness: Study 1 with 10 non-specialists showed strong learning outcomes (average quiz score 7.5/10) and excellent usability (SUS: 83/100). Study 2 with 4 expert anthropologists confirmed pedagogical value, with one senior researcher discovering "new aspects" of Malinowski's work through gameplay. The findings demonstrate that AI-driven educational games can effectively convey complex anthropological concepts while sparking disciplinary curiosity. This study advances AI-native educational game design and provides a replicable model for transforming academic texts into engaging interactive experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06619v1">How Do VLAs Effectively Inherit from VLMs?</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Vision-language-action (VLA) models hold the promise to attain generalizable embodied control. To achieve this, a pervasive paradigm is to leverage the rich vision-semantic priors of large vision-language models (VLMs). However, the fundamental question persists: How do VLAs effectively inherit the prior knowledge from VLMs? To address this critical question, we introduce a diagnostic benchmark, GrinningFace, an emoji tabletop manipulation task where the robot arm is asked to place objects onto printed emojis corresponding to language instructions. This task design is particularly revealing -- knowledge associated with emojis is ubiquitous in Internet-scale datasets used for VLM pre-training, yet emojis themselves are largely absent from standard robotics datasets. Consequently, they provide a clean proxy: successful task completion indicates effective transfer of VLM priors to embodied control. We implement this diagnostic task in both simulated environment and a real robot, and compare various promising techniques for knowledge transfer. Specifically, we investigate the effects of parameter-efficient fine-tuning, VLM freezing, co-training, predicting discretized actions, and predicting latent actions. Through systematic evaluation, our work not only demonstrates the critical importance of preserving VLM priors for the generalization of VLA but also establishes guidelines for future research in developing truly generalizable embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06185v1">Dataforge: A Data Agent Platform for Autonomous Data Engineering</a></div>
    <div class="paper-meta">
      📅 2025-11-09
    </div>
    <details class="paper-abstract">
      The growing demand for AI applications in fields such as materials discovery, molecular modeling, and climate science has made data preparation an important but labor-intensive step. Raw data from diverse sources must be cleaned, normalized, and transformed to become AI-ready, while effective feature transformation and selection are essential for efficient training and inference. To address the challenges of scalability and expertise dependence, we present Data Agent, a fully autonomous system specialized for tabular data. Leveraging large language model (LLM) reasoning and grounded validation, Data Agent automatically performs data cleaning, hierarchical routing, and feature-level optimization through dual feedback loops. It embodies three core principles: automatic, safe, and non-expert friendly, which ensure end-to-end reliability without human supervision. This demo showcases the first practical realization of an autonomous Data Agent, illustrating how raw data can be transformed "From Data to Better Data."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05936v1">10 Open Challenges Steering the Future of Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2025-11-08
      | 💬 AAAI 2026 (Senior Track)
    </div>
    <details class="paper-abstract">
      Due to their ability of follow natural language instructions, vision-language-action (VLA) models are increasingly prevalent in the embodied AI arena, following the widespread success of their precursors -- LLMs and VLMs. In this paper, we discuss 10 principal milestones in the ongoing development of VLA models -- multimodality, reasoning, data, evaluation, cross-robot action generalization, efficiency, whole-body coordination, safety, agents, and coordination with humans. Furthermore, we discuss the emerging trends of using spatial understanding, modeling world dynamics, post training, and data synthesis -- all aiming to reach these milestones. Through these discussions, we hope to bring attention to the research avenues that may accelerate the development of VLA models into wider acceptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05304v1">psiUnity: A Platform for Multimodal Data-Driven XR</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Extended reality (XR) research increasingly relies on the ability to stream and synchronize multimodal data between headsets and immersive applications for data-driven interaction and experimentation. However, developers face a critical gap: the Platform for Situated Intelligence (psi), which excels at deterministic temporal alignment and multimodal data management, has been largely inaccessible to the dominant Unity/MRTK ecosystem used for HoloLens development. We introduce psiUnity, an open-source C# integration that bridges psi's .NET libraries with Unity 2022.3 and MRTK3 for HoloLens 2. psiUnity enables bidirectional, real-time streaming of head pose, hand tracking, gaze, IMU, audio, and depth sensor data (AHAT and long-throw) with microsecond-level temporal precision, allowing Unity applications to both consume and produce synchronized multimodal data streams. By embedding psi's native serialization, logging, and temporal coordination directly within Unity's architecture, psiUnity extends psi beyond its previous StereoKit limitations and empowers the HRI, HCI, and embodied-AI communities to develop reproducible, data-driven XR interactions and experiments within the familiar Unity environment. The integration is available at https://github.com/sailgt/psiUnity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04976v1">iFlyBot-VLM Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      We introduce iFlyBot-VLM, a general-purpose Vision-Language Model (VLM) used to improve the domain of Embodied Intelligence. The central objective of iFlyBot-VLM is to bridge the cross-modal semantic gap between high-dimensional environmental perception and low-level robotic motion control. To this end, the model abstracts complex visual and spatial information into a body-agnostic and transferable Operational Language, thereby enabling seamless perception-action closed-loop coordination across diverse robotic platforms. The architecture of iFlyBot-VLM is systematically designed to realize four key functional capabilities essential for embodied intelligence: 1) Spatial Understanding and Metric Reasoning; 2) Interactive Target Grounding; 3) Action Abstraction and Control Parameter Generation; 4) Task Planning and Skill Sequencing. We envision iFlyBot-VLM as a scalable and generalizable foundation model for embodied AI, facilitating the progression from specialized task-oriented systems toward generalist, cognitively capable agents. We conducted evaluations on 10 current mainstream embodied intelligence-related VLM benchmark datasets, such as Blink and Where2Place, and achieved optimal performance while preserving the model's general capabilities. We will publicly release both the training data and model weights to foster further research and development in the field of Embodied Intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05304v1">psiUnity: A Platform for Multimodal Data-Driven XR</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Extended reality (XR) research increasingly relies on the ability to stream and synchronize multimodal data between headsets and immersive applications for data-driven interaction and experimentation. However, developers face a critical gap: the Platform for Situated Intelligence (psi), which excels at deterministic temporal alignment and multimodal data management, has been largely inaccessible to the dominant Unity/MRTK ecosystem used for HoloLens development. We introduce psiUnity, an open-source C# integration that bridges psi's .NET libraries with Unity 2022.3 and MRTK3 for HoloLens 2. psiUnity enables bidirectional, real-time streaming of head pose, hand tracking, gaze, IMU, audio, and depth sensor data (AHAT and long-throw) with microsecond-level temporal precision, allowing Unity applications to both consume and produce synchronized multimodal data streams. By embedding psi's native serialization, logging, and temporal coordination directly within Unity's architecture, psiUnity extends psi beyond its previous StereoKit limitations and empowers the HRI, HCI, and embodied-AI communities to develop reproducible, data-driven XR interactions and experiments within the familiar Unity environment. The integration is available at https://github.com/sailgt/psiUnity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04976v1">iFlyBot-VLM Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      We introduce iFlyBot-VLM, a general-purpose Vision-Language Model (VLM) used to improve the domain of Embodied Intelligence. The central objective of iFlyBot-VLM is to bridge the cross-modal semantic gap between high-dimensional environmental perception and low-level robotic motion control. To this end, the model abstracts complex visual and spatial information into a body-agnostic and transferable Operational Language, thereby enabling seamless perception-action closed-loop coordination across diverse robotic platforms. The architecture of iFlyBot-VLM is systematically designed to realize four key functional capabilities essential for embodied intelligence: 1) Spatial Understanding and Metric Reasoning; 2) Interactive Target Grounding; 3) Action Abstraction and Control Parameter Generation; 4) Task Planning and Skill Sequencing. We envision iFlyBot-VLM as a scalable and generalizable foundation model for embodied AI, facilitating the progression from specialized task-oriented systems toward generalist, cognitively capable agents. We conducted evaluations on 10 current mainstream embodied intelligence-related VLM benchmark datasets, such as Blink and Where2Place, and achieved optimal performance while preserving the model's general capabilities. We will publicly release both the training data and model weights to foster further research and development in the field of Embodied Intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02851v2">Action Deviation-Aware Inference for Low-Latency Wireless Robots</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      To support latency-sensitive AI applications ranging from autonomous driving to industrial robot manipulation, 6G envisions distributed ML with computational resources in mobile, edge, and cloud connected over hyper-reliable low-latency communication (HRLLC). In this setting, speculative decoding can facilitate collaborative inference of models distributively deployed: a lightweight on-device model locally generates drafts while a more capable remote target model on a server verifies and corrects them in parallel with speculative sampling, thus resulting in lower latency without compromising accuracy. However, unlike autoregressive text generation, behavior cloning policies, typically used for embodied AI applications, cannot parallelize verification and correction for multiple drafts as each generated action depends on observation updated by a previous action. To this end, we propose Action Deviation-Aware Hybrid Inference (ADAHI), wherein drafts are selectively transmitted and verified based on action deviation, which has a strong correlation with action's rejection probability by the target model. By invoking server operation only when necessary, communication and computational overhead can be reduced while accuracy gain from speculative sampling is preserved. Experiments on our testbed show that ADAHI reduces transmission and server operations by approximately 40%, lowers end-to-end latency by 39.2%, and attains up to 97.2% of the task-success rate of baseline that invokes speculative sampling for every draft embedding vector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03997v1">PhysCorr: Dual-Reward DPO for Physics-Constrained Text-to-Video Generation with Automated Preference Selection</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Recent advances in text-to-video generation have achieved impressive perceptual quality, yet generated content often violates fundamental principles of physical plausibility - manifesting as implausible object dynamics, incoherent interactions, and unrealistic motion patterns. Such failures hinder the deployment of video generation models in embodied AI, robotics, and simulation-intensive domains. To bridge this gap, we propose PhysCorr, a unified framework for modeling, evaluating, and optimizing physical consistency in video generation. Specifically, we introduce PhysicsRM, the first dual-dimensional reward model that quantifies both intra-object stability and inter-object interactions. On this foundation, we develop PhyDPO, a novel direct preference optimization pipeline that leverages contrastive feedback and physics-aware reweighting to guide generation toward physically coherent outputs. Our approach is model-agnostic and scalable, enabling seamless integration into a wide range of video diffusion and transformer-based backbones. Extensive experiments across multiple benchmarks demonstrate that PhysCorr achieves significant improvements in physical realism while preserving visual fidelity and semantic alignment. This work takes a critical step toward physically grounded and trustworthy video generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03992v1">CaRF: Enhancing Multi-View Consistency in Referring 3D Gaussian Splatting Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Referring 3D Gaussian Splatting Segmentation (R3DGS) aims to interpret free-form language expressions and localize the corresponding 3D regions in Gaussian fields. While recent advances have introduced cross-modal alignment between language and 3D geometry, existing pipelines still struggle with cross-view consistency due to their reliance on 2D rendered pseudo supervision and view specific feature learning. In this work, we present Camera Aware Referring Field (CaRF), a fully differentiable framework that operates directly in the 3D Gaussian space and achieves multi view consistency. Specifically, CaRF introduces Gaussian Field Camera Encoding (GFCE), which incorporates camera geometry into Gaussian text interactions to explicitly model view dependent variations and enhance geometric reasoning. Building on this, In Training Paired View Supervision (ITPVS) is proposed to align per Gaussian logits across calibrated views during training, effectively mitigating single view overfitting and exposing inter view discrepancies for optimization. Extensive experiments on three representative benchmarks demonstrate that CaRF achieves average improvements of 16.8%, 4.3%, and 2.0% in mIoU over state of the art methods on the Ref LERF, LERF OVS, and 3D OVS datasets, respectively. Moreover, this work promotes more reliable and view consistent 3D scene understanding, with potential benefits for embodied AI, AR/VR interaction, and autonomous perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.03497v1">ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Agentic AI systems and Physical or Embodied AI systems have been two key research verticals at the forefront of Artificial Intelligence and Robotics, with Model Context Protocol (MCP) increasingly becoming a key component and enabler of agentic applications. However, the literature at the intersection of these verticals, i.e., Agentic Embodied AI, remains scarce. This paper introduces an MCP server for analyzing ROS and ROS 2 bags, allowing for analyzing, visualizing and processing robot data with natural language through LLMs and VLMs. We describe specific tooling built with robotics domain knowledge, with our initial release focused on mobile robotics and supporting natively the analysis of trajectories, laser scan data, transforms, or time series data. This is in addition to providing an interface to standard ROS 2 CLI tools ("ros2 bag list" or "ros2 bag info"), as well as the ability to filter bags with a subset of topics or trimmed in time. Coupled with the MCP server, we provide a lightweight UI that allows the benchmarking of the tooling with different LLMs, both proprietary (Anthropic, OpenAI) and open-source (through Groq). Our experimental results include the analysis of tool calling capabilities of eight different state-of-the-art LLM/VLM models, both proprietary and open-source, large and small. Our experiments indicate that there is a large divide in tool calling capabilities, with Kimi K2 and Claude Sonnet 4 demonstrating clearly superior performance. We also conclude that there are multiple factors affecting the success rates, from the tool description schema to the number of arguments, as well as the number of tools available to the models. The code is available with a permissive license at https://github.com/binabik-ai/mcp-rosbags.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.02979v1">Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companion Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Submitted to Neurips 2025 workshop: LLM Persona Workshop
    </div>
    <details class="paper-abstract">
      The design and application of LLM-based personas in AI companionship is a rapidly expanding but fragmented field, spanning from virtual emotional companions and game NPCs to embodied functional robots. This diversity in objectives, modality, and technical stacks creates an urgent need for a unified framework. To address this gap, this paper systematizes the field by proposing a Four-Quadrant Technical Taxonomy for AI companion applications. The framework is structured along two critical axes: Virtual vs. Embodied and Emotional Companionship vs. Functional Augmentation. Quadrant I (Virtual Companionship) explores virtual idols, romantic companions, and story characters, introducing a four-layer technical framework to analyze their challenges in maintaining long-term emotional consistency. Quadrant II (Functional Virtual Assistants) analyzes AI applications in work, gaming, and mental health, highlighting the shift from "feeling" to "thinking and acting" and pinpointing key technologies like enterprise RAG and on-device inference. Quadrants III & IV (Embodied Intelligence) shift from the virtual to the physical world, analyzing home robots and vertical-domain assistants, revealing core challenges in symbol grounding, data privacy, and ethical liability. This taxonomy provides not only a systematic map for researchers and developers to navigate the complex persona design space but also a basis for policymakers to identify and address the unique risks inherent in different application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02851v2">Action Deviation-Aware Inference for Low-Latency Wireless Robots</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      To support latency-sensitive AI applications ranging from autonomous driving to industrial robot manipulation, 6G envisions distributed ML with computational resources in mobile, edge, and cloud connected over hyper-reliable low-latency communication (HRLLC). In this setting, speculative decoding can facilitate collaborative inference of models distributively deployed: a lightweight on-device model locally generates drafts while a more capable remote target model on a server verifies and corrects them in parallel with speculative sampling, thus resulting in lower latency without compromising accuracy. However, unlike autoregressive text generation, behavior cloning policies, typically used for embodied AI applications, cannot parallelize verification and correction for multiple drafts as each generated action depends on observation updated by a previous action. To this end, we propose Action Deviation-Aware Hybrid Inference (ADAHI), wherein drafts are selectively transmitted and verified based on action deviation, which has a strong correlation with action's rejection probability by the target model. By invoking server operation only when necessary, communication and computational overhead can be reduced while accuracy gain from speculative sampling is preserved. Experiments on our testbed show that ADAHI reduces transmission and server operations by approximately 40%, lowers end-to-end latency by 39.2%, and attains up to 97.2% of the task-success rate of baseline that invokes speculative sampling for every draft embedding vector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04157v1">Are We Aligned? A Preliminary Investigation of the Alignment of Responsible AI Values between LLMs and Human Judgment</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in software engineering tasks such as requirements elicitation, design, and evaluation, raising critical questions regarding their alignment with human judgments on responsible AI values. This study investigates how closely LLMs' value preferences align with those of two human groups: a US-representative sample and AI practitioners. We evaluate 23 LLMs across four tasks: (T1) selecting key responsible AI values, (T2) rating their importance in specific contexts, (T3) resolving trade-offs between competing values, and (T4) prioritizing software requirements that embody those values. The results show that LLMs generally align more closely with AI practitioners than with the US-representative sample, emphasizing fairness, privacy, transparency, safety, and accountability. However, inconsistencies appear between the values that LLMs claim to uphold (Tasks 1-3) and the way they prioritize requirements (Task 4), revealing gaps in faithfulness between stated and applied behavior. These findings highlight the practical risk of relying on LLMs in requirements engineering without human oversight and motivate the need for systematic approaches to benchmark, interpret, and monitor value alignment in AI-assisted software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03997v1">PhysCorr: Dual-Reward DPO for Physics-Constrained Text-to-Video Generation with Automated Preference Selection</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Recent advances in text-to-video generation have achieved impressive perceptual quality, yet generated content often violates fundamental principles of physical plausibility - manifesting as implausible object dynamics, incoherent interactions, and unrealistic motion patterns. Such failures hinder the deployment of video generation models in embodied AI, robotics, and simulation-intensive domains. To bridge this gap, we propose PhysCorr, a unified framework for modeling, evaluating, and optimizing physical consistency in video generation. Specifically, we introduce PhysicsRM, the first dual-dimensional reward model that quantifies both intra-object stability and inter-object interactions. On this foundation, we develop PhyDPO, a novel direct preference optimization pipeline that leverages contrastive feedback and physics-aware reweighting to guide generation toward physically coherent outputs. Our approach is model-agnostic and scalable, enabling seamless integration into a wide range of video diffusion and transformer-based backbones. Extensive experiments across multiple benchmarks demonstrate that PhysCorr achieves significant improvements in physical realism while preserving visual fidelity and semantic alignment. This work takes a critical step toward physically grounded and trustworthy video generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03992v1">CaRF: Enhancing Multi-View Consistency in Referring 3D Gaussian Splatting Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Referring 3D Gaussian Splatting Segmentation (R3DGS) aims to interpret free-form language expressions and localize the corresponding 3D regions in Gaussian fields. While recent advances have introduced cross-modal alignment between language and 3D geometry, existing pipelines still struggle with cross-view consistency due to their reliance on 2D rendered pseudo supervision and view specific feature learning. In this work, we present Camera Aware Referring Field (CaRF), a fully differentiable framework that operates directly in the 3D Gaussian space and achieves multi view consistency. Specifically, CaRF introduces Gaussian Field Camera Encoding (GFCE), which incorporates camera geometry into Gaussian text interactions to explicitly model view dependent variations and enhance geometric reasoning. Building on this, In Training Paired View Supervision (ITPVS) is proposed to align per Gaussian logits across calibrated views during training, effectively mitigating single view overfitting and exposing inter view discrepancies for optimization. Extensive experiments on three representative benchmarks demonstrate that CaRF achieves average improvements of 16.8%, 4.3%, and 2.0% in mIoU over state of the art methods on the Ref LERF, LERF OVS, and 3D OVS datasets, respectively. Moreover, this work promotes more reliable and view consistent 3D scene understanding, with potential benefits for embodied AI, AR/VR interaction, and autonomous perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03497v1">ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Agentic AI systems and Physical or Embodied AI systems have been two key research verticals at the forefront of Artificial Intelligence and Robotics, with Model Context Protocol (MCP) increasingly becoming a key component and enabler of agentic applications. However, the literature at the intersection of these verticals, i.e., Agentic Embodied AI, remains scarce. This paper introduces an MCP server for analyzing ROS and ROS 2 bags, allowing for analyzing, visualizing and processing robot data with natural language through LLMs and VLMs. We describe specific tooling built with robotics domain knowledge, with our initial release focused on mobile robotics and supporting natively the analysis of trajectories, laser scan data, transforms, or time series data. This is in addition to providing an interface to standard ROS 2 CLI tools ("ros2 bag list" or "ros2 bag info"), as well as the ability to filter bags with a subset of topics or trimmed in time. Coupled with the MCP server, we provide a lightweight UI that allows the benchmarking of the tooling with different LLMs, both proprietary (Anthropic, OpenAI) and open-source (through Groq). Our experimental results include the analysis of tool calling capabilities of eight different state-of-the-art LLM/VLM models, both proprietary and open-source, large and small. Our experiments indicate that there is a large divide in tool calling capabilities, with Kimi K2 and Claude Sonnet 4 demonstrating clearly superior performance. We also conclude that there are multiple factors affecting the success rates, from the tool description schema to the number of arguments, as well as the number of tools available to the models. The code is available with a permissive license at https://github.com/binabik-ai/mcp-rosbags.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03370v1">EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) in automated negotiation has set a high performance benchmark, but their computational cost and data privacy requirements render them unsuitable for many privacy-sensitive, on-device applications such as mobile assistants, embodied AI agents or private client interactions. While small language models (SLMs) offer a practical alternative, they suffer from a significant performance gap compared to LLMs in playing emotionally charged complex personas, especially for credit negotiation. This paper introduces EQ-Negotiator, a novel framework that bridges this capability gap using emotional personas. Its core is a reasoning system that integrates game theory with a Hidden Markov Model(HMM) to learn and track debtor emotional states online, without pre-training. This allows EQ-Negotiator to equip SLMs with the strategic intelligence to counter manipulation while de-escalating conflict and upholding ethical standards. Through extensive agent-to-agent simulations across diverse credit negotiation scenarios, including adversarial debtor strategies like cheating, threatening, and playing the victim, we show that a 7B parameter language model with EQ-Negotiator achieves better debt recovery and negotiation efficiency than baseline LLMs more than 10 times its size. This work advances persona modeling from descriptive character profiles to dynamic emotional architectures that operate within privacy constraints. Besides, this paper establishes that strategic emotional intelligence, not raw model scale, is the critical factor for success in automated negotiation, paving the way for effective, ethical, and privacy-preserving AI negotiators that can operate on the edge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.02071v1">Human-AI Co-Embodied Intelligence for Scientific Experimentation and Manufacturing</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Scientific experiment and manufacture rely on complex, multi-step procedures that demand continuous human expertise for precise execution and decision-making. Despite advances in machine learning and automation, conventional models remain confined to virtual domains, while real-world experiment and manufacture still rely on human supervision and expertise. This gap between machine intelligence and physical execution limits reproducibility, scalability, and accessibility across scientific and manufacture workflows. Here, we introduce human-AI co-embodied intelligence, a new form of physical AI that unites human users, agentic AI, and wearable hardware into an integrated system for real-world experiment and intelligent manufacture. In this paradigm, humans provide precise execution and control, while agentic AI contributes memory, contextual reasoning, adaptive planning, and real-time feedback. The wearable interface continuously captures the experimental and manufacture processes, facilitates seamless communication between humans and AI for corrective guidance and interpretable collaboration. As a demonstration, we present Agentic-Physical Experimentation (APEX) system, coupling agentic reasoning with physical execution through mixed-reality. APEX observes and interprets human actions, aligns them with standard operating procedures, provides 3D visual guidance, and analyzes every step. Implemented in a cleanroom for flexible electronics fabrication, APEX system achieves context-aware reasoning with accuracy exceeding general multimodal large language models, corrects errors in real time, and transfers expertise to beginners. These results establish a new class of agentic-physical-human intelligence that extends agentic reasoning beyond computation into the physical domain, transforming scientific research and manufacturing into autonomous, traceable, interpretable, and scalable processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03497v1">ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Agentic AI systems and Physical or Embodied AI systems have been two key research verticals at the forefront of Artificial Intelligence and Robotics, with Model Context Protocol (MCP) increasingly becoming a key component and enabler of agentic applications. However, the literature at the intersection of these verticals, i.e., Agentic Embodied AI, remains scarce. This paper introduces an MCP server for analyzing ROS and ROS 2 bags, allowing for analyzing, visualizing and processing robot data with natural language through LLMs and VLMs. We describe specific tooling built with robotics domain knowledge, with our initial release focused on mobile robotics and supporting natively the analysis of trajectories, laser scan data, transforms, or time series data. This is in addition to providing an interface to standard ROS 2 CLI tools ("ros2 bag list" or "ros2 bag info"), as well as the ability to filter bags with a subset of topics or trimmed in time. Coupled with the MCP server, we provide a lightweight UI that allows the benchmarking of the tooling with different LLMs, both proprietary (Anthropic, OpenAI) and open-source (through Groq). Our experimental results include the analysis of tool calling capabilities of eight different state-of-the-art LLM/VLM models, both proprietary and open-source, large and small. Our experiments indicate that there is a large divide in tool calling capabilities, with Kimi K2 and Claude Sonnet 4 demonstrating clearly superior performance. We also conclude that there are multiple factors affecting the success rates, from the tool description schema to the number of arguments, as well as the number of tools available to the models. The code is available with a permissive license at https://github.com/binabik-ai/mcp-rosbags.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03370v1">EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) in automated negotiation has set a high performance benchmark, but their computational cost and data privacy requirements render them unsuitable for many privacy-sensitive, on-device applications such as mobile assistants, embodied AI agents or private client interactions. While small language models (SLMs) offer a practical alternative, they suffer from a significant performance gap compared to LLMs in playing emotionally charged complex personas, especially for credit negotiation. This paper introduces EQ-Negotiator, a novel framework that bridges this capability gap using emotional personas. Its core is a reasoning system that integrates game theory with a Hidden Markov Model(HMM) to learn and track debtor emotional states online, without pre-training. This allows EQ-Negotiator to equip SLMs with the strategic intelligence to counter manipulation while de-escalating conflict and upholding ethical standards. Through extensive agent-to-agent simulations across diverse credit negotiation scenarios, including adversarial debtor strategies like cheating, threatening, and playing the victim, we show that a 7B parameter language model with EQ-Negotiator achieves better debt recovery and negotiation efficiency than baseline LLMs more than 10 times its size. This work advances persona modeling from descriptive character profiles to dynamic emotional architectures that operate within privacy constraints. Besides, this paper establishes that strategic emotional intelligence, not raw model scale, is the critical factor for success in automated negotiation, paving the way for effective, ethical, and privacy-preserving AI negotiators that can operate on the edge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00529v2">On Improvisation and Open-Endedness: Insights for Experiential AI</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 Submitted to AAAI 2026 Creative AI for Live Interactive Performances Workshop (CLIP) as a work-in-progress paper
    </div>
    <details class="paper-abstract">
      Improvisation-the art of spontaneous creation that unfolds moment-to-moment without a scripted outcome-requires practitioners to continuously sense, adapt, and create anew. It is a fundamental mode of human creativity spanning music, dance, and everyday life. The open-ended nature of improvisation produces a stream of novel, unrepeatable moments-an aspect highly valued in artistic creativity. In parallel, open-endedness (OE)-a system's capacity for unbounded novelty and endless "interestingness"-is exemplified in natural or cultural evolution and has been considered "the last grand challenge" in artificial life (ALife). The rise of generative AI now raises the question in computational creativity (CC) research: What makes a "good" improvisation for AI? Can AI learn to improvise in a genuinely open-ended way? In this work-in-progress paper, we report insights from in-depth interviews with 6 experts in improvisation across dance, music, and contact improvisation. We draw systemic connections between human improvisational arts and the design of future experiential AI agents that could improvise alone or alongside humans-or even with other AI agents-embodying qualities of improvisation drawn from practice: active listening (umwelt and awareness), being in the time (mindfulness and ephemerality), embracing the unknown (source of randomness and serendipity), non-judgmental flow (acceptance and dynamical stability, balancing structure and surprise (unpredictable criticality at edge of chaos), imaginative metaphor (synaesthesia and planning), empathy, trust, boundary, and care (mutual theory of mind), and playfulness and intrinsic motivation (maintaining interestingness).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02912v3">Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Published in the Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Scaling Environments for Agents (SEA). Additionally accepted for presentation in the NeurIPS 2025 Workshop: Embodied World Models for Decision Making (EWM) and the NeurIPS 2025 Workshop: Optimization for Machine Learning (OPT)
    </div>
    <details class="paper-abstract">
      Robust coordination is critical for effective decision-making in multi-agent systems, especially under partial observability. A central question in Multi-Agent Reinforcement Learning (MARL) is whether to engineer communication protocols or learn them end-to-end. We investigate this dichotomy using embodied world models. We propose and compare two communication strategies for a cooperative task-allocation problem. The first, Learned Direct Communication (LDC), learns a protocol end-to-end. The second, Intention Communication, uses an engineered inductive bias: a compact, learned world model, the Imagined Trajectory Generation Module (ITGM), which uses the agent's own policy to simulate future states. A Message Generation Network (MGN) then compresses this plan into a message. We evaluate these approaches on goal-directed interaction in a grid world, a canonical abstraction for embodied AI problems, while scaling environmental complexity. Our experiments reveal that while emergent communication is viable in simple settings, the engineered, world model-based approach shows superior performance, sample efficiency, and scalability as complexity increases. These findings advocate for integrating structured, predictive models into MARL agents to enable active, goal-driven coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02225v1">Learning Interactive World Model for Object-Centric Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Agents that understand objects and their interactions can learn policies that are more robust and transferable. However, most object-centric RL methods factor state by individual objects while leaving interactions implicit. We introduce the Factored Interactive Object-Centric World Model (FIOC-WM), a unified framework that learns structured representations of both objects and their interactions within a world model. FIOC-WM captures environment dynamics with disentangled and modular representations of object interactions, improving sample efficiency and generalization for policy learning. Concretely, FIOC-WM first learns object-centric latents and an interaction structure directly from pixels, leveraging pre-trained vision encoders. The learned world model then decomposes tasks into composable interaction primitives, and a hierarchical policy is trained on top: a high level selects the type and order of interactions, while a low level executes them. On simulated robotic and embodied-AI benchmarks, FIOC-WM improves policy-learning sample efficiency and generalization over world-model baselines, indicating that explicit, modular interaction learning is crucial for robust control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02979v1">Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companion Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Submitted to Neurips 2025 workshop: LLM Persona Workshop
    </div>
    <details class="paper-abstract">
      The design and application of LLM-based personas in AI companionship is a rapidly expanding but fragmented field, spanning from virtual emotional companions and game NPCs to embodied functional robots. This diversity in objectives, modality, and technical stacks creates an urgent need for a unified framework. To address this gap, this paper systematizes the field by proposing a Four-Quadrant Technical Taxonomy for AI companion applications. The framework is structured along two critical axes: Virtual vs. Embodied and Emotional Companionship vs. Functional Augmentation. Quadrant I (Virtual Companionship) explores virtual idols, romantic companions, and story characters, introducing a four-layer technical framework to analyze their challenges in maintaining long-term emotional consistency. Quadrant II (Functional Virtual Assistants) analyzes AI applications in work, gaming, and mental health, highlighting the shift from "feeling" to "thinking and acting" and pinpointing key technologies like enterprise RAG and on-device inference. Quadrants III & IV (Embodied Intelligence) shift from the virtual to the physical world, analyzing home robots and vertical-domain assistants, revealing core challenges in symbol grounding, data privacy, and ethical liability. This taxonomy provides not only a systematic map for researchers and developers to navigate the complex persona design space but also a basis for policymakers to identify and address the unique risks inherent in different application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02374v1">AyurParam: A State-of-the-Art Bilingual Language Model for Ayurveda</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Current large language models excel at broad, general-purpose tasks, but consistently underperform when exposed to highly specialized domains that require deep cultural, linguistic, and subject-matter expertise. In particular, traditional medical systems such as Ayurveda embody centuries of nuanced textual and clinical knowledge that mainstream LLMs fail to accurately interpret or apply. We introduce AyurParam-2.9B, a domain-specialized, bilingual language model fine-tuned from Param-1-2.9B using an extensive, expertly curated Ayurveda dataset spanning classical texts and clinical guidance. AyurParam's dataset incorporates context-aware, reasoning, and objective-style Q&A in both English and Hindi, with rigorous annotation protocols for factual precision and instructional clarity. Benchmarked on BhashaBench-Ayur, AyurParam not only surpasses all open-source instruction-tuned models in its size class (1.5--3B parameters), but also demonstrates competitive or superior performance compared to much larger models. The results from AyurParam highlight the necessity for authentic domain adaptation and high-quality supervision in delivering reliable, culturally congruent AI for specialized medical knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02912v3">Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Published in the Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Scaling Environments for Agents (SEA). Additionally accepted for presentation in the NeurIPS 2025 Workshop: Embodied World Models for Decision Making (EWM) and the NeurIPS 2025 Workshop: Optimization for Machine Learning (OPT)
    </div>
    <details class="paper-abstract">
      Robust coordination is critical for effective decision-making in multi-agent systems, especially under partial observability. A central question in Multi-Agent Reinforcement Learning (MARL) is whether to engineer communication protocols or learn them end-to-end. We investigate this dichotomy using embodied world models. We propose and compare two communication strategies for a cooperative task-allocation problem. The first, Learned Direct Communication (LDC), learns a protocol end-to-end. The second, Intention Communication, uses an engineered inductive bias: a compact, learned world model, the Imagined Trajectory Generation Module (ITGM), which uses the agent's own policy to simulate future states. A Message Generation Network (MGN) then compresses this plan into a message. We evaluate these approaches on goal-directed interaction in a grid world, a canonical abstraction for embodied AI problems, while scaling environmental complexity. Our experiments reveal that while emergent communication is viable in simple settings, the engineered, world model-based approach shows superior performance, sample efficiency, and scalability as complexity increases. These findings advocate for integrating structured, predictive models into MARL agents to enable active, goal-driven coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02225v1">Learning Interactive World Model for Object-Centric Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Agents that understand objects and their interactions can learn policies that are more robust and transferable. However, most object-centric RL methods factor state by individual objects while leaving interactions implicit. We introduce the Factored Interactive Object-Centric World Model (FIOC-WM), a unified framework that learns structured representations of both objects and their interactions within a world model. FIOC-WM captures environment dynamics with disentangled and modular representations of object interactions, improving sample efficiency and generalization for policy learning. Concretely, FIOC-WM first learns object-centric latents and an interaction structure directly from pixels, leveraging pre-trained vision encoders. The learned world model then decomposes tasks into composable interaction primitives, and a hierarchical policy is trained on top: a high level selects the type and order of interactions, while a low level executes them. On simulated robotic and embodied-AI benchmarks, FIOC-WM improves policy-learning sample efficiency and generalization over world-model baselines, indicating that explicit, modular interaction learning is crucial for robust control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02071v1">Human-AI Co-Embodied Intelligence for Scientific Experimentation and Manufacturing</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Scientific experiment and manufacture rely on complex, multi-step procedures that demand continuous human expertise for precise execution and decision-making. Despite advances in machine learning and automation, conventional models remain confined to virtual domains, while real-world experiment and manufacture still rely on human supervision and expertise. This gap between machine intelligence and physical execution limits reproducibility, scalability, and accessibility across scientific and manufacture workflows. Here, we introduce human-AI co-embodied intelligence, a new form of physical AI that unites human users, agentic AI, and wearable hardware into an integrated system for real-world experiment and intelligent manufacture. In this paradigm, humans provide precise execution and control, while agentic AI contributes memory, contextual reasoning, adaptive planning, and real-time feedback. The wearable interface continuously captures the experimental and manufacture processes, facilitates seamless communication between humans and AI for corrective guidance and interpretable collaboration. As a demonstration, we present Agentic-Physical Experimentation (APEX) system, coupling agentic reasoning with physical execution through mixed-reality. APEX observes and interprets human actions, aligns them with standard operating procedures, provides 3D visual guidance, and analyzes every step. Implemented in a cleanroom for flexible electronics fabrication, APEX system achieves context-aware reasoning with accuracy exceeding general multimodal large language models, corrects errors in real time, and transfers expertise to beginners. These results establish a new class of agentic-physical-human intelligence that extends agentic reasoning beyond computation into the physical domain, transforming scientific research and manufacturing into autonomous, traceable, interpretable, and scalable processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01663v1">The Ghost in the Keys: A Disklavier Demo for Human-AI Musical Co-Creativity</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      While generative models for music composition are increasingly capable, their adoption by musicians is hindered by text-prompting, an asynchronous workflow disconnected from the embodied, responsive nature of instrumental performance. To address this, we introduce Aria-Duet, an interactive system facilitating a real-time musical duet between a human pianist and Aria, a state-of-the-art generative model, using a Yamaha Disklavier as a shared physical interface. The framework enables a turn-taking collaboration: the user performs, signals a handover, and the model generates a coherent continuation performed acoustically on the piano. Beyond describing the technical architecture enabling this low-latency interaction, we analyze the system's output from a musicological perspective, finding the model can maintain stylistic semantics and develop coherent phrasal ideas, demonstrating that such embodied systems can engage in musically sophisticated dialogue and open a promising new path for human-AI co-creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25760v2">Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Humans possess spatial reasoning abilities that enable them to understand spaces through multimodal observations, such as vision and sound. Large multimodal reasoning models extend these abilities by learning to perceive and reason, showing promising performance across diverse spatial tasks. However, systematic reviews and publicly available benchmarks for these models remain limited. In this survey, we provide a comprehensive review of multimodal spatial reasoning tasks with large models, categorizing recent progress in multimodal large language models (MLLMs) and introducing open benchmarks for evaluation. We begin by outlining general spatial reasoning, focusing on post-training techniques, explainability, and architecture. Beyond classical 2D tasks, we examine spatial relationship reasoning, scene and layout understanding, as well as visual question answering and grounding in 3D space. We also review advances in embodied AI, including vision-language navigation and action models. Additionally, we consider emerging modalities such as audio and egocentric video, which contribute to novel spatial understanding through new sensors. We believe this survey establishes a solid foundation and offers insights into the growing field of multimodal spatial reasoning. Updated information about this survey, codes and implementation of the open benchmarks can be found at https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00940v1">URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Constructing accurate digital twins of articulated objects is essential for robotic simulation training and embodied AI world model building, yet historically requires painstaking manual modeling or multi-stage pipelines. In this work, we propose \textbf{URDF-Anything}, an end-to-end automatic reconstruction framework based on a 3D multimodal large language model (MLLM). URDF-Anything utilizes an autoregressive prediction framework based on point-cloud and text multimodal input to jointly optimize geometric segmentation and kinematic parameter prediction. It implements a specialized $[SEG]$ token mechanism that interacts directly with point cloud features, enabling fine-grained part-level segmentation while maintaining consistency with the kinematic parameter predictions. Experiments on both simulated and real-world datasets demonstrate that our method significantly outperforms existing approaches regarding geometric segmentation (mIoU 17\% improvement), kinematic parameter prediction (average error reduction of 29\%), and physical executability (surpassing baselines by 50\%). Notably, our method exhibits excellent generalization ability, performing well even on objects outside the training set. This work provides an efficient solution for constructing digital twins for robotic simulation, significantly enhancing the sim-to-real transfer capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00940v1">URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Constructing accurate digital twins of articulated objects is essential for robotic simulation training and embodied AI world model building, yet historically requires painstaking manual modeling or multi-stage pipelines. In this work, we propose \textbf{URDF-Anything}, an end-to-end automatic reconstruction framework based on a 3D multimodal large language model (MLLM). URDF-Anything utilizes an autoregressive prediction framework based on point-cloud and text multimodal input to jointly optimize geometric segmentation and kinematic parameter prediction. It implements a specialized $[SEG]$ token mechanism that interacts directly with point cloud features, enabling fine-grained part-level segmentation while maintaining consistency with the kinematic parameter predictions. Experiments on both simulated and real-world datasets demonstrate that our method significantly outperforms existing approaches regarding geometric segmentation (mIoU 17\% improvement), kinematic parameter prediction (average error reduction of 29\%), and physical executability (surpassing baselines by 50\%). Notably, our method exhibits excellent generalization ability, performing well even on objects outside the training set. This work provides an efficient solution for constructing digital twins for robotic simulation, significantly enhancing the sim-to-real transfer capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25760v2">Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Humans possess spatial reasoning abilities that enable them to understand spaces through multimodal observations, such as vision and sound. Large multimodal reasoning models extend these abilities by learning to perceive and reason, showing promising performance across diverse spatial tasks. However, systematic reviews and publicly available benchmarks for these models remain limited. In this survey, we provide a comprehensive review of multimodal spatial reasoning tasks with large models, categorizing recent progress in multimodal large language models (MLLMs) and introducing open benchmarks for evaluation. We begin by outlining general spatial reasoning, focusing on post-training techniques, explainability, and architecture. Beyond classical 2D tasks, we examine spatial relationship reasoning, scene and layout understanding, as well as visual question answering and grounding in 3D space. We also review advances in embodied AI, including vision-language navigation and action models. Additionally, we consider emerging modalities such as audio and egocentric video, which contribute to novel spatial understanding through new sensors. We believe this survey establishes a solid foundation and offers insights into the growing field of multimodal spatial reasoning. Updated information about this survey, codes and implementation of the open benchmarks can be found at https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning.
    </details>
</div>
