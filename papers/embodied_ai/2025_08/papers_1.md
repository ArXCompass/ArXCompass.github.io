# embodied ai - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05614v1">OmniEAR: Benchmarking Agent Reasoning in Embodied Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Project Page: https://zju-real.github.io/OmniEmbodied Code: https://github.com/ZJU-REAL/OmniEmbodied
    </div>
    <details class="paper-abstract">
      Large language models excel at abstract reasoning but their capacity for embodied agent reasoning remains largely unexplored. We present OmniEAR, a comprehensive framework for evaluating how language models reason about physical interactions, tool usage, and multi-agent coordination in embodied tasks. Unlike existing benchmarks that provide predefined tool sets or explicit collaboration directives, OmniEAR requires agents to dynamically acquire capabilities and autonomously determine coordination strategies based on task demands. Through text-based environment representation, we model continuous physical properties and complex spatial relationships across 1,500 scenarios spanning household and industrial domains. Our systematic evaluation reveals severe performance degradation when models must reason from constraints: while achieving 85-96% success with explicit instructions, performance drops to 56-85% for tool reasoning and 63-85% for implicit collaboration, with compound tasks showing over 50% failure rates. Surprisingly, complete environmental information degrades coordination performance, indicating models cannot filter task-relevant constraints. Fine-tuning improves single-agent tasks dramatically (0.6% to 76.3%) but yields minimal multi-agent gains (1.5% to 5.5%), exposing fundamental architectural limitations. These findings demonstrate that embodied reasoning poses fundamentally different challenges than current models can address, establishing OmniEAR as a rigorous benchmark for evaluating and advancing embodied AI systems. Our code and data are included in the supplementary materials and will be open-sourced upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05543v1">CleanUpBench: Embodied Sweeping and Grasping Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Embodied AI benchmarks have advanced navigation, manipulation, and reasoning, but most target complex humanoid agents or large-scale simulations that are far from real-world deployment. In contrast, mobile cleaning robots with dual mode capabilities, such as sweeping and grasping, are rapidly emerging as realistic and commercially viable platforms. However, no benchmark currently exists that systematically evaluates these agents in structured, multi-target cleaning tasks, revealing a critical gap between academic research and real-world applications. We introduce CleanUpBench, a reproducible and extensible benchmark for evaluating embodied agents in realistic indoor cleaning scenarios. Built on NVIDIA Isaac Sim, CleanUpBench simulates a mobile service robot equipped with a sweeping mechanism and a six-degree-of-freedom robotic arm, enabling interaction with heterogeneous objects. The benchmark includes manually designed environments and one procedurally generated layout to assess generalization, along with a comprehensive evaluation suite covering task completion, spatial efficiency, motion quality, and control performance. To support comparative studies, we provide baseline agents based on heuristic strategies and map-based planning. CleanUpBench bridges the gap between low-level skill evaluation and full-scene testing, offering a scalable testbed for grounded, embodied intelligence in everyday settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11773v3">AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      A major challenge in developing robust and generalizable Human Activity Recognition (HAR) systems for smart homes is the lack of large and diverse labeled datasets. Variations in home layouts, sensor configurations, and individual behaviors further exacerbate this issue. To address this, we leverage the idea of embodied AI agents-virtual agents that perceive and act within simulated environments guided by internal world models. We introduce AgentSense, a virtual data generation pipeline in which agents live out daily routines in simulated smart homes, with behavior guided by Large Language Models (LLMs). The LLM generates diverse synthetic personas and realistic routines grounded in the environment, which are then decomposed into fine-grained actions. These actions are executed in an extended version of the VirtualHome simulator, which we augment with virtual ambient sensors that record the agents' activities. Our approach produces rich, privacy-preserving sensor data that reflects real-world diversity. We evaluate AgentSense on five real HAR datasets. Models pretrained on the generated data consistently outperform baselines, especially in low-resource settings. Furthermore, combining the generated virtual sensor data with a small amount of real data achieves performance comparable to training on full real-world datasets. These results highlight the potential of using LLM-guided embodied agents for scalable and cost-effective sensor data generation in HAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16402v2">IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. Code and data are released under [this https URL](https://github.com/AI45Lab/IS-Bench).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02029v4">RoboBrain 2.0 Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model, featuring a heterogeneous architecture with a vision encoder and a language model. Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models. In particular, it supports key real-world embodied AI capabilities, including spatial understanding (e.g., affordance prediction, spatial referring, trajectory forecasting) and temporal decision-making (e.g., closed-loop interaction, multi-agent long-horizon planning, and scene graph updating). This report details the model architecture, data construction, multi-stage training strategies, infrastructure and practical applications. We hope RoboBrain 2.0 advances embodied AI research and serves as a practical step toward building generalist embodied agents. The code, checkpoint and benchmark are available at https://superrobobrain.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21045v2">Reconstructing 4D Spatial Intelligence: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 Project page: https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence
    </div>
    <details class="paper-abstract">
      Reconstructing 4D spatial intelligence from visual observations has long been a central yet challenging task in computer vision, with broad real-world applications. These range from entertainment domains like movies, where the focus is often on reconstructing fundamental visual elements, to embodied AI, which emphasizes interaction modeling and physical realism. Fueled by rapid advances in 3D representations and deep learning architectures, the field has evolved quickly, outpacing the scope of previous surveys. Additionally, existing surveys rarely offer a comprehensive analysis of the hierarchical structure of 4D scene reconstruction. To address this gap, we present a new perspective that organizes existing methods into five progressive levels of 4D spatial intelligence: (1) Level 1 -- reconstruction of low-level 3D attributes (e.g., depth, pose, and point maps); (2) Level 2 -- reconstruction of 3D scene components (e.g., objects, humans, structures); (3) Level 3 -- reconstruction of 4D dynamic scenes; (4) Level 4 -- modeling of interactions among scene components; and (5) Level 5 -- incorporation of physical laws and constraints. We conclude the survey by discussing the key challenges at each level and highlighting promising directions for advancing toward even richer levels of 4D spatial intelligence. To track ongoing developments, we maintain an up-to-date project page: https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12974v3">Exploring 3D Reasoning-Driven Planning: From Implicit Human Intentions to Route-Aware Activity Planning</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      3D task planning has attracted increasing attention in human-robot interaction and embodied AI thanks to the recent advances in multimodal learning. However, most existing studies are facing two common challenges: 1) heavy reliance on explicit instructions with little reasoning on implicit user intention; 2) negligence of inter-step route planning on robot moves. We address the above challenges by proposing 3D Reasoning-Driven Planning, a novel 3D task that reasons the intended activities from implicit instructions and decomposes them into steps with inter-step routes and planning under the guidance of fine-grained 3D object shapes and locations from scene segmentation. We tackle the new 3D task from two perspectives. First, we construct ReasonPlan3D, a large-scale benchmark that covers diverse 3D scenes with rich implicit instructions and detailed annotations for multi-step task planning, inter-step route planning, and fine-grained segmentation. Second, we design a novel framework that introduces progressive plan generation with contextual consistency across multiple steps, as well as a scene graph that is updated dynamically for capturing critical objects and their spatial relations. Extensive experiments demonstrate the effectiveness of our benchmark and framework in reasoning activities from implicit human instructions, producing accurate stepwise task plans and seamlessly integrating route planning for multi-step moves. The dataset and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13927v4">Multimodal 3D Reasoning Segmentation with Complex Scenes</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      The recent development in multimodal learning has greatly advanced the research in 3D scene understanding in various real-world tasks such as embodied AI. However, most existing studies are facing two common challenges: 1) they are short of reasoning ability for interaction and interpretation of human intentions and 2) they focus on scenarios with single-category objects and over-simplified textual descriptions and neglect multi-object scenarios with complicated spatial relations among objects. We address the above challenges by proposing a 3D reasoning segmentation task for reasoning segmentation with multiple objects in scenes. The task allows producing 3D segmentation masks and detailed textual explanations as enriched by 3D spatial relations among objects. To this end, we create ReasonSeg3D, a large-scale and high-quality benchmark that integrates 3D segmentation masks and 3D spatial relations with generated question-answer pairs. In addition, we design MORE3D, a novel 3D reasoning network that works with queries of multiple objects and is tailored for 3D scene understanding. MORE3D learns detailed explanations on 3D relations and employs them to capture spatial information of objects and reason textual outputs. Extensive experiments show that MORE3D excels in reasoning and segmenting complex multi-object 3D scenes. In addition, the created ReasonSeg3D offers a valuable platform for future exploration of 3D reasoning segmentation. The data and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19516v2">Boosting Robotic Manipulation Generalization with Minimal Costly Data</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      The growing adoption of Vision-Language-Action (VLA) models in embodied AI intensifies the demand for diverse manipulation demonstrations. However, high costs associated with data collection often result in insufficient data coverage across all scenarios, which limits the performance of the models. It is observed that the spatial reasoning phase (SRP) in large workspace dominates the failure cases. Fortunately, this data can be collected with low cost, underscoring the potential of leveraging inexpensive data to improve model performance. In this paper, we introduce the RoboTron-Craft, a stage-divided and cost-effective pipeline for realistic manipulation generation. Base on this, the RoboTron-Platter method is introduced, a framework that decouples training trajectories into distinct task stages and leverages abundant easily collectible SRP data to enhance VLA model's generalization. Through analysis we demonstrate that sub-task-specific training with additional SRP data with proper proportion can act as a performance catalyst for robot manipulation, maximizing the utilization of costly physical interaction phase (PIP) data. Experiments show that through introducing large proportion of cost-effective SRP trajectories into a limited set of PIP data, we can achieve a maximum improvement of 41\% on success rate in zero-shot scenes, while with the ability to transfer manipulation skill to novel targets. Project available at https://github.com/ notFoundThisPerson/RoboTron-Craft.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01184v1">Object Affordance Recognition and Grounding via Multi-scale Cross-modal Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-02
    </div>
    <details class="paper-abstract">
      A core problem of Embodied AI is to learn object manipulation from observation, as humans do. To achieve this, it is important to localize 3D object affordance areas through observation such as images (3D affordance grounding) and understand their functionalities (affordance classification). Previous attempts usually tackle these two tasks separately, leading to inconsistent predictions due to lacking proper modeling of their dependency. In addition, these methods typically only ground the incomplete affordance areas depicted in images, failing to predict the full potential affordance areas, and operate at a fixed scale, resulting in difficulty in coping with affordances significantly varying in scale with respect to the whole object. To address these issues, we propose a novel approach that learns an affordance-aware 3D representation and employs a stage-wise inference strategy leveraging the dependency between grounding and classification tasks. Specifically, we first develop a cross-modal 3D representation through efficient fusion and multi-scale geometric feature propagation, enabling inference of full potential affordance areas at a suitable regional scale. Moreover, we adopt a simple two-stage prediction mechanism, effectively coupling grounding and classification for better affordance understanding. Experiments demonstrate the effectiveness of our method, showing improved performance in both affordance grounding and classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00400v1">Sari Sandbox: A Virtual Retail Store Environment for Embodied AI Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 14 pages, accepted in ICCV 2025 Workshop on RetailVision
    </div>
    <details class="paper-abstract">
      We present Sari Sandbox, a high-fidelity, photorealistic 3D retail store simulation for benchmarking embodied agents against human performance in shopping tasks. Addressing a gap in retail-specific sim environments for embodied agent training, Sari Sandbox features over 250 interactive grocery items across three store configurations, controlled via an API. It supports both virtual reality (VR) for human interaction and a vision language model (VLM)-powered embodied agent. We also introduce SariBench, a dataset of annotated human demonstrations across varied task difficulties. Our sandbox enables embodied agents to navigate, inspect, and manipulate retail items, providing baselines against human performance. We conclude with benchmarks, performance analysis, and recommendations for enhancing realism and scalability. The source code can be accessed via https://github.com/upeee/sari-sandbox-env.
    </details>
</div>
