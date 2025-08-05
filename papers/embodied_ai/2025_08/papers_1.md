# embodied ai - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

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
