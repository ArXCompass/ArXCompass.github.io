# embodied ai - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07446v2">TPT-Bench: A Large-Scale, Long-Term and Robot-Egocentric Dataset for Benchmarking Target Person Tracking</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Under review. web: https://medlartea.github.io/tpt-bench/
    </div>
    <details class="paper-abstract">
      Tracking a target person from robot-egocentric views is crucial for developing autonomous robots that provide continuous personalized assistance or collaboration in Human-Robot Interaction (HRI) and Embodied AI. However, most existing target person tracking (TPT) benchmarks are limited to controlled laboratory environments with few distractions, clean backgrounds, and short-term occlusions. In this paper, we introduce a large-scale dataset designed for TPT in crowded and unstructured environments, demonstrated through a robot-person following task. The dataset is collected by a human pushing a sensor-equipped cart while following a target person, capturing human-like following behavior and emphasizing long-term tracking challenges, including frequent occlusions and the need for re-identification from numerous pedestrians. It includes multi-modal data streams, including odometry, 3D LiDAR, IMU, panoramic images, and RGB-D images, along with exhaustively annotated 2D bounding boxes of the target person across 48 sequences, both indoors and outdoors. Using this dataset and visual annotations, we perform extensive experiments with existing SOTA TPT methods, offering a thorough analysis of their limitations and suggesting future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06719v1">A Neural Representation Framework with LLM-Driven Spatial Reasoning for Open-Vocabulary 3D Visual Grounding</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Open-vocabulary 3D visual grounding aims to localize target objects based on free-form language queries, which is crucial for embodied AI applications such as autonomous navigation, robotics, and augmented reality. Learning 3D language fields through neural representations enables accurate understanding of 3D scenes from limited viewpoints and facilitates the localization of target objects in complex environments. However, existing language field methods struggle to accurately localize instances using spatial relations in language queries, such as ``the book on the chair.'' This limitation mainly arises from inadequate reasoning about spatial relations in both language queries and 3D scenes. In this work, we propose SpatialReasoner, a novel neural representation-based framework with large language model (LLM)-driven spatial reasoning that constructs a visual properties-enhanced hierarchical feature field for open-vocabulary 3D visual grounding. To enable spatial reasoning in language queries, SpatialReasoner fine-tunes an LLM to capture spatial relations and explicitly infer instructions for the target, anchor, and spatial relation. To enable spatial reasoning in 3D scenes, SpatialReasoner incorporates visual properties (opacity and color) to construct a hierarchical feature field. This field represents language and instance features using distilled CLIP features and masks extracted via the Segment Anything Model (SAM). The field is then queried using the inferred instructions in a hierarchical manner to localize the target 3D instance based on the spatial relation in the language query. Extensive experiments show that our framework can be seamlessly integrated into different neural representations, outperforming baseline models in 3D visual grounding while empowering their spatial reasoning capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02359v2">Attribution Regularization for Multimodal Paradigms</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Multimodal machine learning has gained significant attention in recent years due to its potential for integrating information from multiple modalities to enhance learning and decision-making processes. However, it is commonly observed that unimodal models outperform multimodal models, despite the latter having access to richer information. Additionally, the influence of a single modality often dominates the decision-making process, resulting in suboptimal performance. This research project aims to address these challenges by proposing a novel regularization term that encourages multimodal models to effectively utilize information from all modalities when making decisions. The focus of this project lies in the video-audio domain, although the proposed regularization technique holds promise for broader applications in embodied AI research, where multiple modalities are involved. By leveraging this regularization term, the proposed approach aims to mitigate the issue of unimodal dominance and improve the performance of multimodal machine learning systems. Through extensive experimentation and evaluation, the effectiveness and generalizability of the proposed technique will be assessed. The findings of this research project have the potential to significantly contribute to the advancement of multimodal machine learning and facilitate its application in various domains, including multimedia analysis, human-computer interaction, and embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05763v1">DreamArt: Generating Interactable Articulated Objects from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Generating articulated objects, such as laptops and microwaves, is a crucial yet challenging task with extensive applications in Embodied AI and AR/VR. Current image-to-3D methods primarily focus on surface geometry and texture, neglecting part decomposition and articulation modeling. Meanwhile, neural reconstruction approaches (e.g., NeRF or Gaussian Splatting) rely on dense multi-view or interaction data, limiting their scalability. In this paper, we introduce DreamArt, a novel framework for generating high-fidelity, interactable articulated assets from single-view images. DreamArt employs a three-stage pipeline: firstly, it reconstructs part-segmented and complete 3D object meshes through a combination of image-to-3D generation, mask-prompted 3D segmentation, and part amodal completion. Second, we fine-tune a video diffusion model to capture part-level articulation priors, leveraging movable part masks as prompt and amodal images to mitigate ambiguities caused by occlusion. Finally, DreamArt optimizes the articulation motion, represented by a dual quaternion, and conducts global texture refinement and repainting to ensure coherent, high-quality textures across all parts. Experimental results demonstrate that DreamArt effectively generates high-quality articulated objects, possessing accurate part shape, high appearance fidelity, and plausible articulation, thereby providing a scalable solution for articulated asset generation. Our project page is available at https://dream-art-0.github.io/DreamArt/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05555v1">PAPRLE (Plug-And-Play Robotic Limb Environment): A Modular Ecosystem for Robotic Limbs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      We introduce PAPRLE (Plug-And-Play Robotic Limb Environment), a modular ecosystem that enables flexible placement and control of robotic limbs. With PAPRLE, a user can change the arrangement of the robotic limbs, and control them using a variety of input devices, including puppeteers, gaming controllers, and VR-based interfaces. This versatility supports a wide range of teleoperation scenarios and promotes adaptability to different task requirements. To further enhance configurability, we introduce a pluggable puppeteer device that can be easily mounted and adapted to match the target robot configurations. PAPRLE supports bilateral teleoperation through these puppeteer devices, agnostic to the type or configuration of the follower robot. By supporting both joint-space and task-space control, the system provides real-time force feedback, improving user fidelity and physical interaction awareness. The modular design of PAPRLE facilitates novel spatial arrangements of the limbs and enables scalable data collection, thereby advancing research in embodied AI and learning-based control. We validate PAPRLE in various real-world settings, demonstrating its versatility across diverse combinations of leader devices and follower robots. The system will be released as open source, including both hardware and software components, to support broader adoption and community-driven extension. Additional resources and demonstrations are available at the project website: https://uiuckimlab.github.io/paprle-pages
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05198v1">EmbodieDreamer: Advancing Real2Sim2Real Transfer for Policy Training via Embodied World Modeling</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Project Page: https://embodiedreamer.github.io/
    </div>
    <details class="paper-abstract">
      The rapid advancement of Embodied AI has led to an increasing demand for large-scale, high-quality real-world data. However, collecting such embodied data remains costly and inefficient. As a result, simulation environments have become a crucial surrogate for training robot policies. Yet, the significant Real2Sim2Real gap remains a critical bottleneck, particularly in terms of physical dynamics and visual appearance. To address this challenge, we propose EmbodieDreamer, a novel framework that reduces the Real2Sim2Real gap from both the physics and appearance perspectives. Specifically, we propose PhysAligner, a differentiable physics module designed to reduce the Real2Sim physical gap. It jointly optimizes robot-specific parameters such as control gains and friction coefficients to better align simulated dynamics with real-world observations. In addition, we introduce VisAligner, which incorporates a conditional video diffusion model to bridge the Sim2Real appearance gap by translating low-fidelity simulated renderings into photorealistic videos conditioned on simulation states, enabling high-fidelity visual transfer. Extensive experiments validate the effectiveness of EmbodieDreamer. The proposed PhysAligner reduces physical parameter estimation error by 3.74% compared to simulated annealing methods while improving optimization speed by 89.91\%. Moreover, training robot policies in the generated photorealistic environment leads to a 29.17% improvement in the average task success rate across real-world tasks after reinforcement learning. Code, model and data will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22355v3">Embodied AI Agents: Modeling the World</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      This paper describes our research on AI agents embodied in visual, virtual or physical forms, enabling them to interact with both users and their environments. These agents, which include virtual avatars, wearable devices, and robots, are designed to perceive, learn and act within their surroundings, which makes them more similar to how humans learn and interact with the environments as compared to disembodied agents. We propose that the development of world models is central to reasoning and planning of embodied AI agents, allowing these agents to understand and predict their environment, to understand user intentions and social contexts, thereby enhancing their ability to perform complex tasks autonomously. World modeling encompasses the integration of multimodal perception, planning through reasoning for action and control, and memory to create a comprehensive understanding of the physical world. Beyond the physical world, we also propose to learn the mental world model of users to enable better human-agent collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04789v1">Training-free Generation of Temporally Consistent Rewards from VLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Recent advances in vision-language models (VLMs) have significantly improved performance in embodied tasks such as goal decomposition and visual comprehension. However, providing accurate rewards for robotic manipulation without fine-tuning VLMs remains challenging due to the absence of domain-specific robotic knowledge in pre-trained datasets and high computational costs that hinder real-time applicability. To address this, we propose $\mathrm{T}^2$-VLM, a novel training-free, temporally consistent framework that generates accurate rewards through tracking the status changes in VLM-derived subgoals. Specifically, our method first queries the VLM to establish spatially aware subgoals and an initial completion estimate before each round of interaction. We then employ a Bayesian tracking algorithm to update the goal completion status dynamically, using subgoal hidden states to generate structured rewards for reinforcement learning (RL) agents. This approach enhances long-horizon decision-making and improves failure recovery capabilities with RL. Extensive experiments indicate that $\mathrm{T}^2$-VLM achieves state-of-the-art performance in two robot manipulation benchmarks, demonstrating superior reward accuracy with reduced computation consumption. We believe our approach not only advances reward generation techniques but also contributes to the broader field of embodied AI. Project website: https://t2-vlm.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01366v2">ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 24 pages with 8 figures and 14 tables; updated for ACL 2025 camera-ready with additional discussion and figures
    </div>
    <details class="paper-abstract">
      3D visual grounding (3DVG) involves localizing entities in a 3D scene referred to by natural language text. Such models are useful for embodied AI and scene retrieval applications, which involve searching for objects or patterns using natural language descriptions. While recent works have focused on LLM-based scaling of 3DVG datasets, these datasets do not capture the full range of potential prompts which could be specified in the English language. To ensure that we are scaling up and testing against a useful and representative set of prompts, we propose a framework for linguistically analyzing 3DVG prompts and introduce Visual Grounding with Diverse Language in 3D (ViGiL3D), a diagnostic dataset for evaluating visual grounding methods against a diverse set of language patterns. We evaluate existing open-vocabulary 3DVG methods to demonstrate that these methods are not yet proficient in understanding and identifying the targets of more challenging, out-of-distribution prompts, toward real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04452v1">SimLauncher: Launching Sample-Efficient Real-world Robotic Reinforcement Learning via Simulation Pre-training</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Autonomous learning of dexterous, long-horizon robotic skills has been a longstanding pursuit of embodied AI. Recent advances in robotic reinforcement learning (RL) have demonstrated remarkable performance and robustness in real-world visuomotor control tasks. However, applying RL in the real world faces challenges such as low sample efficiency, slow exploration, and significant reliance on human intervention. In contrast, simulators offer a safe and efficient environment for extensive exploration and data collection, while the visual sim-to-real gap, often a limiting factor, can be mitigated using real-to-sim techniques. Building on these, we propose SimLauncher, a novel framework that combines the strengths of real-world RL and real-to-sim-to-real approaches to overcome these challenges. Specifically, we first pre-train a visuomotor policy in the digital twin simulation environment, which then benefits real-world RL in two ways: (1) bootstrapping target values using extensive simulated demonstrations and real-world demonstrations derived from pre-trained policy rollouts, and (2) Incorporating action proposals from the pre-trained policy for better exploration. We conduct comprehensive experiments across multi-stage, contact-rich, and dexterous hand manipulation tasks. Compared to prior real-world RL approaches, SimLauncher significantly improves sample efficiency and achieves near-perfect success rates. We hope this work serves as a proof of concept and inspires further research on leveraging large-scale simulation pre-training to benefit real-world robotic RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02029v2">RoboBrain 2.0 Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model, featuring a heterogeneous architecture with a vision encoder and a language model. Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models. In particular, it supports key real-world embodied AI capabilities, including spatial understanding (e.g., affordance prediction, spatial referring, trajectory forecasting) and temporal decision-making (e.g., closed-loop interaction, multi-agent long-horizon planning, and scene graph updating). This report details the model architecture, data construction, multi-stage training strategies, infrastructure and practical applications. We hope RoboBrain 2.0 advances embodied AI research and serves as a practical step toward building generalist embodied agents. The code, checkpoint and benchmark are available at https://superrobobrain.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20129v3">Agentic 3D Scene Generation with Spatially Contextualized VLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Project page: https://spatctxvlm.github.io/project_page/
    </div>
    <details class="paper-abstract">
      Despite recent advances in multimodal content generation enabled by vision-language models (VLMs), their ability to reason about and generate structured 3D scenes remains largely underexplored. This limitation constrains their utility in spatially grounded tasks such as embodied AI, immersive simulations, and interactive 3D applications. We introduce a new paradigm that enables VLMs to generate, understand, and edit complex 3D environments by injecting a continually evolving spatial context. Constructed from multimodal input, this context consists of three components: a scene portrait that provides a high-level semantic blueprint, a semantically labeled point cloud capturing object-level geometry, and a scene hypergraph that encodes rich spatial relationships, including unary, binary, and higher-order constraints. Together, these components provide the VLM with a structured, geometry-aware working memory that integrates its inherent multimodal reasoning capabilities with structured 3D understanding for effective spatial reasoning. Building on this foundation, we develop an agentic 3D scene generation pipeline in which the VLM iteratively reads from and updates the spatial context. The pipeline features high-quality asset generation with geometric restoration, environment setup with automatic verification, and ergonomic adjustment guided by the scene hypergraph. Experiments show that our framework can handle diverse and challenging inputs, achieving a level of generalization not observed in prior work. Further results demonstrate that injecting spatial context enables VLMs to perform downstream tasks such as interactive scene editing and path planning, suggesting strong potential for spatially intelligent systems in computer graphics, 3D vision, and embodied applications. Project page: https://spatctxvlm.github.io/project_page/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22355v2">Embodied AI Agents: Modeling the World</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      This paper describes our research on AI agents embodied in visual, virtual or physical forms, enabling them to interact with both users and their environments. These agents, which include virtual avatars, wearable devices, and robots, are designed to perceive, learn and act within their surroundings, which makes them more similar to how humans learn and interact with the environments as compared to disembodied agents. We propose that the development of world models is central to reasoning and planning of embodied AI agents, allowing these agents to understand and predict their environment, to understand user intentions and social contexts, thereby enhancing their ability to perform complex tasks autonomously. World modeling encompasses the integration of multimodal perception, planning through reasoning for action and control, and memory to create a comprehensive understanding of the physical world. Beyond the physical world, we also propose to learn the mental world model of users to enable better human-agent collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23351v2">Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Challenge Webpage: https://robotwin-benchmark.github.io/cvpr-2025-challenge/
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (Embodied AI) is an emerging frontier in robotics, driven by the need for autonomous systems that can perceive, reason, and act in complex physical environments. While single-arm systems have shown strong task performance, collaborative dual-arm systems are essential for handling more intricate tasks involving rigid, deformable, and tactile-sensitive objects. To advance this goal, we launched the RoboTwin Dual-Arm Collaboration Challenge at the 2nd MEIS Workshop, CVPR 2025. Built on the RoboTwin Simulation platform (1.0 and 2.0) and the AgileX COBOT-Magic Robot platform, the competition consisted of three stages: Simulation Round 1, Simulation Round 2, and a final Real-World Round. Participants totally tackled 17 dual-arm manipulation tasks, covering rigid, deformable, and tactile-based scenarios. The challenge attracted 64 global teams and over 400 participants, producing top-performing solutions like SEM and AnchorDP3 and generating valuable insights into generalizable bimanual policy learning. This report outlines the competition setup, task design, evaluation methodology, key findings and future direction, aiming to support future research on robust and generalizable bimanual manipulation policies. The Challenge Webpage is available at https://robotwin-benchmark.github.io/cvpr-2025-challenge/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18904v2">TC-Light: Temporally Coherent Generative Rendering for Realistic World Transfer</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Project Page: https://dekuliutesla.github.io/tclight/ Code: https://github.com/Linketic/TC-Light
    </div>
    <details class="paper-abstract">
      Illumination and texture editing are critical dimensions for world-to-world transfer, which is valuable for applications including sim2real and real2real visual data scaling up for embodied AI. Existing techniques generatively re-render the input video to realize the transfer, such as video relighting models and conditioned world generation models. Nevertheless, these models are predominantly limited to the domain of training data (e.g., portrait) or fall into the bottleneck of temporal consistency and computation efficiency, especially when the input video involves complex dynamics and long durations. In this paper, we propose TC-Light, a novel generative renderer to overcome these problems. Starting from the video preliminarily relighted by an inflated video relighting model, it optimizes appearance embedding in the first stage to align global illumination. Then it optimizes the proposed canonical video representation, i.e., Unique Video Tensor (UVT), to align fine-grained texture and lighting in the second stage. To comprehensively evaluate performance, we also establish a long and highly dynamic video benchmark. Extensive experiments show that our method enables physically plausible re-rendering results with superior temporal coherence and low computation cost. The code and video demos are available at https://dekuliutesla.github.io/tclight/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01667v1">What does really matter in image goal navigation?</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Image goal navigation requires two different skills: firstly, core navigation skills, including the detection of free space and obstacles, and taking decisions based on an internal representation; and secondly, computing directional information by comparing visual observations to the goal image. Current state-of-the-art methods either rely on dedicated image-matching, or pre-training of computer vision modules on relative pose estimation. In this paper, we study whether this task can be efficiently solved with end-to-end training of full agents with RL, as has been claimed by recent work. A positive answer would have impact beyond Embodied AI and allow training of relative pose estimation from reward for navigation alone. In a large study we investigate the effect of architectural choices like late fusion, channel stacking, space-to-depth projections and cross-attention, and their role in the emergence of relative pose estimators from navigation training. We show that the success of recent methods is influenced up to a certain extent by simulator settings, leading to shortcuts in simulation. However, we also show that these capabilities can be transferred to more realistic setting, up to some extend. We also find evidence for correlations between navigation performance and probed (emerging) relative pose estimation performance, an important sub skill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01398v2">Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      3D scene understanding is a long-standing challenge in computer vision and a key component in enabling mixed reality, wearable computing, and embodied AI. Providing a solution to these applications requires a multifaceted approach that covers scene-centric, object-centric, as well as interaction-centric capabilities. While there exist numerous datasets and algorithms approaching the former two problems, the task of understanding interactable and articulated objects is underrepresented and only partly covered in the research field. In this work, we address this shortcoming by introducing: (1) Articulate3D, an expertly curated 3D dataset featuring high-quality manual annotations on 280 indoor scenes. Articulate3D provides 8 types of annotations for articulated objects, covering parts and detailed motion information, all stored in a standardized scene representation format designed for scalable 3D content creation, exchange and seamless integration into simulation environments. (2) USDNet, a novel unified framework capable of simultaneously predicting part segmentation along with a full specification of motion attributes for articulated objects. We evaluate USDNet on Articulate3D as well as two existing datasets, demonstrating the advantage of our unified dense prediction approach. Furthermore, we highlight the value of Articulate3D through cross-dataset and cross-domain evaluations and showcase its applicability in downstream tasks such as scene editing through LLM prompting and robotic policy training for articulated object manipulation. We provide open access to our dataset, benchmark, and method's source code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02029v1">RoboBrain 2.0 Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model, featuring a heterogeneous architecture with a vision encoder and a language model. Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models. In particular, it supports key real-world embodied AI capabilities, including spatial understanding (e.g., affordance prediction, spatial referring, trajectory forecasting) and temporal decision-making (e.g., closed-loop interaction, multi-agent long-horizon planning, and scene graph updating). This report details the model architecture, data construction, multi-stage training strategies, infrastructure and practical applications. We hope RoboBrain 2.0 advances embodied AI research and serves as a practical step toward building generalist embodied agents. The code, checkpoint and benchmark are available at https://superrobobrain.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23153v2">Conceptual Framework Toward Embodied Collective Adaptive Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Collective Adaptive Intelligence (CAI) represent a transformative approach in embodied AI, wherein numerous autonomous agents collaborate, adapt, and self-organize to navigate complex, dynamic environments. By enabling systems to reconfigure themselves in response to unforeseen challenges, CAI facilitate robust performance in real-world scenarios. This article introduces a conceptual framework for designing and analyzing CAI. It delineates key attributes including task generalization, resilience, scalability, and self-assembly, aiming to bridge theoretical foundations with practical methodologies for engineering adaptive, emergent intelligence. By providing a structured foundation for understanding and implementing CAI, this work seeks to guide researchers and practitioners in developing more resilient, scalable, and adaptable AI systems across various domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00917v1">A Survey: Learning Embodied Intelligence from Physical Simulators and World Models</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey
    </div>
    <details class="paper-abstract">
      The pursuit of artificial general intelligence (AGI) has placed embodied intelligence at the forefront of robotics research. Embodied intelligence focuses on agents capable of perceiving, reasoning, and acting within the physical world. Achieving robust embodied intelligence requires not only advanced perception and control, but also the ability to ground abstract cognition in real-world interactions. Two foundational technologies, physical simulators and world models, have emerged as critical enablers in this quest. Physical simulators provide controlled, high-fidelity environments for training and evaluating robotic agents, allowing safe and efficient development of complex behaviors. In contrast, world models empower robots with internal representations of their surroundings, enabling predictive planning and adaptive decision-making beyond direct sensory input. This survey systematically reviews recent advances in learning embodied AI through the integration of physical simulators and world models. We analyze their complementary roles in enhancing autonomy, adaptability, and generalization in intelligent robots, and discuss the interplay between external simulation and internal modeling in bridging the gap between simulated training and real-world deployment. By synthesizing current progress and identifying open challenges, this survey aims to provide a comprehensive perspective on the path toward more capable and generalizable embodied AI systems. We also maintain an active repository that contains up-to-date literature and open-source projects at https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey.
    </details>
</div>
