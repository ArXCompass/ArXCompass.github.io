# embodied ai - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

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
