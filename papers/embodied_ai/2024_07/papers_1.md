# embodied ai - 2024_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11377v2">HabiCrowd: A High Performance Simulator for Crowd-Aware Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 Accepted to IROS 2024
    </div>
    <details class="paper-abstract">
      Visual navigation, a foundational aspect of Embodied AI (E-AI), has been significantly studied in the past few years. While many 3D simulators have been introduced to support visual navigation tasks, scarcely works have been directed towards combining human dynamics, creating the gap between simulation and real-world applications. Furthermore, current 3D simulators incorporating human dynamics have several limitations, particularly in terms of computational efficiency, which is a promise of E-AI simulators. To overcome these shortcomings, we introduce HabiCrowd, the first standard benchmark for crowd-aware visual navigation that integrates a crowd dynamics model with diverse human settings into photorealistic environments. Empirical evaluations demonstrate that our proposed human dynamics model achieves state-of-the-art performance in collision avoidance, while exhibiting superior computational efficiency compared to its counterparts. We leverage HabiCrowd to conduct several comprehensive studies on crowd-aware visual navigation tasks and human-robot interactions. The source code and data can be found at https://habicrowd.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17135v4">TLControl: Trajectory and Language Control for Human Motion Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-07-24
    </div>
    <details class="paper-abstract">
      Controllable human motion synthesis is essential for applications in AR/VR, gaming and embodied AI. Existing methods often focus solely on either language or full trajectory control, lacking precision in synthesizing motions aligned with user-specified trajectories, especially for multi-joint control. To address these issues, we present TLControl, a novel method for realistic human motion synthesis, incorporating both low-level Trajectory and high-level Language semantics controls, through the integration of neural-based and optimization-based techniques. Specifically, we begin with training a VQ-VAE for a compact and well-structured latent motion space organized by body parts. We then propose a Masked Trajectories Transformer (MTT) for predicting a motion distribution conditioned on language and trajectory. Once trained, we use MTT to sample initial motion predictions given user-specified partial trajectories and text descriptions as conditioning. Finally, we introduce a test-time optimization to refine these coarse predictions for precise trajectory control, which offers flexibility by allowing users to specify various optimization goals and ensures high runtime efficiency. Comprehensive experiments show that TLControl significantly outperforms the state-of-the-art in trajectory accuracy and time efficiency, making it practical for interactive and high-quality animation generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14758v1">DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-level Control</a></div>
    <div class="paper-meta">
      📅 2024-07-20
      | 💬 ECCV 2024
    </div>
    <details class="paper-abstract">
      Building a general-purpose intelligent home-assistant agent skilled in diverse tasks by human commands is a long-term blueprint of embodied AI research, which poses requirements on task planning, environment modeling, and object interaction. In this work, we study primitive mobile manipulations for embodied agents, i.e. how to navigate and interact based on an instructed verb-noun pair. We propose DISCO, which features non-trivial advancements in contextualized scene modeling and efficient controls. In particular, DISCO incorporates differentiable scene representations of rich semantics in object and affordance, which is dynamically learned on the fly and facilitates navigation planning. Besides, we propose dual-level coarse-to-fine action controls leveraging both global and local cues to accomplish mobile manipulation tasks efficiently. DISCO easily integrates into embodied tasks such as embodied instruction following. To validate our approach, we take the ALFRED benchmark of large-scale long-horizon vision-language navigation and interaction tasks as a test bed. In extensive experiments, we make comprehensive evaluations and demonstrate that DISCO outperforms the art by a sizable +8.6% success rate margin in unseen scenes, even without step-by-step instructions. Our code is publicly released at https://github.com/AllenXuuu/DISCO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11325v1">VISA: Reasoning Video Object Segmentation via Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-07-16
    </div>
    <details class="paper-abstract">
      Existing Video Object Segmentation (VOS) relies on explicit user instructions, such as categories, masks, or short phrases, restricting their ability to perform complex video segmentation requiring reasoning with world knowledge. In this paper, we introduce a new task, Reasoning Video Object Segmentation (ReasonVOS). This task aims to generate a sequence of segmentation masks in response to implicit text queries that require complex reasoning abilities based on world knowledge and video contexts, which is crucial for structured environment understanding and object-centric interactions, pivotal in the development of embodied AI. To tackle ReasonVOS, we introduce VISA (Video-based large language Instructed Segmentation Assistant), to leverage the world knowledge reasoning capabilities of multi-modal LLMs while possessing the ability to segment and track objects in videos with a mask decoder. Moreover, we establish a comprehensive benchmark consisting of 35,074 instruction-mask sequence pairs from 1,042 diverse videos, which incorporates complex world knowledge reasoning into segmentation tasks for instruction-tuning and evaluation purposes of ReasonVOS models. Experiments conducted on 8 datasets demonstrate the effectiveness of VISA in tackling complex reasoning segmentation and vanilla referring segmentation in both video and image domains. The code and dataset are available at https://github.com/cilinyan/VISA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10943v1">GRUtopia: Dream General Robots in a City at Scale</a></div>
    <div class="paper-meta">
      📅 2024-07-15
    </div>
    <details class="paper-abstract">
      Recent works have been exploring the scaling laws in the field of Embodied AI. Given the prohibitive costs of collecting real-world data, we believe the Simulation-to-Real (Sim2Real) paradigm is a crucial step for scaling the learning of embodied models. This paper introduces project GRUtopia, the first simulated interactive 3D society designed for various robots. It features several advancements: (a) The scene dataset, GRScenes, includes 100k interactive, finely annotated scenes, which can be freely combined into city-scale environments. In contrast to previous works mainly focusing on home, GRScenes covers 89 diverse scene categories, bridging the gap of service-oriented environments where general robots would be initially deployed. (b) GRResidents, a Large Language Model (LLM) driven Non-Player Character (NPC) system that is responsible for social interaction, task generation, and task assignment, thus simulating social scenarios for embodied AI applications. (c) The benchmark, GRBench, supports various robots but focuses on legged robots as primary agents and poses moderately challenging tasks involving Object Loco-Navigation, Social Loco-Navigation, and Loco-Manipulation. We hope that this work can alleviate the scarcity of high-quality data in this field and provide a more comprehensive assessment of Embodied AI research. The project is available at https://github.com/OpenRobotLab/GRUtopia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10062v1">SpikeGS: 3D Gaussian Splatting from Spike Streams with High-Speed Camera Motion</a></div>
    <div class="paper-meta">
      📅 2024-07-14
    </div>
    <details class="paper-abstract">
      Novel View Synthesis plays a crucial role by generating new 2D renderings from multi-view images of 3D scenes. However, capturing high-speed scenes with conventional cameras often leads to motion blur, hindering the effectiveness of 3D reconstruction. To address this challenge, high-frame-rate dense 3D reconstruction emerges as a vital technique, enabling detailed and accurate modeling of real-world objects or scenes in various fields, including Virtual Reality or embodied AI. Spike cameras, a novel type of neuromorphic sensor, continuously record scenes with an ultra-high temporal resolution, showing potential for accurate 3D reconstruction. Despite their promise, existing approaches, such as applying Neural Radiance Fields (NeRF) to spike cameras, encounter challenges due to the time-consuming rendering process. To address this issue, we make the first attempt to introduce the 3D Gaussian Splatting (3DGS) into spike cameras in high-speed capture, providing 3DGS as dense and continuous clues of views, then constructing SpikeGS. Specifically, to train SpikeGS, we establish computational equations between the rendering process of 3DGS and the processes of instantaneous imaging and exposing-like imaging of the continuous spike stream. Besides, we build a very lightweight but effective mapping process from spikes to instant images to support training. Furthermore, we introduced a new spike-based 3D rendering dataset for validation. Extensive experiments have demonstrated our method possesses the high quality of novel view rendering, proving the tremendous potential of spike cameras in modeling 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19741v3">ROS-LLM: A ROS framework for embodied AI with task feedback and structured reasoning</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 This document contains 26 pages and 13 figures
    </div>
    <details class="paper-abstract">
      We present a framework for intuitive robot programming by non-experts, leveraging natural language prompts and contextual information from the Robot Operating System (ROS). Our system integrates large language models (LLMs), enabling non-experts to articulate task requirements to the system through a chat interface. Key features of the framework include: integration of ROS with an AI agent connected to a plethora of open-source and commercial LLMs, automatic extraction of a behavior from the LLM output and execution of ROS actions/services, support for three behavior modes (sequence, behavior tree, state machine), imitation learning for adding new robot actions to the library of possible actions, and LLM reflection via human and environment feedback. Extensive experiments validate the framework, showcasing robustness, scalability, and versatility in diverse scenarios, including long-horizon tasks, tabletop rearrangements, and remote supervisory control. To facilitate the adoption of our framework and support the reproduction of our results, we have made our code open-source. You can access it at: https://github.com/huawei-noah/HEBO/tree/master/ROSLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07061v2">Internet of Agents: Weaving a Web of Heterogeneous Agents for Collaborative Intelligence</a></div>
    <div class="paper-meta">
      📅 2024-07-10
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has paved the way for the development of highly capable autonomous agents. However, existing multi-agent frameworks often struggle with integrating diverse capable third-party agents due to reliance on agents defined within their own ecosystems. They also face challenges in simulating distributed environments, as most frameworks are limited to single-device setups. Furthermore, these frameworks often rely on hard-coded communication pipelines, limiting their adaptability to dynamic task requirements. Inspired by the concept of the Internet, we propose the Internet of Agents (IoA), a novel framework that addresses these limitations by providing a flexible and scalable platform for LLM-based multi-agent collaboration. IoA introduces an agent integration protocol, an instant-messaging-like architecture design, and dynamic mechanisms for agent teaming and conversation flow control. Through extensive experiments on general assistant tasks, embodied AI tasks, and retrieval-augmented generation benchmarks, we demonstrate that IoA consistently outperforms state-of-the-art baselines, showcasing its ability to facilitate effective collaboration among heterogeneous agents. IoA represents a step towards linking diverse agents in an Internet-like environment, where agents can seamlessly collaborate to achieve greater intelligence and capabilities. Our codebase has been released at \url{https://github.com/OpenBMB/IoA}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09465v2">PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-07-10
      | 💬 Accepted by CVPR 2024 (Highlight), 18 pages
    </div>
    <details class="paper-abstract">
      With recent developments in Embodied Artificial Intelligence (EAI) research, there has been a growing demand for high-quality, large-scale interactive scene generation. While prior methods in scene synthesis have prioritized the naturalness and realism of the generated scenes, the physical plausibility and interactivity of scenes have been largely left unexplored. To address this disparity, we introduce PhyScene, a novel method dedicated to generating interactive 3D scenes characterized by realistic layouts, articulated objects, and rich physical interactivity tailored for embodied agents. Based on a conditional diffusion model for capturing scene layouts, we devise novel physics- and interactivity-based guidance mechanisms that integrate constraints from object collision, room layout, and object reachability. Through extensive experiments, we demonstrate that PhyScene effectively leverages these guidance functions for physically interactable scene synthesis, outperforming existing state-of-the-art scene synthesis methods by a large margin. Our findings suggest that the scenes generated by PhyScene hold considerable potential for facilitating diverse skill acquisition among agents within interactive environments, thereby catalyzing further advancements in embodied AI research. Project website: http://physcene.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.19007v2">DOZE: A Dataset for Open-Vocabulary Zero-Shot Object Navigation in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2024-07-08
      | 💬 This version of the paper has been accepted for publication in IEEE Robotics and Automation Letters (RA-L)
    </div>
    <details class="paper-abstract">
      Zero-Shot Object Navigation (ZSON) requires agents to autonomously locate and approach unseen objects in unfamiliar environments and has emerged as a particularly challenging task within the domain of Embodied AI. Existing datasets for developing ZSON algorithms lack consideration of dynamic obstacles, object attribute diversity, and scene texts, thus exhibiting noticeable discrepancies from real-world situations. To address these issues, we propose a Dataset for Open-Vocabulary Zero-Shot Object Navigation in Dynamic Environments (DOZE) that comprises ten high-fidelity 3D scenes with over 18k tasks, aiming to mimic complex, dynamic real-world scenarios. Specifically, DOZE scenes feature multiple moving humanoid obstacles, a wide array of open-vocabulary objects, diverse distinct-attribute objects, and valuable textual hints. Besides, different from existing datasets that only provide collision checking between the agent and static obstacles, we enhance DOZE by integrating capabilities for detecting collisions between the agent and moving obstacles. This novel functionality enables the evaluation of the agents' collision avoidance abilities in dynamic environments. We test four representative ZSON methods on DOZE, revealing substantial room for improvement in existing approaches concerning navigation efficiency, safety, and object recognition accuracy. Our dataset can be found at https://DOZE-Dataset.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14868v2">Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-07-05
      | 💬 Accepted to ECCV 2024. Project webpage is available at: https://gcd.cs.columbia.edu/
    </div>
    <details class="paper-abstract">
      Accurate reconstruction of complex dynamic scenes from just a single viewpoint continues to be a challenging task in computer vision. Current dynamic novel view synthesis methods typically require videos from many different camera viewpoints, necessitating careful recording setups, and significantly restricting their utility in the wild as well as in terms of embodied AI applications. In this paper, we propose $\textbf{GCD}$, a controllable monocular dynamic view synthesis pipeline that leverages large-scale diffusion priors to, given a video of any scene, generate a synchronous video from any other chosen perspective, conditioned on a set of relative camera pose parameters. Our model does not require depth as input, and does not explicitly model 3D scene geometry, instead performing end-to-end video-to-video translation in order to achieve its goal efficiently. Despite being trained on synthetic multi-view video data only, zero-shot real-world generalization experiments show promising results in multiple domains, including robotics, object permanence, and driving environments. We believe our framework can potentially unlock powerful applications in rich dynamic scene understanding, perception for robotics, and interactive 3D video viewing experiences for virtual reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02220v2">Embodied AI in Mobile Robots: Coverage Path Planning with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-07-04
      | 💬 7 pages, 2 figures, conference
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and solving mathematical problems, leading to advancements in various fields. We propose an LLM-embodied path planning framework for mobile agents, focusing on solving high-level coverage path planning issues and low-level control. Our proposed multi-layer architecture uses prompted LLMs in the path planning phase and integrates them with the mobile agents' low-level actuators. To evaluate the performance of various LLMs, we propose a coverage-weighted path planning metric to assess the performance of the embodied models. Our experiments show that the proposed framework improves LLMs' spatial inference abilities. We demonstrate that the proposed multi-layer framework significantly enhances the efficiency and accuracy of these tasks by leveraging the natural language understanding and generative capabilities of LLMs. Our experiments show that this framework can improve LLMs' 2D plane reasoning abilities and complete coverage path planning tasks. We also tested three LLM kernels: gpt-4o, gemini-1.5-flash, and claude-3.5-sonnet. The experimental results show that claude-3.5 can complete the coverage planning task in different scenarios, and its indicators are better than those of the other models.
    </details>
</div>
