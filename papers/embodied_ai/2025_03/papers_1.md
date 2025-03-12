# embodied ai - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06843v4">HO-Cap: A Capture System and Dataset for 3D Reconstruction and Pose Tracking of Hand-Object Interaction</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      We introduce a data capture system and a new dataset, HO-Cap, for 3D reconstruction and pose tracking of hands and objects in videos. The system leverages multiple RGBD cameras and a HoloLens headset for data collection, avoiding the use of expensive 3D scanners or mocap systems. We propose a semi-automatic method for annotating the shape and pose of hands and objects in the collected videos, significantly reducing the annotation time compared to manual labeling. With this system, we captured a video dataset of humans interacting with objects to perform various tasks, including simple pick-and-place actions, handovers between hands, and using objects according to their affordance, which can serve as human demonstrations for research in embodied AI and robot manipulation. Our data capture setup and annotation framework will be available for the community to use in reconstructing 3D shapes of objects and human hands and tracking their poses in videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10439v2">CogNav: Cognitive Process Modeling for Object Goal Navigation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Object goal navigation (ObjectNav) is a fundamental task in embodied AI, requiring an agent to locate a target object in previously unseen environments. This task is particularly challenging because it requires both perceptual and cognitive processes, including object recognition and decision-making. While substantial advancements in perception have been driven by the rapid development of visual foundation models, progress on the cognitive aspect remains constrained, primarily limited to either implicit learning through simulator rollouts or explicit reliance on predefined heuristic rules. Inspired by neuroscientific findings demonstrating that humans maintain and dynamically update fine-grained cognitive states during object search tasks in novel environments, we propose CogNav, a framework designed to mimic this cognitive process using large language models. Specifically, we model the cognitive process using a finite state machine comprising fine-grained cognitive states, ranging from exploration to identification. Transitions between states are determined by a large language model based on a dynamically constructed heterogeneous cognitive map, which contains spatial and semantic information about the scene being explored. Extensive evaluations on the HM3D, MP3D, and RoboTHOR benchmarks demonstrate that our cognitive process modeling significantly improves the success rate of ObjectNav at least by relative 14% over the state-of-the-arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08306v1">Reasoning in visual navigation of end-to-end trained agents: a dynamical systems approach</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Progress in Embodied AI has made it possible for end-to-end-trained agents to navigate in photo-realistic environments with high-level reasoning and zero-shot or language-conditioned behavior, but benchmarks are still dominated by simulation. In this work, we focus on the fine-grained behavior of fast-moving real robots and present a large-scale experimental study involving \numepisodes{} navigation episodes in a real environment with a physical robot, where we analyze the type of reasoning emerging from end-to-end training. In particular, we study the presence of realistic dynamics which the agent learned for open-loop forecasting, and their interplay with sensing. We analyze the way the agent uses latent memory to hold elements of the scene structure and information gathered during exploration. We probe the planning capabilities of the agent, and find in its memory evidence for somewhat precise plans over a limited horizon. Furthermore, we show in a post-hoc analysis that the value function learned by the agent relates to long-term planning. Put together, our experiments paint a new picture on how using tools from computer vision and sequential decision making have led to new capabilities in robotics and control. An interactive tool is available at europe.naverlabs.com/research/publications/reasoning-in-visual-navigation-of-end-to-end-trained-agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08174v1">Investigating the Effectiveness of a Socratic Chain-of-Thoughts Reasoning Method for Task Planning in Robotics, A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated unprecedented capability in reasoning with natural language. Coupled with this development is the emergence of embodied AI in robotics. Despite showing promise for verbal and written reasoning tasks, it remains unknown whether LLMs are capable of navigating complex spatial tasks with physical actions in the real world. To this end, it is of interest to investigate applying LLMs to robotics in zero-shot learning scenarios, and in the absence of fine-tuning - a feat which could significantly improve human-robot interaction, alleviate compute cost, and eliminate low-level programming tasks associated with robot tasks. To explore this question, we apply GPT-4(Omni) with a simulated Tiago robot in Webots engine for an object search task. We evaluate the effectiveness of three reasoning strategies based on Chain-of-Thought (CoT) sub-task list generation with the Socratic method (SocraCoT) (in order of increasing rigor): (1) Non-CoT/Non-SocraCoT, (2) CoT only, and (3) SocraCoT. Performance was measured in terms of the proportion of tasks successfully completed and execution time (N = 20). Our preliminary results show that when combined with chain-of-thought reasoning, the Socratic method can be used for code generation for robotic tasks that require spatial awareness. In extension of this finding, we propose EVINCE-LoC; a modified EVINCE method that could further enhance performance in highly complex and or dynamic testing scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v4">OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents through an environment by leveraging both language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising a versatile toolchain and large-scale benchmark for aerial VLN. Firstly, we develop a highly automated toolchain for data collection, enabling automatic point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Secondly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. The corresponding visual data are generated using various rendering engines and advanced techniques, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). All data exhibit high visual quality. Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of the dataset. Thirdly, we propose OpenFly-Agent, a keyframe-aware VLN model, which takes language instructions, current observations, and historical keyframes as input, and outputs flight actions directly. Extensive analyses and experiments are conducted, showcasing the superiority of our OpenFly platform and OpenFly-Agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01481v2">SonicSim: A customizable simulation platform for speech processing in moving sound source scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Systematic evaluation of speech separation and enhancement models under moving sound source conditions requires extensive and diverse data. However, real-world datasets often lack sufficient data for training and evaluation, and synthetic datasets, while larger, lack acoustic realism. Consequently, neither effectively meets practical needs. To address this issue, we introduce SonicSim, a synthetic toolkit based on the embodied AI simulation platform Habitat-sim, designed to generate highly customizable data for moving sound sources. SonicSim supports multi-level adjustments, including scene-level, microphone-level, and source-level adjustments, enabling the creation of more diverse synthetic data. Leveraging SonicSim, we constructed a benchmark dataset called SonicSet, utilizing LibriSpeech, Freesound Dataset 50k (FSD50K), Free Music Archive (FMA), and 90 scenes from Matterport3D to evaluate speech separation and enhancement models. Additionally, to investigate the differences between synthetic and real-world data, we selected 5 hours of raw, non-reverberant data from the SonicSet validation set and recorded a real-world speech separation dataset, providing a reference for comparing SonicSet with other synthetic datasets. For speech enhancement, we utilized the real-world dataset RealMAN to validate the acoustic gap between SonicSet and existing synthetic datasets. The results indicate that models trained on SonicSet generalize better to real-world scenarios compared to other synthetic datasets. The code is publicly available at https://cslikai.cn/SonicSim/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04879v1">Modeling Dynamic Hand-Object Interactions with Applications to Human-Robot Handovers</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 PhD Thesis
    </div>
    <details class="paper-abstract">
      Humans frequently grasp, manipulate, and move objects. Interactive systems assist humans in these tasks, enabling applications in Embodied AI, human-robot interaction, and virtual reality. However, current methods in hand-object synthesis often neglect dynamics and focus on generating static grasps. The first part of this dissertation introduces dynamic grasp synthesis, where a hand grasps and moves an object to a target pose. We approach this task using physical simulation and reinforcement learning. We then extend this to bimanual manipulation and articulated objects, requiring fine-grained coordination between hands. In the second part of this dissertation, we study human-to-robot handovers. We integrate captured human motion into simulation and introduce a student-teacher framework that adapts to human behavior and transfers from sim to real. To overcome data scarcity, we generate synthetic interactions, increasing training diversity by 100x. Our user study finds no difference between policies trained on synthetic vs. real motions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v3">OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents through an environment by leveraging both language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising a versatile toolchain and large-scale benchmark for aerial VLN. Firstly, we develop a highly automated toolchain for data collection, enabling automatic point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Secondly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. The corresponding visual data are generated using various rendering engines and advanced techniques, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). All data exhibit high visual quality. Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of the dataset. Thirdly, we propose OpenFly-Agent, a keyframe-aware VLN model, which takes language instructions, current observations, and historical keyframes as input, and outputs flight actions directly. Extensive analyses and experiments are conducted, showcasing the superiority of our OpenFly platform and OpenFly-Agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14093v4">A Survey on Vision-Language-Action Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Project page: https://github.com/yueen-ma/Awesome-VLA
    </div>
    <details class="paper-abstract">
      Embodied AI is widely recognized as a key element of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models -- referred to as vision-language-action models (VLAs) -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. In recent years, a myriad of VLAs have been developed, making it imperative to capture the rapidly evolving landscape through a comprehensive survey. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges faced by VLAs and outline promising future directions in embodied AI. We have created a project associated with this survey, which is available at https://github.com/yueen-ma/Awesome-VLA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02247v1">WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Object Goal Navigation-requiring an agent to locate a specific object in an unseen environment-remains a core challenge in embodied AI. Although recent progress in Vision-Language Model (VLM)-based agents has demonstrated promising perception and decision-making abilities through prompting, none has yet established a fully modular world model design that reduces risky and costly interactions with the environment by predicting the future state of the world. We introduce WMNav, a novel World Model-based Navigation framework powered by Vision-Language Models (VLMs). It predicts possible outcomes of decisions and builds memories to provide feedback to the policy module. To retain the predicted state of the environment, WMNav proposes the online maintained Curiosity Value Map as part of the world model memory to provide dynamic configuration for navigation policy. By decomposing according to a human-like thinking process, WMNav effectively alleviates the impact of model hallucination by making decisions based on the feedback difference between the world model plan and observation. To further boost efficiency, we implement a two-stage action proposer strategy: broad exploration followed by precise localization. Extensive evaluation on HM3D and MP3D validates WMNav surpasses existing zero-shot benchmarks in both success rate and exploration efficiency (absolute improvement: +3.2% SR and +3.2% SPL on HM3D, +13.5% SR and +1.1% SPL on MP3D). Project page: https://b0b8k1ng.github.io/WMNav/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14093v3">A Survey on Vision-Language-Action Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 16 pages, a survey of vision-language-action models
    </div>
    <details class="paper-abstract">
      Embodied AI is widely recognized as a key element of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models -- referred to as vision-language-action models (VLAs) -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. In recent years, a myriad of VLAs have been developed, making it imperative to capture the rapidly evolving landscape through a comprehensive survey. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges faced by VLAs and outline promising future directions in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07468v3">From Screens to Scenes: A Survey of Embodied AI in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 56 pages, 11 figures, manuscript accepted by Information Fusion
    </div>
    <details class="paper-abstract">
      Healthcare systems worldwide face persistent challenges in efficiency, accessibility, and personalization. Powered by modern AI technologies such as multimodal large language models and world models, Embodied AI (EmAI) represents a transformative frontier, offering enhanced autonomy and the ability to interact with the physical world to address these challenges. As an interdisciplinary and rapidly evolving research domain, "EmAI in healthcare" spans diverse fields such as algorithms, robotics, and biomedicine. This complexity underscores the importance of timely reviews and analyses to track advancements, address challenges, and foster cross-disciplinary collaboration. In this paper, we provide a comprehensive overview of the "brain" of EmAI for healthcare, wherein we introduce foundational AI algorithms for perception, actuation, planning, and memory, and focus on presenting the healthcare applications spanning clinical interventions, daily care & companionship, infrastructure support, and biomedical research. Despite its promise, the development of EmAI for healthcare is hindered by critical challenges such as safety concerns, gaps between simulation platforms and real-world applications, the absence of standardized benchmarks, and uneven progress across interdisciplinary domains. We discuss the technical barriers and explore ethical considerations, offering a forward-looking perspective on the future of EmAI in healthcare. A hierarchical framework of intelligent levels for EmAI systems is also introduced to guide further development. By providing systematic insights, this work aims to inspire innovation and practical applications, paving the way for a new era of intelligent, patient-centered healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00778v1">AffordGrasp: In-Context Affordance Reasoning for Open-Vocabulary Task-Oriented Grasping in Clutter</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Inferring the affordance of an object and grasping it in a task-oriented manner is crucial for robots to successfully complete manipulation tasks. Affordance indicates where and how to grasp an object by taking its functionality into account, serving as the foundation for effective task-oriented grasping. However, current task-oriented methods often depend on extensive training data that is confined to specific tasks and objects, making it difficult to generalize to novel objects and complex scenes. In this paper, we introduce AffordGrasp, a novel open-vocabulary grasping framework that leverages the reasoning capabilities of vision-language models (VLMs) for in-context affordance reasoning. Unlike existing methods that rely on explicit task and object specifications, our approach infers tasks directly from implicit user instructions, enabling more intuitive and seamless human-robot interaction in everyday scenarios. Building on the reasoning outcomes, our framework identifies task-relevant objects and grounds their part-level affordances using a visual grounding module. This allows us to generate task-oriented grasp poses precisely within the affordance regions of the object, ensuring both functional and context-aware robotic manipulation. Extensive experiments demonstrate that AffordGrasp achieves state-of-the-art performance in both simulation and real-world scenarios, highlighting the effectiveness of our method. We believe our approach advances robotic manipulation techniques and contributes to the broader field of embodied AI. Project website: https://eqcy.github.io/affordgrasp/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08208v3">SPA: 3D Spatial-Awareness Enables Effective Embodied Representation</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 Project Page: https://haoyizhu.github.io/spa/
    </div>
    <details class="paper-abstract">
      In this paper, we introduce SPA, a novel representation learning framework that emphasizes the importance of 3D spatial awareness in embodied AI. Our approach leverages differentiable neural rendering on multi-view images to endow a vanilla Vision Transformer (ViT) with intrinsic spatial understanding. We present the most comprehensive evaluation of embodied representation learning to date, covering 268 tasks across 8 simulators with diverse policies in both single-task and language-conditioned multi-task scenarios. The results are compelling: SPA consistently outperforms more than 10 state-of-the-art representation methods, including those specifically designed for embodied AI, vision-centric tasks, and multi-modal applications, while using less training data. Furthermore, we conduct a series of real-world experiments to confirm its effectiveness in practical scenarios. These results highlight the critical role of 3D spatial awareness for embodied representation learning. Our strongest model takes more than 6000 GPU hours to train and we are committed to open-sourcing all code and model weights to foster future research in embodied representation learning. Project Page: https://haoyizhu.github.io/spa/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17735v4">3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      Constructing compact and informative 3D scene representations is essential for effective embodied exploration and reasoning, especially in complex environments over extended periods. Existing representations, such as object-centric 3D scene graphs, oversimplify spatial relationships by modeling scenes as isolated objects with restrictive textual relationships, making it difficult to address queries requiring nuanced spatial understanding. Moreover, these representations lack natural mechanisms for active exploration and memory management, hindering their application to lifelong autonomy. In this work, we propose 3D-Mem, a novel 3D scene memory framework for embodied agents. 3D-Mem employs informative multi-view images, termed Memory Snapshots, to represent the scene and capture rich visual information of explored regions. It further integrates frontier-based exploration by introducing Frontier Snapshots-glimpses of unexplored areas-enabling agents to make informed decisions by considering both known and potential new information. To support lifelong memory in active exploration settings, we present an incremental construction pipeline for 3D-Mem, as well as a memory retrieval technique for memory management. Experimental results on three benchmarks demonstrate that 3D-Mem significantly enhances agents' exploration and reasoning capabilities in 3D environments, highlighting its potential for advancing applications in embodied AI.
    </details>
</div>
