# embodied ai - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v6">OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-07-31
      | 💬 20 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents by leveraging language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising various rendering engines, a versatile toolchain, and a large-scale benchmark for aerial VLN. Firstly, we integrate diverse rendering engines and advanced techniques for environment simulation, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of our environments. Secondly, we develop a highly automated toolchain for aerial VLN data collection, streamlining point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Thirdly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. Moreover, we propose OpenFly-Agent, a keyframe-aware VLN model emphasizing key observations during flight. For benchmarking, extensive experiments and analyses are conducted, evaluating several recent VLN methods and showcasing the superiority of our OpenFly platform and agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.15677v3">Embodied Web Agents: Bridging Physical-Digital Realms for Integrated Agent Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-31
    </div>
    <details class="paper-abstract">
      AI agents today are mostly siloed - they either retrieve and reason over vast amount of digital information and knowledge obtained online; or interact with the physical world through embodied perception, planning and action - but rarely both. This separation limits their ability to solve tasks that require integrated physical and digital intelligence, such as cooking from online recipes, navigating with dynamic map data, or interpreting real-world landmarks using web knowledge. We introduce Embodied Web Agents, a novel paradigm for AI agents that fluidly bridge embodiment and web-scale reasoning. To operationalize this concept, we first develop the Embodied Web Agents task environments, a unified simulation platform that tightly integrates realistic 3D indoor and outdoor environments with functional web interfaces. Building upon this platform, we construct and release the Embodied Web Agents Benchmark, which encompasses a diverse suite of tasks including cooking, navigation, shopping, tourism, and geolocation - all requiring coordinated reasoning across physical and digital realms for systematic assessment of cross-domain intelligence. Experimental results reveal significant performance gaps between state-of-the-art AI systems and human capabilities, establishing both challenges and opportunities at the intersection of embodied cognition and web-scale knowledge access. All datasets, codes and websites are publicly available at our project page https://embodied-web-agent.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23042v1">Early Goal-Guided Multi-Scale Fusion for Real-Time Vision-Language Driving</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      Autonomous vehicles must react in milliseconds while reasoning about road geometry and traffic intent to navigate complex situations. We introduce NovaDrive, a single-branch vision-language architecture that processes front-camera images, HD-map tiles, LiDAR depth, and textual waypoints in a single branch. A lightweight, two-stage cross-attention block first aligns waypoint tokens with the HD map, then refines attention over fine-grained image and depth patches. Coupled with a novel smoothness loss that discourages abrupt steering and speed changes, this design eliminates the need for recurrent memory. We fine-tune the top 15 layers of an 11B LLaMA-3.2 vision-language backbone, enabling real-time inference. On the nuScenes / Waymo subset of the MD-NEX Outdoor benchmark, NovaDrive raises success rate to 84% (+4%), boosts path-efficiency (SPL) to 0.66 (+0.11), and reduces collision frequency from 2.6% to 1.2% (-1.4%) relative to the previous state-of-the-art. Our ablations confirm that waypoint tokens, partial VLM fine-tuning, and the cross-attention fusion each contribute the most to these gains. Beyond safety, NovaDrive's shorter routes (resulting from the novel smoothness loss) translate to lower fuel or battery usage, pointing toward leaner, more easily updated driving stacks. NovaDrive can be extended to other embodied-AI domains as well.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2507.21589v1">Exploring the Link Between Bayesian Inference and Embodied Intelligence: Toward Open Physical-World Embodied AI Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Embodied intelligence posits that cognitive capabilities fundamentally emerge from - and are shaped by - an agent's real-time sensorimotor interactions with its environment. Such adaptive behavior inherently requires continuous inference under uncertainty. Bayesian statistics offers a principled probabilistic framework to address this challenge by representing knowledge as probability distributions and updating beliefs in response to new evidence. The core computational processes underlying embodied intelligence - including perception, action selection, learning, and even higher-level cognition - can be effectively understood and modeled as forms of Bayesian inference. Despite the deep conceptual connection between Bayesian statistics and embodied intelligence, Bayesian principles have not been widely or explicitly applied in today's embodied intelligence systems. In this work, we examine both Bayesian and contemporary embodied intelligence approaches through two fundamental lenses: search and learning - the two central themes in modern AI, as highlighted in Rich Sutton's influential essay "The Bitter Lesson". This analysis sheds light on why Bayesian inference has not played a central role in the development of modern embodied intelligence. At the same time, it reveals that current embodied intelligence systems remain largely confined to closed-physical-world environments, and highlights the potential for Bayesian methods to play a key role in extending these systems toward truly open physical-world embodied intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21589v1">Exploring the Link Between Bayesian Inference and Embodied Intelligence: Toward Open Physical-World Embodied AI Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Embodied intelligence posits that cognitive capabilities fundamentally emerge from - and are shaped by - an agent's real-time sensorimotor interactions with its environment. Such adaptive behavior inherently requires continuous inference under uncertainty. Bayesian statistics offers a principled probabilistic framework to address this challenge by representing knowledge as probability distributions and updating beliefs in response to new evidence. The core computational processes underlying embodied intelligence - including perception, action selection, learning, and even higher-level cognition - can be effectively understood and modeled as forms of Bayesian inference. Despite the deep conceptual connection between Bayesian statistics and embodied intelligence, Bayesian principles have not been widely or explicitly applied in today's embodied intelligence systems. In this work, we examine both Bayesian and contemporary embodied intelligence approaches through two fundamental lenses: search and learning - the two central themes in modern AI, as highlighted in Rich Sutton's influential essay "The Bitter Lesson". This analysis sheds light on why Bayesian inference has not played a central role in the development of modern embodied intelligence. At the same time, it reveals that current embodied intelligence systems remain largely confined to closed-physical-world environments, and highlights the potential for Bayesian methods to play a key role in extending these systems toward truly open physical-world embodied intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21496v1">Multifunctional physical reservoir computing in soft tensegrity robots</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 25 pages, 12 figures. The following article has been accepted by Chaos: An Interdisciplinary Journal of Nonlinear Science
    </div>
    <details class="paper-abstract">
      Recent studies have demonstrated that the dynamics of physical systems can be utilized for the desired information processing under the framework of physical reservoir computing (PRC). Robots with soft bodies are examples of such physical systems, and their nonlinear body-environment dynamics can be used to compute and generate the motor signals necessary for the control of their own behavior. In this simulation study, we extend this approach to control and embed not only one but also multiple behaviors into a type of soft robot called a tensegrity robot. The resulting system, consisting of the robot and the environment, is a multistable dynamical system that converges to different attractors from varying initial conditions. Furthermore, attractor analysis reveals that there exist "untrained attractors" in the state space of the system outside the training data. These untrained attractors reflect the intrinsic properties and structures of the tensegrity robot and its interactions with the environment. The impacts of these recent findings in PRC remain unexplored in embodied AI research. We here illustrate their potential to understand various features of embodied cognition that have not been fully addressed to date.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21045v1">Reconstructing 4D Spatial Intelligence: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Project page: https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence
    </div>
    <details class="paper-abstract">
      Reconstructing 4D spatial intelligence from visual observations has long been a central yet challenging task in computer vision, with broad real-world applications. These range from entertainment domains like movies, where the focus is often on reconstructing fundamental visual elements, to embodied AI, which emphasizes interaction modeling and physical realism. Fueled by rapid advances in 3D representations and deep learning architectures, the field has evolved quickly, outpacing the scope of previous surveys. Additionally, existing surveys rarely offer a comprehensive analysis of the hierarchical structure of 4D scene reconstruction. To address this gap, we present a new perspective that organizes existing methods into five progressive levels of 4D spatial intelligence: (1) Level 1 -- reconstruction of low-level 3D attributes (e.g., depth, pose, and point maps); (2) Level 2 -- reconstruction of 3D scene components (e.g., objects, humans, structures); (3) Level 3 -- reconstruction of 4D dynamic scenes; (4) Level 4 -- modeling of interactions among scene components; and (5) Level 5 -- incorporation of physical laws and constraints. We conclude the survey by discussing the key challenges at each level and highlighting promising directions for advancing toward even richer levels of 4D spatial intelligence. To track ongoing developments, we maintain an up-to-date project page: https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20395v1">MazeEval: A Benchmark for Testing Sequential Decision-Making in Language Models</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly power autonomous agents in robotics and embodied AI, understanding their spatial reasoning capabilities becomes crucial for ensuring reliable real-world deployment. Despite advances in language understanding, current research lacks evaluation of how LLMs perform spatial navigation without visual cues, a fundamental requirement for agents operating with limited sensory information. This paper addresses this gap by introducing MazeEval, a benchmark designed to isolate and evaluate pure spatial reasoning in LLMs through coordinate-based maze navigation tasks. Our methodology employs a function-calling interface where models navigate mazes of varying complexity ($5\times 5$ to $15\times 15$ grids) using only coordinate feedback and distance-to-wall information, excluding visual input to test fundamental spatial cognition. We evaluate eight state-of-the-art LLMs across identical mazes in both English and Icelandic to assess cross-linguistic transfer of spatial abilities. Our findings reveal striking disparities: while OpenAI's O3 achieves perfect navigation for mazes up to size $30\times 30$, other models exhibit catastrophic failure beyond $9\times 9$ mazes, with 100% of failures attributed to excessive looping behavior where models revisit a cell at least 10 times. We document a significant performance degradation in Icelandic, with models solving mazes 3-4 sizes smaller than in English, suggesting spatial reasoning in LLMs emerges from linguistic patterns rather than language-agnostic mechanisms. These results have important implications for global deployment of LLM-powered autonomous systems, showing spatial intelligence remains fundamentally constrained by training data availability and highlighting the need for architectural innovations to achieve reliable navigation across linguistic contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19684v1">Salsa as a Nonverbal Embodied Language -- The CoMPAS3D Dataset and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 https://rosielab.github.io/compas3d
    </div>
    <details class="paper-abstract">
      Imagine a humanoid that can safely and creatively dance with a human, adapting to its partner's proficiency, using haptic signaling as a primary form of communication. While today's AI systems excel at text or voice-based interaction with large language models, human communication extends far beyond text-it includes embodied movement, timing, and physical coordination. Modeling coupled interaction between two agents poses a formidable challenge: it is continuous, bidirectionally reactive, and shaped by individual variation. We present CoMPAS3D, the largest and most diverse motion capture dataset of improvised salsa dancing, designed as a challenging testbed for interactive, expressive humanoid AI. The dataset includes 3 hours of leader-follower salsa dances performed by 18 dancers spanning beginner, intermediate, and professional skill levels. For the first time, we provide fine-grained salsa expert annotations, covering over 2,800 move segments, including move types, combinations, execution errors and stylistic elements. We draw analogies between partner dance communication and natural language, evaluating CoMPAS3D on two benchmark tasks for synthetic humans that parallel key problems in spoken language and dialogue processing: leader or follower generation with proficiency levels (speaker or listener synthesis), and duet (conversation) generation. Towards a long-term goal of partner dance with humans, we release the dataset, annotations, and code, along with a multitask SalsaAgent model capable of performing all benchmark tasks, alongside additional baselines to encourage research in socially interactive embodied AI and creative, expressive humanoid motion generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14501v3">Advances in 4D Generation: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Generative artificial intelligence has recently progressed from static image and video synthesis to 3D content generation, culminating in the emergence of 4D generation-the task of synthesizing temporally coherent dynamic 3D assets guided by user input. As a burgeoning research frontier, 4D generation enables richer interactive and immersive experiences, with applications ranging from digital humans to autonomous driving. Despite rapid progress, the field lacks a unified understanding of 4D representations, generative frameworks, basic paradigms, and the core technical challenges it faces. This survey provides a systematic and in-depth review of the 4D generation landscape. To comprehensively characterize 4D generation, we first categorize fundamental 4D representations and outline associated techniques for 4D generation. We then present an in-depth analysis of representative generative pipelines based on conditions and representation methods. Subsequently, we discuss how motion and geometry priors are integrated into 4D outputs to ensure spatio-temporal consistency under various control schemes. From an application perspective, this paper summarizes 4D generation tasks in areas such as dynamic object/scene generation, digital human synthesis, editable 4D content, and embodied AI. Furthermore, we summarize and multi-dimensionally compare four basic paradigms for 4D generation: End-to-End, Generated-Data-Based, Implicit-Distillation-Based, and Explicit-Supervision-Based. Concluding our analysis, we highlight five key challenges-consistency, controllability, diversity, efficiency, and fidelity-and contextualize these with current approaches.By distilling recent advances and outlining open problems, this work offers a comprehensive and forward-looking perspective to guide future research in 4D generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15610v2">BoxFusion: Reconstruction-Free Open-Vocabulary 3D Object Detection via Real-Time Multi-View Box Fusion</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 Project page: https://lanlan96.github.io/BoxFusion/
    </div>
    <details class="paper-abstract">
      Open-vocabulary 3D object detection has gained significant interest due to its critical applications in autonomous driving and embodied AI. Existing detection methods, whether offline or online, typically rely on dense point cloud reconstruction, which imposes substantial computational overhead and memory constraints, hindering real-time deployment in downstream tasks. To address this, we propose a novel reconstruction-free online framework tailored for memory-efficient and real-time 3D detection. Specifically, given streaming posed RGB-D video input, we leverage Cubify Anything as a pre-trained visual foundation model (VFM) for single-view 3D object detection by bounding boxes, coupled with CLIP to capture open-vocabulary semantics of detected objects. To fuse all detected bounding boxes across different views into a unified one, we employ an association module for correspondences of multi-views and an optimization module to fuse the 3D bounding boxes of the same instance predicted in multi-views. The association module utilizes 3D Non-Maximum Suppression (NMS) and a box correspondence matching module, while the optimization module uses an IoU-guided efficient random optimization technique based on particle filtering to enforce multi-view consistency of the 3D bounding boxes while minimizing computational complexity. Extensive experiments on ScanNetV2 and CA-1M datasets demonstrate that our method achieves state-of-the-art performance among online methods. Benefiting from this novel reconstruction-free paradigm for 3D object detection, our method exhibits great generalization abilities in various scenarios, enabling real-time perception even in environments exceeding 1000 square meters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.05171v2">Towards provable probabilistic safety for scalable embodied AI systems</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Embodied AI systems, comprising AI models and physical plants, are increasingly prevalent across various applications. Due to the rarity of system failures, ensuring their safety in complex operating environments remains a major challenge, which severely hinders their large-scale deployment in safety-critical domains, such as autonomous vehicles, medical devices, and robotics. While achieving provable deterministic safety--verifying system safety across all possible scenarios--remains theoretically ideal, the rarity and complexity of corner cases make this approach impractical for scalable embodied AI systems. Instead, empirical safety evaluation is employed as an alternative, but the absence of provable guarantees imposes significant limitations. To address these issues, we argue for a paradigm shift to provable probabilistic safety that integrates provable guarantees with progressive achievement toward a probabilistic safety boundary on overall system performance. The new paradigm better leverages statistical methods to enhance feasibility and scalability, and a well-defined probabilistic safety boundary enables embodied AI systems to be deployed at scale. In this Perspective, we outline a roadmap for provable probabilistic safety, along with corresponding challenges and potential solutions. By bridging the gap between theoretical safety assurance and practical deployment, this Perspective offers a pathway toward safer, large-scale adoption of embodied AI systems in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2507.16398v1">AI or Human? Understanding Perceptions of Embodied Robots with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      The pursuit of artificial intelligence has long been associated to the the challenge of effectively measuring intelligence. Even if the Turing Test was introduced as a means of assessing a system intelligence, its relevance and application within the field of human-robot interaction remain largely underexplored. This study investigates the perception of intelligence in embodied robots by performing a Turing Test within a robotic platform. A total of 34 participants were tasked with distinguishing between AI- and human-operated robots while engaging in two interactive tasks: an information retrieval and a package handover. These tasks assessed the robot perception and navigation abilities under both static and dynamic conditions. Results indicate that participants were unable to reliably differentiate between AI- and human-controlled robots beyond chance levels. Furthermore, analysis of participant responses reveals key factors influencing the perception of artificial versus human intelligence in embodied robotic systems. These findings provide insights into the design of future interactive robots and contribute to the ongoing discourse on intelligence assessment in AI-driven systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16815v1">ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Project page: https://jasper0314-huang.github.io/thinkact-vla/
    </div>
    <details class="paper-abstract">
      Vision-language-action (VLA) reasoning tasks require agents to interpret multimodal instructions, perform long-horizon planning, and act adaptively in dynamic environments. Existing approaches typically train VLA models in an end-to-end fashion, directly mapping inputs to actions without explicit reasoning, which hinders their ability to plan over multiple steps or adapt to complex task variations. In this paper, we propose ThinkAct, a dual-system framework that bridges high-level reasoning with low-level action execution via reinforced visual latent planning. ThinkAct trains a multimodal LLM to generate embodied reasoning plans guided by reinforcing action-aligned visual rewards based on goal completion and trajectory consistency. These reasoning plans are compressed into a visual plan latent that conditions a downstream action model for robust action execution on target environments. Extensive experiments on embodied reasoning and robot manipulation benchmarks demonstrate that ThinkAct enables few-shot adaptation, long-horizon planning, and self-correction behaviors in complex embodied AI tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05171v2">Towards provable probabilistic safety for scalable embodied AI systems</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Embodied AI systems, comprising AI models and physical plants, are increasingly prevalent across various applications. Due to the rarity of system failures, ensuring their safety in complex operating environments remains a major challenge, which severely hinders their large-scale deployment in safety-critical domains, such as autonomous vehicles, medical devices, and robotics. While achieving provable deterministic safety--verifying system safety across all possible scenarios--remains theoretically ideal, the rarity and complexity of corner cases make this approach impractical for scalable embodied AI systems. Instead, empirical safety evaluation is employed as an alternative, but the absence of provable guarantees imposes significant limitations. To address these issues, we argue for a paradigm shift to provable probabilistic safety that integrates provable guarantees with progressive achievement toward a probabilistic safety boundary on overall system performance. The new paradigm better leverages statistical methods to enhance feasibility and scalability, and a well-defined probabilistic safety boundary enables embodied AI systems to be deployed at scale. In this Perspective, we outline a roadmap for provable probabilistic safety, along with corresponding challenges and potential solutions. By bridging the gap between theoretical safety assurance and practical deployment, this Perspective offers a pathway toward safer, large-scale adoption of embodied AI systems in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20041v2">Learning Streaming Video Representation via Multitask Training</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Technical Report. Project Page: https://go2heart.github.io/streamformer
    </div>
    <details class="paper-abstract">
      Understanding continuous video streams plays a fundamental role in real-time applications including embodied AI and autonomous driving. Unlike offline video understanding, streaming video understanding requires the ability to process video streams frame by frame, preserve historical information, and make low-latency decisions. To address these challenges, our main contributions are three-fold. (i) We develop a novel streaming video backbone, termed as StreamFormer, by incorporating causal temporal attention into a pre-trained vision transformer. This enables efficient streaming video processing while maintaining image representation capability. (ii) To train StreamFormer, we propose to unify diverse spatial-temporal video understanding tasks within a multitask visual-language alignment framework. Hence, StreamFormer learns global semantics, temporal dynamics, and fine-grained spatial relationships simultaneously. (iii) We conduct extensive experiments on online action detection, online video instance segmentation, and video question answering. StreamFormer achieves competitive results while maintaining efficiency, demonstrating its potential for real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01559v2">PR2: A Physics- and Photo-realistic Humanoid Testbed with Pilot Study in Competition</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      This paper presents the development of a Physics-realistic and Photo-realistic humanoid robot testbed, PR2, to facilitate collaborative research between Embodied Artificial Intelligence (Embodied AI) and robotics. PR2 offers high-quality scene rendering and robot dynamic simulation, enabling (i) the creation of diverse scenes using various digital assets, (ii) the integration of advanced perception or foundation models, and (iii) the implementation of planning and control algorithms for dynamic humanoid robot behaviors based on environmental feedback. The beta version of PR2 has been deployed for the simulation track of a nationwide full-size humanoid robot competition for college students, attracting 137 teams and over 400 participants within four months. This competition covered traditional tasks in bipedal walking, as well as novel challenges in loco-manipulation and language-instruction-based object search, marking a first for public college robotics competitions. A retrospective analysis of the competition suggests that future events should emphasize the integration of locomotion with manipulation and perception. By making the PR2 testbed publicly available at https://github.com/pr2-humanoid/PR2-Platform, we aim to further advance education and training in humanoid robotics. Video demonstration: https://pr2-humanoid.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15428v1">EgoPrune: Efficient Token Pruning for Egomotion Video Reasoning in Embodied Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Egomotion videos are first-person recordings where the view changes continuously due to the agent's movement. As they serve as the primary visual input for embodied AI agents, making egomotion video reasoning more efficient is therefore essential for real-world deployment. Recent advances in vision-language models have enabled strong multimodal reasoning capabilities, but their computational cost remains prohibitive for long, redundant video inputs. Existing token pruning methods, typically designed for third-person videos, fail to leverage the spatiotemporal continuity and motion constraints inherent in egomotion settings. To address this, we propose EgoPrune, a training-free token pruning method tailored for egomotion video reasoning. EgoPrune comprises three components: a keyframe selector adapted from EmbodiedR for temporally efficient sampling; Perspective-Aware Redundancy Filtering (PARF), which aligns visual tokens using perspective transformations and removes redundant tokens; and a Maximal Marginal Relevance (MMR)-based token selector that jointly considers visual-text relevance and intra-frame diversity. Experiments on two egomotion video benchmarks show that EgoPrune consistently outperforms prior training-free methods across various pruning ratios while significantly reducing FLOPs, memory usage, and latency. Moreover, we deploy EgoPrune on an embodied agent equipped with a Jetson Orin NX 16GB edge device, demonstrating its real-world efficiency and suitability for on-device egomotion video reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12465v3">PhysX-3D: Physical-Grounded 3D Asset Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 Project page: https://physx-3d.github.io/
    </div>
    <details class="paper-abstract">
      3D modeling is moving from virtual to physical. Existing 3D generation primarily emphasizes geometries and textures while neglecting physical-grounded modeling. Consequently, despite the rapid development of 3D generative models, the synthesized 3D assets often overlook rich and important physical properties, hampering their real-world application in physical domains like simulation and embodied AI. As an initial attempt to address this challenge, we propose \textbf{PhysX-3D}, an end-to-end paradigm for physical-grounded 3D asset generation. 1) To bridge the critical gap in physics-annotated 3D datasets, we present PhysXNet - the first physics-grounded 3D dataset systematically annotated across five foundational dimensions: absolute scale, material, affordance, kinematics, and function description. In particular, we devise a scalable human-in-the-loop annotation pipeline based on vision-language models, which enables efficient creation of physics-first assets from raw 3D assets.2) Furthermore, we propose \textbf{PhysXGen}, a feed-forward framework for physics-grounded image-to-3D asset generation, injecting physical knowledge into the pre-trained 3D structural space. Specifically, PhysXGen employs a dual-branch architecture to explicitly model the latent correlations between 3D structures and physical properties, thereby producing 3D assets with plausible physical predictions while preserving the native geometry quality. Extensive experiments validate the superior performance and promising generalization capability of our framework. All the code, data, and models will be released to facilitate future research in generative physical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02247v5">WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Object Goal Navigation-requiring an agent to locate a specific object in an unseen environment-remains a core challenge in embodied AI. Although recent progress in Vision-Language Model (VLM)-based agents has demonstrated promising perception and decision-making abilities through prompting, none has yet established a fully modular world model design that reduces risky and costly interactions with the environment by predicting the future state of the world. We introduce WMNav, a novel World Model-based Navigation framework powered by Vision-Language Models (VLMs). It predicts possible outcomes of decisions and builds memories to provide feedback to the policy module. To retain the predicted state of the environment, WMNav proposes the online maintained Curiosity Value Map as part of the world model memory to provide dynamic configuration for navigation policy. By decomposing according to a human-like thinking process, WMNav effectively alleviates the impact of model hallucination by making decisions based on the feedback difference between the world model plan and observation. To further boost efficiency, we implement a two-stage action proposer strategy: broad exploration followed by precise localization. Extensive evaluation on HM3D and MP3D validates WMNav surpasses existing zero-shot benchmarks in both success rate and exploration efficiency (absolute improvement: +3.2% SR and +3.2% SPL on HM3D, +13.5% SR and +1.1% SPL on MP3D). Project page: https://b0b8k1ng.github.io/WMNav/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16411v2">The Duality of Generative AI and Reinforcement Learning in Robotics: A Review</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Submitted for publication to Information Fusion
    </div>
    <details class="paper-abstract">
      Recently, generative AI and reinforcement learning (RL) have been redefining what is possible for AI agents that take information flows as input and produce intelligent behavior. As a result, we are seeing similar advancements in embodied AI and robotics for control policy generation. Our review paper examines the integration of generative AI models with RL to advance robotics. Our primary focus is on the duality between generative AI and RL for robotics downstream tasks. Specifically, we investigate: (1) The role of prominent generative AI tools as modular priors for multi-modal input fusion in RL tasks. (2) How RL can train, fine-tune and distill generative models for policy generation, such as VLA models, similarly to RL applications in large language models. We then propose a new taxonomy based on a considerable amount of selected papers. Lastly, we identify open challenges accounting for model scalability, adaptation and grounding, giving recommendations and insights on future research directions. We reflect on which generative AI models best fit the RL tasks and why. On the other side, we reflect on important issues inherent to RL-enhanced generative policies, such as safety concerns and failure modes, and what are the limitations of current methods. A curated collection of relevant research papers is maintained on our GitHub repository, serving as a resource for ongoing research and development in this field: https://github.com/clmoro/Robotics-RL-FMs-Integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12465v2">PhysX: Physical-Grounded 3D Asset Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 Project page: https://physx-3d.github.io/
    </div>
    <details class="paper-abstract">
      3D modeling is moving from virtual to physical. Existing 3D generation primarily emphasizes geometries and textures while neglecting physical-grounded modeling. Consequently, despite the rapid development of 3D generative models, the synthesized 3D assets often overlook rich and important physical properties, hampering their real-world application in physical domains like simulation and embodied AI. As an initial attempt to address this challenge, we propose \textbf{PhysX}, an end-to-end paradigm for physical-grounded 3D asset generation. 1) To bridge the critical gap in physics-annotated 3D datasets, we present PhysXNet - the first physics-grounded 3D dataset systematically annotated across five foundational dimensions: absolute scale, material, affordance, kinematics, and function description. In particular, we devise a scalable human-in-the-loop annotation pipeline based on vision-language models, which enables efficient creation of physics-first assets from raw 3D assets.2) Furthermore, we propose \textbf{PhysXGen}, a feed-forward framework for physics-grounded image-to-3D asset generation, injecting physical knowledge into the pre-trained 3D structural space. Specifically, PhysXGen employs a dual-branch architecture to explicitly model the latent correlations between 3D structures and physical properties, thereby producing 3D assets with plausible physical predictions while preserving the native geometry quality. Extensive experiments validate the superior performance and promising generalization capability of our framework. All the code, data, and models will be released to facilitate future research in generative physical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23765v6">STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      The use of Multimodal Large Language Models (MLLMs) as an end-to-end solution for Embodied AI and Autonomous Driving has become a prevailing trend. While MLLMs have been extensively studied for visual semantic understanding tasks, their ability to perform precise and quantitative spatial-temporal understanding in real-world applications remains largely unexamined, leading to uncertain prospects. To evaluate models' Spatial-Temporal Intelligence, we introduce STI-Bench, a benchmark designed to evaluate MLLMs' spatial-temporal understanding through challenging tasks such as estimating and predicting the appearance, pose, displacement, and motion of objects. Our benchmark encompasses a wide range of robot and vehicle operations across desktop, indoor, and outdoor scenarios. The extensive experiments reveals that the state-of-the-art MLLMs still struggle in real-world spatial-temporal understanding, especially in tasks requiring precise distance estimation and motion analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12465v1">PhysX: Physical-Grounded 3D Asset Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Project page: https://physx-3d.github.io/
    </div>
    <details class="paper-abstract">
      3D modeling is moving from virtual to physical. Existing 3D generation primarily emphasizes geometries and textures while neglecting physical-grounded modeling. Consequently, despite the rapid development of 3D generative models, the synthesized 3D assets often overlook rich and important physical properties, hampering their real-world application in physical domains like simulation and embodied AI. As an initial attempt to address this challenge, we propose \textbf{PhysX}, an end-to-end paradigm for physical-grounded 3D asset generation. 1) To bridge the critical gap in physics-annotated 3D datasets, we present PhysXNet - the first physics-grounded 3D dataset systematically annotated across five foundational dimensions: absolute scale, material, affordance, kinematics, and function description. In particular, we devise a scalable human-in-the-loop annotation pipeline based on vision-language models, which enables efficient creation of physics-first assets from raw 3D assets.2) Furthermore, we propose \textbf{PhysXGen}, a feed-forward framework for physics-grounded image-to-3D asset generation, injecting physical knowledge into the pre-trained 3D structural space. Specifically, PhysXGen employs a dual-branch architecture to explicitly model the latent correlations between 3D structures and physical properties, thereby producing 3D assets with plausible physical predictions while preserving the native geometry quality. Extensive experiments validate the superior performance and promising generalization capability of our framework. All the code, data, and models will be released to facilitate future research in generative physical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12955v2">HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      We propose a new task to benchmark human-in-scene understanding for embodied agents: Human-In-Scene Question Answering (HIS-QA). Given a human motion within a 3D scene, HIS-QA requires the agent to comprehend human states and behaviors, reason about its surrounding environment, and answer human-related questions within the scene. To support this new task, we present HIS-Bench, a multimodal benchmark that systematically evaluates HIS understanding across a broad spectrum, from basic perception to commonsense reasoning and planning. Our evaluation of various vision-language models on HIS-Bench reveals significant limitations in their ability to handle HIS-QA tasks. To this end, we propose HIS-GPT, the first foundation model for HIS understanding. HIS-GPT integrates 3D scene context and human motion dynamics into large language models while incorporating specialized mechanisms to capture human-scene interactions. Extensive experiments demonstrate that HIS-GPT sets a new state-of-the-art on HIS-QA tasks. We hope this work inspires future research on human behavior analysis in 3D scenes, advancing embodied AI and world models. The codes and data: https://github.com/ZJHTerry18/HumanInScene.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2507.10812v1">React to This (RTT): A Nonverbal Turing Test for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 5 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We propose an approach to test embodied AI agents for interaction awareness and believability, particularly in scenarios where humans push them to their limits. Turing introduced the Imitation Game as a way to explore the question: "Can machines think?" The Total Turing Test later expanded this concept beyond purely verbal communication, incorporating perceptual and physical interaction. Building on this, we propose a new guiding question: "Can machines react?" and introduce the React to This (RTT) test for nonverbal behaviors, presenting results from an initial experiment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00917v2">A Survey: Learning Embodied Intelligence from Physical Simulators and World Models</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 49pages, 25figures, 6tables, github repository avalible in https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey
    </div>
    <details class="paper-abstract">
      The pursuit of artificial general intelligence (AGI) has placed embodied intelligence at the forefront of robotics research. Embodied intelligence focuses on agents capable of perceiving, reasoning, and acting within the physical world. Achieving robust embodied intelligence requires not only advanced perception and control, but also the ability to ground abstract cognition in real-world interactions. Two foundational technologies, physical simulators and world models, have emerged as critical enablers in this quest. Physical simulators provide controlled, high-fidelity environments for training and evaluating robotic agents, allowing safe and efficient development of complex behaviors. In contrast, world models empower robots with internal representations of their surroundings, enabling predictive planning and adaptive decision-making beyond direct sensory input. This survey systematically reviews recent advances in learning embodied AI through the integration of physical simulators and world models. We analyze their complementary roles in enhancing autonomy, adaptability, and generalization in intelligent robots, and discuss the interplay between external simulation and internal modeling in bridging the gap between simulated training and real-world deployment. By synthesizing current progress and identifying open challenges, this survey aims to provide a comprehensive perspective on the path toward more capable and generalizable embodied AI systems. We also maintain an active repository that contains up-to-date literature and open-source projects at https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02029v3">RoboBrain 2.0 Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model, featuring a heterogeneous architecture with a vision encoder and a language model. Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models. In particular, it supports key real-world embodied AI capabilities, including spatial understanding (e.g., affordance prediction, spatial referring, trajectory forecasting) and temporal decision-making (e.g., closed-loop interaction, multi-agent long-horizon planning, and scene graph updating). This report details the model architecture, data construction, multi-stage training strategies, infrastructure and practical applications. We hope RoboBrain 2.0 advances embodied AI research and serves as a practical step toward building generalist embodied agents. The code, checkpoint and benchmark are available at https://superrobobrain.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08831v2">View Invariant Learning for Vision-Language Navigation in Continuous Environments</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation in Continuous Environments (VLNCE), where an agent follows instructions and moves freely to reach a destination, is a key research problem in embodied AI. However, most navigation policies are sensitive to viewpoint changes, i.e., variations in camera height and viewing angle that alter the agent's observation. In this paper, we introduce a generalized scenario, V2-VLNCE (VLNCE with Varied Viewpoints), and propose VIL (View Invariant Learning), a view-invariant post-training strategy that enhances the robustness of existing navigation policies to changes in camera viewpoint. VIL employs a contrastive learning framework to learn sparse and view-invariant features. Additionally, we introduce a teacher-student framework for the Waypoint Predictor Module, a core component of most VLNCE baselines, where a view-dependent teacher model distills knowledge into a view-invariant student model. We employ an end-to-end training paradigm to jointly optimize these components, thus eliminating the cost for individual module training. Empirical results show that our method outperforms state-of-the-art approaches on V2-VLNCE by 8-15% measured on Success Rate for two standard benchmark datasets R2R-CE and RxR-CE. Furthermore, we evaluate VIL under the standard VLNCE setting and find that, despite being trained for varied viewpoints, it often still improves performance. On the more challenging RxR-CE dataset, our method also achieved state-of-the-art performance across all metrics when compared to other map-free methods. This suggests that adding VIL does not diminish the standard viewpoint performance and can serve as a plug-and-play post-training method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10894v1">NavComposer: Composing Language Instructions for Navigation Trajectories through Action-Scene-Object Modularization</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Language-guided navigation is a cornerstone of embodied AI, enabling agents to interpret language instructions and navigate complex environments. However, expert-provided instructions are limited in quantity, while synthesized annotations often lack quality, making them insufficient for large-scale research. To address this, we propose NavComposer, a novel framework for automatically generating high-quality navigation instructions. NavComposer explicitly decomposes semantic entities such as actions, scenes, and objects, and recomposes them into natural language instructions. Its modular architecture allows flexible integration of state-of-the-art techniques, while the explicit use of semantic entities enhances both the richness and accuracy of instructions. Moreover, it operates in a data-agnostic manner, supporting adaptation to diverse navigation trajectories without domain-specific training. Complementing NavComposer, we introduce NavInstrCritic, a comprehensive annotation-free evaluation system that assesses navigation instructions on three dimensions: contrastive matching, semantic consistency, and linguistic diversity. NavInstrCritic provides a holistic evaluation of instruction quality, addressing limitations of traditional metrics that rely heavily on expert annotations. By decoupling instruction generation and evaluation from specific navigation agents, our method enables more scalable and generalizable research. Extensive experiments provide direct and practical evidence for the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09876v1">ViTCoT: Video-Text Interleaved Chain-of-Thought for Boosting Video Understanding in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted by ACM MM 2025
    </div>
    <details class="paper-abstract">
      Video understanding plays a vital role in bridging low-level visual signals with high-level cognitive reasoning, and is fundamental to applications such as autonomous driving, embodied AI, and the broader pursuit of AGI. The rapid development of large language models (LLMs), particularly those utilizing Chain-of-Thought (CoT) technology, has significantly advanced video reasoning capabilities. However, current approaches primarily depend on textual information for reasoning, overlooking the visual modality in the actual video reasoning process. In contrast, humans naturally re-examine visual content while reasoning. Motivated by this, we introduce a novel video reasoning paradigm: Video-Text Interleaved CoT (ViTCoT), which facilitates more intuitive and cognitively aligned reasoning. To the end, first, we construct the Video-Text Interleaved Benchmark (ViTIB), which is created using MLLMs for key-video selection and manually verified. Furthermore, we extensively explore the potential of the ViTCoT paradigm in the video understanding field. Extensive experiments demonstrate that ViTCoT significantly enhances performance compared to the traditional text-only CoT paradigm and effectively activates more neuron values in MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10812v1">React to This (RTT): A Nonverbal Turing Test for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 5 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We propose an approach to test embodied AI agents for interaction awareness and believability, particularly in scenarios where humans push them to their limits. Turing introduced the Imitation Game as a way to explore the question: "Can machines think?" The Total Turing Test later expanded this concept beyond purely verbal communication, incorporating perceptual and physical interaction. Building on this, we propose a new guiding question: "Can machines react?" and introduce the React to This (RTT) test for nonverbal behaviors, presenting results from an initial experiment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22050v2">Reinforced Reasoning for Embodied Planning</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Embodied planning requires agents to make coherent multi-step decisions based on dynamic visual observations and natural language goals. While recent vision-language models (VLMs) excel at static perception tasks, they struggle with the temporal reasoning, spatial understanding, and commonsense grounding needed for planning in interactive environments. In this work, we introduce a reinforcement fine-tuning framework that brings R1-style reasoning enhancement into embodied planning. We first distill a high-quality dataset from a powerful closed-source model and perform supervised fine-tuning (SFT) to equip the model with structured decision-making priors. We then design a rule-based reward function tailored to multi-step action quality and optimize the policy via Generalized Reinforced Preference Optimization (GRPO). Our approach is evaluated on Embench, a recent benchmark for interactive embodied tasks, covering both in-domain and out-of-domain scenarios. Experimental results show that our method significantly outperforms models of similar or larger scale, including GPT-4o-mini and 70B+ open-source baselines, and exhibits strong generalization to unseen environments. This work highlights the potential of reinforcement-driven reasoning to advance long-horizon planning in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09217v1">Online Long-term Point Tracking in the Foundation Model Era</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 arXiv admin note: substantial text overlap with arXiv:2501.18487
    </div>
    <details class="paper-abstract">
      Point tracking aims to identify the same physical point across video frames and serves as a geometry-aware representation of motion. This representation supports a wide range of applications, from robotics to augmented reality, by enabling accurate modeling of dynamic environments. Most existing long-term tracking approaches operate in an offline setting, where future frames are available to refine predictions and recover from occlusions. However, real-world scenarios often demand online predictions: the model must operate causally, using only current and past frames. This constraint is critical in streaming video and embodied AI, where decisions must be made immediately based on past observations. Under such constraints, viewpoint invariance becomes essential. Visual foundation models, trained on diverse large-scale datasets, offer the potential for robust geometric representations. While they lack temporal reasoning on their own, they can be integrated into tracking pipelines to enrich spatial features. In this thesis, we address the problem of long-term point tracking in an online setting, where frames are processed sequentially without access to future information or sliding windows. We begin by evaluating the suitability of visual foundation models for this task and find that they can serve as useful initializations and be integrated into tracking pipelines. However, to enable long-term tracking in an online setting, a dedicated design is still required. In particular, maintaining coherence over time in this causal regime requires memory to propagate appearance and context across frames. To address this, we introduce Track-On, a transformer-based model that treats each tracked point as a query and processes video frames one at a time. Track-On sets a new state of the art across seven public benchmarks, demonstrating the feasibility of long-term tracking without future access.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08496v1">LLaPa: A Vision-Language Model Framework for Counterfactual-Aware Procedural Planning</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have advanced procedural planning for embodied AI systems through strong reasoning abilities, the integration of multimodal inputs and counterfactual reasoning remains underexplored. To tackle these challenges, we introduce LLaPa, a vision-language model framework designed for multimodal procedural planning. LLaPa generates executable action sequences from textual task descriptions and visual environmental images using vision-language models (VLMs). Furthermore, we enhance LLaPa with two auxiliary modules to improve procedural planning. The first module, the Task-Environment Reranker (TER), leverages task-oriented segmentation to create a task-sensitive feature space, aligning textual descriptions with visual environments and emphasizing critical regions for procedural execution. The second module, the Counterfactual Activities Retriever (CAR), identifies and emphasizes potential counterfactual conditions, enhancing the model's reasoning capability in counterfactual scenarios. Extensive experiments on ActPlan-1K and ALFRED benchmarks demonstrate that LLaPa generates higher-quality plans with superior LCS and correctness, outperforming advanced models. The code and models are available https://github.com/sunshibo1234/LLaPa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07781v1">SURPRISE3D: A Dataset for Spatial Understanding and Reasoning in Complex 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      The integration of language and 3D perception is critical for embodied AI and robotic systems to perceive, understand, and interact with the physical world. Spatial reasoning, a key capability for understanding spatial relationships between objects, remains underexplored in current 3D vision-language research. Existing datasets often mix semantic cues (e.g., object name) with spatial context, leading models to rely on superficial shortcuts rather than genuinely interpreting spatial relationships. To address this gap, we introduce S\textsc{urprise}3D, a novel dataset designed to evaluate language-guided spatial reasoning segmentation in complex 3D scenes. S\textsc{urprise}3D consists of more than 200k vision language pairs across 900+ detailed indoor scenes from ScanNet++ v2, including more than 2.8k unique object classes. The dataset contains 89k+ human-annotated spatial queries deliberately crafted without object name, thereby mitigating shortcut biases in spatial understanding. These queries comprehensively cover various spatial reasoning skills, such as relative position, narrative perspective, parametric perspective, and absolute distance reasoning. Initial benchmarks demonstrate significant challenges for current state-of-the-art expert 3D visual grounding methods and 3D-LLMs, underscoring the necessity of our dataset and the accompanying 3D Spatial Reasoning Segmentation (3D-SRS) benchmark suite. S\textsc{urprise}3D and 3D-SRS aim to facilitate advancements in spatially aware AI, paving the way for effective embodied interaction and robotic planning. The code and datasets can be found in https://github.com/liziwennba/SUPRISE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13451v4">MapNav: A Novel Memory Representation via Annotated Semantic Maps for Vision-and-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Vision-and-language navigation (VLN) is a key task in Embodied AI, requiring agents to navigate diverse and unseen environments while following natural language instructions. Traditional approaches rely heavily on historical observations as spatio-temporal contexts for decision making, leading to significant storage and computational overhead. In this paper, we introduce MapNav, a novel end-to-end VLN model that leverages Annotated Semantic Map (ASM) to replace historical frames. Specifically, our approach constructs a top-down semantic map at the start of each episode and update it at each timestep, allowing for precise object mapping and structured navigation information. Then, we enhance this map with explicit textual labels for key regions, transforming abstract semantics into clear navigation cues and generate our ASM. MapNav agent using the constructed ASM as input, and use the powerful end-to-end capabilities of VLM to empower VLN. Extensive experiments demonstrate that MapNav achieves state-of-the-art (SOTA) performance in both simulated and real-world environments, validating the effectiveness of our method. Moreover, we will release our ASM generation source code and dataset to ensure reproducibility, contributing valuable resources to the field. We believe that our proposed MapNav can be used as a new memory representation method in VLN, paving the way for future research in this field.
    </details>
</div>
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
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.22355v3">Embodied AI Agents: Modeling the World</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      This paper describes our research on AI agents embodied in visual, virtual or physical forms, enabling them to interact with both users and their environments. These agents, which include virtual avatars, wearable devices, and robots, are designed to perceive, learn and act within their surroundings, which makes them more similar to how humans learn and interact with the environments as compared to disembodied agents. We propose that the development of world models is central to reasoning and planning of embodied AI agents, allowing these agents to understand and predict their environment, to understand user intentions and social contexts, thereby enhancing their ability to perform complex tasks autonomously. World modeling encompasses the integration of multimodal perception, planning through reasoning for action and control, and memory to create a comprehensive understanding of the physical world. Beyond the physical world, we also propose to learn the mental world model of users to enable better human-agent collaboration.
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08831v1">View Invariant Learning for Vision-Language Navigation in Continuous Environments</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation in Continuous Environments (VLNCE), where an agent follows instructions and moves freely to reach a destination, is a key research problem in embodied AI. However, most navigation policies are sensitive to viewpoint changes, i.e., variations in camera height and viewing angle that alter the agent's observation. In this paper, we introduce a generalized scenario, V2-VLNCE (VLNCE with Varied Viewpoints), and propose VIL (View Invariant Learning), a view-invariant post-training strategy that enhances the robustness of existing navigation policies to changes in camera viewpoint. VIL employs a contrastive learning framework to learn sparse and view-invariant features. Additionally, we introduce a teacher-student framework for the Waypoint Predictor Module, a core component of most VLNCE baselines, where a view-dependent teacher model distills knowledge into a view-invariant student model. We employ an end-to-end training paradigm to jointly optimize these components, thus eliminating the cost for individual module training. Empirical results show that our method outperforms state-of-the-art approaches on V2-VLNCE by 8-15% measured on Success Rate for two standard benchmark datasets R2R-CE and RxR-CE. Furthermore, we evaluate VIL under the standard VLNCE setting and find that, despite being trained for varied viewpoints, it often still improves performance. On the more challenging RxR-CE dataset, our method also achieved state-of-the-art performance across all metrics when compared to other map-free methods. This suggests that adding VIL does not diminish the standard viewpoint performance and can serve as a plug-and-play post-training method.
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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.24009v1">Bridging Physical and Digital Worlds: Embodied Large AI for Future Wireless Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large artificial intelligence (AI) models offer revolutionary potential for future wireless systems, promising unprecedented capabilities in network optimization and performance. However, current paradigms largely overlook crucial physical interactions. This oversight means they primarily rely on offline datasets, leading to difficulties in handling real-time wireless dynamics and non-stationary environments. Furthermore, these models often lack the capability for active environmental probing. This paper proposes a fundamental paradigm shift towards wireless embodied large AI (WELAI), moving from passive observation to active embodiment. We first identify key challenges faced by existing models, then we explore the design principles and system structure of WELAI. Besides, we outline prospective applications in next-generation wireless. Finally, through an illustrative case study, we demonstrate the effectiveness of WELAI and point out promising research directions for realizing adaptive, robust, and autonomous wireless systems.
    </details>
</div>
