# embodied ai - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25760v1">Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Humans possess spatial reasoning abilities that enable them to understand spaces through multimodal observations, such as vision and sound. Large multimodal reasoning models extend these abilities by learning to perceive and reason, showing promising performance across diverse spatial tasks. However, systematic reviews and publicly available benchmarks for these models remain limited. In this survey, we provide a comprehensive review of multimodal spatial reasoning tasks with large models, categorizing recent progress in multimodal large language models (MLLMs) and introducing open benchmarks for evaluation. We begin by outlining general spatial reasoning, focusing on post-training techniques, explainability, and architecture. Beyond classical 2D tasks, we examine spatial relationship reasoning, scene and layout understanding, as well as visual question answering and grounding in 3D space. We also review advances in embodied AI, including vision-language navigation and action models. Additionally, we consider emerging modalities such as audio and egocentric video, which contribute to novel spatial understanding through new sensors. We believe this survey establishes a solid foundation and offers insights into the growing field of multimodal spatial reasoning. Updated information about this survey, codes and implementation of the open benchmarks can be found at https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25268v1">SynHLMA:Synthesizing Hand Language Manipulation for Articulated Object with Discrete Human Object Interaction Representation</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Generating hand grasps with language instructions is a widely studied topic that benefits from embodied AI and VR/AR applications. While transferring into hand articulatied object interaction (HAOI), the hand grasps synthesis requires not only object functionality but also long-term manipulation sequence along the object deformation. This paper proposes a novel HAOI sequence generation framework SynHLMA, to synthesize hand language manipulation for articulated objects. Given a complete point cloud of an articulated object, we utilize a discrete HAOI representation to model each hand object interaction frame. Along with the natural language embeddings, the representations are trained by an HAOI manipulation language model to align the grasping process with its language description in a shared representation space. A joint-aware loss is employed to ensure hand grasps follow the dynamic variations of articulated object joints. In this way, our SynHLMA achieves three typical hand manipulation tasks for articulated objects of HAOI generation, HAOI prediction and HAOI interpolation. We evaluate SynHLMA on our built HAOI-lang dataset and experimental results demonstrate the superior hand grasp sequence generation performance comparing with state-of-the-art. We also show a robotics grasp application that enables dexterous grasps execution from imitation learning using the manipulation sequence provided by our SynHLMA. Our codes and datasets will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20414v2">SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 Accepted by NeurIPS 2025, 26 pages
    </div>
    <details class="paper-abstract">
      Indoor scene synthesis has become increasingly important with the rise of Embodied AI, which requires 3D environments that are not only visually realistic but also physically plausible and functionally diverse. While recent approaches have advanced visual fidelity, they often remain constrained to fixed scene categories, lack sufficient object-level detail and physical consistency, and struggle to align with complex user instructions. In this work, we present SceneWeaver, a reflective agentic framework that unifies diverse scene synthesis paradigms through tool-based iterative refinement. At its core, SceneWeaver employs a language model-based planner to select from a suite of extensible scene generation tools, ranging from data-driven generative models to visual- and LLM-based methods, guided by self-evaluation of physical plausibility, visual realism, and semantic alignment with user input. This closed-loop reason-act-reflect design enables the agent to identify semantic inconsistencies, invoke targeted tools, and update the environment over successive iterations. Extensive experiments on both common and open-vocabulary room types demonstrate that SceneWeaver not only outperforms prior methods on physical, visual, and semantic metrics, but also generalizes effectively to complex scenes with diverse instructions, marking a step toward general-purpose 3D environment generation. Project website: https://scene-weaver.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16258v4">MEReQ: Max-Ent Residual-Q Inverse RL for Sample-Efficient Alignment from Intervention</a></div>
    <div class="paper-meta">
      📅 2025-10-24
    </div>
    <details class="paper-abstract">
      Aligning robot behavior with human preferences is crucial for deploying embodied AI agents in human-centered environments. A promising solution is interactive imitation learning from human intervention, where a human expert observes the policy's execution and provides interventions as feedback. However, existing methods often fail to utilize the prior policy efficiently to facilitate learning, thus hindering sample efficiency. In this work, we introduce MEReQ (Maximum-Entropy Residual-Q Inverse Reinforcement Learning), designed for sample-efficient alignment from human intervention. Instead of inferring the complete human behavior characteristics, MEReQ infers a residual reward function that captures the discrepancy between the human expert's and the prior policy's underlying reward functions. It then employs Residual Q-Learning (RQL) to align the policy with human preferences using this residual reward function. Extensive evaluations on simulated and real-world tasks demonstrate that MEReQ achieves sample-efficient policy alignment from human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21991v1">Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising</a></div>
    <div class="paper-meta">
      📅 2025-10-24
      | 💬 16 pages, 11 figure, 2 tables, accepted at Neurips 2025
    </div>
    <details class="paper-abstract">
      Diffusion models, such as diffusion policy, have achieved state-of-the-art results in robotic manipulation by imitating expert demonstrations. While diffusion models were originally developed for vision tasks like image and video generation, many of their inference strategies have been directly transferred to control domains without adaptation. In this work, we show that by tailoring the denoising process to the specific characteristics of embodied AI tasks -- particularly structured, low-dimensional nature of action distributions -- diffusion policies can operate effectively with as few as 5 neural function evaluations (NFE). Building on this insight, we propose a population-based sampling strategy, genetic denoising, which enhances both performance and stability by selecting denoising trajectories with low out-of-distribution risk. Our method solves challenging tasks with only 2 NFE while improving or matching performance. We evaluate our approach across 14 robotic manipulation tasks from D4RL and Robomimic, spanning multiple action horizons and inference budgets. In over 2 million evaluations, our method consistently outperforms standard diffusion-based policies, achieving up to 20\% performance gains with significantly fewer inference steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20578v1">EmbodiedBrain: Expanding Performance Boundaries of Task Planning for Embodied Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-10-23
    </div>
    <details class="paper-abstract">
      The realization of Artificial General Intelligence (AGI) necessitates Embodied AI agents capable of robust spatial perception, effective task planning, and adaptive execution in physical environments. However, current large language models (LLMs) and multimodal LLMs (MLLMs) for embodied tasks suffer from key limitations, including a significant gap between model design and agent requirements, an unavoidable trade-off between real-time latency and performance, and the use of unauthentic, offline evaluation metrics. To address these challenges, we propose EmbodiedBrain, a novel vision-language foundation model available in both 7B and 32B parameter sizes. Our framework features an agent-aligned data structure and employs a powerful training methodology that integrates large-scale Supervised Fine-Tuning (SFT) with Step-Augumented Group Relative Policy Optimization (Step-GRPO), which boosts long-horizon task success by integrating preceding steps as Guided Precursors. Furthermore, we incorporate a comprehensive reward system, including a Generative Reward Model (GRM) accelerated at the infrastructure level, to improve training efficiency. For enable thorough validation, we establish a three-part evaluation system encompassing General, Planning, and End-to-End Simulation Benchmarks, highlighted by the proposal and open-sourcing of a novel, challenging simulation environment. Experimental results demonstrate that EmbodiedBrain achieves superior performance across all metrics, establishing a new state-of-the-art for embodied foundation models. Towards paving the way for the next generation of generalist embodied agents, we open-source all of our data, model weight, and evaluating methods, which are available at https://zterobot.github.io/EmbodiedBrain.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05341v2">Direct Numerical Layout Generation for 3D Indoor Scene Synthesis via Spatial Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-23
      | 💬 Project Page: https://directlayout.github.io/
    </div>
    <details class="paper-abstract">
      Realistic 3D indoor scene synthesis is vital for embodied AI and digital content creation. It can be naturally divided into two subtasks: object generation and layout generation. While recent generative models have significantly advanced object-level quality and controllability, layout generation remains challenging due to limited datasets. Existing methods either overfit to these datasets or rely on predefined constraints to optimize numerical layout that sacrifice flexibility. As a result, they fail to generate scenes that are both open-vocabulary and aligned with fine-grained user instructions. We introduce DirectLayout, a framework that directly generates numerical 3D layouts from text descriptions using generalizable spatial reasoning of large language models (LLMs). DirectLayout decomposes the generation into three stages: producing a Bird's-Eye View (BEV) layout, lifting it into 3D space, and refining object placements. To enable explicit spatial reasoning and help the model grasp basic principles of object placement, we employ Chain-of-Thought (CoT) Activation based on the 3D-Front dataset. Additionally, we design CoT-Grounded Generative Layout Reward to enhance generalization and spatial planning. During inference, DirectLayout addresses asset-layout mismatches via Iterative Asset-Layout Alignment through in-context learning. Extensive experiments demonstrate that DirectLayout achieves impressive semantic consistency, generalization and physical plausibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19400v1">Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 The project and benchmark are publicly available at https://github.com/microsoft/MV-RoboBench
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) are essential to Embodied AI, enabling robots to perceive, reason, and act in complex environments. They also serve as the foundation for the recent Vision-Language-Action (VLA) models. Yet most evaluations of VLMs focus on single-view settings, leaving their ability to integrate multi-view information underexplored. At the same time, multi-camera setups are increasingly standard in robotic platforms, as they provide complementary perspectives to mitigate occlusion and depth ambiguity. Whether VLMs can effectively leverage such multi-view inputs for robotic reasoning therefore remains an open question. To bridge this gap, we introduce MV-RoboBench, a benchmark specifically designed to evaluate the multi-view spatial reasoning capabilities of VLMs in robotic manipulation. MV-RoboBench consists of 1.7k manually curated QA items across eight subtasks, divided into two primary categories: spatial understanding and robotic execution. We evaluate a diverse set of existing VLMs, including both open-source and closed-source models, along with enhanced versions incorporating CoT-inspired techniques. The results show that state-of-the-art models remain far below human performance, underscoring the substantial challenges VLMs face in multi-view robotic perception. Additionally, our analysis uncovers two key findings: (i) spatial intelligence and robotic task execution are positively correlated in multi-view robotic scenarios; and (ii) strong performance on existing general-purpose single-view spatial understanding benchmarks does not reliably translate to success in the robotic spatial tasks assessed by our benchmark. We release MV-RoboBench as an open resource to foster progress in spatially grounded VLMs and VLAs, providing not only data but also a standardized evaluation protocol for multi-view embodied reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19944v1">Seed3D 1.0: From Images to High-Fidelity Simulation-Ready 3D Assets</a></div>
    <div class="paper-meta">
      📅 2025-10-22
      | 💬 Seed3D 1.0 Technical Report; Official Page on https://seed.bytedance.com/seed3d
    </div>
    <details class="paper-abstract">
      Developing embodied AI agents requires scalable training environments that balance content diversity with physics accuracy. World simulators provide such environments but face distinct limitations: video-based methods generate diverse content but lack real-time physics feedback for interactive learning, while physics-based engines provide accurate dynamics but face scalability limitations from costly manual asset creation. We present Seed3D 1.0, a foundation model that generates simulation-ready 3D assets from single images, addressing the scalability challenge while maintaining physics rigor. Unlike existing 3D generation models, our system produces assets with accurate geometry, well-aligned textures, and realistic physically-based materials. These assets can be directly integrated into physics engines with minimal configuration, enabling deployment in robotic manipulation and simulation training. Beyond individual objects, the system scales to complete scene generation through assembling objects into coherent environments. By enabling scalable simulation-ready content creation, Seed3D 1.0 provides a foundation for advancing physics-based world simulators. Seed3D 1.0 is now available on https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?modelId=doubao-seed3d-1-0-250928&tab=Gen3D
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00441v3">Seeing through Uncertainty: Robust Task-Oriented Optimization in Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2025-10-21
    </div>
    <details class="paper-abstract">
      Visual navigation is a fundamental problem in embodied AI, yet practical deployments demand long-horizon planning capabilities to address multi-objective tasks. A major bottleneck is data scarcity: policies learned from limited data often overfit and fail to generalize OOD. Existing neural network-based agents typically increase architectural complexity that paradoxically become counterproductive in the small-sample regime. This paper introduce NeuRO, a integrated learning-to-optimize framework that tightly couples perception networks with downstream task-level robust optimization. Specifically, NeuRO addresses core difficulties in this integration: (i) it transforms noisy visual predictions under data scarcity into convex uncertainty sets using Partially Input Convex Neural Networks (PICNNs) with conformal calibration, which directly parameterize the optimization constraints; and (ii) it reformulates planning under partial observability as a robust optimization problem, enabling uncertainty-aware policies that transfer across environments. Extensive experiments on both unordered and sequential multi-object navigation tasks demonstrate that NeuRO establishes SoTA performance, particularly in generalization to unseen environments. Our work thus presents a significant advancement for developing robust, generalizable autonomous agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09049v2">VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-21
      | 💬 Project page: https://faceong.github.io/VIKI-R/
    </div>
    <details class="paper-abstract">
      Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17369v1">Bridging Embodiment Gaps: Deploying Vision-Language-Action Models on Soft Robots</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Accepted by NeurIPS 2025 SpaVLE workshop. 4 pages, 2 figures(in main paper, excluding references and supplements)
    </div>
    <details class="paper-abstract">
      Robotic systems are increasingly expected to operate in human-centered, unstructured environments where safety, adaptability, and generalization are essential. Vision-Language-Action (VLA) models have been proposed as a language guided generalized control framework for real robots. However, their deployment has been limited to conventional serial link manipulators. Coupled by their rigidity and unpredictability of learning based control, the ability to safely interact with the environment is missing yet critical. In this work, we present the deployment of a VLA model on a soft continuum manipulator to demonstrate autonomous safe human-robot interaction. We present a structured finetuning and deployment pipeline evaluating two state-of-the-art VLA models (OpenVLA-OFT and $\pi_0$) across representative manipulation tasks, and show while out-of-the-box policies fail due to embodiment mismatch, through targeted finetuning the soft robot performs equally to the rigid counterpart. Our findings highlight the necessity of finetuning for bridging embodiment gaps, and demonstrate that coupling VLA models with soft robots enables safe and flexible embodied AI in human-shared environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09956v4">DeepSeek-Inspired Exploration of RL-based LLMs and Synergy with Wireless Networks: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 45 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL)-based large language models (LLMs), such as ChatGPT, DeepSeek, and Grok-3, have attracted widespread attention for their remarkable capabilities in multimodal data understanding. Meanwhile, the rapid expansion of information services has led to a growing demand for AI-enabled wireless networks. The open-source DeepSeek models are famous for their innovative designs, such as large-scale pure RL and cost-efficient training, which make them well-suited for practical deployment in wireless networks. By integrating DeepSeek-style LLMs with wireless infrastructures, a synergistic opportunity arises: the DeepSeek-style LLMs enhance network optimization with strong reasoning and decision-making abilities, while wireless infrastructure enables the broad deployment of these models. Motivated by this convergence, this survey presents a comprehensive DeepSeek-inspired exploration of RL-based LLMs in the context of wireless networks. We begin by reviewing key techniques behind network optimization to establish a foundation for understanding DeepSeek-style LLM integration. Next, we examine recent advancements in RL-based LLMs, using DeepSeek models as a representative example. Building on this, we explore the synergy between the two domains, highlighting motivations, challenges, and potential solutions. Finally, we highlight emerging directions for integrating LLMs with wireless networks, such as quantum, on-device, and neural-symbolic LLM models, as well as embodied AI agents. Overall, this survey offers a comprehensive examination of the interplay between DeepSeek-style LLMs and wireless networks, demonstrating how these domains can mutually enhance each other to drive innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02236v2">3D Audio-Visual Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Accepted at the NeurIPS 2024 Workshop on Audio Imagination; this version updates the project page link
    </div>
    <details class="paper-abstract">
      Recognizing the sounding objects in scenes is a longstanding objective in embodied AI, with diverse applications in robotics and AR/VR/MR. To that end, Audio-Visual Segmentation (AVS), taking as condition an audio signal to identify the masks of the target sounding objects in an input image with synchronous camera and microphone sensors, has been recently advanced. However, this paradigm is still insufficient for real-world operation, as the mapping from 2D images to 3D scenes is missing. To address this fundamental limitation, we introduce a novel research problem, 3D Audio-Visual Segmentation, extending the existing AVS to the 3D output space. This problem poses more challenges due to variations in camera extrinsics, audio scattering, occlusions, and diverse acoustics across sounding object categories. To facilitate this research, we create the very first simulation based benchmark, 3DAVS-S34-O7, providing photorealistic 3D scene environments with grounded spatial audio under single-instance and multi-instance settings, across 34 scenes and 7 object categories. This is made possible by re-purposing the Habitat simulator to generate comprehensive annotations of sounding object locations and corresponding 3D masks. Subsequently, we propose a new approach, EchoSegnet, characterized by integrating the ready-to-use knowledge from pretrained 2D audio-visual foundation models synergistically with 3D visual scene representation through spatial audio-aware mask alignment and refinement. Extensive experiments demonstrate that EchoSegnet can effectively segment sounding objects in 3D space on our new benchmark, representing a significant advancement in the field of embodied AI. Project page: https://x-up-lab.github.io/research/3d-audio-visual-segmentation/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16732v1">A Comprehensive Survey on World Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 https://github.com/Li-Zn-H/AwesomeWorldModels
    </div>
    <details class="paper-abstract">
      Embodied AI requires agents that perceive, act, and anticipate how actions reshape future world states. World models serve as internal simulators that capture environment dynamics, enabling forward and counterfactual rollouts to support perception, prediction, and decision making. This survey presents a unified framework for world models in embodied AI. Specifically, we formalize the problem setting and learning objectives, and propose a three-axis taxonomy encompassing: (1) Functionality, Decision-Coupled vs. General-Purpose; (2) Temporal Modeling, Sequential Simulation and Inference vs. Global Difference Prediction; (3) Spatial Representation, Global Latent Vector, Token Feature Sequence, Spatial Latent Grid, and Decomposed Rendering Representation. We systematize data resources and metrics across robotics, autonomous driving, and general video settings, covering pixel prediction quality, state-level understanding, and task performance. Furthermore, we offer a quantitative comparison of state-of-the-art models and distill key open challenges, including the scarcity of unified datasets and the need for evaluation metrics that assess physical consistency over pixel fidelity, the trade-off between model performance and the computational efficiency required for real-time control, and the core modeling difficulty of achieving long-horizon temporal consistency while mitigating error accumulation. Finally, we maintain a curated bibliography at https://github.com/Li-Zn-H/AwesomeWorldModels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09585v3">Elevating Visual Perception in Multimodal LLMs with Visual Embedding Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Project Page: https://praeclarumjj3.github.io/visper_lm/
    </div>
    <details class="paper-abstract">
      In recent times, the standard practice for developing MLLMs is to feed features from vision encoder(s) into the LLM and train with natural language supervision. This approach often causes models to lean towards language comprehension and undermine the rich visual perception signals present in the data, which are critical for tasks involving spatial reasoning in the domain of embodied AI and robotics. Is it possible to optimize both at the same time? In this work, we propose VisPer-LM, the first approach that infuses visual perception knowledge from expert vision encoders into the LLM's (of an MLLM) hidden representations. We start by investigating MLLMs trained solely with natural language supervision and identify a positive correlation between the quality of visual representations within these models and their downstream performance. Given this insight, we formulate the objective during the pretraining stage in MLLMs as a coupled optimization of predictive visual embedding and next (text) token prediction. Moreover, through extensive probing, we observe improved visual representation quality due to embedding optimization, underscoring the effectiveness of our probing setup. We demonstrate that our VisPer-LM outperforms the single and multi-encoder baselines, proving our approach's superiority over explicitly feeding the corresponding features to the LLM. In particular, VisPer-LM boosts performance by an average margin of up to 2.5% on various benchmarks, with a notable improvement of 8.7% on the Depth task in CV-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15018v1">UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Technical report. Project page: https://urbanverseproject.github.io/
    </div>
    <details class="paper-abstract">
      Urban embodied AI agents, ranging from delivery robots to quadrupeds, are increasingly populating our cities, navigating chaotic streets to provide last-mile connectivity. Training such agents requires diverse, high-fidelity urban environments to scale, yet existing human-crafted or procedurally generated simulation scenes either lack scalability or fail to capture real-world complexity. We introduce UrbanVerse, a data-driven real-to-sim system that converts crowd-sourced city-tour videos into physics-aware, interactive simulation scenes. UrbanVerse consists of: (i) UrbanVerse-100K, a repository of 100k+ annotated urban 3D assets with semantic and physical attributes, and (ii) UrbanVerse-Gen, an automatic pipeline that extracts scene layouts from video and instantiates metric-scale 3D simulations using retrieved assets. Running in IsaacSim, UrbanVerse offers 160 high-quality constructed scenes from 24 countries, along with a curated benchmark of 10 artist-designed test scenes. Experiments show that UrbanVerse scenes preserve real-world semantics and layouts, achieving human-evaluated realism comparable to manually crafted scenes. In urban navigation, policies trained in UrbanVerse exhibit scaling power laws and strong generalization, improving success by +6.3% in simulation and +30.1% in zero-shot sim-to-real transfer comparing to prior methods, accomplishing a 300 m real-world mission with only two interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10414v2">HUMOTO: A 4D Dataset of Mocap Human Object Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 ICCV 2025, 19 pages, 15 figures
    </div>
    <details class="paper-abstract">
      We present Human Motions with Objects (HUMOTO), a high-fidelity dataset of human-object interactions for motion generation, computer vision, and robotics applications. Featuring 735 sequences (7,875 seconds at 30 fps), HUMOTO captures interactions with 63 precisely modeled objects and 72 articulated parts. Our innovations include a scene-driven LLM scripting pipeline creating complete, purposeful tasks with natural progression, and a mocap-and-camera recording setup to effectively handle occlusions. Spanning diverse activities from cooking to outdoor picnics, HUMOTO preserves both physical accuracy and logical task flow. Professional artists rigorously clean and verify each sequence, minimizing foot sliding and object penetrations. We also provide benchmarks compared to other datasets. HUMOTO's comprehensive full-body motion and simultaneous multi-object interactions address key data-capturing challenges and provide opportunities to advance realistic human-object interaction modeling across research domains with practical applications in animation, robotics, and embodied AI systems. Project: https://jiaxin-lu.github.io/humoto/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21117v2">CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 ICCV 2025, Project Page: https://cl-splats.github.io
    </div>
    <details class="paper-abstract">
      In dynamic 3D environments, accurately updating scene representations over time is crucial for applications in robotics, mixed reality, and embodied AI. As scenes evolve, efficient methods to incorporate changes are needed to maintain up-to-date, high-quality reconstructions without the computational overhead of re-optimizing the entire scene. This paper introduces CL-Splats, which incrementally updates Gaussian splatting-based 3D representations from sparse scene captures. CL-Splats integrates a robust change-detection module that segments updated and static components within the scene, enabling focused, local optimization that avoids unnecessary re-computation. Moreover, CL-Splats supports storing and recovering previous scene states, facilitating temporal segmentation and new scene-analysis applications. Our extensive experiments demonstrate that CL-Splats achieves efficient updates with improved reconstruction quality over the state-of-the-art. This establishes a robust foundation for future real-time adaptation in 3D scene reconstruction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00441v2">Seeing through Uncertainty: Robust Task-Oriented Optimization in Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Visual navigation is a fundamental problem in embodied AI, yet practical deployments demand long-horizon planning capabilities to address multi-objective tasks. A major bottleneck is data scarcity: policies learned from limited data often overfit and fail to generalize OOD. Existing neural network-based agents typically increase architectural complexity that paradoxically become counterproductive in the small-sample regime. This paper introduce NeuRO, a integrated learning-to-optimize framework that tightly couples perception networks with downstream task-level robust optimization. Specifically, NeuRO addresses core difficulties in this integration: (i) it transforms noisy visual predictions under data scarcity into convex uncertainty sets using Partially Input Convex Neural Networks (PICNNs) with conformal calibration, which directly parameterize the optimization constraints; and (ii) it reformulates planning under partial observability as a robust optimization problem, enabling uncertainty-aware policies that transfer across environments. Extensive experiments on both unordered and sequential multi-object navigation tasks demonstrate that NeuRO establishes SoTA performance, particularly in generalization to unseen environments. Our work thus presents a significant advancement for developing robust, generalizable autonomous agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12749v1">SPORTS: Simultaneous Panoptic Odometry, Rendering, Tracking and Segmentation for Urban Scenes Understanding</a></div>
    <div class="paper-meta">
      📅 2025-10-14
      | 💬 Accepted by IEEE Transactions on Multimedia
    </div>
    <details class="paper-abstract">
      The scene perception, understanding, and simulation are fundamental techniques for embodied-AI agents, while existing solutions are still prone to segmentation deficiency, dynamic objects' interference, sensor data sparsity, and view-limitation problems. This paper proposes a novel framework, named SPORTS, for holistic scene understanding via tightly integrating Video Panoptic Segmentation (VPS), Visual Odometry (VO), and Scene Rendering (SR) tasks into an iterative and unified perspective. Firstly, VPS designs an adaptive attention-based geometric fusion mechanism to align cross-frame features via enrolling the pose, depth, and optical flow modality, which automatically adjust feature maps for different decoding stages. And a post-matching strategy is integrated to improve identities tracking. In VO, panoptic segmentation results from VPS are combined with the optical flow map to improve the confidence estimation of dynamic objects, which enhances the accuracy of the camera pose estimation and completeness of the depth map generation via the learning-based paradigm. Furthermore, the point-based rendering of SR is beneficial from VO, transforming sparse point clouds into neural fields to synthesize high-fidelity RGB views and twin panoptic views. Extensive experiments on three public datasets demonstrate that our attention-based feature fusion outperforms most existing state-of-the-art methods on the odometry, tracking, segmentation, and novel view synthesis tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12693v1">ERA: Transforming VLMs into Embodied Agents via Embodied Prior Learning and Online Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Recent advances in embodied AI highlight the potential of vision language models (VLMs) as agents capable of perception, reasoning, and interaction in complex environments. However, top-performing systems rely on large-scale models that are costly to deploy, while smaller VLMs lack the necessary knowledge and skills to succeed. To bridge this gap, we present \textit{Embodied Reasoning Agent (ERA)}, a two-stage framework that integrates prior knowledge learning and online reinforcement learning (RL). The first stage, \textit{Embodied Prior Learning}, distills foundational knowledge from three types of data: (1) Trajectory-Augmented Priors, which enrich existing trajectory data with structured reasoning generated by stronger models; (2) Environment-Anchored Priors, which provide in-environment knowledge and grounding supervision; and (3) External Knowledge Priors, which transfer general knowledge from out-of-environment datasets. In the second stage, we develop an online RL pipeline that builds on these priors to further enhance agent performance. To overcome the inherent challenges in agent RL, including long horizons, sparse rewards, and training instability, we introduce three key designs: self-summarization for context management, dense reward shaping, and turn-level policy optimization. Extensive experiments on both high-level planning (EB-ALFRED) and low-level control (EB-Manipulation) tasks demonstrate that ERA-3B surpasses both prompting-based large models and previous training-based baselines. Specifically, it achieves overall improvements of 8.4\% on EB-ALFRED and 19.4\% on EB-Manipulation over GPT-4o, and exhibits strong generalization to unseen tasks. Overall, ERA offers a practical path toward scalable embodied intelligence, providing methodological insights for future embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16815v2">Image Quality Assessment for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      Embodied AI has developed rapidly in recent years, but it is still mainly deployed in laboratories, with various distortions in the Real-world limiting its application. Traditionally, Image Quality Assessment (IQA) methods are applied to predict human preferences for distorted images; however, there is no IQA method to assess the usability of an image in embodied tasks, namely, the perceptual quality for robots. To provide accurate and reliable quality indicators for future embodied scenarios, we first propose the topic: IQA for Embodied AI. Specifically, we (1) based on the Mertonian system and meta-cognitive theory, constructed a perception-cognition-decision-execution pipeline and defined a comprehensive subjective score collection process; (2) established the Embodied-IQA database, containing over 36k reference/distorted image pairs, with more than 5m fine-grained annotations provided by Vision Language Models/Vision Language Action-models/Real-world robots; (3) trained and validated the performance of mainstream IQA methods on Embodied-IQA, demonstrating the need to develop more accurate quality indicators for Embodied AI. We sincerely hope that through evaluation, we can promote the application of Embodied AI under complex distortions in the Real-world. Project page: https://github.com/lcysyzxdxc/EmbodiedIQA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10813v2">InternScenes: A Large-scale Simulatable Indoor Scene Dataset with Realistic Layouts</a></div>
    <div class="paper-meta">
      📅 2025-10-14
    </div>
    <details class="paper-abstract">
      The advancement of Embodied AI heavily relies on large-scale, simulatable 3D scene datasets characterized by scene diversity and realistic layouts. However, existing datasets typically suffer from limitations in data scale or diversity, sanitized layouts lacking small items, and severe object collisions. To address these shortcomings, we introduce \textbf{InternScenes}, a novel large-scale simulatable indoor scene dataset comprising approximately 40,000 diverse scenes by integrating three disparate scene sources, real-world scans, procedurally generated scenes, and designer-created scenes, including 1.96M 3D objects and covering 15 common scene types and 288 object classes. We particularly preserve massive small items in the scenes, resulting in realistic and complex layouts with an average of 41.5 objects per region. Our comprehensive data processing pipeline ensures simulatability by creating real-to-sim replicas for real-world scans, enhances interactivity by incorporating interactive objects into these scenes, and resolves object collisions by physical simulations. We demonstrate the value of InternScenes with two benchmark applications: scene layout generation and point-goal navigation. Both show the new challenges posed by the complex and realistic layouts. More importantly, InternScenes paves the way for scaling up the model training for both tasks, making the generation and navigation in such complex scenes possible. We commit to open-sourcing the data, models, and benchmarks to benefit the whole community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11687v1">Beyond 'Templates': Category-Agnostic Object Pose, Size, and Shape Estimation from a Single View</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Estimating an object's 6D pose, size, and shape from visual input is a fundamental problem in computer vision, with critical applications in robotic grasping and manipulation. Existing methods either rely on object-specific priors such as CAD models or templates, or suffer from limited generalization across categories due to pose-shape entanglement and multi-stage pipelines. In this work, we propose a unified, category-agnostic framework that simultaneously predicts 6D pose, size, and dense shape from a single RGB-D image, without requiring templates, CAD models, or category labels at test time. Our model fuses dense 2D features from vision foundation models with partial 3D point clouds using a Transformer encoder enhanced by a Mixture-of-Experts, and employs parallel decoders for pose-size estimation and shape reconstruction, achieving real-time inference at 28 FPS. Trained solely on synthetic data from 149 categories in the SOPE dataset, our framework is evaluated on four diverse benchmarks SOPE, ROPE, ObjaversePose, and HANDAL, spanning over 300 categories. It achieves state-of-the-art accuracy on seen categories while demonstrating remarkably strong zero-shot generalization to unseen real-world objects, establishing a new standard for open-set 6D understanding in robotics and embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18361v4">Task-Optimized Convolutional Recurrent Networks Align with Tactile Processing in the Rodent Brain</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 10 pages, 8 figures, 7 tables, NeurIPS 2025 Camera Ready Version (oral)
    </div>
    <details class="paper-abstract">
      Tactile sensing remains far less understood in neuroscience and less effective in artificial systems compared to more mature modalities such as vision and language. We bridge these gaps by introducing a novel Encoder-Attender-Decoder (EAD) framework to systematically explore the space of task-optimized temporal neural networks trained on realistic tactile input sequences from a customized rodent whisker-array simulator. We identify convolutional recurrent neural networks (ConvRNNs) as superior encoders to purely feedforward and state-space architectures for tactile categorization. Crucially, these ConvRNN-encoder-based EAD models achieve neural representations closely matching rodent somatosensory cortex, saturating the explainable neural variability and revealing a clear linear relationship between supervised categorization performance and neural alignment. Furthermore, contrastive self-supervised ConvRNN-encoder-based EADs, trained with tactile-specific augmentations, match supervised neural fits, serving as an ethologically-relevant, label-free proxy. For neuroscience, our findings highlight nonlinear recurrent processing as important for general-purpose tactile representations in somatosensory cortex, providing the first quantitative characterization of the underlying inductive biases in this system. For embodied AI, our results emphasize the importance of recurrent EAD architectures to handle realistic tactile inputs, along with tailored self-supervised learning methods for achieving robust tactile perception with the same type of sensors animals use to sense in unstructured environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11307v1">FOSSIL: Harnessing Feedback on Suboptimal Samples for Data-Efficient Generalisation with Imitation Learning for Embodied Vision-and-Language Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Current approaches to embodied AI tend to learn policies from expert demonstrations. However, without a mechanism to evaluate the quality of demonstrated actions, they are limited to learning from optimal behaviour, or they risk replicating errors and inefficiencies. While reinforcement learning offers one alternative, the associated exploration typically results in sacrificing data efficiency. This work explores how agents trained with imitation learning can learn robust representations from both optimal and suboptimal demonstrations when given access to constructive language feedback as a means to contextualise different modes of behaviour. We directly provide language feedback embeddings as part of the input sequence into a Transformer-based policy, and optionally complement the traditional next action prediction objective with auxiliary self-supervised learning objectives for feedback prediction. We test our approach on a range of embodied Vision-and-Language tasks in our custom BabyAI-XGen environment and show significant improvements in agents' compositional generalisation abilities and robustness, suggesting that our data-efficient method allows models to successfully convert suboptimal behaviour into learning opportunities. Overall, our results suggest that language feedback is a competitive and intuitive alternative to intermediate scalar rewards for language-specified embodied tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16012v2">DualTHOR: A Dual-Arm Humanoid Simulation Platform for Contingency-Aware Planning</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 The experiments in the paper need to be further supplemented, and more methods should be considered for expansion
    </div>
    <details class="paper-abstract">
      Developing embodied agents capable of performing complex interactive tasks in real-world scenarios remains a fundamental challenge in embodied AI. Although recent advances in simulation platforms have greatly enhanced task diversity to train embodied Vision Language Models (VLMs), most platforms rely on simplified robot morphologies and bypass the stochastic nature of low-level execution, which limits their transferability to real-world robots. To address these issues, we present a physics-based simulation platform DualTHOR for complex dual-arm humanoid robots, built upon an extended version of AI2-THOR. Our simulator includes real-world robot assets, a task suite for dual-arm collaboration, and inverse kinematics solvers for humanoid robots. We also introduce a contingency mechanism that incorporates potential failures through physics-based low-level execution, bridging the gap to real-world scenarios. Our simulator enables a more comprehensive evaluation of the robustness and generalization of VLMs in household environments. Extensive evaluations reveal that current VLMs struggle with dual-arm coordination and exhibit limited robustness in realistic environments with contingencies, highlighting the importance of using our simulator to develop more capable VLMs for embodied tasks. The code is available at https://github.com/ds199895/DualTHOR.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23042v2">Goal-Based Vision-Language Driving</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      Autonomous vehicles must react in milliseconds while reasoning about road geometry and traffic intent to navigate complex situations. We introduce NovaDrive, a single-branch vision-language architecture that processes front-camera images, HD-map tiles, LiDAR depth, and textual waypoints in a single branch. A lightweight, two-stage cross-attention block first aligns waypoint tokens with the HD map, then refines attention over fine-grained image and depth patches. Coupled with a novel smoothness loss that discourages abrupt steering and speed changes, this design eliminates the need for recurrent memory. We fine-tune the top 15 layers of an 11B LLaMA-3.2 vision-language backbone, enabling real-time inference. On the nuScenes / Waymo subset of the MD-NEX Outdoor benchmark, NovaDrive raises success rate to 84% (+4%), boosts path-efficiency (SPL) to 0.66 (+0.11), and reduces collision frequency from 2.6% to 1.2% (-1.4%) relative to the previous state-of-the-art. Our ablations confirm that waypoint tokens, partial VLM fine-tuning, and the cross-attention fusion each contribute the most to these gains. Beyond safety, NovaDrive's shorter routes (resulting from the novel smoothness loss) translate to lower fuel or battery usage, pointing toward leaner, more easily updated driving stacks. NovaDrive can be extended to other embodied-AI domains as well.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10932v1">TabVLA: Targeted Backdoor Attacks on Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2025-10-13
      | 💬 8 pages, 8 tables, 1 figure. Under review
    </div>
    <details class="paper-abstract">
      With the growing deployment of Vision-Language-Action (VLA) models in real-world embodied AI systems, their increasing vulnerability to backdoor attacks poses a serious safety threat. A backdoored VLA agent can be covertly triggered by a pre-injected backdoor to execute adversarial actions, potentially causing system failures or even physical harm. Although backdoor attacks on VLA models have been explored, prior work has focused only on untargeted attacks, leaving the more practically threatening scenario of targeted manipulation unexamined. In this paper, we study targeted backdoor attacks on VLA models and introduce TabVLA, a novel framework that enables such attacks via black-box fine-tuning. TabVLA explores two deployment-relevant inference-time threat models: input-stream editing and in-scene triggering. It formulates poisoned data generation as an optimization problem to improve attack effectivess. Experiments with OpenVLA-7B on the LIBERO benchmark reveal that the vision channel is the principal attack surface: targeted backdoors succeed with minimal poisoning, remain robust across variations in trigger design, and are degraded only by positional mismatches between fine-tuning and inference triggers. We also investigate a potential detection-based defense against TabVLA, which reconstructs latent visual triggers from the input stream to flag activation-conditioned backdoor samples. Our work highlights the vulnerability of VLA models to targeted backdoor manipulation and underscores the need for more advanced defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07501v3">FormCoach: Lift Smarter, Not Harder</a></div>
    <div class="paper-meta">
      📅 2025-10-13
    </div>
    <details class="paper-abstract">
      Good form is the difference between strength and strain, yet for the fast-growing community of at-home fitness enthusiasts, expert feedback is often out of reach. FormCoach transforms a simple camera into an always-on, interactive AI training partner, capable of spotting subtle form errors and delivering tailored corrections in real time, leveraging vision-language models (VLMs). We showcase this capability through a web interface and benchmark state-of-the-art VLMs on a dataset of 1,700 expert-annotated user-reference video pairs spanning 22 strength and mobility exercises. To accelerate research in AI-driven coaching, we release both the dataset and an automated, rubric-based evaluation pipeline, enabling standardized comparison across models. Our benchmarks reveal substantial gaps compared to human-level coaching, underscoring both the challenges and opportunities in integrating nuanced, context-aware movement analysis into interactive AI systems. By framing form correction as a collaborative and creative process between humans and machines, FormCoach opens a new frontier in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10823v1">The Irrational Machine: Neurosis and the Limits of Algorithmic Safety</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 41 pages, 17 figures, 5 tables
    </div>
    <details class="paper-abstract">
      We present a framework for characterizing neurosis in embodied AI: behaviors that are internally coherent yet misaligned with reality, arising from interactions among planning, uncertainty handling, and aversive memory. In a grid navigation stack we catalogue recurrent modalities including flip-flop, plan churn, perseveration loops, paralysis and hypervigilance, futile search, belief incoherence, tie break thrashing, corridor thrashing, optimality compulsion, metric mismatch, policy oscillation, and limited-visibility variants. For each we give lightweight online detectors and reusable escape policies (short commitments, a margin to switch, smoothing, principled arbitration). We then show that durable phobic avoidance can persist even under full visibility when learned aversive costs dominate local choice, producing long detours despite globally safe routes. Using First/Second/Third Law as engineering shorthand for safety latency, command compliance, and resource efficiency, we argue that local fixes are insufficient; global failures can remain. To surface them, we propose genetic-programming based destructive testing that evolves worlds and perturbations to maximize law pressure and neurosis scores, yielding adversarial curricula and counterfactual traces that expose where architectural revision, not merely symptom-level patches, is required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17040v2">Multimodal Alignment and Fusion: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Accepted to IJCV 2025
    </div>
    <details class="paper-abstract">
      This survey provides a comprehensive overview of recent advances in multimodal alignment and fusion within the field of machine learning, driven by the increasing availability and diversity of data modalities such as text, images, audio, and video. Unlike previous surveys that often focus on specific modalities or limited fusion strategies, our work presents a structure-centric and method-driven framework that emphasizes generalizable techniques. We systematically categorize and analyze key approaches to alignment and fusion through both structural perspectives -- data-level, feature-level, and output-level fusion -- and methodological paradigms -- including statistical, kernel-based, graphical, generative, contrastive, attention-based, and large language model (LLM)-based methods, drawing insights from an extensive review of over 260 relevant studies. Furthermore, this survey highlights critical challenges such as cross-modal misalignment, computational bottlenecks, data quality issues, and the modality gap, along with recent efforts to address them. Applications ranging from social media analysis and medical imaging to emotion recognition and embodied AI are explored to illustrate the real-world impact of robust multimodal systems. The insights provided aim to guide future research toward optimizing multimodal learning systems for improved scalability, robustness, and generalizability across diverse domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09507v1">PhysToolBench: Benchmarking Physical Tool Understanding for MLLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      The ability to use, understand, and create tools is a hallmark of human intelligence, enabling sophisticated interaction with the physical world. For any general-purpose intelligent agent to achieve true versatility, it must also master these fundamental skills. While modern Multimodal Large Language Models (MLLMs) leverage their extensive common knowledge for high-level planning in embodied AI and in downstream Vision-Language-Action (VLA) models, the extent of their true understanding of physical tools remains unquantified. To bridge this gap, we present PhysToolBench, the first benchmark dedicated to evaluating the comprehension of physical tools by MLLMs. Our benchmark is structured as a Visual Question Answering (VQA) dataset comprising over 1,000 image-text pairs. It assesses capabilities across three distinct difficulty levels: (1) Tool Recognition: Requiring the recognition of a tool's primary function. (2) Tool Understanding: Testing the ability to grasp the underlying principles of a tool's operation. (3) Tool Creation: Challenging the model to fashion a new tool from surrounding objects when conventional options are unavailable. Our comprehensive evaluation of 32 MLLMs-spanning proprietary, open-source, specialized embodied, and backbones in VLAs-reveals a significant deficiency in tool understanding. Furthermore, we provide an in-depth analysis and propose preliminary solutions. Code and dataset are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09483v1">FOGMACHINE -- Leveraging Discrete-Event Simulation and Scene Graphs for Modeling Hierarchical, Interconnected Environments under Partial Observations from Mobile Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 submitted to the IEEE for possible publication; 8 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Dynamic Scene Graphs (DSGs) provide a structured representation of hierarchical, interconnected environments, but current approaches struggle to capture stochastic dynamics, partial observability, and multi-agent activity. These aspects are critical for embodied AI, where agents must act under uncertainty and delayed perception. We introduce FOGMACHINE , an open-source framework that fuses DSGs with discrete-event simulation to model object dynamics, agent observations, and interactions at scale. This setup enables the study of uncertainty propagation, planning under limited perception, and emergent multi-agent behavior. Experiments in urban scenarios illustrate realistic temporal and spatial patterns while revealing the challenges of belief estimation under sparse observations. By combining structured representations with efficient simulation, FOGMACHINE establishes an effective tool for benchmarking, model training, and advancing embodied AI in complex, uncertain environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09269v1">Goal-oriented Backdoor Attack against Vision-Language-Action Models via Physical Objects</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Recent advances in vision-language-action (VLA) models have greatly improved embodied AI, enabling robots to follow natural language instructions and perform diverse tasks. However, their reliance on uncurated training datasets raises serious security concerns. Existing backdoor attacks on VLAs mostly assume white-box access and result in task failures instead of enforcing specific actions. In this work, we reveal a more practical threat: attackers can manipulate VLAs by simply injecting physical objects as triggers into the training dataset. We propose goal-oriented backdoor attacks (GoBA), where the VLA behaves normally in the absence of physical triggers but executes predefined and goal-oriented actions in the presence of physical triggers. Specifically, based on a popular VLA benchmark LIBERO, we introduce BadLIBERO that incorporates diverse physical triggers and goal-oriented backdoor actions. In addition, we propose a three-level evaluation that categorizes the victim VLA's actions under GoBA into three states: nothing to do, try to do, and success to do. Experiments show that GoBA enables the victim VLA to successfully achieve the backdoor goal in 97 percentage of inputs when the physical trigger is present, while causing zero performance degradation on clean inputs. Finally, by investigating factors related to GoBA, we find that the action trajectory and trigger color significantly influence attack performance, while trigger size has surprisingly little effect. The code and BadLIBERO dataset are accessible via the project page at https://goba-attack.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07791v2">GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Recently spatial-temporal intelligence of Visual-Language Models (VLMs) has attracted much attention due to its importance for Autonomous Driving, Embodied AI and General Artificial Intelligence. Existing spatial-temporal benchmarks mainly focus on egocentric perspective reasoning with images/video context, or geographic perspective reasoning with graphics context (eg. a map), thus fail to assess VLMs' geographic spatial-temporal intelligence with both images/video and graphics context, which is important for areas like traffic management and emergency response. To address the gaps, we introduce Geo-Temporal Reasoning benchmark (GTR-Bench), a novel challenge for geographic temporal reasoning of moving targets in a large-scale camera network. GTR-Bench is more challenging as it requires multiple perspective switches between maps and videos, joint reasoning across multiple videos with non-overlapping fields of view, and inference over spatial-temporal regions that are unobserved by any video context. Evaluations of more than 10 popular VLMs on GTR-Bench demonstrate that even the best proprietary model, Gemini-2.5-Pro (34.9%), significantly lags behind human performance (78.61%) on geo-temporal reasoning. Moreover, our comprehensive analysis on GTR-Bench reveals three primary deficiencies of current models for geo-temporal reasoning. (1) VLMs' reasoning is impaired by an imbalanced utilization of spatial-temporal context. (2) VLMs are weak in temporal forecasting, which leads to worse performance on temporal-emphasized tasks than on spatial-emphasized tasks. (3) VLMs lack the proficiency to comprehend or align the map data with multi-view video inputs. We believe GTR-Bench offers valuable insights and opens up new opportunities for research and applications in spatial-temporal intelligence. Benchmark and code will be released at https://github.com/X-Luffy/GTR-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08316v1">Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Work in process
    </div>
    <details class="paper-abstract">
      Affordance segmentation aims to parse 3D objects into functionally distinct parts, bridging recognition and interaction for applications in robotic manipulation, embodied AI, and AR. While recent studies leverage visual or textual prompts to guide this process, they often rely on point cloud encoders as generic feature extractors, overlooking the intrinsic challenges of 3D data such as sparsity, noise, and geometric ambiguity. As a result, 3D features learned in isolation frequently lack clear and semantically consistent functional boundaries. To address this bottleneck, we propose a semantic-grounded learning paradigm that transfers rich semantic knowledge from large-scale 2D Vision Foundation Models (VFMs) into the 3D domain. Specifically, We introduce Cross-Modal Affinity Transfer (CMAT), a pre-training strategy that aligns a 3D encoder with lifted 2D semantics and jointly optimizes reconstruction, affinity, and diversity to yield semantically organized representations. Building on this backbone, we further design the Cross-modal Affordance Segmentation Transformer (CAST), which integrates multi-modal prompts with CMAT-pretrained features to generate precise, prompt-aware segmentation maps. Extensive experiments on standard benchmarks demonstrate that our framework establishes new state-of-the-art results for 3D affordance segmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08227v1">Practicing a Second Language Without Fear: Mixed Reality Agents for Interactive Group Conversation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Developing speaking proficiency in a second language can be cognitively demanding and emotionally taxing, often triggering fear of making mistakes or being excluded from larger groups. While current learning tools show promise for speaking practice, most focus on dyadic, scripted scenarios, limiting opportunities for dynamic group interactions. To address this gap, we present ConversAR, a Mixed Reality system that leverages Generative AI and XR to support situated and personalized group conversations. It integrates embodied AI agents, scene recognition, and generative 3D props anchored to real-world surroundings. Based on a formative study with experts in language acquisition, we developed and tested this system with a user study with 21 second-language learners. Results indicate that the system enhanced learner engagement, increased willingness to communicate, and offered a safe space for speaking. We discuss the implications for integrating Generative AI and XR into the design of future language learning applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07975v1">Executable Analytic Concepts as the Missing Link Between VLM Insight and Precise Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Enabling robots to perform precise and generalized manipulation in unstructured environments remains a fundamental challenge in embodied AI. While Vision-Language Models (VLMs) have demonstrated remarkable capabilities in semantic reasoning and task planning, a significant gap persists between their high-level understanding and the precise physical execution required for real-world manipulation. To bridge this "semantic-to-physical" gap, we introduce GRACE, a novel framework that grounds VLM-based reasoning through executable analytic concepts (EAC)-mathematically defined blueprints that encode object affordances, geometric constraints, and semantics of manipulation. Our approach integrates a structured policy scaffolding pipeline that turn natural language instructions and visual information into an instantiated EAC, from which we derive grasp poses, force directions and plan physically feasible motion trajectory for robot execution. GRACE thus provides a unified and interpretable interface between high-level instruction understanding and low-level robot control, effectively enabling precise and generalizable manipulation through semantic-physical grounding. Extensive experiments demonstrate that GRACE achieves strong zero-shot generalization across a variety of articulated objects in both simulated and real-world environments, without requiring task-specific training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24528v2">CORE-3D: Context-aware Open-vocabulary Retrieval by Embeddings in 3D</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 9 pages, 4 figures, submitted for ICLR 2026 conference
    </div>
    <details class="paper-abstract">
      3D scene understanding is fundamental for embodied AI and robotics, supporting reliable perception for interaction and navigation. Recent approaches achieve zero-shot, open-vocabulary 3D semantic mapping by assigning embedding vectors to 2D class-agnostic masks generated via vision-language models (VLMs) and projecting these into 3D. However, these methods often produce fragmented masks and inaccurate semantic assignments due to the direct use of raw masks, limiting their effectiveness in complex environments. To address this, we leverage SemanticSAM with progressive granularity refinement to generate more accurate and numerous object-level masks, mitigating the over-segmentation commonly observed in mask generation models such as vanilla SAM, and improving downstream 3D semantic segmentation. To further enhance semantic context, we employ a context-aware CLIP encoding strategy that integrates multiple contextual views of each mask using empirically determined weighting, providing much richer visual context. We evaluate our approach on multiple 3D scene understanding tasks, including 3D semantic segmentation and object retrieval from language queries, across several benchmark datasets. Experimental results demonstrate significant improvements over existing methods, highlighting the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07791v1">GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Recently spatial-temporal intelligence of Visual-Language Models (VLMs) has attracted much attention due to its importance for Autonomous Driving, Embodied AI and General Artificial Intelligence. Existing spatial-temporal benchmarks mainly focus on egocentric perspective reasoning with images/video context, or geographic perspective reasoning with graphics context (eg. a map), thus fail to assess VLMs' geographic spatial-temporal intelligence with both images/video and graphics context, which is important for areas like traffic management and emergency response. To address the gaps, we introduce Geo-Temporal Reasoning benchmark (GTR-Bench), a novel challenge for geographic temporal reasoning of moving targets in a large-scale camera network. GTR-Bench is more challenging as it requires multiple perspective switches between maps and videos, joint reasoning across multiple videos with non-overlapping fields of view, and inference over spatial-temporal regions that are unobserved by any video context. Evaluations of more than 10 popular VLMs on GTR-Bench demonstrate that even the best proprietary model, Gemini-2.5-Pro (34.9%), significantly lags behind human performance (78.61%) on geo-temporal reasoning. Moreover, our comprehensive analysis on GTR-Bench reveals three primary deficiencies of current models for geo-temporal reasoning. (1) VLMs' reasoning is impaired by an imbalanced utilization of spatial-temporal context. (2) VLMs are weak in temporal forecasting, which leads to worse performance on temporal-emphasized tasks than on spatial-emphasized tasks. (3) VLMs lack the proficiency to comprehend or align the map data with multi-view video inputs. We believe GTR-Bench offers valuable insights and opens up new opportunities for research and applications in spatial-temporal intelligence. Benchmark and code will be released at https://github.com/X-Luffy/GTR-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07067v1">Bring the Apple, Not the Sofa: Impact of Irrelevant Context in Embodied AI Commands on VLA Models</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Vision Language Action (VLA) models are widely used in Embodied AI, enabling robots to interpret and execute language instructions. However, their robustness to natural language variability in real-world scenarios has not been thoroughly investigated. In this work, we present a novel systematic study of the robustness of state-of-the-art VLA models under linguistic perturbations. Specifically, we evaluate model performance under two types of instruction noise: (1) human-generated paraphrasing and (2) the addition of irrelevant context. We further categorize irrelevant contexts into two groups according to their length and their semantic and lexical proximity to robot commands. In this study, we observe consistent performance degradation as context size expands. We also demonstrate that the model can exhibit relative robustness to random context, with a performance drop within 10%, while semantically and lexically similar context of the same length can trigger a quality decline of around 50%. Human paraphrases of instructions lead to a drop of nearly 20%. To mitigate this, we propose an LLM-based filtering framework that extracts core commands from noisy inputs. Incorporating our filtering step allows models to recover up to 98.5% of their original performance under noisy conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05865v1">The Safety Challenge of World Models for Embodied AI Agents: A Review</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      The rapid progress in embodied artificial intelligence has highlighted the necessity for more advanced and integrated models that can perceive, interpret, and predict environmental dynamics. In this context, World Models (WMs) have been introduced to provide embodied agents with the abilities to anticipate future environmental states and fill in knowledge gaps, thereby enhancing agents' ability to plan and execute actions. However, when dealing with embodied agents it is fundamental to ensure that predictions are safe for both the agent and the environment. In this article, we conduct a comprehensive literature review of World Models in the domains of autonomous driving and robotics, with a specific focus on the safety implications of scene and control generation tasks. Our review is complemented by an empirical analysis, wherein we collect and examine predictions from state-of-the-art models, identify and categorize common faults (herein referred to as pathologies), and provide a quantitative evaluation of the results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05684v1">D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations, and 1K+ hours of pseudo-labeled gameplay), we achieve a total of 96.6% success rate on LIBERO manipulation and 83.3% on CANVAS navigation benchmarks. This validates that sensorimotor primitives in digital interactions exhibit sufficient invariance to transfer meaningfully to physical embodied tasks, establishing desktop pretraining as a practical paradigm for robotics. We will make all our work public, including the OWA toolkit, datasets of human-collected and pseudo-labeled, and VAPT-trained models available at https://worv-ai.github.io/d2e/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07634v3">Neural Brain: A Neuroscience-inspired Framework for Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 51 pages, 17 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04401v1">Your Vision-Language Model Can't Even Count to 20: Exposing the Failures of VLMs in Compositional Counting</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) have become a central focus of today's AI community, owing to their impressive abilities gained from training on large-scale vision-language data from the Web. These models have demonstrated strong performance across diverse tasks, including image understanding, video understanding, complex visual reasoning, and embodied AI. Despite these noteworthy successes, a fundamental question remains: Can VLMs count objects correctly? In this paper, we introduce a simple yet effective benchmark, VLMCountBench, designed under a minimalist setting with only basic geometric shapes (e.g., triangles, circles) and their compositions, focusing exclusively on counting tasks without interference from other factors. We adopt strict independent variable control and systematically study the effects of simple properties such as color, size, and prompt refinement in a controlled ablation. Our empirical results reveal that while VLMs can count reliably when only one shape type is present, they exhibit substantial failures when multiple shape types are combined (i.e., compositional counting). This highlights a fundamental empirical limitation of current VLMs and motivates important directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03842v3">INGRID: Intelligent Generative Robotic Design Using Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 We are revising it
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into robotic systems has accelerated progress in embodied artificial intelligence, yet current approaches remain constrained by existing robotic architectures, particularly serial mechanisms. This hardware dependency fundamentally limits the scope of robotic intelligence. Here, we present INGRID (Intelligent Generative Robotic Design), a framework that enables the automated design of parallel robotic mechanisms through deep integration with reciprocal screw theory and kinematic synthesis methods. We decompose the design challenge into four progressive tasks: constraint analysis, kinematic joint generation, chain construction, and complete mechanism design. INGRID demonstrates the ability to generate novel parallel mechanisms with both fixed and variable mobility, discovering kinematic configurations not previously documented in the literature. We validate our approach through three case studies demonstrating how INGRID assists users in designing task-specific parallel robots based on desired mobility requirements. By bridging the gap between mechanism theory and machine learning, INGRID enables researchers without specialized robotics training to create custom parallel mechanisms, thereby decoupling advances in robotic intelligence from hardware constraints. This work establishes a foundation for mechanism intelligence, where AI systems actively design robotic hardware, potentially transforming the development of embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03909v1">Generating Human Motion Videos using a Cascaded Text-to-Video Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 18 pages, 7 figures, Project Page:https://hyelinnam.github.io/Cameo/
    </div>
    <details class="paper-abstract">
      Human video generation is becoming an increasingly important task with broad applications in graphics, entertainment, and embodied AI. Despite the rapid progress of video diffusion models (VDMs), their use for general-purpose human video generation remains underexplored, with most works constrained to image-to-video setups or narrow domains like dance videos. In this work, we propose CAMEO, a cascaded framework for general human motion video generation. It seamlessly bridges Text-to-Motion (T2M) models and conditional VDMs, mitigating suboptimal factors that may arise in this process across both training and inference through carefully designed components. Specifically, we analyze and prepare both textual prompts and visual conditions to effectively train the VDM, ensuring robust alignment between motion descriptions, conditioning signals, and the generated videos. Furthermore, we introduce a camera-aware conditioning module that connects the two stages, automatically selecting viewpoints aligned with the input text to enhance coherence and reduce manual intervention. We demonstrate the effectiveness of our approach on both the MovieGen benchmark and a newly introduced benchmark tailored to the T2M-VDM combination, while highlighting its versatility across diverse use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03153v1">Improving Cooperation in Collaborative Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 In proceedings of UKCI 2025
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into multiagent systems has opened new possibilities for collaborative reasoning and cooperation with AI agents. This paper explores different prompting methods and evaluates their effectiveness in enhancing agent collaborative behaviour and decision-making. We enhance CoELA, a framework designed for building Collaborative Embodied Agents that leverage LLMs for multi-agent communication, reasoning, and task coordination in shared virtual spaces. Through systematic experimentation, we examine different LLMs and prompt engineering strategies to identify optimised combinations that maximise collaboration performance. Furthermore, we extend our research by integrating speech capabilities, enabling seamless collaborative voice-based interactions. Our findings highlight the effectiveness of prompt optimisation in enhancing collaborative agent performance; for example, our best combination improved the efficiency of the system running with Gemma3 by 22% compared to the original CoELA system. In addition, the speech integration provides a more engaging user interface for iterative system development and demonstrations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02851v1">Action Deviation-Aware Inference for Low-Latency Wireless Robots</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      To support latency-sensitive AI applications ranging from autonomous driving to industrial robot manipulation, 6G envisions distributed ML, connecting distributed computational resources in edge and cloud over hyper-reliable low-latency communication (HRLLC). In this setting, speculative decoding can facilitate collaborative inference of models distributively deployed: an on-device draft model locally generates drafts and a remote server-based target model verifies and corrects them, resulting lower latency. However, unlike autoregressive text generation, behavior cloning policies, typically used for embodied AI applications like robot manipulation and autonomous driving, cannot parallelize verification and correction for multiple drafts as each action depends on observation which needs to be updated by a previous action. To this end, we propose Action Deviation-Aware Hybrid Inference, wherein the draft model estimates an action's need for verification and correction by the target model and selectively skips communication and computation for server operations. Action deviation shows a strong correlation with action's rejection probability by the target model, enabling selective skipping. We derive the path deviation threshold that balances the transmission rate and the inference performance, and we empirically show that action deviation-aware hybrid inference reduces uplink transmission and server operation by 40%, while lowering end-to-end latency by 33.32% relative to hybrid inference without skipping and achieving task success rate up to 97.03% of that of target model only inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01623v1">VLA-R1: Enhancing Reasoning in Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models aim to unify perception, language understanding, and action generation, offering strong cross-task and cross-scene generalization with broad impact on embodied AI. However, current VLA models often lack explicit step-by-step reasoning, instead emitting final actions without considering affordance constraints or geometric relations. Their post-training pipelines also rarely reinforce reasoning quality, relying primarily on supervised fine-tuning with weak reward design. To address these challenges, we present VLA-R1, a reasoning-enhanced VLA that integrates Reinforcement Learning from Verifiable Rewards (RLVR) with Group Relative Policy Optimization (GRPO) to systematically optimize both reasoning and execution. Specifically, we design an RLVR-based post-training strategy with verifiable rewards for region alignment, trajectory consistency, and output formatting, thereby strengthening reasoning robustness and execution accuracy. Moreover, we develop VLA-CoT-13K, a high-quality dataset that provides chain-of-thought supervision explicitly aligned with affordance and trajectory annotations. Furthermore, extensive evaluations on in-domain, out-of-domain, simulation, and real-robot platforms demonstrate that VLA-R1 achieves superior generalization and real-world performance compared to prior VLA methods. We plan to release the model, code, and dataset following the publication of this work. Code: https://github.com/GigaAI-research/VLA-R1. Website: https://gigaai-research.github.io/VLA-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16928v2">Beyond Needle(s) in the Embodied Haystack: Environment, Architecture, and Training Considerations for Long Context Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI. $\infty$-THOR provides: (1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories; (2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents' long-context reasoning ability; and (3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences. To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction. Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions. Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00441v1">Seeing through Uncertainty: Robust Task-Oriented Optimization in Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Visual navigation is a fundamental problem in embodied AI, yet practical deployments demand long-horizon planning capabilities to address multi-objective tasks. A major bottleneck is data scarcity: policies learned from limited data often overfit and fail to generalize OOD. Existing neural network-based agents typically increase architectural complexity that paradoxically become counterproductive in the small-sample regime. This paper introduce NeuRO, a integrated learning-to-optimize framework that tightly couples perception networks with downstream task-level robust optimization. Specifically, NeuRO addresses core difficulties in this integration: (i) it transforms noisy visual predictions under data scarcity into convex uncertainty sets using Partially Input Convex Neural Networks (PICNNs) with conformal calibration, which directly parameterize the optimization constraints; and (ii) it reformulates planning under partial observability as a robust optimization problem, enabling uncertainty-aware policies that transfer across environments. Extensive experiments on both unordered and sequential multi-object navigation tasks demonstrate that NeuRO establishes SoTA performance, particularly in generalization to unseen environments. Our work thus presents a significant advancement for developing robust, generalizable autonomous agents.
    </details>
</div>
