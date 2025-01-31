# embodied ai - 2024_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06874v3">Learning Reward and Policy Jointly from Demonstration and Preference Improves Alignment</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      Aligning human preference and value is an important requirement for building contemporary foundation models and embodied AI. However, popular approaches such as reinforcement learning with human feedback (RLHF) break down the task into successive stages, such as supervised fine-tuning (SFT), reward modeling (RM), and reinforcement learning (RL), each performing one specific learning task. Such a sequential approach results in serious issues such as significant under-utilization of data and distribution mismatch between the learned reward model and generated policy, which eventually lead to poor alignment performance. We develop a single stage approach named Alignment with Integrated Human Feedback (AIHF), capable of integrating both human preference and demonstration to train reward models and the policy. The proposed approach admits a suite of efficient algorithms, which can easily reduce to, and leverage, popular alignment algorithms such as RLHF and Directly Policy Optimization (DPO), and only requires minor changes to the existing alignment pipelines. We demonstrate the efficiency of the proposed solutions with extensive experiments involving alignment problems in LLMs and robotic control problems in MuJoCo. We observe that the proposed solutions outperform the existing alignment algorithms such as RLHF and DPO by large margins, especially when the amount of high-quality preference data is relatively limited.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04449v5">EARBench: Towards Evaluating Physical Risk Awareness for Task Planning of Foundation Model-based Embodied AI Agents</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      Embodied artificial intelligence (EAI) integrates advanced AI models into physical entities for real-world interaction. The emergence of foundation models as the "brain" of EAI agents for high-level task planning has shown promising results. However, the deployment of these agents in physical environments presents significant safety challenges. For instance, a housekeeping robot lacking sufficient risk awareness might place a metal container in a microwave, potentially causing a fire. To address these critical safety concerns, comprehensive pre-deployment risk assessments are imperative. This study introduces EARBench, a novel framework for automated physical risk assessment in EAI scenarios. EAIRiskBench employs a multi-agent cooperative system that leverages various foundation models to generate safety guidelines, create risk-prone scenarios, make task planning, and evaluate safety systematically. Utilizing this framework, we construct EARDataset, comprising diverse test cases across various domains, encompassing both textual and visual scenarios. Our comprehensive evaluation of state-of-the-art foundation models reveals alarming results: all models exhibit high task risk rates (TRR), with an average of 95.75% across all evaluated models. To address these challenges, we further propose two prompting-based risk mitigation strategies. While these strategies demonstrate some efficacy in reducing TRR, the improvements are limited, still indicating substantial safety concerns. This study provides the first large-scale assessment of physical risk awareness in EAI agents. Our findings underscore the critical need for enhanced safety measures in EAI systems and provide valuable insights for future research directions in developing safer embodied artificial intelligence system. Data and code are available at https://github.com/zihao-ai/EARBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14093v2">A Survey on Vision-Language-Action Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 17 pages, a survey of vision-language-action models
    </div>
    <details class="paper-abstract">
      Deep learning has demonstrated remarkable success across many domains, including computer vision, natural language processing, and reinforcement learning. Representative artificial neural networks in these fields span convolutional neural networks, Transformers, and deep Q-networks. Built upon unimodal neural networks, numerous multi-modal models have been introduced to address a range of tasks such as visual question answering, image captioning, and speech recognition. The rise of instruction-following robotic policies in embodied AI has spurred the development of a novel category of multi-modal models known as vision-language-action models (VLAs). Their multi-modality capability has become a foundational element in robot learning. Various methods have been proposed to enhance traits such as versatility, dexterity, and generalizability. Some models focus on refining specific components. Others aim to develop control policies adept at predicting low-level actions. Certain VLAs serve as high-level task planners capable of decomposing long-horizon tasks into executable subtasks. Over the past few years, a myriad of VLAs have emerged, reflecting the rapid advancement of embodied AI. Therefore, it is imperative to capture the evolving landscape through a comprehensive survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12977v2">MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Collaborative Learning</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Contemporary embodied agents, such as Voyager in Minecraft, have demonstrated promising capabilities in open-ended individual learning. However, when powered with open large language models (LLMs), these agents often struggle with rudimentary tasks, even when fine-tuned on domain-specific knowledge. Inspired by human cultural learning, we present \collabvoyager, a novel framework that enhances Voyager with lifelong collaborative learning through explicit perspective-taking. \collabvoyager introduces three key innovations: (1) theory of mind representations linking percepts, beliefs, desires, and actions; (2) natural language communication between agents; and (3) semantic memory of task and environment knowledge and episodic memory of collaboration episodes. These advancements enable agents to reason about their and others' mental states, empirically addressing two prevalent failure modes: false beliefs and faulty task executions. In mixed-expertise Minecraft experiments, \collabvoyager agents outperform Voyager counterparts, significantly improving task completion rate by $66.6\% (+39.4\%)$ for collecting one block of dirt and $70.8\% (+20.8\%)$ for collecting one wood block. They exhibit emergent behaviors like knowledge transfer from expert to novice agents and collaborative code correction. \collabvoyager agents also demonstrate the ability to adapt to out-of-distribution tasks by using their previous experiences and beliefs obtained through collaboration. In this open-ended social learning paradigm, \collabvoyager paves the way for the democratic development of embodied AI, where agents learn in deployment from both peer and environmental feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12822v2">AVID: Adapting Video Diffusion Models to World Models</a></div>
    <div class="paper-meta">
      📅 2024-11-24
      | 💬 Project Webpage: https://sites.google.com/view/avid-world-model-adapters/home
    </div>
    <details class="paper-abstract">
      Large-scale generative models have achieved remarkable success in a number of domains. However, for sequential decision-making problems, such as robotics, action-labelled data is often scarce and therefore scaling-up foundation models for decision-making remains a challenge. A potential solution lies in leveraging widely-available unlabelled videos to train world models that simulate the consequences of actions. If the world model is accurate, it can be used to optimize decision-making in downstream tasks. Image-to-video diffusion models are already capable of generating highly realistic synthetic videos. However, these models are not action-conditioned, and the most powerful models are closed-source which means they cannot be finetuned. In this work, we propose to adapt pretrained video diffusion models to action-conditioned world models, without access to the parameters of the pretrained model. Our approach, AVID, trains an adapter on a small domain-specific dataset of action-labelled videos. AVID uses a learned mask to modify the intermediate outputs of the pretrained model and generate accurate action-conditioned videos. We evaluate AVID on video game and real-world robotics data, and show that it outperforms existing baselines for diffusion model adaptation.1 Our results demonstrate that if utilized correctly, pretrained video models have the potential to be powerful tools for embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15714v1">ROOT: VLM based System for Indoor Scene Understanding and Beyond</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      Recently, Vision Language Models (VLMs) have experienced significant advancements, yet these models still face challenges in spatial hierarchical reasoning within indoor scenes. In this study, we introduce ROOT, a VLM-based system designed to enhance the analysis of indoor scenes. Specifically, we first develop an iterative object perception algorithm using GPT-4V to detect object entities within indoor scenes. This is followed by employing vision foundation models to acquire additional meta-information about the scene, such as bounding boxes. Building on this foundational data, we propose a specialized VLM, SceneVLM, which is capable of generating spatial hierarchical scene graphs and providing distance information for objects within indoor environments. This information enhances our understanding of the spatial arrangement of indoor scenes. To train our SceneVLM, we collect over 610,000 images from various public indoor datasets and implement a scene data generation pipeline with a semi-automated technique to establish relationships and estimate distances among indoor objects. By utilizing this enriched data, we conduct various training recipes and finish SceneVLM. Our experiments demonstrate that \rootname facilitates indoor scene understanding and proves effective in diverse downstream applications, such as 3D scene generation and embodied AI. The code will be released at \url{https://github.com/harrytea/ROOT}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13927v1">Multimodal 3D Reasoning Segmentation with Complex Scenes</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      The recent development in multimodal learning has greatly advanced the research in 3D scene understanding in various real-world tasks such as embodied AI. However, most existing work shares two typical constraints: 1) they are short of reasoning ability for interaction and interpretation of human intension and 2) they focus on scenarios with single-category objects only which leads to over-simplified textual descriptions due to the negligence of multi-object scenarios and spatial relations among objects. We bridge the research gaps by proposing a 3D reasoning segmentation task for multiple objects in scenes. The task allows producing 3D segmentation masks and detailed textual explanations as enriched by 3D spatial relations among objects. To this end, we create ReasonSeg3D, a large-scale and high-quality benchmark that integrates 3D spatial relations with generated question-answer pairs and 3D segmentation masks. In addition, we design MORE3D, a simple yet effective method that enables multi-object 3D reasoning segmentation with user questions and textual outputs. Extensive experiments show that MORE3D excels in reasoning and segmenting complex multi-object 3D scenes, and the created ReasonSeg3D offers a valuable platform for future exploration of 3D reasoning segmentation. The dataset and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11844v2">Generative World Explorer</a></div>
    <div class="paper-meta">
      📅 2024-11-19
      | 💬 Website: generative-world-explorer.github.io
    </div>
    <details class="paper-abstract">
      Planning with partial observation is a central challenge in embodied AI. A majority of prior works have tackled this challenge by developing agents that physically explore their environment to update their beliefs about the world state. In contrast, humans can $\textit{imagine}$ unseen parts of the world through a mental exploration and $\textit{revise}$ their beliefs with imagined observations. Such updated beliefs can allow them to make more informed decisions, without necessitating the physical exploration of the world at all times. To achieve this human-like ability, we introduce the $\textit{Generative World Explorer (Genex)}$, an egocentric world exploration framework that allows an agent to mentally explore a large-scale 3D world (e.g., urban scenes) and acquire imagined observations to update its belief. This updated belief will then help the agent to make a more informed decision at the current step. To train $\textit{Genex}$, we create a synthetic urban scene dataset, Genex-DB. Our experimental results demonstrate that (1) $\textit{Genex}$ can generate high-quality and consistent observations during long-horizon exploration of a large virtual physical world and (2) the beliefs updated with the generated observations can inform an existing decision-making model (e.g., an LLM agent) to make better plans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02389v2">Multi-modal Situated Reasoning in 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 Accepted by NeurIPS 2024 Datasets and Benchmarks Track. Project page: https://msr3d.github.io/
    </div>
    <details class="paper-abstract">
      Situation awareness is essential for understanding and reasoning about 3D scenes in embodied AI agents. However, existing datasets and benchmarks for situated understanding are limited in data modality, diversity, scale, and task scope. To address these limitations, we propose Multi-modal Situated Question Answering (MSQA), a large-scale multi-modal situated reasoning dataset, scalably collected leveraging 3D scene graphs and vision-language models (VLMs) across a diverse range of real-world 3D scenes. MSQA includes 251K situated question-answering pairs across 9 distinct question categories, covering complex scenarios within 3D scenes. We introduce a novel interleaved multi-modal input setting in our benchmark to provide text, image, and point cloud for situation and question description, resolving ambiguity in previous single-modality convention (e.g., text). Additionally, we devise the Multi-modal Situated Next-step Navigation (MSNN) benchmark to evaluate models' situated reasoning for navigation. Comprehensive evaluations on MSQA and MSNN highlight the limitations of existing vision-language models and underscore the importance of handling multi-modal interleaved inputs and situation modeling. Experiments on data scaling and cross-domain transfer further demonstrate the efficacy of leveraging MSQA as a pre-training dataset for developing more powerful situated reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04292v4">Software-Hardware Co-Design For Embodied AI Robots</a></div>
    <div class="paper-meta">
      📅 2024-11-16
    </div>
    <details class="paper-abstract">
      Embodied AI robots have the potential to fundamentally improve the way human beings live and manufacture. Continued progress in the burgeoning field of using large language models to control robots depends critically on an efficient computing substrate. In particular, today's computing systems for embodied AI robots are designed purely based on the interest of algorithm developers, where robot actions are divided into a discrete frame-basis. Such an execution pipeline creates high latency and energy consumption. This paper proposes Corki, an algorithm-architecture co-design framework for real-time embodied AI robot control. Our idea is to decouple LLM inference, robotic control and data communication in the embodied AI robots compute pipeline. Instead of predicting action for one single frame, Corki predicts the trajectory for the near future to reduce the frequency of LLM inference. The algorithm is coupled with a hardware that accelerates transforming trajectory into actual torque signals used to control robots and an execution pipeline that parallels data communication with computation. Corki largely reduces LLM inference frequency by up to 8.0x, resulting in up to 3.6x speed up. The success rate improvement can be up to 17.3%. Code is provided for re-implementation. https://github.com/hyy0613/Corki
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18515v2">Atlas3D: Physically Constrained Self-Supporting Text-to-3D for Simulation and Fabrication</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 Project Page: https://yunuoch.github.io/Atlas3D/
    </div>
    <details class="paper-abstract">
      Existing diffusion-based text-to-3D generation methods primarily focus on producing visually realistic shapes and appearances, often neglecting the physical constraints necessary for downstream tasks. Generated models frequently fail to maintain balance when placed in physics-based simulations or 3D printed. This balance is crucial for satisfying user design intentions in interactive gaming, embodied AI, and robotics, where stable models are needed for reliable interaction. Additionally, stable models ensure that 3D-printed objects, such as figurines for home decoration, can stand on their own without requiring additional supports. To fill this gap, we introduce Atlas3D, an automatic and easy-to-implement method that enhances existing Score Distillation Sampling (SDS)-based text-to-3D tools. Atlas3D ensures the generation of self-supporting 3D models that adhere to physical laws of stability under gravity, contact, and friction. Our approach combines a novel differentiable simulation-based loss function with physically inspired regularization, serving as either a refinement or a post-processing module for existing frameworks. We verify Atlas3D's efficacy through extensive generation tasks and validate the resulting 3D models in both simulated and real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09823v1">Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      Creating large-scale interactive 3D environments is essential for the development of Robotics and Embodied AI research. Current methods, including manual design, procedural generation, diffusion-based scene generation, and large language model (LLM) guided scene design, are hindered by limitations such as excessive human effort, reliance on predefined rules or training datasets, and limited 3D spatial reasoning ability. Since pre-trained 2D image generative models better capture scene and object configuration than LLMs, we address these challenges by introducing Architect, a generative framework that creates complex and realistic 3D embodied environments leveraging diffusion-based 2D image inpainting. In detail, we utilize foundation visual perception models to obtain each generated object from the image and leverage pre-trained depth estimation models to lift the generated 2D image to 3D space. Our pipeline is further extended to a hierarchical and iterative inpainting process to continuously generate placement of large furniture and small objects to enrich the scene. This iterative structure brings the flexibility for our method to generate or refine scenes from various starting points, such as text, floor plans, or pre-arranged environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05902v1">Autoregressive Models in Vision: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-11-08
    </div>
    <details class="paper-abstract">
      Autoregressive modeling has been a huge success in the field of natural language processing (NLP). Recently, autoregressive models have emerged as a significant area of focus in computer vision, where they excel in producing high-quality visual content. Autoregressive models in NLP typically operate on subword tokens. However, the representation strategy in computer vision can vary in different levels, \textit{i.e.}, pixel-level, token-level, or scale-level, reflecting the diverse and hierarchical nature of visual data compared to the sequential structure of language. This survey comprehensively examines the literature on autoregressive models applied to vision. To improve readability for researchers from diverse research backgrounds, we start with preliminary sequence representation and modeling in vision. Next, we divide the fundamental frameworks of visual autoregressive models into three general sub-categories, including pixel-based, token-based, and scale-based models based on the strategy of representation. We then explore the interconnections between autoregressive models and other generative models. Furthermore, we present a multi-faceted categorization of autoregressive models in computer vision, including image generation, video generation, 3D generation, and multi-modal generation. We also elaborate on their applications in diverse domains, including emerging domains such as embodied AI and 3D medical AI, with about 250 related references. Finally, we highlight the current challenges to autoregressive models in vision with suggestions about potential research directions. We have also set up a Github repository to organize the papers included in this survey at: \url{https://github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14565v2">"Where am I?" Scene Retrieval with Language</a></div>
    <div class="paper-meta">
      📅 2024-11-08
    </div>
    <details class="paper-abstract">
      Natural language interfaces to embodied AI are becoming more ubiquitous in our daily lives. This opens up further opportunities for language-based interaction with embodied agents, such as a user verbally instructing an agent to execute some task in a specific location. For example, "put the bowls back in the cupboard next to the fridge" or "meet me at the intersection under the red sign." As such, we need methods that interface between natural language and map representations of the environment. To this end, we explore the question of whether we can use an open-set natural language query to identify a scene represented by a 3D scene graph. We define this task as "language-based scene-retrieval" and it is closely related to "coarse-localization," but we are instead searching for a match from a collection of disjoint scenes and not necessarily a large-scale continuous map. We present Text2SceneGraphMatcher, a "scene-retrieval" pipeline that learns joint embeddings between text descriptions and scene graphs to determine if they are a match. The code, trained models, and datasets will be made public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.07918v5">Unified Human-Scene Interaction via Prompted Chain-of-Contacts</a></div>
    <div class="paper-meta">
      📅 2024-11-05
      | 💬 A unified Human-Scene Interaction framework that supports versatile interactions through language commands.Project URL: https://xizaoqu.github.io/unihsi/ . Code: https://github.com/OpenRobotLab/UniHSI
    </div>
    <details class="paper-abstract">
      Human-Scene Interaction (HSI) is a vital component of fields like embodied AI and virtual reality. Despite advancements in motion quality and physical plausibility, two pivotal factors, versatile interaction control and the development of a user-friendly interface, require further exploration before the practical application of HSI. This paper presents a unified HSI framework, UniHSI, which supports unified control of diverse interactions through language commands. This framework is built upon the definition of interaction as Chain of Contacts (CoC): steps of human joint-object part pairs, which is inspired by the strong correlation between interaction types and human-object contact regions. Based on the definition, UniHSI constitutes a Large Language Model (LLM) Planner to translate language prompts into task plans in the form of CoC, and a Unified Controller that turns CoC into uniform task execution. To facilitate training and evaluation, we collect a new dataset named ScenePlan that encompasses thousands of task plans generated by LLMs based on diverse scenarios. Comprehensive experiments demonstrate the effectiveness of our framework in versatile task execution and generalizability to real scanned scenes. The project page is at https://github.com/OpenRobotLab/UniHSI .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02236v1">3D Audio-Visual Segmentation</a></div>
    <div class="paper-meta">
      📅 2024-11-04
      | 💬 Accepted at the NeurIPS 2024 Workshop on Audio Imagination
    </div>
    <details class="paper-abstract">
      Recognizing the sounding objects in scenes is a longstanding objective in embodied AI, with diverse applications in robotics and AR/VR/MR. To that end, Audio-Visual Segmentation (AVS), taking as condition an audio signal to identify the masks of the target sounding objects in an input image with synchronous camera and microphone sensors, has been recently advanced. However, this paradigm is still insufficient for real-world operation, as the mapping from 2D images to 3D scenes is missing. To address this fundamental limitation, we introduce a novel research problem, 3D Audio-Visual Segmentation, extending the existing AVS to the 3D output space. This problem poses more challenges due to variations in camera extrinsics, audio scattering, occlusions, and diverse acoustics across sounding object categories. To facilitate this research, we create the very first simulation based benchmark, 3DAVS-S34-O7, providing photorealistic 3D scene environments with grounded spatial audio under single-instance and multi-instance settings, across 34 scenes and 7 object categories. This is made possible by re-purposing the Habitat simulator to generate comprehensive annotations of sounding object locations and corresponding 3D masks. Subsequently, we propose a new approach, EchoSegnet, characterized by integrating the ready-to-use knowledge from pretrained 2D audio-visual foundation models synergistically with 3D visual scene representation through spatial audio-aware mask alignment and refinement. Extensive experiments demonstrate that EchoSegnet can effectively segment sounding objects in 3D space on our new benchmark, representing a significant advancement in the field of embodied AI. Project page: https://surrey-uplab.github.io/research/3d-audio-visual-segmentation/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19236v3">Human-Aware Vision-and-Language Navigation: Bridging Simulation to Reality with Dynamic Human Interactions</a></div>
    <div class="paper-meta">
      📅 2024-11-02
      | 💬 Spotlight at NeurIPS 2024 D&B Track. 32 pages, 18 figures, Project Page: https://lpercc.github.io/HA3D_simulator/
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) aims to develop embodied agents that navigate based on human instructions. However, current VLN frameworks often rely on static environments and optimal expert supervision, limiting their real-world applicability. To address this, we introduce Human-Aware Vision-and-Language Navigation (HA-VLN), extending traditional VLN by incorporating dynamic human activities and relaxing key assumptions. We propose the Human-Aware 3D (HA3D) simulator, which combines dynamic human activities with the Matterport3D dataset, and the Human-Aware Room-to-Room (HA-R2R) dataset, extending R2R with human activity descriptions. To tackle HA-VLN challenges, we present the Expert-Supervised Cross-Modal (VLN-CM) and Non-Expert-Supervised Decision Transformer (VLN-DT) agents, utilizing cross-modal fusion and diverse training strategies for effective navigation in dynamic human environments. A comprehensive evaluation, including metrics considering human activities, and systematic analysis of HA-VLN's unique challenges, underscores the need for further research to enhance HA-VLN agents' real-world robustness and adaptability. Ultimately, this work provides benchmarks and insights for future research on embodied AI and Sim2Real transfer, paving the way for more realistic and applicable VLN systems in human-populated environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03570v3">Embodied AI with Two Arms: Zero-shot Learning, Safety and Modularity</a></div>
    <div class="paper-meta">
      📅 2024-11-01
    </div>
    <details class="paper-abstract">
      We present an embodied AI system which receives open-ended natural language instructions from a human, and controls two arms to collaboratively accomplish potentially long-horizon tasks over a large workspace. Our system is modular: it deploys state of the art Large Language Models for task planning,Vision-Language models for semantic perception, and Point Cloud transformers for grasping. With semantic and physical safety in mind, these modules are interfaced with a real-time trajectory optimizer and a compliant tracking controller to enable human-robot proximity. We demonstrate performance for the following tasks: bi-arm sorting, bottle opening, and trash disposal tasks. These are done zero-shot where the models used have not been trained with any real world data from this bi-arm robot, scenes or workspace. Composing both learning- and non-learning-based components in a modular fashion with interpretable inputs and outputs allows the user to easily debug points of failures and fragilities. One may also in-place swap modules to improve the robustness of the overall platform, for instance with imitation-learned policies. Please see https://sites.google.com/corp/view/safe-robots .
    </details>
</div>
