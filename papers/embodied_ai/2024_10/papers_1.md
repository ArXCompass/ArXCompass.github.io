# embodied ai - 2024_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.23535v1">Simulating User Agents for Embodied Conversational-AI</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 8 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Embodied agents designed to assist users with tasks must engage in natural language interactions, interpret instructions, execute actions, and communicate effectively to resolve issues. However, collecting large-scale, diverse datasets of situated human-robot dialogues to train and evaluate such agents is expensive, labor-intensive, and time-consuming. To address this challenge, we propose building a large language model (LLM)-based user agent that can simulate user behavior during interactions with an embodied agent in a virtual environment. Given a user goal (e.g., make breakfast), at each time step, the user agent may observe" the robot actions or speak" to either intervene with the robot or answer questions. Such a user agent assists in improving the scalability and efficiency of embodied dialogues dataset generation and is critical for enhancing and evaluating the robot's interaction and task completion ability, as well as for research in reinforcement learning using AI feedback. We evaluate our user agent's ability to generate human-like behaviors by comparing its simulated dialogues with the TEACh dataset. We perform three experiments: zero-shot prompting to predict dialogue acts, few-shot prompting, and fine-tuning on the TEACh training subset. Results show the LLM-based user agent achieves an F-measure of 42% with zero-shot prompting and 43.4% with few-shot prompting in mimicking human speaking behavior. Through fine-tuning, performance in deciding when to speak remained stable, while deciding what to say improved from 51.1% to 62.5%. These findings showcase the feasibility of the proposed approach for assessing and enhancing the effectiveness of robot task completion through natural language communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2302.03385v2">NeuronsGym: A Hybrid Framework and Benchmark for Robot Tasks with Sim2Real Policy Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The rise of embodied AI has greatly improved the possibility of general mobile agent systems. At present, many evaluation platforms with rich scenes, high visual fidelity and various application scenarios have been developed. In this paper, we present a hybrid framework named NeuronsGym that can be used for policy learning of robot tasks, covering a simulation platform for training policy, and a physical system for studying sim2real problems. Unlike most current single-task, slow-moving robotic platforms, our framework provides agile physical robots with a wider range of speeds, and can be employed to train robotic navigation and confrontation policies. At the same time, in order to evaluate the safety of robot navigation, we propose a safety-weighted path length (SFPL) to improve the safety evaluation in the current mobile robot navigation. Based on this platform, we build a new benchmark for navigation and confrontation tasks under this platform by comparing the current mainstream sim2real methods, and hold the 2022 IEEE Conference on Games (CoG) RoboMaster sim2real challenge. We release the codes of this framework\footnote{\url{https://github.com/DRL-CASIA/NeuronsGym}} and hope that this platform can promote the development of more flexible and agile general mobile agent algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16258v2">MEReQ: Max-Ent Residual-Q Inverse RL for Sample-Efficient Alignment from Intervention</a></div>
    <div class="paper-meta">
      📅 2024-10-28
    </div>
    <details class="paper-abstract">
      Aligning robot behavior with human preferences is crucial for deploying embodied AI agents in human-centered environments. A promising solution is interactive imitation learning from human intervention, where a human expert observes the policy's execution and provides interventions as feedback. However, existing methods often fail to utilize the prior policy efficiently to facilitate learning, thus hindering sample efficiency. In this work, we introduce MEReQ (Maximum-Entropy Residual-Q Inverse Reinforcement Learning), designed for sample-efficient alignment from human intervention. Instead of inferring the complete human behavior characteristics, MEReQ infers a residual reward function that captures the discrepancy between the human expert's and the prior policy's underlying reward functions. It then employs Residual Q-Learning (RQL) to align the policy with human preferences using this residual reward function. Extensive evaluations on simulated and real-world tasks demonstrate that MEReQ achieves sample-efficient policy alignment from human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16919v1">EnvBridge: Bridging Diverse Environments with Cross-Environment Knowledge Transfer for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have demonstrated high reasoning capabilities, drawing attention for their applications as agents in various decision-making processes. One notably promising application of LLM agents is robotic manipulation. Recent research has shown that LLMs can generate text planning or control code for robots, providing substantial flexibility and interaction capabilities. However, these methods still face challenges in terms of flexibility and applicability across different environments, limiting their ability to adapt autonomously. Current approaches typically fall into two categories: those relying on environment-specific policy training, which restricts their transferability, and those generating code actions based on fixed prompts, which leads to diminished performance when confronted with new environments. These limitations significantly constrain the generalizability of agents in robotic manipulation. To address these limitations, we propose a novel method called EnvBridge. This approach involves the retention and transfer of successful robot control codes from source environments to target environments. EnvBridge enhances the agent's adaptability and performance across diverse settings by leveraging insights from multiple environments. Notably, our approach alleviates environmental constraints, offering a more flexible and generalizable solution for robotic manipulation tasks. We validated the effectiveness of our method using robotic manipulation benchmarks: RLBench, MetaWorld, and CALVIN. Our experiments demonstrate that LLM agents can successfully leverage diverse knowledge sources to solve complex tasks. Consequently, our approach significantly enhances the adaptability and robustness of robotic manipulation agents in planning across diverse environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.14590v1">Embodied Exploration of Latent Spaces and Explainable AI</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 In Proceedings of Explainable AI for the Arts Workshop 2024 (XAIxArts 2024) arXiv:2406.14485
    </div>
    <details class="paper-abstract">
      In this paper, we explore how performers' embodied interactions with a Neural Audio Synthesis model allow the exploration of the latent space of such a model, mediated through movements sensed by e-textiles. We provide background and context for the performance, highlighting the potential of embodied practices to contribute to developing explainable AI systems. By integrating various artistic domains with explainable AI principles, our interdisciplinary exploration contributes to the discourse on art, embodiment, and AI, offering insights into intuitive approaches found through bodily expression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.08588v2">Octopus: Embodied Vision-Language Programmer from Environmental Feedback</a></div>
    <div class="paper-meta">
      📅 2024-10-20
      | 💬 Project Page: https://choiszt.github.io/Octopus/, Codebase: https://github.com/dongyh20/Octopus
    </div>
    <details class="paper-abstract">
      Large vision-language models (VLMs) have achieved substantial progress in multimodal perception and reasoning. When integrated into an embodied agent, existing embodied VLM works either output detailed action sequences at the manipulation level or only provide plans at an abstract level, leaving a gap between high-level planning and real-world manipulation. To bridge this gap, we introduce Octopus, an embodied vision-language programmer that uses executable code generation as a medium to connect planning and manipulation. Octopus is designed to 1) proficiently comprehend an agent's visual and textual task objectives, 2) formulate intricate action sequences, and 3) generate executable code. To facilitate Octopus model development, we introduce OctoVerse: a suite of environments tailored for benchmarking vision-based code generators on a wide spectrum of tasks, ranging from mundane daily chores in simulators to sophisticated interactions in complex video games such as Grand Theft Auto (GTA) and Minecraft. To train Octopus, we leverage GPT-4 to control an explorative agent that generates training data, i.e., action blueprints and corresponding executable code. We also collect feedback that enables an enhanced training scheme called Reinforcement Learning with Environmental Feedback (RLEF). Through a series of experiments, we demonstrate Octopus's functionality and present compelling results, showing that the proposed RLEF refines the agent's decision-making. By open-sourcing our simulation environments, dataset, and model architecture, we aspire to ignite further innovation and foster collaborative applications within the broader embodied AI community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10918v5">Multi-LLM QA with Embodied Exploration</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 16 pages, 9 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have grown in popularity due to their natural language interface and pre trained knowledge, leading to rapidly increasing success in question-answering (QA) tasks. More recently, multi-agent systems with LLM-based agents (Multi-LLM) have been utilized increasingly more for QA. In these scenarios, the models may each answer the question and reach a consensus or each model is specialized to answer different domain questions. However, most prior work dealing with Multi-LLM QA has focused on scenarios where the models are asked in a zero-shot manner or are given information sources to extract the answer. For question answering of an unknown environment, embodied exploration of the environment is first needed to answer the question. This skill is necessary for personalizing embodied AI to environments such as households. There is a lack of insight into whether a Multi-LLM system can handle question-answering based on observations from embodied exploration. In this work, we address this gap by investigating the use of Multi-Embodied LLM Explorers (MELE) for QA in an unknown environment. Multiple LLM-based agents independently explore and then answer queries about a household environment. We analyze different aggregation methods to generate a single, final answer for each query: debating, majority voting, and training a central answer module (CAM). Using CAM, we observe a $46\%$ higher accuracy compared against the other non-learning-based aggregation methods. We provide code and the query dataset for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.13407v1">BestMan: A Modular Mobile Manipulator Platform for Embodied AI with Unified Simulation-Hardware APIs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (Embodied AI) emphasizes agents' ability to perceive, understand, and act in physical environments. Simulation platforms play a crucial role in advancing this field by enabling the validation and optimization of algorithms. However, existing platforms face challenges such as multilevel technical integration complexity, insufficient modularity, interface heterogeneity, and adaptation to diverse hardware. We present BestMan, a simulation platform based on PyBullet, designed to address these issues. BestMan introduces an integrated multilevel skill chain for seamless coordination across perception, planning, and control; a highly modular architecture for flexible algorithm integration; unified interfaces for smooth simulation-to-reality transfer; and a hardware-agnostic approach for adapting to various mobile manipulator configurations. These features collectively simplify development and enhance platform expandability, making BestMan a valuable tool for Embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.14590v1">Embodied Exploration of Latent Spaces and Explainable AI</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 In Proceedings of Explainable AI for the Arts Workshop 2024 (XAIxArts 2024) arXiv:2406.14485
    </div>
    <details class="paper-abstract">
      In this paper, we explore how performers' embodied interactions with a Neural Audio Synthesis model allow the exploration of the latent space of such a model, mediated through movements sensed by e-textiles. We provide background and context for the performance, highlighting the potential of embodied practices to contribute to developing explainable AI systems. By integrating various artistic domains with explainable AI principles, our interdisciplinary exploration contributes to the discourse on art, embodiment, and AI, offering insights into intuitive approaches found through bodily expression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00785v1">IGOR: Image-GOal Representations are the Atomic Control Units for Foundation Models in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      We introduce Image-GOal Representations (IGOR), aiming to learn a unified, semantically consistent action space across human and various robots. Through this unified latent action space, IGOR enables knowledge transfer among large-scale robot and human activity data. We achieve this by compressing visual changes between an initial image and its goal state into latent actions. IGOR allows us to generate latent action labels for internet-scale video data. This unified latent action space enables the training of foundation policy and world models across a wide variety of tasks performed by both robots and humans. We demonstrate that: (1) IGOR learns a semantically consistent action space for both human and robots, characterizing various possible motions of objects representing the physical interaction knowledge; (2) IGOR can "migrate" the movements of the object in the one video to other videos, even across human and robots, by jointly using the latent action model and world model; (3) IGOR can learn to align latent actions with natural language through the foundation policy model, and integrate latent actions with a low-level policy model to achieve effective robot control. We believe IGOR opens new possibilities for human-to-robot knowledge transfer and control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13407v1">BestMan: A Modular Mobile Manipulator Platform for Embodied AI with Unified Simulation-Hardware APIs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (Embodied AI) emphasizes agents' ability to perceive, understand, and act in physical environments. Simulation platforms play a crucial role in advancing this field by enabling the validation and optimization of algorithms. However, existing platforms face challenges such as multilevel technical integration complexity, insufficient modularity, interface heterogeneity, and adaptation to diverse hardware. We present BestMan, a simulation platform based on PyBullet, designed to address these issues. BestMan introduces an integrated multilevel skill chain for seamless coordination across perception, planning, and control; a highly modular architecture for flexible algorithm integration; unified interfaces for smooth simulation-to-reality transfer; and a hardware-agnostic approach for adapting to various mobile manipulator configurations. These features collectively simplify development and enhance platform expandability, making BestMan a valuable tool for Embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.13407v1">BestMan: A Modular Mobile Manipulator Platform for Embodied AI with Unified Simulation-Hardware APIs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (Embodied AI) emphasizes agents' ability to perceive, understand, and act in physical environments. Simulation platforms play a crucial role in advancing this field by enabling the validation and optimization of algorithms. However, existing platforms face challenges such as multilevel technical integration complexity, insufficient modularity, interface heterogeneity, and adaptation to diverse hardware. We present BestMan, a simulation platform based on PyBullet, designed to address these issues. BestMan introduces an integrated multilevel skill chain for seamless coordination across perception, planning, and control; a highly modular architecture for flexible algorithm integration; unified interfaces for smooth simulation-to-reality transfer; and a hardware-agnostic approach for adapting to various mobile manipulator configurations. These features collectively simplify development and enhance platform expandability, making BestMan a valuable tool for Embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09016v3">Closed-Loop Visuomotor Control with Generative Expectation for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Accepted at NeurIPS 2024. Code and models: https://github.com/OpenDriveLab/CLOVER
    </div>
    <details class="paper-abstract">
      Despite significant progress in robotics and embodied AI in recent years, deploying robots for long-horizon tasks remains a great challenge. Majority of prior arts adhere to an open-loop philosophy and lack real-time feedback, leading to error accumulation and undesirable robustness. A handful of approaches have endeavored to establish feedback mechanisms leveraging pixel-level differences or pre-trained visual representations, yet their efficacy and adaptability have been found to be constrained. Inspired by classic closed-loop control systems, we propose CLOVER, a closed-loop visuomotor control framework that incorporates feedback mechanisms to improve adaptive robotic control. CLOVER consists of a text-conditioned video diffusion model for generating visual plans as reference inputs, a measurable embedding space for accurate error quantification, and a feedback-driven controller that refines actions from feedback and initiates replans as needed. Our framework exhibits notable advancement in real-world robotic tasks and achieves state-of-the-art on CALVIN benchmark, improving by 8% over previous open-loop counterparts. Code and checkpoints are maintained at https://github.com/OpenDriveLab/CLOVER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.11623v1">VidEgoThink: Assessing Egocentric Video Understanding Capabilities for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Recent advancements in Multi-modal Large Language Models (MLLMs) have opened new avenues for applications in Embodied AI. Building on previous work, EgoThink, we introduce VidEgoThink, a comprehensive benchmark for evaluating egocentric video understanding capabilities. To bridge the gap between MLLMs and low-level control in Embodied AI, we design four key interrelated tasks: video question-answering, hierarchy planning, visual grounding and reward modeling. To minimize manual annotation costs, we develop an automatic data generation pipeline based on the Ego4D dataset, leveraging the prior knowledge and multimodal capabilities of GPT-4o. Three human annotators then filter the generated data to ensure diversity and quality, resulting in the VidEgoThink benchmark. We conduct extensive experiments with three types of models: API-based MLLMs, open-source image-based MLLMs, and open-source video-based MLLMs. Experimental results indicate that all MLLMs, including GPT-4o, perform poorly across all tasks related to egocentric video understanding. These findings suggest that foundation models still require significant advancements to be effectively applied to first-person scenarios in Embodied AI. In conclusion, VidEgoThink reflects a research trend towards employing MLLMs for egocentric vision, akin to human capabilities, enabling active observation and interaction in the complex real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11623v1">VidEgoThink: Assessing Egocentric Video Understanding Capabilities for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Recent advancements in Multi-modal Large Language Models (MLLMs) have opened new avenues for applications in Embodied AI. Building on previous work, EgoThink, we introduce VidEgoThink, a comprehensive benchmark for evaluating egocentric video understanding capabilities. To bridge the gap between MLLMs and low-level control in Embodied AI, we design four key interrelated tasks: video question-answering, hierarchy planning, visual grounding and reward modeling. To minimize manual annotation costs, we develop an automatic data generation pipeline based on the Ego4D dataset, leveraging the prior knowledge and multimodal capabilities of GPT-4o. Three human annotators then filter the generated data to ensure diversity and quality, resulting in the VidEgoThink benchmark. We conduct extensive experiments with three types of models: API-based MLLMs, open-source image-based MLLMs, and open-source video-based MLLMs. Experimental results indicate that all MLLMs, including GPT-4o, perform poorly across all tasks related to egocentric video understanding. These findings suggest that foundation models still require significant advancements to be effectively applied to first-person scenarios in Embodied AI. In conclusion, VidEgoThink reflects a research trend towards employing MLLMs for egocentric vision, akin to human capabilities, enabling active observation and interaction in the complex real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11402v1">M2Diffuser: Diffusion-based Trajectory Optimization for Mobile Manipulation in 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Recent advances in diffusion models have opened new avenues for research into embodied AI agents and robotics. Despite significant achievements in complex robotic locomotion and skills, mobile manipulation-a capability that requires the coordination of navigation and manipulation-remains a challenge for generative AI techniques. This is primarily due to the high-dimensional action space, extended motion trajectories, and interactions with the surrounding environment. In this paper, we introduce M2Diffuser, a diffusion-based, scene-conditioned generative model that directly generates coordinated and efficient whole-body motion trajectories for mobile manipulation based on robot-centric 3D scans. M2Diffuser first learns trajectory-level distributions from mobile manipulation trajectories provided by an expert planner. Crucially, it incorporates an optimization module that can flexibly accommodate physical constraints and task objectives, modeled as cost and energy functions, during the inference process. This enables the reduction of physical violations and execution errors at each denoising step in a fully differentiable manner. Through benchmarking on three types of mobile manipulation tasks across over 20 scenes, we demonstrate that M2Diffuser outperforms state-of-the-art neural planners and successfully transfers the generated trajectories to a real-world robot. Our evaluations underscore the potential of generative AI to enhance the generalization of traditional planning and learning-based robotic methods, while also highlighting the critical role of enforcing physical constraints for safe and robust execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.09054v1">Circuits and Systems for Embodied AI: Exploring uJ Multi-Modal Perception for Nano-UAVs on the Kraken Shield</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 5 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Embodied artificial intelligence (AI) requires pushing complex multi-modal models to the extreme edge for time-constrained tasks such as autonomous navigation of robots and vehicles. On small form-factor devices, e.g., nano-sized unmanned aerial vehicles (UAVs), such challenges are exacerbated by stringent constraints on energy efficiency and weight. In this paper, we explore embodied multi-modal AI-based perception for Nano-UAVs with the Kraken shield, a 7g multi-sensor (frame-based and event-based imagers) board based on Kraken, a 22 nm SoC featuring multiple acceleration engines for multi-modal event and frame-based inference based on spiking (SNN) and ternary (TNN) neural networks, respectively. Kraken can execute SNN real-time inference for depth estimation at 1.02k inf/s, 18 μJ/inf, TNN real-time inference for object classification at 10k inf/s, 6 μJ/inf, and real-time inference for obstacle avoidance at 221 frame/s, 750 μJ/inf.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.11623v1">VidEgoThink: Assessing Egocentric Video Understanding Capabilities for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Recent advancements in Multi-modal Large Language Models (MLLMs) have opened new avenues for applications in Embodied AI. Building on previous work, EgoThink, we introduce VidEgoThink, a comprehensive benchmark for evaluating egocentric video understanding capabilities. To bridge the gap between MLLMs and low-level control in Embodied AI, we design four key interrelated tasks: video question-answering, hierarchy planning, visual grounding and reward modeling. To minimize manual annotation costs, we develop an automatic data generation pipeline based on the Ego4D dataset, leveraging the prior knowledge and multimodal capabilities of GPT-4o. Three human annotators then filter the generated data to ensure diversity and quality, resulting in the VidEgoThink benchmark. We conduct extensive experiments with three types of models: API-based MLLMs, open-source image-based MLLMs, and open-source video-based MLLMs. Experimental results indicate that all MLLMs, including GPT-4o, perform poorly across all tasks related to egocentric video understanding. These findings suggest that foundation models still require significant advancements to be effectively applied to first-person scenarios in Embodied AI. In conclusion, VidEgoThink reflects a research trend towards employing MLLMs for egocentric vision, akin to human capabilities, enabling active observation and interaction in the complex real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2407.08725v2">MetaUrban: An Embodied AI Simulation Platform for Urban Micromobility</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 Technical report. Project page: https://metadriverse.github.io/metaurban/
    </div>
    <details class="paper-abstract">
      Public urban spaces like streetscapes and plazas serve residents and accommodate social life in all its vibrant variations. Recent advances in Robotics and Embodied AI make public urban spaces no longer exclusive to humans. Food delivery bots and electric wheelchairs have started sharing sidewalks with pedestrians, while robot dogs and humanoids have recently emerged in the street. Micromobility enabled by AI for short-distance travel in public urban spaces plays a crucial component in the future transportation system. Ensuring the generalizability and safety of AI models maneuvering mobile machines is essential. In this work, we present MetaUrban, a compositional simulation platform for the AI-driven urban micromobility research. MetaUrban can construct an infinite number of interactive urban scenes from compositional elements, covering a vast array of ground plans, object placements, pedestrians, vulnerable road users, and other mobile agents' appearances and dynamics. We design point navigation and social navigation tasks as the pilot study using MetaUrban for urban micromobility research and establish various baselines of Reinforcement Learning and Imitation Learning. We conduct extensive evaluation across mobile machines, demonstrating that heterogeneous mechanical structures significantly influence the learning and execution of AI policies. We perform a thorough ablation study, showing that the compositional nature of the simulated environments can substantially improve the generalizability and safety of the trained mobile agents. MetaUrban will be made publicly available to provide research opportunities and foster safe and trustworthy embodied AI and micromobility in cities. The code and dataset will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13659v2">EgoChoir: Capturing 3D Human-Object Interaction Regions from Egocentric Views</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 NeurIPS2024, project: https://yyvhang.github.io/EgoChoir/
    </div>
    <details class="paper-abstract">
      Understanding egocentric human-object interaction (HOI) is a fundamental aspect of human-centric perception, facilitating applications like AR/VR and embodied AI. For the egocentric HOI, in addition to perceiving semantics e.g., ''what'' interaction is occurring, capturing ''where'' the interaction specifically manifests in 3D space is also crucial, which links the perception and operation. Existing methods primarily leverage observations of HOI to capture interaction regions from an exocentric view. However, incomplete observations of interacting parties in the egocentric view introduce ambiguity between visual observations and interaction contents, impairing their efficacy. From the egocentric view, humans integrate the visual cortex, cerebellum, and brain to internalize their intentions and interaction concepts of objects, allowing for the pre-formulation of interactions and making behaviors even when interaction regions are out of sight. In light of this, we propose harmonizing the visual appearance, head motion, and 3D object to excavate the object interaction concept and subject intention, jointly inferring 3D human contact and object affordance from egocentric videos. To achieve this, we present EgoChoir, which links object structures with interaction contexts inherent in appearance and head motion to reveal object affordance, further utilizing it to model human contact. Additionally, a gradient modulation is employed to adopt appropriate clues for capturing interaction regions across various egocentric scenarios. Moreover, 3D contact and affordance are annotated for egocentric videos collected from Ego-Exo4D and GIMO to support the task. Extensive experiments on them demonstrate the effectiveness and superiority of EgoChoir.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10179v3">Scaling Instructable Agents Across Many Simulated Worlds</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Building embodied AI systems that can follow arbitrary language instructions in any 3D environment is a key challenge for creating general AI. Accomplishing this goal requires learning to ground language in perception and embodied actions, in order to accomplish complex tasks. The Scalable, Instructable, Multiworld Agent (SIMA) project tackles this by training agents to follow free-form instructions across a diverse range of virtual 3D environments, including curated research environments as well as open-ended, commercial video games. Our goal is to develop an instructable agent that can accomplish anything a human can do in any simulated 3D environment. Our approach focuses on language-driven generality while imposing minimal assumptions. Our agents interact with environments in real-time using a generic, human-like interface: the inputs are image observations and language instructions and the outputs are keyboard-and-mouse actions. This general approach is challenging, but it allows agents to ground language across many visually complex and semantically rich environments while also allowing us to readily run agents in new environments. In this paper we describe our motivation and goal, the initial progress we have made, and promising preliminary results on several diverse research environments and a variety of commercial video games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08725v2">MetaUrban: An Embodied AI Simulation Platform for Urban Micromobility</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 Technical report. Project page: https://metadriverse.github.io/metaurban/
    </div>
    <details class="paper-abstract">
      Public urban spaces like streetscapes and plazas serve residents and accommodate social life in all its vibrant variations. Recent advances in Robotics and Embodied AI make public urban spaces no longer exclusive to humans. Food delivery bots and electric wheelchairs have started sharing sidewalks with pedestrians, while robot dogs and humanoids have recently emerged in the street. Micromobility enabled by AI for short-distance travel in public urban spaces plays a crucial component in the future transportation system. Ensuring the generalizability and safety of AI models maneuvering mobile machines is essential. In this work, we present MetaUrban, a compositional simulation platform for the AI-driven urban micromobility research. MetaUrban can construct an infinite number of interactive urban scenes from compositional elements, covering a vast array of ground plans, object placements, pedestrians, vulnerable road users, and other mobile agents' appearances and dynamics. We design point navigation and social navigation tasks as the pilot study using MetaUrban for urban micromobility research and establish various baselines of Reinforcement Learning and Imitation Learning. We conduct extensive evaluation across mobile machines, demonstrating that heterogeneous mechanical structures significantly influence the learning and execution of AI policies. We perform a thorough ablation study, showing that the compositional nature of the simulated environments can substantially improve the generalizability and safety of the trained mobile agents. MetaUrban will be made publicly available to provide research opportunities and foster safe and trustworthy embodied AI and micromobility in cities. The code and dataset will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08208v2">SPA: 3D Spatial-Awareness Enables Effective Embodied Representation</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 Project Page: https://haoyizhu.github.io/spa/
    </div>
    <details class="paper-abstract">
      In this paper, we introduce SPA, a novel representation learning framework that emphasizes the importance of 3D spatial awareness in embodied AI. Our approach leverages differentiable neural rendering on multi-view images to endow a vanilla Vision Transformer (ViT) with intrinsic spatial understanding. We present the most comprehensive evaluation of embodied representation learning to date, covering 268 tasks across 8 simulators with diverse policies in both single-task and language-conditioned multi-task scenarios. The results are compelling: SPA consistently outperforms more than 10 state-of-the-art representation methods, including those specifically designed for embodied AI, vision-centric tasks, and multi-modal applications, while using less training data. Furthermore, we conduct a series of real-world experiments to confirm its effectiveness in practical scenarios. These results highlight the critical role of 3D spatial awareness for embodied representation learning. Our strongest model takes more than 6000 GPU hours to train and we are committed to open-sourcing all code and model weights to foster future research in embodied representation learning. Project Page: https://haoyizhu.github.io/spa/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.08725v2">MetaUrban: An Embodied AI Simulation Platform for Urban Micromobility</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 Technical report. Project page: https://metadriverse.github.io/metaurban/
    </div>
    <details class="paper-abstract">
      Public urban spaces like streetscapes and plazas serve residents and accommodate social life in all its vibrant variations. Recent advances in Robotics and Embodied AI make public urban spaces no longer exclusive to humans. Food delivery bots and electric wheelchairs have started sharing sidewalks with pedestrians, while robot dogs and humanoids have recently emerged in the street. Micromobility enabled by AI for short-distance travel in public urban spaces plays a crucial component in the future transportation system. Ensuring the generalizability and safety of AI models maneuvering mobile machines is essential. In this work, we present MetaUrban, a compositional simulation platform for the AI-driven urban micromobility research. MetaUrban can construct an infinite number of interactive urban scenes from compositional elements, covering a vast array of ground plans, object placements, pedestrians, vulnerable road users, and other mobile agents' appearances and dynamics. We design point navigation and social navigation tasks as the pilot study using MetaUrban for urban micromobility research and establish various baselines of Reinforcement Learning and Imitation Learning. We conduct extensive evaluation across mobile machines, demonstrating that heterogeneous mechanical structures significantly influence the learning and execution of AI policies. We perform a thorough ablation study, showing that the compositional nature of the simulated environments can substantially improve the generalizability and safety of the trained mobile agents. MetaUrban will be made publicly available to provide research opportunities and foster safe and trustworthy embodied AI and micromobility in cities. The code and dataset will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.05756v1">Learning the Generalizable Manipulation Skills on Soft-body Tasks via Guided Self-attention Behavior Cloning Policy</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Embodied AI represents a paradigm in AI research where artificial agents are situated within and interact with physical or virtual environments. Despite the recent progress in Embodied AI, it is still very challenging to learn the generalizable manipulation skills that can handle large deformation and topological changes on soft-body objects, such as clay, water, and soil. In this work, we proposed an effective policy, namely GP2E behavior cloning policy, which can guide the agent to learn the generalizable manipulation skills from soft-body tasks, including pouring, filling, hanging, excavating, pinching, and writing. Concretely, we build our policy from three insights:(1) Extracting intricate semantic features from point cloud data and seamlessly integrating them into the robot's end-effector frame; (2) Capturing long-distance interactions in long-horizon tasks through the incorporation of our guided self-attention module; (3) Mitigating overfitting concerns and facilitating model convergence to higher accuracy levels via the introduction of our two-stage fine-tuning strategy. Through extensive experiments, we demonstrate the effectiveness of our approach by achieving the 1st prize in the soft-body track of the ManiSkill2 Challenge at the CVPR 2023 4th Embodied AI workshop. Our findings highlight the potential of our method to improve the generalization abilities of Embodied AI models and pave the way for their practical applications in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05791v1">FürElise: Capturing and Physically Synthesizing Hand Motions of Piano Performance</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 SIGGRAPH Asia 2024. Project page: https://for-elise.github.io/
    </div>
    <details class="paper-abstract">
      Piano playing requires agile, precise, and coordinated hand control that stretches the limits of dexterity. Hand motion models with the sophistication to accurately recreate piano playing have a wide range of applications in character animation, embodied AI, biomechanics, and VR/AR. In this paper, we construct a first-of-its-kind large-scale dataset that contains approximately 10 hours of 3D hand motion and audio from 15 elite-level pianists playing 153 pieces of classical music. To capture natural performances, we designed a markerless setup in which motions are reconstructed from multi-view videos using state-of-the-art pose estimation models. The motion data is further refined via inverse kinematics using the high-resolution MIDI key-pressing data obtained from sensors in a specialized Yamaha Disklavier piano. Leveraging the collected dataset, we developed a pipeline that can synthesize physically-plausible hand motions for musical scores outside of the dataset. Our approach employs a combination of imitation learning and reinforcement learning to obtain policies for physics-based bimanual control involving the interaction between hands and piano keys. To solve the sampling efficiency problem with the large motion dataset, we use a diffusion model to generate natural reference motions, which provide high-level trajectory and fingering (finger order and placement) information. However, the generated reference motion alone does not provide sufficient accuracy for piano performance modeling. We then further augmented the data by using musical similarity to retrieve similar motions from the captured dataset to boost the precision of the RL policy. With the proposed method, our model generates natural, dexterous motions that generalize to music from outside the training dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05756v1">Learning the Generalizable Manipulation Skills on Soft-body Tasks via Guided Self-attention Behavior Cloning Policy</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Embodied AI represents a paradigm in AI research where artificial agents are situated within and interact with physical or virtual environments. Despite the recent progress in Embodied AI, it is still very challenging to learn the generalizable manipulation skills that can handle large deformation and topological changes on soft-body objects, such as clay, water, and soil. In this work, we proposed an effective policy, namely GP2E behavior cloning policy, which can guide the agent to learn the generalizable manipulation skills from soft-body tasks, including pouring, filling, hanging, excavating, pinching, and writing. Concretely, we build our policy from three insights:(1) Extracting intricate semantic features from point cloud data and seamlessly integrating them into the robot's end-effector frame; (2) Capturing long-distance interactions in long-horizon tasks through the incorporation of our guided self-attention module; (3) Mitigating overfitting concerns and facilitating model convergence to higher accuracy levels via the introduction of our two-stage fine-tuning strategy. Through extensive experiments, we demonstrate the effectiveness of our approach by achieving the 1st prize in the soft-body track of the ManiSkill2 Challenge at the CVPR 2023 4th Embodied AI workshop. Our findings highlight the potential of our method to improve the generalization abilities of Embodied AI models and pave the way for their practical applications in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2407.05047v3">MFE-ETP: A Comprehensive Evaluation Benchmark for Multi-modal Foundation Models on Embodied Task Planning</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      In recent years, Multi-modal Foundation Models (MFMs) and Embodied Artificial Intelligence (EAI) have been advancing side by side at an unprecedented pace. The integration of the two has garnered significant attention from the AI research community. In this work, we attempt to provide an in-depth and comprehensive evaluation of the performance of MFM s on embodied task planning, aiming to shed light on their capabilities and limitations in this domain. To this end, based on the characteristics of embodied task planning, we first develop a systematic evaluation framework, which encapsulates four crucial capabilities of MFMs: object understanding, spatio-temporal perception, task understanding, and embodied reasoning. Following this, we propose a new benchmark, named MFE-ETP, characterized its complex and variable task scenarios, typical yet diverse task types, task instances of varying difficulties, and rich test case types ranging from multiple embodied question answering to embodied task reasoning. Finally, we offer a simple and easy-to-use automatic evaluation platform that enables the automated testing of multiple MFMs on the proposed benchmark. Using the benchmark and evaluation platform, we evaluated several state-of-the-art MFMs and found that they significantly lag behind human-level performance. The MFE-ETP is a high-quality, large-scale, and challenging benchmark relevant to real-world tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.05756v1">Learning the Generalizable Manipulation Skills on Soft-body Tasks via Guided Self-attention Behavior Cloning Policy</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Embodied AI represents a paradigm in AI research where artificial agents are situated within and interact with physical or virtual environments. Despite the recent progress in Embodied AI, it is still very challenging to learn the generalizable manipulation skills that can handle large deformation and topological changes on soft-body objects, such as clay, water, and soil. In this work, we proposed an effective policy, namely GP2E behavior cloning policy, which can guide the agent to learn the generalizable manipulation skills from soft-body tasks, including pouring, filling, hanging, excavating, pinching, and writing. Concretely, we build our policy from three insights:(1) Extracting intricate semantic features from point cloud data and seamlessly integrating them into the robot's end-effector frame; (2) Capturing long-distance interactions in long-horizon tasks through the incorporation of our guided self-attention module; (3) Mitigating overfitting concerns and facilitating model convergence to higher accuracy levels via the introduction of our two-stage fine-tuning strategy. Through extensive experiments, we demonstrate the effectiveness of our approach by achieving the 1st prize in the soft-body track of the ManiSkill2 Challenge at the CVPR 2023 4th Embodied AI workshop. Our findings highlight the potential of our method to improve the generalization abilities of Embodied AI models and pave the way for their practical applications in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.05047v3">MFE-ETP: A Comprehensive Evaluation Benchmark for Multi-modal Foundation Models on Embodied Task Planning</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      In recent years, Multi-modal Foundation Models (MFMs) and Embodied Artificial Intelligence (EAI) have been advancing side by side at an unprecedented pace. The integration of the two has garnered significant attention from the AI research community. In this work, we attempt to provide an in-depth and comprehensive evaluation of the performance of MFM s on embodied task planning, aiming to shed light on their capabilities and limitations in this domain. To this end, based on the characteristics of embodied task planning, we first develop a systematic evaluation framework, which encapsulates four crucial capabilities of MFMs: object understanding, spatio-temporal perception, task understanding, and embodied reasoning. Following this, we propose a new benchmark, named MFE-ETP, characterized its complex and variable task scenarios, typical yet diverse task types, task instances of varying difficulties, and rich test case types ranging from multiple embodied question answering to embodied task reasoning. Finally, we offer a simple and easy-to-use automatic evaluation platform that enables the automated testing of multiple MFMs on the proposed benchmark. Using the benchmark and evaluation platform, we evaluated several state-of-the-art MFMs and found that they significantly lag behind human-level performance. The MFE-ETP is a high-quality, large-scale, and challenging benchmark relevant to real-world tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02669v2">Causality-Aware Transformer Networks for Robotic Navigation</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Current research in Visual Navigation reveals opportunities for improvement. First, the direct adoption of RNNs and Transformers often overlooks the specific differences between Embodied AI and traditional sequential data modelling, potentially limiting its performance in Embodied AI tasks. Second, the reliance on task-specific configurations, such as pre-trained modules and dataset-specific logic, compromises the generalizability of these methods. We address these constraints by initially exploring the unique differences between Navigation tasks and other sequential data tasks through the lens of Causality, presenting a causal framework to elucidate the inadequacies of conventional sequential methods for Navigation. By leveraging this causal perspective, we propose Causality-Aware Transformer (CAT) Networks for Navigation, featuring a Causal Understanding Module to enhance the models's Environmental Understanding capability. Meanwhile, our method is devoid of task-specific inductive biases and can be trained in an End-to-End manner, which enhances the method's generalizability across various contexts. Empirical evaluations demonstrate that our methodology consistently surpasses benchmark performances across a spectrum of settings, tasks and simulation environments. Extensive ablation studies reveal that the performance gains can be attributed to the Causal Understanding Module, which demonstrates effectiveness and efficiency in both Reinforcement Learning and Supervised Learning settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20774v2">Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 31 pages, including main paper, references, and appendix
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03907v1">ActPlan-1K: Benchmarking the Procedural Planning Ability of Visual Language Models in Household Activities</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 13 pages, 9 figures, 8 tables, accepted to EMNLP 2024 main conference
    </div>
    <details class="paper-abstract">
      Large language models~(LLMs) have been adopted to process textual task description and accomplish procedural planning in embodied AI tasks because of their powerful reasoning ability. However, there is still lack of study on how vision language models~(VLMs) behave when multi-modal task inputs are considered. Counterfactual planning that evaluates the model's reasoning ability over alternative task situations are also under exploited. In order to evaluate the planning ability of both multi-modal and counterfactual aspects, we propose ActPlan-1K. ActPlan-1K is a multi-modal planning benchmark constructed based on ChatGPT and household activity simulator iGibson2. The benchmark consists of 153 activities and 1,187 instances. Each instance describing one activity has a natural language task description and multiple environment images from the simulator. The gold plan of each instance is action sequences over the objects in provided scenes. Both the correctness and commonsense satisfaction are evaluated on typical VLMs. It turns out that current VLMs are still struggling at generating human-level procedural plans for both normal activities and counterfactual activities. We further provide automatic evaluation metrics by finetuning over BLEURT model to facilitate future research on our benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03488v1">MO-DDN: A Coarse-to-Fine Attribute-based Exploration Agent for Multi-object Demand-driven Navigation</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted at NeurIPS 2024; 39 pages, 11 figures;
    </div>
    <details class="paper-abstract">
      The process of satisfying daily demands is a fundamental aspect of humans' daily lives. With the advancement of embodied AI, robots are increasingly capable of satisfying human demands. Demand-driven navigation (DDN) is a task in which an agent must locate an object to satisfy a specified demand instruction, such as ``I am thirsty.'' The previous study typically assumes that each demand instruction requires only one object to be fulfilled and does not consider individual preferences. However, the realistic human demand may involve multiple objects. In this paper, we introduce the Multi-object Demand-driven Navigation (MO-DDN) benchmark, which addresses these nuanced aspects, including multi-object search and personal preferences, thus making the MO-DDN task more reflective of real-life scenarios compared to DDN. Building upon previous work, we employ the concept of ``attribute'' to tackle this new task. However, instead of solely relying on attribute features in an end-to-end manner like DDN, we propose a modular method that involves constructing a coarse-to-fine attribute-based exploration agent (C2FAgent). Our experimental results illustrate that this coarse-to-fine exploration strategy capitalizes on the advantages of attributes at various decision-making levels, resulting in superior performance compared to baseline methods. Code and video can be found at https://sites.google.com/view/moddn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02751v1">ReLIC: A Recipe for 64k Steps of In-Context Reinforcement Learning for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Intelligent embodied agents need to quickly adapt to new scenarios by integrating long histories of experience into decision-making. For instance, a robot in an unfamiliar house initially wouldn't know the locations of objects needed for tasks and might perform inefficiently. However, as it gathers more experience, it should learn the layout of its environment and remember where objects are, allowing it to complete new tasks more efficiently. To enable such rapid adaptation to new tasks, we present ReLIC, a new approach for in-context reinforcement learning (RL) for embodied agents. With ReLIC, agents are capable of adapting to new environments using 64,000 steps of in-context experience with full attention while being trained through self-generated experience via RL. We achieve this by proposing a novel policy update scheme for on-policy RL called "partial updates'' as well as a Sink-KV mechanism that enables effective utilization of a long observation history for embodied agents. Our method outperforms a variety of meta-RL baselines in adapting to unseen houses in an embodied multi-object navigation task. In addition, we find that ReLIC is capable of few-shot imitation learning despite never being trained with expert demonstrations. We also provide a comprehensive analysis of ReLIC, highlighting that the combination of large-scale RL training, the proposed partial updates scheme, and the Sink-KV are essential for effective in-context learning. The code for ReLIC and all our experiments is at https://github.com/aielawady/relic
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20242v3">BadRobot: Manipulating Embodied LLMs in the Physical World</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 38 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Embodied AI represents systems where AI is integrated into physical entities, enabling them to perceive and interact with their surroundings. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot. Warning: This paper contains harmful AI-generated language and aggressive actions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.01176v1">Generative Diffusion-based Contract Design for Efficient AI Twins Migration in Vehicular Embodied AI Networks</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Embodied AI is a rapidly advancing field that bridges the gap between cyberspace and physical space, enabling a wide range of applications. This evolution has led to the development of the Vehicular Embodied AI NETwork (VEANET), where advanced AI capabilities are integrated into vehicular systems to enhance autonomous operations and decision-making. Embodied agents, such as Autonomous Vehicles (AVs), are autonomous entities that can perceive their environment and take actions to achieve specific goals, actively interacting with the physical world. Embodied twins are digital models of these embodied agents, with various embodied AI twins for intelligent applications in cyberspace. In VEANET, embodied AI twins act as in-vehicle AI assistants to perform diverse tasks supporting autonomous driving using generative AI models. Due to limited computational resources of AVs, these AVs often offload computationally intensive tasks, such as constructing and updating embodied AI twins, to nearby RSUs. However, since the rapid mobility of AVs and the limited provision coverage of a single RSU, embodied AI twins require dynamic migrations from current RSU to other RSUs in real-time, resulting in the challenge of selecting suitable RSUs for efficient embodied AI twins migrations. Given information asymmetry, AVs cannot know the detailed information of RSUs. To this end, in this paper, we construct a multi-dimensional contract theoretical model between AVs and alternative RSUs. Considering that AVs may exhibit irrational behavior, we utilize prospect theory instead of expected utility theory to model the actual utilities of AVs. Finally, we employ a generative diffusion model-based algorithm to identify the optimal contract designs. Compared with traditional deep reinforcement learning algorithms, numerical results demonstrate the effectiveness of the proposed scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01481v1">SonicSim: A customizable simulation platform for speech processing in moving sound source scenarios</a></div>
    <div class="paper-meta">
      📅 2024-10-02
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      The systematic evaluation of speech separation and enhancement models under moving sound source conditions typically requires extensive data comprising diverse scenarios. However, real-world datasets often contain insufficient data to meet the training and evaluation requirements of models. Although synthetic datasets offer a larger volume of data, their acoustic simulations lack realism. Consequently, neither real-world nor synthetic datasets effectively fulfill practical needs. To address these issues, we introduce SonicSim, a synthetic toolkit de-designed to generate highly customizable data for moving sound sources. SonicSim is developed based on the embodied AI simulation platform, Habitat-sim, supporting multi-level adjustments, including scene-level, microphone-level, and source-level, thereby generating more diverse synthetic data. Leveraging SonicSim, we constructed a moving sound source benchmark dataset, SonicSet, using the Librispeech, the Freesound Dataset 50k (FSD50K) and Free Music Archive (FMA), and 90 scenes from the Matterport3D to evaluate speech separation and enhancement models. Additionally, to validate the differences between synthetic data and real-world data, we randomly selected 5 hours of raw data without reverberation from the SonicSet validation set to record a real-world speech separation dataset, which was then compared with the corresponding synthetic datasets. Similarly, we utilized the real-world speech enhancement dataset RealMAN to validate the acoustic gap between other synthetic datasets and the SonicSet dataset for speech enhancement. The results indicate that the synthetic data generated by SonicSim can effectively generalize to real-world scenarios. Demo and code are publicly available at https://cslikai.cn/SonicSim/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01176v1">Generative Diffusion-based Contract Design for Efficient AI Twins Migration in Vehicular Embodied AI Networks</a></div>
    <div class="paper-meta">
      📅 2024-10-02
    </div>
    <details class="paper-abstract">
      Embodied AI is a rapidly advancing field that bridges the gap between cyberspace and physical space, enabling a wide range of applications. This evolution has led to the development of the Vehicular Embodied AI NETwork (VEANET), where advanced AI capabilities are integrated into vehicular systems to enhance autonomous operations and decision-making. Embodied agents, such as Autonomous Vehicles (AVs), are autonomous entities that can perceive their environment and take actions to achieve specific goals, actively interacting with the physical world. Embodied twins are digital models of these embodied agents, with various embodied AI twins for intelligent applications in cyberspace. In VEANET, embodied AI twins act as in-vehicle AI assistants to perform diverse tasks supporting autonomous driving using generative AI models. Due to limited computational resources of AVs, these AVs often offload computationally intensive tasks, such as constructing and updating embodied AI twins, to nearby RSUs. However, since the rapid mobility of AVs and the limited provision coverage of a single RSU, embodied AI twins require dynamic migrations from current RSU to other RSUs in real-time, resulting in the challenge of selecting suitable RSUs for efficient embodied AI twins migrations. Given information asymmetry, AVs cannot know the detailed information of RSUs. To this end, in this paper, we construct a multi-dimensional contract theoretical model between AVs and alternative RSUs. Considering that AVs may exhibit irrational behavior, we utilize prospect theory instead of expected utility theory to model the actual utilities of AVs. Finally, we employ a generative diffusion model-based algorithm to identify the optimal contract designs. Compared with traditional deep reinforcement learning algorithms, numerical results demonstrate the effectiveness of the proposed scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.01176v1">Generative Diffusion-based Contract Design for Efficient AI Twins Migration in Vehicular Embodied AI Networks</a></div>
    <div class="paper-meta">
      📅 2024-10-02
    </div>
    <details class="paper-abstract">
      Embodied AI is a rapidly advancing field that bridges the gap between cyberspace and physical space, enabling a wide range of applications. This evolution has led to the development of the Vehicular Embodied AI NETwork (VEANET), where advanced AI capabilities are integrated into vehicular systems to enhance autonomous operations and decision-making. Embodied agents, such as Autonomous Vehicles (AVs), are autonomous entities that can perceive their environment and take actions to achieve specific goals, actively interacting with the physical world. Embodied twins are digital models of these embodied agents, with various embodied AI twins for intelligent applications in cyberspace. In VEANET, embodied AI twins act as in-vehicle AI assistants to perform diverse tasks supporting autonomous driving using generative AI models. Due to limited computational resources of AVs, these AVs often offload computationally intensive tasks, such as constructing and updating embodied AI twins, to nearby RSUs. However, since the rapid mobility of AVs and the limited provision coverage of a single RSU, embodied AI twins require dynamic migrations from current RSU to other RSUs in real-time, resulting in the challenge of selecting suitable RSUs for efficient embodied AI twins migrations. Given information asymmetry, AVs cannot know the detailed information of RSUs. To this end, in this paper, we construct a multi-dimensional contract theoretical model between AVs and alternative RSUs. Considering that AVs may exhibit irrational behavior, we utilize prospect theory instead of expected utility theory to model the actual utilities of AVs. Finally, we employ a generative diffusion model-based algorithm to identify the optimal contract designs. Compared with traditional deep reinforcement learning algorithms, numerical results demonstrate the effectiveness of the proposed scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00425v1">ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-10-01
      | 💬 Project website: http://maniskill.ai/
    </div>
    <details class="paper-abstract">
      Simulation has enabled unprecedented compute-scalable approaches to robot learning. However, many existing simulation frameworks typically support a narrow range of scenes/tasks and lack features critical for scaling generalizable robotics and sim2real. We introduce and open source ManiSkill3, the fastest state-visual GPU parallelized robotics simulator with contact-rich physics targeting generalizable manipulation. ManiSkill3 supports GPU parallelization of many aspects including simulation+rendering, heterogeneous simulation, pointclouds/voxels visual input, and more. Simulation with rendering on ManiSkill3 can run 10-1000x faster with 2-3x less GPU memory usage than other platforms, achieving up to 30,000+ FPS in benchmarked environments due to minimal python/pytorch overhead in the system, simulation on the GPU, and the use of the SAPIEN parallel rendering system. Tasks that used to take hours to train can now take minutes. We further provide the most comprehensive range of GPU parallelized environments/tasks spanning 12 distinct domains including but not limited to mobile manipulation for tasks such as drawing, humanoids, and dextrous manipulation in realistic scenes designed by artists or real-world digital twins. In addition, millions of demonstration frames are provided from motion planning, RL, and teleoperation. ManiSkill3 also provides a comprehensive set of baselines that span popular RL and learning-from-demonstrations algorithms.
    </details>
</div>
