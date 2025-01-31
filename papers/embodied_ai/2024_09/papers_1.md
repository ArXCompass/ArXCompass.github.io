# embodied ai - 2024_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19459v1">Language-guided Robust Navigation for Mobile Robots in Dynamically-changing Environments</a></div>
    <div class="paper-meta">
      📅 2024-09-28
    </div>
    <details class="paper-abstract">
      In this paper, we develop an embodied AI system for human-in-the-loop navigation with a wheeled mobile robot. We propose a direct yet effective method of monitoring the robot's current plan to detect changes in the environment that impact the intended trajectory of the robot significantly and then query a human for feedback. We also develop a means to parse human feedback expressed in natural language into local navigation waypoints and integrate it into a global planning system, by leveraging a map of semantic features and an aligned obstacle map. Extensive testing in simulation and physical hardware experiments with a resource-constrained wheeled robot tasked to navigate in a real-world environment validate the efficacy and robustness of our method. This work can support applications like precision agriculture and construction, where persistent monitoring of the environment provides a human with information about the environment state.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18800v1">MiniVLN: Efficient Vision-and-Language Navigation by Progressive Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      In recent years, Embodied Artificial Intelligence (Embodied AI) has advanced rapidly, yet the increasing size of models conflicts with the limited computational capabilities of Embodied AI platforms. To address this challenge, we aim to achieve both high model performance and practical deployability. Specifically, we focus on Vision-and-Language Navigation (VLN), a core task in Embodied AI. This paper introduces a two-stage knowledge distillation framework, producing a student model, MiniVLN, and showcasing the significant potential of distillation techniques in developing lightweight models. The proposed method aims to capture fine-grained knowledge during the pretraining phase and navigation-specific knowledge during the fine-tuning phase. Our findings indicate that the two-stage distillation approach is more effective in narrowing the performance gap between the teacher model and the student model compared to single-stage distillation. On the public R2R and REVERIE benchmarks, MiniVLN achieves performance on par with the teacher model while having only about 12% of the teacher model's parameter count.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09054v1">Circuits and Systems for Embodied AI: Exploring uJ Multi-Modal Perception for Nano-UAVs on the Kraken Shield</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 5 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Embodied artificial intelligence (AI) requires pushing complex multi-modal models to the extreme edge for time-constrained tasks such as autonomous navigation of robots and vehicles. On small form-factor devices, e.g., nano-sized unmanned aerial vehicles (UAVs), such challenges are exacerbated by stringent constraints on energy efficiency and weight. In this paper, we explore embodied multi-modal AI-based perception for Nano-UAVs with the Kraken shield, a 7g multi-sensor (frame-based and event-based imagers) board based on Kraken, a 22 nm SoC featuring multiple acceleration engines for multi-modal event and frame-based inference based on spiking (SNN) and ternary (TNN) neural networks, respectively. Kraken can execute SNN real-time inference for depth estimation at 1.02k inf/s, 18 {\mu}J/inf, TNN real-time inference for object classification at 10k inf/s, 6 {\mu}J/inf, and real-time inference for obstacle avoidance at 221 frame/s, 750 {\mu}J/inf.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16019v1">AIR-Embodied: An Efficient Active 3DGS-based Interaction and Reconstruction Framework with Embodied Large Language Model</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D reconstruction and neural rendering have enhanced the creation of high-quality digital assets, yet existing methods struggle to generalize across varying object shapes, textures, and occlusions. While Next Best View (NBV) planning and Learning-based approaches offer solutions, they are often limited by predefined criteria and fail to manage occlusions with human-like common sense. To address these problems, we present AIR-Embodied, a novel framework that integrates embodied AI agents with large-scale pretrained multi-modal language models to improve active 3DGS reconstruction. AIR-Embodied utilizes a three-stage process: understanding the current reconstruction state via multi-modal prompts, planning tasks with viewpoint selection and interactive actions, and employing closed-loop reasoning to ensure accurate execution. The agent dynamically refines its actions based on discrepancies between the planned and actual outcomes. Experimental evaluations across virtual and real-world environments demonstrate that AIR-Embodied significantly enhances reconstruction efficiency and quality, providing a robust solution to challenges in active 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15658v1">ReLEP: A Novel Framework for Real-world Long-horizon Embodied Planning</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      Real-world long-horizon embodied planning underpins embodied AI. To accomplish long-horizon tasks, agents need to decompose abstract instructions into detailed steps. Prior works mostly rely on GPT-4V for task decomposition into predefined actions, which limits task diversity due to GPT-4V's finite understanding of larger skillsets. Therefore, we present ReLEP, a groundbreaking framework for Real world Long-horizon Embodied Planning, which can accomplish a wide range of daily tasks. At its core lies a fine-tuned large vision language model that formulates plans as sequences of skill functions according to input instruction and scene image. These functions are selected from a carefully designed skill library. ReLEP is also equipped with a Memory module for plan and status recall, and a Robot Configuration module for versatility across robot types. In addition, we propose a semi-automatic data generation pipeline to tackle dataset scarcity. Real-world off-line experiments across eight daily embodied tasks demonstrate that ReLEP is able to accomplish long-horizon embodied tasks and outperforms other state-of-the-art baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14908v1">KARMA: Augmenting Embodied AI Agents with Long-and-short Term Memory Systems</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Embodied AI agents responsible for executing interconnected, long-sequence household tasks often face difficulties with in-context memory, leading to inefficiencies and errors in task execution. To address this issue, we introduce KARMA, an innovative memory system that integrates long-term and short-term memory modules, enhancing large language models (LLMs) for planning in embodied agents through memory-augmented prompting. KARMA distinguishes between long-term and short-term memory, with long-term memory capturing comprehensive 3D scene graphs as representations of the environment, while short-term memory dynamically records changes in objects' positions and states. This dual-memory structure allows agents to retrieve relevant past scene experiences, thereby improving the accuracy and efficiency of task planning. Short-term memory employs strategies for effective and adaptive memory replacement, ensuring the retention of critical information while discarding less pertinent data. Compared to state-of-the-art embodied agents enhanced with memory, our memory-augmented embodied AI agent improves success rates by 1.3x and 2.3x in Composite Tasks and Complex Tasks within the AI2-THOR simulator, respectively, and enhances task execution efficiency by 3.4x and 62.7x. Furthermore, we demonstrate that KARMA's plug-and-play capability allows for seamless deployment on real-world robotic systems, such as mobile manipulation platforms.Through this plug-and-play memory system, KARMA significantly enhances the ability of embodied agents to generate coherent and contextually appropriate plans, making the execution of complex household tasks more efficient. The experimental videos from the work can be found at https://youtu.be/4BT7fnw9ehs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14692v1">Dynamic Realms: 4D Content Analysis, Recovery and Generation with Geometric, Topological and Physical Priors</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 Research Summary - DC
    </div>
    <details class="paper-abstract">
      My research focuses on the analysis, recovery, and generation of 4D content, where 4D includes three spatial dimensions (x, y, z) and a temporal dimension t, such as shape and motion. This focus goes beyond static objects to include dynamic changes over time, providing a comprehensive understanding of both spatial and temporal variations. These techniques are critical in applications like AR/VR, embodied AI, and robotics. My research aims to make 4D content generation more efficient, accessible, and higher in quality by incorporating geometric, topological, and physical priors. I also aim to develop effective methods for 4D content recovery and analysis using these priors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02522v2">Cog-GA: A Large Language Models-based Generative Agent for Vision-Language Navigation in Continuous Environments</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Vision Language Navigation in Continuous Environments (VLN-CE) represents a frontier in embodied AI, demanding agents to navigate freely in unbounded 3D spaces solely guided by natural language instructions. This task introduces distinct challenges in multimodal comprehension, spatial reasoning, and decision-making. To address these challenges, we introduce Cog-GA, a generative agent founded on large language models (LLMs) tailored for VLN-CE tasks. Cog-GA employs a dual-pronged strategy to emulate human-like cognitive processes. Firstly, it constructs a cognitive map, integrating temporal, spatial, and semantic elements, thereby facilitating the development of spatial memory within LLMs. Secondly, Cog-GA employs a predictive mechanism for waypoints, strategically optimizing the exploration trajectory to maximize navigational efficiency. Each waypoint is accompanied by a dual-channel scene description, categorizing environmental cues into 'what' and 'where' streams as the brain. This segregation enhances the agent's attentional focus, enabling it to discern pertinent spatial information for navigation. A reflective mechanism complements these strategies by capturing feedback from prior navigation experiences, facilitating continual learning and adaptive replanning. Extensive evaluations conducted on VLN-CE benchmarks validate Cog-GA's state-of-the-art performance and ability to simulate human-like navigation behaviors. This research significantly contributes to the development of strategic and interpretable VLN-CE agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13642v6">SpatialBot: Precise Spatial Understanding with Vision Language Models</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      Vision Language Models (VLMs) have achieved impressive performance in 2D image understanding, however they are still struggling with spatial understanding which is the foundation of Embodied AI. In this paper, we propose SpatialBot for better spatial understanding by feeding both RGB and depth images. Additionally, we have constructed the SpatialQA dataset, which involves multi-level depth-related questions to train VLMs for depth understanding. Finally, we present SpatialBench to comprehensively evaluate VLMs' capabilities in spatial understanding at different levels. Extensive experiments on our spatial-understanding benchmark, general VLM benchmarks and Embodied AI tasks, demonstrate the remarkable improvements of SpatialBot trained on SpatialQA. The model, code and data are available at https://github.com/BAAI-DCAI/SpatialBot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11279v1">P-RAG: Progressive Retrieval Augmented Generation For Planning on Embodied Everyday Task</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      Embodied Everyday Task is a popular task in the embodied AI community, requiring agents to make a sequence of actions based on natural language instructions and visual observations. Traditional learning-based approaches face two challenges. Firstly, natural language instructions often lack explicit task planning. Secondly, extensive training is required to equip models with knowledge of the task environment. Previous works based on Large Language Model (LLM) either suffer from poor performance due to the lack of task-specific knowledge or rely on ground truth as few-shot samples. To address the above limitations, we propose a novel approach called Progressive Retrieval Augmented Generation (P-RAG), which not only effectively leverages the powerful language processing capabilities of LLMs but also progressively accumulates task-specific knowledge without ground-truth. Compared to the conventional RAG methods, which retrieve relevant information from the database in a one-shot manner to assist generation, P-RAG introduces an iterative approach to progressively update the database. In each iteration, P-RAG retrieves the latest database and obtains historical information from the previous interaction as experiential references for the current interaction. Moreover, we also introduce a more granular retrieval scheme that not only retrieves similar tasks but also incorporates retrieval of similar situations to provide more valuable reference experiences. Extensive experiments reveal that P-RAG achieves competitive results without utilizing ground truth and can even further improve performance through self-iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11347v2">Multimodal Datasets and Benchmarks for Reasoning about Dynamic Spatio-Temporality in Everyday Environments</a></div>
    <div class="paper-meta">
      📅 2024-09-17
      | 💬 5 pages, 1 figure, 1 table, accepted in Embodied AI 2024 Workshop held in conjunction with CVPR 2024
    </div>
    <details class="paper-abstract">
      We used a 3D simulator to create artificial video data with standardized annotations, aiming to aid in the development of Embodied AI. Our question answering (QA) dataset measures the extent to which a robot can understand human behavior and the environment in a home setting. Preliminary experiments suggest our dataset is useful in measuring AI's comprehension of daily life. \end{abstract}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14367v2">PoseBench: Benchmarking the Robustness of Pose Estimation Models under Corruptions</a></div>
    <div class="paper-meta">
      📅 2024-09-14
      | 💬 Technical report. Project page: https://xymsh.github.io/PoseBench/
    </div>
    <details class="paper-abstract">
      Pose estimation aims to accurately identify anatomical keypoints in humans and animals using monocular images, which is crucial for various applications such as human-machine interaction, embodied AI, and autonomous driving. While current models show promising results, they are typically trained and tested on clean data, potentially overlooking the corruption during real-world deployment and thus posing safety risks in practical scenarios. To address this issue, we introduce PoseBench, a comprehensive benchmark designed to evaluate the robustness of pose estimation models against real-world corruption. We evaluated 60 representative models, including top-down, bottom-up, heatmap-based, regression-based, and classification-based methods, across three datasets for human and animal pose estimation. Our evaluation involves 10 types of corruption in four categories: 1) blur and noise, 2) compression and color loss, 3) severe lighting, and 4) masks. Our findings reveal that state-of-the-art models are vulnerable to common real-world corruptions and exhibit distinct behaviors when tackling human and animal pose estimation tasks. To improve model robustness, we delve into various design considerations, including input resolution, pre-training datasets, backbone capacity, post-processing, and data augmentations. We hope that our benchmark will serve as a foundation for advancing research in robust pose estimation. The benchmark and source code will be released at https://xymsh.github.io/PoseBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03824v4">A call for embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-09-13
      | 💬 Published in ICML 2024 Position paper track
    </div>
    <details class="paper-abstract">
      We propose Embodied AI as the next fundamental step in the pursuit of Artificial General Intelligence, juxtaposing it against current AI advancements, particularly Large Language Models. We traverse the evolution of the embodiment concept across diverse fields - philosophy, psychology, neuroscience, and robotics - to highlight how EAI distinguishes itself from the classical paradigm of static learning. By broadening the scope of Embodied AI, we introduce a theoretical framework based on cognitive architectures, emphasizing perception, action, memory, and learning as essential components of an embodied agent. This framework is aligned with Friston's active inference principle, offering a comprehensive approach to EAI development. Despite the progress made in the field of AI, substantial challenges, such as the formulation of a novel AI learning theory and the innovation of advanced hardware, persist. Our discussion lays down a foundational guideline for future Embodied AI research. Highlighting the importance of creating Embodied AI agents capable of seamless communication, collaboration, and coexistence with humans and other intelligent entities within real-world environments, we aim to steer the AI community towards addressing the multifaceted challenges and seizing the opportunities that lie ahead in the quest for AGI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05486v2">Artificial social influence via human-embodied AI agent interaction in immersive virtual reality (VR): Effects of similarity-matching during health conversations</a></div>
    <div class="paper-meta">
      📅 2024-09-12
      | 💬 11 pages, 4 figures, manuscript submitted to a journal
    </div>
    <details class="paper-abstract">
      Interactions with artificial intelligence (AI) based agents can positively influence human behavior and judgment. However, studies to date focus on text-based conversational agents (CA) with limited embodiment, restricting our understanding of how social influence principles, such as similarity, apply to AI agents (i.e., artificial social influence). We address this gap by leveraging the latest advances in AI (language models) and combining them with immersive virtual reality (VR). Specifically, we built VR-ECAs, or embodied conversational agents that can naturally converse with humans about health-related topics in a virtual environment. Then we manipulated interpersonal similarity via gender matching and examined its effects on biobehavioral (i.e., gaze), social (e.g., agent likeability), and behavioral outcomes (i.e., healthy snack selection). We found an interesting interaction effect between agent and participant gender on biobehavioral outcomes: discussing health with opposite-gender agents tended to enhance gaze duration, with the effect stronger for male participants compared to their female counterparts. A similar directional pattern was observed for healthy snack selection, though it was not statistically significant. In addition, female participants liked the VR-ECAs more than their male counterparts, regardless of the gender of the VR-ECAs. Finally, participants experienced greater presence while conversing with VR-embodied agents than chatting with text-only agents. Overall, our findings highlight embodiment as a crucial factor of influence of AI on human behavior, and our paradigm enables new experimental research at the intersection of social influence, human-AI communication, and immersive virtual reality (VR).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05583v1">Spatially-Aware Speaker for Vision-and-Language Navigation Instruction Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      Embodied AI aims to develop robots that can \textit{understand} and execute human language instructions, as well as communicate in natural languages. On this front, we study the task of generating highly detailed navigational instructions for the embodied robots to follow. Although recent studies have demonstrated significant leaps in the generation of step-by-step instructions from sequences of images, the generated instructions lack variety in terms of their referral to objects and landmarks. Existing speaker models learn strategies to evade the evaluation metrics and obtain higher scores even for low-quality sentences. In this work, we propose SAS (Spatially-Aware Speaker), an instruction generator or \textit{Speaker} model that utilises both structural and semantic knowledge of the environment to produce richer instructions. For training, we employ a reward learning method in an adversarial setting to avoid systematic bias introduced by language evaluation metrics. Empirically, our method outperforms existing instruction generation models, evaluated using standard metrics. Our code is available at \url{https://github.com/gmuraleekrishna/SAS}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01630v1">SafeEmbodAI: a Safety Framework for Mobile Robots in Embodied AI Systems</a></div>
    <div class="paper-meta">
      📅 2024-09-03
    </div>
    <details class="paper-abstract">
      Embodied AI systems, including AI-powered robots that autonomously interact with the physical world, stand to be significantly advanced by Large Language Models (LLMs), which enable robots to better understand complex language commands and perform advanced tasks with enhanced comprehension and adaptability, highlighting their potential to improve embodied AI capabilities. However, this advancement also introduces safety challenges, particularly in robotic navigation tasks. Improper safety management can lead to failures in complex environments and make the system vulnerable to malicious command injections, resulting in unsafe behaviours such as detours or collisions. To address these issues, we propose \textit{SafeEmbodAI}, a safety framework for integrating mobile robots into embodied AI systems. \textit{SafeEmbodAI} incorporates secure prompting, state management, and safety validation mechanisms to secure and assist LLMs in reasoning through multi-modal data and validating responses. We designed a metric to evaluate mission-oriented exploration, and evaluations in simulated environments demonstrate that our framework effectively mitigates threats from malicious commands and improves performance in various environment settings, ensuring the safety of embodied AI systems. Notably, In complex environments with mixed obstacles, our method demonstrates a significant performance increase of 267\% compared to the baseline in attack scenarios, highlighting its robustness in challenging conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01559v1">PR2: A Physics- and Photo-realistic Testbed for Embodied AI and Humanoid Robots</a></div>
    <div class="paper-meta">
      📅 2024-09-03
    </div>
    <details class="paper-abstract">
      This paper presents the development of a Physics-realistic and Photo-\underline{r}ealistic humanoid robot testbed, PR2, to facilitate collaborative research between Embodied Artificial Intelligence (Embodied AI) and robotics. PR2 offers high-quality scene rendering and robot dynamic simulation, enabling (i) the creation of diverse scenes using various digital assets, (ii) the integration of advanced perception or foundation models, and (iii) the implementation of planning and control algorithms for dynamic humanoid robot behaviors based on environmental feedback. The beta version of PR2 has been deployed for the simulation track of a nationwide full-size humanoid robot competition for college students, attracting 137 teams and over 400 participants within four months. This competition covered traditional tasks in bipedal walking, as well as novel challenges in loco-manipulation and language-instruction-based object search, marking a first for public college robotics competitions. A retrospective analysis of the competition suggests that future events should emphasize the integration of locomotion with manipulation and perception. By making the PR2 testbed publicly available at https://github.com/pr2-humanoid/PR2-Platform, we aim to further advance education and training in humanoid robotics.
    </details>
</div>
