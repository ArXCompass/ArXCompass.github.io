# embodied ai - 2024_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08963v2">LEMON: Learning 3D Human-Object Interaction Relation from 2D Images</a></div>
    <div class="paper-meta">
      📅 2024-03-31
      | 💬 accept by CVPR2024
    </div>
    <details class="paper-abstract">
      Learning 3D human-object interaction relation is pivotal to embodied AI and interaction modeling. Most existing methods approach the goal by learning to predict isolated interaction elements, e.g., human contact, object affordance, and human-object spatial relation, primarily from the perspective of either the human or the object. Which underexploit certain correlations between the interaction counterparts (human and object), and struggle to address the uncertainty in interactions. Actually, objects' functionalities potentially affect humans' interaction intentions, which reveals what the interaction is. Meanwhile, the interacting humans and objects exhibit matching geometric structures, which presents how to interact. In light of this, we propose harnessing these inherent correlations between interaction counterparts to mitigate the uncertainty and jointly anticipate the above interaction elements in 3D space. To achieve this, we present LEMON (LEarning 3D huMan-Object iNteraction relation), a unified model that mines interaction intentions of the counterparts and employs curvatures to guide the extraction of geometric correlations, combining them to anticipate the interaction elements. Besides, the 3D Interaction Relation dataset (3DIR) is collected to serve as the test bed for training and evaluation. Extensive experiments demonstrate the superiority of LEMON over methods estimating each element in isolation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.04670v2">Rapid Motor Adaptation for Robotic Manipulator Arms</a></div>
    <div class="paper-meta">
      📅 2024-03-29
      | 💬 Accepted at CVPR 2024. 12 pages
    </div>
    <details class="paper-abstract">
      Developing generalizable manipulation skills is a core challenge in embodied AI. This includes generalization across diverse task configurations, encompassing variations in object shape, density, friction coefficient, and external disturbances such as forces applied to the robot. Rapid Motor Adaptation (RMA) offers a promising solution to this challenge. It posits that essential hidden variables influencing an agent's task performance, such as object mass and shape, can be effectively inferred from the agent's action and proprioceptive history. Drawing inspiration from RMA in locomotion and in-hand rotation, we use depth perception to develop agents tailored for rapid motor adaptation in a variety of manipulation tasks. We evaluated our agents on four challenging tasks from the Maniskill2 benchmark, namely pick-and-place operations with hundreds of objects from the YCB and EGAD datasets, peg insertion with precise position and orientation, and operating a variety of faucets and handles, with customized environment variations. Empirical results demonstrate that our agents surpass state-of-the-art methods like automatic domain randomization and vision-based policies, obtaining better generalization performance and sample efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.12188v2">SG-Bot: Object Rearrangement via Coarse-to-Fine Robotic Imagination on Scene Graphs</a></div>
    <div class="paper-meta">
      📅 2024-03-24
      | 💬 ICRA 2024 accepted. Project website: https://sites.google.com/view/sg-bot
    </div>
    <details class="paper-abstract">
      Object rearrangement is pivotal in robotic-environment interactions, representing a significant capability in embodied AI. In this paper, we present SG-Bot, a novel rearrangement framework that utilizes a coarse-to-fine scheme with a scene graph as the scene representation. Unlike previous methods that rely on either known goal priors or zero-shot large models, SG-Bot exemplifies lightweight, real-time, and user-controllable characteristics, seamlessly blending the consideration of commonsense knowledge with automatic generation capabilities. SG-Bot employs a three-fold procedure--observation, imagination, and execution--to adeptly address the task. Initially, objects are discerned and extracted from a cluttered scene during the observation. These objects are first coarsely organized and depicted within a scene graph, guided by either commonsense or user-defined criteria. Then, this scene graph subsequently informs a generative model, which forms a fine-grained goal scene considering the shape information from the initial scene and object semantics. Finally, for execution, the initial and envisioned goal scenes are matched to formulate robotic action policies. Experimental results demonstrate that SG-Bot outperforms competitors by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15223v1">TriHelper: Zero-Shot Object Navigation with Dynamic Assistance</a></div>
    <div class="paper-meta">
      📅 2024-03-22
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Navigating toward specific objects in unknown environments without additional training, known as Zero-Shot object navigation, poses a significant challenge in the field of robotics, which demands high levels of auxiliary information and strategic planning. Traditional works have focused on holistic solutions, overlooking the specific challenges agents encounter during navigation such as collision, low exploration efficiency, and misidentification of targets. To address these challenges, our work proposes TriHelper, a novel framework designed to assist agents dynamically through three primary navigation challenges: collision, exploration, and detection. Specifically, our framework consists of three innovative components: (i) Collision Helper, (ii) Exploration Helper, and (iii) Detection Helper. These components work collaboratively to solve these challenges throughout the navigation process. Experiments on the Habitat-Matterport 3D (HM3D) and Gibson datasets demonstrate that TriHelper significantly outperforms all existing baseline methods in Zero-Shot object navigation, showcasing superior success rates and exploration efficiency. Our ablation studies further underscore the effectiveness of each helper in addressing their respective challenges, notably enhancing the agent's navigation capabilities. By proposing TriHelper, we offer a fresh perspective on advancing the object navigation task, paving the way for future research in the domain of Embodied AI and visual-based navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08282v2">Hierarchical Auto-Organizing System for Open-Ended Multi-Agent Navigation</a></div>
    <div class="paper-meta">
      📅 2024-03-18
      | 💬 ICLR 2024 Workshop on LLM Agents
    </div>
    <details class="paper-abstract">
      Due to the dynamic and unpredictable open-world setting, navigating complex environments in Minecraft poses significant challenges for multi-agent systems. Agents must interact with the environment and coordinate their actions with other agents to achieve common objectives. However, traditional approaches often struggle to efficiently manage inter-agent communication and task distribution, crucial for effective multi-agent navigation. Furthermore, processing and integrating multi-modal information (such as visual, textual, and auditory data) is essential for agents to comprehend their goals and navigate the environment successfully and fully. To address this issue, we design the HAS framework to auto-organize groups of LLM-based agents to complete navigation tasks. In our approach, we devise a hierarchical auto-organizing navigation system, which is characterized by 1) a hierarchical system for multi-agent organization, ensuring centralized planning and decentralized execution; 2) an auto-organizing and intra-communication mechanism, enabling dynamic group adjustment under subtasks; 3) a multi-modal information platform, facilitating multi-modal perception to perform the three navigation tasks with one system. To assess organizational behavior, we design a series of navigation tasks in the Minecraft environment, which includes searching and exploring. We aim to develop embodied organizations that push the boundaries of embodied AI, moving it towards a more human-like organizational structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09194v2">Intention-driven Ego-to-Exo Video Generation</a></div>
    <div class="paper-meta">
      📅 2024-03-17
    </div>
    <details class="paper-abstract">
      Ego-to-exo video generation refers to generating the corresponding exocentric video according to the egocentric video, providing valuable applications in AR/VR and embodied AI. Benefiting from advancements in diffusion model techniques, notable progress has been achieved in video generation. However, existing methods build upon the spatiotemporal consistency assumptions between adjacent frames, which cannot be satisfied in the ego-to-exo scenarios due to drastic changes in views. To this end, this paper proposes an Intention-Driven Ego-to-exo video generation framework (IDE) that leverages action intention consisting of human movement and action description as view-independent representation to guide video generation, preserving the consistency of content and motion. Specifically, the egocentric head trajectory is first estimated through multi-view stereo matching. Then, cross-view feature perception module is introduced to establish correspondences between exo- and ego- views, guiding the trajectory transformation module to infer human full-body movement from the head trajectory. Meanwhile, we present an action description unit that maps the action semantics into the feature space consistent with the exocentric image. Finally, the inferred human movement and high-level action descriptions jointly guide the generation of exocentric motion and interaction content (i.e., corresponding optical flow and occlusion maps) in the backward process of the diffusion model, ultimately warping them into the corresponding exocentric video. We conduct extensive experiments on the relevant dataset with diverse exo-ego video pairs, and our IDE outperforms state-of-the-art models in both subjective and objective assessments, demonstrating its efficacy in ego-to-exo video generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.09227v1">BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation</a></div>
    <div class="paper-meta">
      📅 2024-03-15
      | 💬 A preliminary version was published at 6th Conference on Robot Learning (CoRL 2022)
    </div>
    <details class="paper-abstract">
      We present BEHAVIOR-1K, a comprehensive simulation benchmark for human-centered robotics. BEHAVIOR-1K includes two components, guided and motivated by the results of an extensive survey on "what do you want robots to do for you?". The first is the definition of 1,000 everyday activities, grounded in 50 scenes (houses, gardens, restaurants, offices, etc.) with more than 9,000 objects annotated with rich physical and semantic properties. The second is OMNIGIBSON, a novel simulation environment that supports these activities via realistic physics simulation and rendering of rigid bodies, deformable bodies, and liquids. Our experiments indicate that the activities in BEHAVIOR-1K are long-horizon and dependent on complex manipulation skills, both of which remain a challenge for even state-of-the-art robot learning solutions. To calibrate the simulation-to-reality gap of BEHAVIOR-1K, we provide an initial study on transferring solutions learned with a mobile manipulator in a simulated apartment to its real-world counterpart. We hope that BEHAVIOR-1K's human-grounded nature, diversity, and realism make it valuable for embodied AI and robot learning research. Project website: https://behavior.stanford.edu.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09227v1">BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation</a></div>
    <div class="paper-meta">
      📅 2024-03-14
      | 💬 A preliminary version was published at 6th Conference on Robot Learning (CoRL 2022)
    </div>
    <details class="paper-abstract">
      We present BEHAVIOR-1K, a comprehensive simulation benchmark for human-centered robotics. BEHAVIOR-1K includes two components, guided and motivated by the results of an extensive survey on "what do you want robots to do for you?". The first is the definition of 1,000 everyday activities, grounded in 50 scenes (houses, gardens, restaurants, offices, etc.) with more than 9,000 objects annotated with rich physical and semantic properties. The second is OMNIGIBSON, a novel simulation environment that supports these activities via realistic physics simulation and rendering of rigid bodies, deformable bodies, and liquids. Our experiments indicate that the activities in BEHAVIOR-1K are long-horizon and dependent on complex manipulation skills, both of which remain a challenge for even state-of-the-art robot learning solutions. To calibrate the simulation-to-reality gap of BEHAVIOR-1K, we provide an initial study on transferring solutions learned with a mobile manipulator in a simulated apartment to its real-world counterpart. We hope that BEHAVIOR-1K's human-grounded nature, diversity, and realism make it valuable for embodied AI and robot learning research. Project website: https://behavior.stanford.edu.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.09227v1">BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation</a></div>
    <div class="paper-meta">
      📅 2024-03-14
      | 💬 A preliminary version was published at 6th Conference on Robot Learning (CoRL 2022)
    </div>
    <details class="paper-abstract">
      We present BEHAVIOR-1K, a comprehensive simulation benchmark for human-centered robotics. BEHAVIOR-1K includes two components, guided and motivated by the results of an extensive survey on "what do you want robots to do for you?". The first is the definition of 1,000 everyday activities, grounded in 50 scenes (houses, gardens, restaurants, offices, etc.) with more than 9,000 objects annotated with rich physical and semantic properties. The second is OMNIGIBSON, a novel simulation environment that supports these activities via realistic physics simulation and rendering of rigid bodies, deformable bodies, and liquids. Our experiments indicate that the activities in BEHAVIOR-1K are long-horizon and dependent on complex manipulation skills, both of which remain a challenge for even state-of-the-art robot learning solutions. To calibrate the simulation-to-reality gap of BEHAVIOR-1K, we provide an initial study on transferring solutions learned with a mobile manipulator in a simulated apartment to its real-world counterpart. We hope that BEHAVIOR-1K's human-grounded nature, diversity, and realism make it valuable for embodied AI and robot learning research. Project website: https://behavior.stanford.edu.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07376v1">NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-03-12
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN), as a crucial research problem of Embodied AI, requires an embodied agent to navigate through complex 3D environments following natural language instructions. Recent research has highlighted the promising capacity of large language models (LLMs) in VLN by improving navigational reasoning accuracy and interpretability. However, their predominant use in an offline manner usually suffers from substantial domain gap between the VLN task and the LLM training corpus. This paper introduces a novel strategy called Navigational Chain-of-Thought (NavCoT), where we fulfill parameter-efficient in-domain training to enable self-guided navigational decision, leading to a significant mitigation of the domain gap in a cost-effective manner. Specifically, at each timestep, the LLM is prompted to forecast the navigational chain-of-thought by: 1) acting as a world model to imagine the next observation according to the instruction, 2) selecting the candidate observation that best aligns with the imagination, and 3) determining the action based on the reasoning from the prior steps. Through constructing formalized labels for training, the LLM can learn to generate desired and reasonable chain-of-thought outputs for improving the action decision. Experimental results across various training settings and popular VLN benchmarks (e.g., Room-to-Room (R2R), Room-across-Room (RxR), Room-for-Room (R4R)) show the significant superiority of NavCoT over the direct action prediction variants. Through simple parameter-efficient finetuning, our NavCoT outperforms a recent GPT4-based approach with ~7% relative improvement on the R2R dataset. We believe that NavCoT will help unlock more task-adaptive and scalable LLM-based embodied agents, which are helpful for developing real-world robotics applications. Code is available at https://github.com/expectorlin/NavCoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2311.04193v2">Selective Visual Representations Improve Convergence and Generalization for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-03-12
      | 💬 See project website: https://embodied-codebook.github.io
    </div>
    <details class="paper-abstract">
      Embodied AI models often employ off the shelf vision backbones like CLIP to encode their visual observations. Although such general purpose representations encode rich syntactic and semantic information about the scene, much of this information is often irrelevant to the specific task at hand. This introduces noise within the learning process and distracts the agent's focus from task-relevant visual cues. Inspired by selective attention in humans-the process through which people filter their perception based on their experiences, knowledge, and the task at hand-we introduce a parameter-efficient approach to filter visual stimuli for embodied AI. Our approach induces a task-conditioned bottleneck using a small learnable codebook module. This codebook is trained jointly to optimize task reward and acts as a task-conditioned selective filter over the visual observation. Our experiments showcase state-of-the-art performance for object goal navigation and object displacement across 5 benchmarks, ProcTHOR, ArchitecTHOR, RoboTHOR, AI2-iTHOR, and ManipulaTHOR. The filtered representations produced by the codebook are also able generalize better and converge faster when adapted to other simulation environments such as Habitat. Our qualitative analyses show that agents explore their environments more effectively and their representations retain task-relevant information like target object recognition while ignoring superfluous information about other objects. Code and pretrained models are available at our project website: https://embodied-codebook.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.04193v2">Selective Visual Representations Improve Convergence and Generalization for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-03-10
      | 💬 See project website: https://embodied-codebook.github.io
    </div>
    <details class="paper-abstract">
      Embodied AI models often employ off the shelf vision backbones like CLIP to encode their visual observations. Although such general purpose representations encode rich syntactic and semantic information about the scene, much of this information is often irrelevant to the specific task at hand. This introduces noise within the learning process and distracts the agent's focus from task-relevant visual cues. Inspired by selective attention in humans-the process through which people filter their perception based on their experiences, knowledge, and the task at hand-we introduce a parameter-efficient approach to filter visual stimuli for embodied AI. Our approach induces a task-conditioned bottleneck using a small learnable codebook module. This codebook is trained jointly to optimize task reward and acts as a task-conditioned selective filter over the visual observation. Our experiments showcase state-of-the-art performance for object goal navigation and object displacement across 5 benchmarks, ProcTHOR, ArchitecTHOR, RoboTHOR, AI2-iTHOR, and ManipulaTHOR. The filtered representations produced by the codebook are also able generalize better and converge faster when adapted to other simulation environments such as Habitat. Our qualitative analyses show that agents explore their environments more effectively and their representations retain task-relevant information like target object recognition while ignoring superfluous information about other objects. Code and pretrained models are available at our project website: https://embodied-codebook.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2311.04193v2">Selective Visual Representations Improve Convergence and Generalization for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-03-10
      | 💬 See project website: https://embodied-codebook.github.io
    </div>
    <details class="paper-abstract">
      Embodied AI models often employ off the shelf vision backbones like CLIP to encode their visual observations. Although such general purpose representations encode rich syntactic and semantic information about the scene, much of this information is often irrelevant to the specific task at hand. This introduces noise within the learning process and distracts the agent's focus from task-relevant visual cues. Inspired by selective attention in humans-the process through which people filter their perception based on their experiences, knowledge, and the task at hand-we introduce a parameter-efficient approach to filter visual stimuli for embodied AI. Our approach induces a task-conditioned bottleneck using a small learnable codebook module. This codebook is trained jointly to optimize task reward and acts as a task-conditioned selective filter over the visual observation. Our experiments showcase state-of-the-art performance for object goal navigation and object displacement across 5 benchmarks, ProcTHOR, ArchitecTHOR, RoboTHOR, AI2-iTHOR, and ManipulaTHOR. The filtered representations produced by the codebook are also able generalize better and converge faster when adapted to other simulation environments such as Habitat. Our qualitative analyses show that agents explore their environments more effectively and their representations retain task-relevant information like target object recognition while ignoring superfluous information about other objects. Code and pretrained models are available at our project website: https://embodied-codebook.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.00833v1">Position Paper: Agent AI Towards a Holistic Intelligence</a></div>
    <div class="paper-meta">
      📅 2024-03-05
      | 💬 22 pages, 4 figures. arXiv admin note: substantial text overlap with arXiv:2401.03568
    </div>
    <details class="paper-abstract">
      Recent advancements in large foundation models have remarkably enhanced our understanding of sensory information in open-world environments. In leveraging the power of foundation models, it is crucial for AI research to pivot away from excessive reductionism and toward an emphasis on systems that function as cohesive wholes. Specifically, we emphasize developing Agent AI -- an embodied system that integrates large foundation models into agent actions. The emerging field of Agent AI spans a wide range of existing embodied and agent-based multimodal interactions, including robotics, gaming, and healthcare systems, etc. In this paper, we propose a novel large action model to achieve embodied intelligent behavior, the Agent Foundation Model. On top of this idea, we discuss how agent AI exhibits remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. Furthermore, we discuss the potential of Agent AI from an interdisciplinary perspective, underscoring AI cognition and consciousness within scientific discourse. We believe that those discussions serve as a basis for future research directions and encourage broader societal engagement.
    </details>
</div>
