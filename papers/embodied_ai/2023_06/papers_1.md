# embodied ai - 2023_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.13760v1">Task-Driven Graph Attention for Hierarchical Relational Object Navigation</a></div>
    <div class="paper-meta">
      📅 2023-06-23
    </div>
    <details class="paper-abstract">
      Embodied AI agents in large scenes often need to navigate to find objects. In this work, we study a naturally emerging variant of the object navigation task, hierarchical relational object navigation (HRON), where the goal is to find objects specified by logical predicates organized in a hierarchical structure - objects related to furniture and then to rooms - such as finding an apple on top of a table in the kitchen. Solving such a task requires an efficient representation to reason about object relations and correlate the relations in the environment and in the task goal. HRON in large scenes (e.g. homes) is particularly challenging due to its partial observability and long horizon, which invites solutions that can compactly store the past information while effectively exploring the scene. We demonstrate experimentally that scene graphs are the best-suited representation compared to conventional representations such as images or 2D maps. We propose a solution that uses scene graphs as part of its input and integrates graph neural networks as its backbone, with an integrated task-driven attention mechanism, and demonstrate its better scalability and learning efficiency than state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2306.09776v1">Inspire creativity with ORIBA: Transform Artists' Original Characters into Chatbots through Large Language Model</a></div>
    <div class="paper-meta">
      📅 2023-06-19
      | 💬 5 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      This research delves into the intersection of illustration art and artificial intelligence (AI), focusing on how illustrators engage with AI agents that embody their original characters (OCs). We introduce 'ORIBA', a customizable AI chatbot that enables illustrators to converse with their OCs. This approach allows artists to not only receive responses from their OCs but also to observe their inner monologues and behavior. Despite the existing tension between artists and AI, our study explores innovative collaboration methods that are inspiring to illustrators. By examining the impact of AI on the creative process and the boundaries of authorship, we aim to enhance human-AI interactions in creative fields, with potential applications extending beyond illustration to interactive storytelling and more.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2306.09643v1">BISCUIT: Causal Representation Learning from Binary Interactions</a></div>
    <div class="paper-meta">
      📅 2023-06-19
      | 💬 Published in: Uncertainty in Artificial Intelligence (UAI 2023). Project page: https://phlippe.github.io/BISCUIT/
    </div>
    <details class="paper-abstract">
      Identifying the causal variables of an environment and how to intervene on them is of core value in applications such as robotics and embodied AI. While an agent can commonly interact with the environment and may implicitly perturb the behavior of some of these causal variables, often the targets it affects remain unknown. In this paper, we show that causal variables can still be identified for many common setups, e.g., additive Gaussian noise models, if the agent's interactions with a causal variable can be described by an unknown binary variable. This happens when each causal variable has two different mechanisms, e.g., an observational and an interventional one. Using this identifiability result, we propose BISCUIT, a method for simultaneously learning causal variables and their corresponding binary interaction variables. On three robotic-inspired datasets, BISCUIT accurately identifies causal variables and can even be scaled to complex, realistic environments for embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.09643v1">BISCUIT: Causal Representation Learning from Binary Interactions</a></div>
    <div class="paper-meta">
      📅 2023-06-16
      | 💬 Published in: Uncertainty in Artificial Intelligence (UAI 2023). Project page: https://phlippe.github.io/BISCUIT/
    </div>
    <details class="paper-abstract">
      Identifying the causal variables of an environment and how to intervene on them is of core value in applications such as robotics and embodied AI. While an agent can commonly interact with the environment and may implicitly perturb the behavior of some of these causal variables, often the targets it affects remain unknown. In this paper, we show that causal variables can still be identified for many common setups, e.g., additive Gaussian noise models, if the agent's interactions with a causal variable can be described by an unknown binary variable. This happens when each causal variable has two different mechanisms, e.g., an observational and an interventional one. Using this identifiability result, we propose BISCUIT, a method for simultaneously learning causal variables and their corresponding binary interaction variables. On three robotic-inspired datasets, BISCUIT accurately identifies causal variables and can even be scaled to complex, realistic environments for embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2304.09349v4">LLM as A Robotic Brain: Unifying Egocentric Memory and Control</a></div>
    <div class="paper-meta">
      📅 2023-06-13
      | 💬 This early project is now integrated to: Mindstorms in Natural Language-Based Societies of Mind, arXiv:2305.17066
    </div>
    <details class="paper-abstract">
      Embodied AI focuses on the study and development of intelligent systems that possess a physical or virtual embodiment (i.e. robots) and are able to dynamically interact with their environment. Memory and control are the two essential parts of an embodied system and usually require separate frameworks to model each of them. In this paper, we propose a novel and generalizable framework called LLM-Brain: using Large-scale Language Model as a robotic brain to unify egocentric memory and control. The LLM-Brain framework integrates multiple multimodal language models for robotic tasks, utilizing a zero-shot learning approach. All components within LLM-Brain communicate using natural language in closed-loop multi-round dialogues that encompass perception, planning, control, and memory. The core of the system is an embodied LLM to maintain egocentric memory and control the robot. We demonstrate LLM-Brain by examining two downstream tasks: active exploration and embodied question answering. The active exploration tasks require the robot to extensively explore an unknown environment within a limited number of actions. Meanwhile, the embodied question answering tasks necessitate that the robot answers questions based on observations acquired during prior explorations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.17537v4">Modeling Dynamic Environments with Scene Graph Memory</a></div>
    <div class="paper-meta">
      📅 2023-06-12
    </div>
    <details class="paper-abstract">
      Embodied AI agents that search for objects in large environments such as households often need to make efficient decisions by predicting object locations based on partial information. We pose this as a new type of link prediction problem: link prediction on partially observable dynamic graphs. Our graph is a representation of a scene in which rooms and objects are nodes, and their relationships are encoded in the edges; only parts of the changing graph are known to the agent at each timestep. This partial observability poses a challenge to existing link prediction approaches, which we address. We propose a novel state representation -- Scene Graph Memory (SGM) -- with captures the agent's accumulated set of observations, as well as a neural net architecture called a Node Edge Predictor (NEP) that extracts information from the SGM to search efficiently. We evaluate our method in the Dynamic House Simulator, a new benchmark that creates diverse dynamic graphs following the semantic patterns typically seen at homes, and show that NEP can be trained to predict the locations of objects in a variety of environments with diverse object movement dynamics, outperforming baselines both in terms of new scene adaptability and overall accuracy. The codebase and more can be found at https://www.scenegraphmemory.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.09349v4">LLM as A Robotic Brain: Unifying Egocentric Memory and Control</a></div>
    <div class="paper-meta">
      📅 2023-06-12
      | 💬 This early project is now integrated to: Mindstorms in Natural Language-Based Societies of Mind, arXiv:2305.17066
    </div>
    <details class="paper-abstract">
      Embodied AI focuses on the study and development of intelligent systems that possess a physical or virtual embodiment (i.e. robots) and are able to dynamically interact with their environment. Memory and control are the two essential parts of an embodied system and usually require separate frameworks to model each of them. In this paper, we propose a novel and generalizable framework called LLM-Brain: using Large-scale Language Model as a robotic brain to unify egocentric memory and control. The LLM-Brain framework integrates multiple multimodal language models for robotic tasks, utilizing a zero-shot learning approach. All components within LLM-Brain communicate using natural language in closed-loop multi-round dialogues that encompass perception, planning, control, and memory. The core of the system is an embodied LLM to maintain egocentric memory and control the robot. We demonstrate LLM-Brain by examining two downstream tasks: active exploration and embodied question answering. The active exploration tasks require the robot to extensively explore an unknown environment within a limited number of actions. Meanwhile, the embodied question answering tasks necessitate that the robot answers questions based on observations acquired during prior explorations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.05582v1">A newborn embodied Turing test for view-invariant object recognition</a></div>
    <div class="paper-meta">
      📅 2023-06-08
      | 💬 7 Pages. 4 figures, 1 table. This paper was accepted to the CogSci 2023 Conference. (https://cognitivesciencesociety.org/)
    </div>
    <details class="paper-abstract">
      Recent progress in artificial intelligence has renewed interest in building machines that learn like animals. Almost all of the work comparing learning across biological and artificial systems comes from studies where animals and machines received different training data, obscuring whether differences between animals and machines emerged from differences in learning mechanisms versus training data. We present an experimental approach-a "newborn embodied Turing Test"-that allows newborn animals and machines to be raised in the same environments and tested with the same tasks, permitting direct comparison of their learning abilities. To make this platform, we first collected controlled-rearing data from newborn chicks, then performed "digital twin" experiments in which machines were raised in virtual environments that mimicked the rearing conditions of the chicks. We found that (1) machines (deep reinforcement learning agents with intrinsic motivation) can spontaneously develop visually guided preference behavior, akin to imprinting in newborn chicks, and (2) machines are still far from newborn-level performance on object recognition tasks. Almost all of the chicks developed view-invariant object recognition, whereas the machines tended to develop view-dependent recognition. The learning outcomes were also far more constrained in the chicks versus machines. Ultimately, we anticipate that this approach will help researchers develop embodied AI systems that learn like newborn animals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2303.01586v2">Alexa Arena: A User-Centric Interactive Platform for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-06-08
    </div>
    <details class="paper-abstract">
      We introduce Alexa Arena, a user-centric simulation platform for Embodied AI (EAI) research. Alexa Arena provides a variety of multi-room layouts and interactable objects, for the creation of human-robot interaction (HRI) missions. With user-friendly graphics and control mechanisms, Alexa Arena supports the development of gamified robotic tasks readily accessible to general human users, thus opening a new venue for high-efficiency HRI data collection and EAI system evaluation. Along with the platform, we introduce a dialog-enabled instruction-following benchmark and provide baseline results for it. We make Alexa Arena publicly available to facilitate research in building generalizable and assistive embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2303.01586v2">Alexa Arena: A User-Centric Interactive Platform for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-06-07
    </div>
    <details class="paper-abstract">
      We introduce Alexa Arena, a user-centric simulation platform for Embodied AI (EAI) research. Alexa Arena provides a variety of multi-room layouts and interactable objects, for the creation of human-robot interaction (HRI) missions. With user-friendly graphics and control mechanisms, Alexa Arena supports the development of gamified robotic tasks readily accessible to general human users, thus opening a new venue for high-efficiency HRI data collection and EAI system evaluation. Along with the platform, we introduce a dialog-enabled instruction-following benchmark and provide baseline results for it. We make Alexa Arena publicly available to facilitate research in building generalizable and assistive embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2301.00452v2">Human-in-the-loop Embodied Intelligence with Interactive Simulation Environment for Surgical Robot Learning</a></div>
    <div class="paper-meta">
      📅 2023-06-06
    </div>
    <details class="paper-abstract">
      Surgical robot automation has attracted increasing research interest over the past decade, expecting its potential to benefit surgeons, nurses and patients. Recently, the learning paradigm of embodied intelligence has demonstrated promising ability to learn good control policies for various complex tasks, where embodied AI simulators play an essential role to facilitate relevant research. However, existing open-sourced simulators for surgical robot are still not sufficiently supporting human interactions through physical input devices, which further limits effective investigations on how the human demonstrations would affect policy learning. In this work, we study human-in-the-loop embodied intelligence with a new interactive simulation platform for surgical robot learning. Specifically, we establish our platform based on our previously released SurRoL simulator with several new features co-developed to allow high-quality human interaction via an input device. We showcase the improvement of our simulation environment with the designed new features, and validate effectiveness of incorporating human factors in embodied intelligence through the use of human demonstrations and reinforcement learning as a representative example. Promising results are obtained in terms of learning efficiency. Lastly, five new surgical robot training tasks are developed and released, with which we hope to pave the way for future research on surgical embodied intelligence. Our learning platform is publicly released and will be continuously updated in the website: https://med-air.github.io/SurRoL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.01295v1">Egocentric Planning for Scalable Embodied Task Achievement</a></div>
    <div class="paper-meta">
      📅 2023-06-02
    </div>
    <details class="paper-abstract">
      Embodied agents face significant challenges when tasked with performing actions in diverse environments, particularly in generalizing across object types and executing suitable actions to accomplish tasks. Furthermore, agents should exhibit robustness, minimizing the execution of illegal actions. In this work, we present Egocentric Planning, an innovative approach that combines symbolic planning and Object-oriented POMDPs to solve tasks in complex environments, harnessing existing models for visual perception and natural language processing. We evaluated our approach in ALFRED, a simulated environment designed for domestic tasks, and demonstrated its high scalability, achieving an impressive 36.07% unseen success rate in the ALFRED benchmark and winning the ALFRED challenge at CVPR Embodied AI workshop. Our method requires reliable perception and the specification or learning of a symbolic description of the preconditions and effects of the agent's actions, as well as what object types reveal information about others. It is capable of naturally scaling to solve new tasks beyond ALFRED, as long as they can be solved using the available skills. This work offers a solid baseline for studying end-to-end and hybrid methods that aim to generalize to new tasks, including recent approaches relying on LLMs, but often struggle to scale to long sequences of actions or produce robust plans for novel tasks.
    </details>
</div>
