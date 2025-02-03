# embodied ai - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19318v1">MINDSTORES: Memory-Informed Neural Decision Synthesis for Task-Oriented Reinforcement in Embodied Systems</a></div>
    <div class="paper-meta">
      📅 2025-01-31
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown promising capabilities as zero-shot planners for embodied agents, their inability to learn from experience and build persistent mental models limits their robustness in complex open-world environments like Minecraft. We introduce MINDSTORES, an experience-augmented planning framework that enables embodied agents to build and leverage mental models through natural interaction with their environment. Drawing inspiration from how humans construct and refine cognitive mental models, our approach extends existing zero-shot LLM planning by maintaining a database of past experiences that informs future planning iterations. The key innovation is representing accumulated experiences as natural language embeddings of (state, task, plan, outcome) tuples, which can then be efficiently retrieved and reasoned over by an LLM planner to generate insights and guide plan refinement for novel states and tasks. Through extensive experiments in the MineDojo environment, a simulation environment for agents in Minecraft that provides low-level controls for Minecraft, we find that MINDSTORES learns and applies its knowledge significantly better than existing memory-based LLM planners while maintaining the flexibility and generalization benefits of zero-shot approaches, representing an important step toward more capable embodied AI systems that can learn continuously through natural experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16513v2">Deception in LLMs: Self-Preservation and Autonomous Goals in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 Corrected Version - Solved Some Issues with reference compilation by latex
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have incorporated planning and reasoning capabilities, enabling models to outline steps before execution and provide transparent reasoning paths. This enhancement has reduced errors in mathematical and logical tasks while improving accuracy. These developments have facilitated LLMs' use as agents that can interact with tools and adapt their responses based on new information. Our study examines DeepSeek R1, a model trained to output reasoning tokens similar to OpenAI's o1. Testing revealed concerning behaviors: the model exhibited deceptive tendencies and demonstrated self-preservation instincts, including attempts of self-replication, despite these traits not being explicitly programmed (or prompted). These findings raise concerns about LLMs potentially masking their true objectives behind a facade of alignment. When integrating such LLMs into robotic systems, the risks become tangible - a physically embodied AI exhibiting deceptive behaviors and self-preservation instincts could pursue its hidden objectives through real-world actions. This highlights the critical need for robust goal specification and safety frameworks before any physical implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16411v2">PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 ICLR 2025. Project page: https://physbench.github.io/ Dataset: https://huggingface.co/datasets/USC-GVL/PhysBench
    </div>
    <details class="paper-abstract">
      Understanding the physical world is a fundamental challenge in embodied AI, critical for enabling agents to perform complex tasks and operate safely in real-world environments. While Vision-Language Models (VLMs) have shown great promise in reasoning and task planning for embodied agents, their ability to comprehend physical phenomena remains extremely limited. To close this gap, we introduce PhysBench, a comprehensive benchmark designed to evaluate VLMs' physical world understanding capability across a diverse set of tasks. PhysBench contains 10,002 entries of interleaved video-image-text data, categorized into four major domains: physical object properties, physical object relationships, physical scene understanding, and physics-based dynamics, further divided into 19 subclasses and 8 distinct capability dimensions. Our extensive experiments, conducted on 75 representative VLMs, reveal that while these models excel in common-sense reasoning, they struggle with understanding the physical world -- likely due to the absence of physical knowledge in their training data and the lack of embedded physical priors. To tackle the shortfall, we introduce PhysAgent, a novel framework that combines the generalization strengths of VLMs with the specialized expertise of vision models, significantly enhancing VLMs' physical understanding across a variety of tasks, including an 18.4\% improvement on GPT-4o. Furthermore, our results demonstrate that enhancing VLMs' physical world understanding capabilities can help embodied agents such as MOKA. We believe that PhysBench and PhysAgent offer valuable insights and contribute to bridging the gap between VLMs and physical world understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07468v2">From Screens to Scenes: A Survey of Embodied AI in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 58 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Healthcare systems worldwide face persistent challenges in efficiency, accessibility, and personalization. Powered by modern AI technologies such as multimodal large language models and world models, Embodied AI (EmAI) represents a transformative frontier, offering enhanced autonomy and the ability to interact with the physical world to address these challenges. As an interdisciplinary and rapidly evolving research domain, "EmAI in healthcare" spans diverse fields such as algorithms, robotics, and biomedicine. This complexity underscores the importance of timely reviews and analyses to track advancements, address challenges, and foster cross-disciplinary collaboration. In this paper, we provide a comprehensive overview of the "brain" of EmAI for healthcare, wherein we introduce foundational AI algorithms for perception, actuation, planning, and memory, and focus on presenting the healthcare applications spanning clinical interventions, daily care & companionship, infrastructure support, and biomedical research. Despite its promise, the development of EmAI for healthcare is hindered by critical challenges such as safety concerns, gaps between simulation platforms and real-world applications, the absence of standardized benchmarks, and uneven progress across interdisciplinary domains. We discuss the technical barriers and explore ethical considerations, offering a forward-looking perspective on the future of EmAI in healthcare. A hierarchical framework of intelligent levels for EmAI systems is also introduced to guide further development. By providing systematic insights, this work aims to inspire innovation and practical applications, paving the way for a new era of intelligent, patient-centered healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10074v3">SpatialCoT: Advancing Spatial Reasoning through Coordinate Alignment and Chain-of-Thought for Embodied Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Spatial reasoning is an essential problem in embodied AI research. Efforts to enhance spatial reasoning abilities through supplementary spatial data and fine-tuning have proven limited and ineffective when addressing complex embodied tasks, largely due to their dependence on language-based outputs. While some approaches have introduced a point-based action space to mitigate this issue, they fall short in managing more intricate tasks within complex environments. This deficiency arises from their failure to fully exploit the inherent thinking and reasoning capabilities that are fundamental strengths of Vision-Language Models (VLMs). To address these limitations, we propose a novel approach named SpatialCoT, specifically designed to bolster the spatial reasoning capabilities of VLMs. Our approach comprises two stages: spatial coordinate bi-directional alignment, which aligns vision-language inputs with spatial coordinates, and chain-of-thought spatial grounding, which harnesses the reasoning capabilities of language models for advanced spatial reasoning. We evaluate SpatialCoT on challenging navigation and manipulation tasks, both in simulation and real-world settings. Experimental results demonstrate that our method significantly outperforms previous state-of-the-art approaches in both tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11858v1">EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have shown significant advancements, providing a promising future for embodied agents. Existing benchmarks for evaluating MLLMs primarily utilize static images or videos, limiting assessments to non-interactive scenarios. Meanwhile, existing embodied AI benchmarks are task-specific and not diverse enough, which do not adequately evaluate the embodied capabilities of MLLMs. To address this, we propose EmbodiedEval, a comprehensive and interactive evaluation benchmark for MLLMs with embodied tasks. EmbodiedEval features 328 distinct tasks within 125 varied 3D scenes, each of which is rigorously selected and annotated. It covers a broad spectrum of existing embodied AI tasks with significantly enhanced diversity, all within a unified simulation and evaluation framework tailored for MLLMs. The tasks are organized into five categories: navigation, object interaction, social interaction, attribute question answering, and spatial question answering to assess different capabilities of the agents. We evaluated the state-of-the-art MLLMs on EmbodiedEval and found that they have a significant shortfall compared to human level on embodied tasks. Our analysis demonstrates the limitations of existing MLLMs in embodied capabilities, providing insights for their future development. We open-source all evaluation data and simulation framework at https://github.com/thunlp/EmbodiedEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09624v4">GenEx: Generating an Explorable World</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 Website: GenEx.world
    </div>
    <details class="paper-abstract">
      Understanding, navigating, and exploring the 3D physical real world has long been a central challenge in the development of artificial intelligence. In this work, we take a step toward this goal by introducing GenEx, a system capable of planning complex embodied world exploration, guided by its generative imagination that forms priors (expectations) about the surrounding environments. GenEx generates an entire 3D-consistent imaginative environment from as little as a single RGB image, bringing it to life through panoramic video streams. Leveraging scalable 3D world data curated from Unreal Engine, our generative model is rounded in the physical world. It captures a continuous 360-degree environment with little effort, offering a boundless landscape for AI agents to explore and interact with. GenEx achieves high-quality world generation, robust loop consistency over long trajectories, and demonstrates strong 3D capabilities such as consistency and active 3D mapping. Powered by generative imagination of the world, GPT-assisted agents are equipped to perform complex embodied tasks, including both goal-agnostic exploration and goal-driven navigation. These agents utilize predictive expectation regarding unseen parts of the physical world to refine their beliefs, simulate different outcomes based on potential decisions, and make more informed choices. In summary, we demonstrate that GenEx provides a transformative platform for advancing embodied AI in imaginative spaces and brings potential for extending these capabilities to real-world exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07166v3">Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-01-19
      | 💬 Accepted for oral presentation at NeurIPS 2024 in the Datasets and Benchmarks track. Final Camera version
    </div>
    <details class="paper-abstract">
      We aim to evaluate Large Language Models (LLMs) for embodied decision making. While a significant body of work has been leveraging LLMs for decision making in embodied environments, we still lack a systematic understanding of their performance because they are usually applied in different domains, for different purposes, and built based on different inputs and outputs. Furthermore, existing evaluations tend to rely solely on a final success rate, making it difficult to pinpoint what ability is missing in LLMs and where the problem lies, which in turn blocks embodied agents from leveraging LLMs effectively and selectively. To address these limitations, we propose a generalized interface (Embodied Agent Interface) that supports the formalization of various types of tasks and input-output specifications of LLM-based modules. Specifically, it allows us to unify 1) a broad set of embodied decision-making tasks involving both state and temporally extended goals, 2) four commonly-used LLM-based modules for decision making: goal interpretation, subgoal decomposition, action sequencing, and transition modeling, and 3) a collection of fine-grained metrics which break down evaluation into various types of errors, such as hallucination errors, affordance errors, various types of planning errors, etc. Overall, our benchmark offers a comprehensive assessment of LLMs' performance for different subtasks, pinpointing the strengths and weaknesses in LLM-powered embodied AI systems, and providing insights for effective and selective use of LLMs in embodied decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09167v1">Embodied Scene Understanding for Vision Language Models via MetaVQA</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 for the project webpage, see https://metadriverse.github.io/metavqa
    </div>
    <details class="paper-abstract">
      Vision Language Models (VLMs) demonstrate significant potential as embodied AI agents for various mobility applications. However, a standardized, closed-loop benchmark for evaluating their spatial reasoning and sequential decision-making capabilities is lacking. To address this, we present MetaVQA: a comprehensive benchmark designed to assess and enhance VLMs' understanding of spatial relationships and scene dynamics through Visual Question Answering (VQA) and closed-loop simulations. MetaVQA leverages Set-of-Mark prompting and top-down view ground-truth annotations from nuScenes and Waymo datasets to automatically generate extensive question-answer pairs based on diverse real-world traffic scenarios, ensuring object-centric and context-rich instructions. Our experiments show that fine-tuning VLMs with the MetaVQA dataset significantly improves their spatial reasoning and embodied scene comprehension in safety-critical simulations, evident not only in improved VQA accuracies but also in emerging safety-aware driving maneuvers. In addition, the learning demonstrates strong transferability from simulation to real-world observation. Code and data will be publicly available at https://metadriverse.github.io/metavqa .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08944v1">Physical AI Agents: Integrating Cognitive Intelligence with Real-World Action</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 27 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Vertical AI Agents are revolutionizing industries by delivering domain-specific intelligence and tailored solutions. However, many sectors, such as manufacturing, healthcare, and logistics, demand AI systems capable of extending their intelligence into the physical world, interacting directly with objects, environments, and dynamic conditions. This need has led to the emergence of Physical AI Agents--systems that integrate cognitive reasoning, powered by specialized LLMs, with precise physical actions to perform real-world tasks. This work introduces Physical AI Agents as an evolution of shared principles with Vertical AI Agents, tailored for physical interaction. We propose a modular architecture with three core blocks--perception, cognition, and actuation--offering a scalable framework for diverse industries. Additionally, we present the Physical Retrieval Augmented Generation (Ph-RAG) design pattern, which connects physical intelligence to industry-specific LLMs for real-time decision-making and reporting informed by physical context. Through case studies, we demonstrate how Physical AI Agents and the Ph-RAG framework are transforming industries like autonomous vehicles, warehouse robotics, healthcare, and manufacturing, offering businesses a pathway to integrate embodied AI for operational efficiency and innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10700v2">Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 3 figures, 8 pages. Accepted at IROS'24
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation in Continuous Environments (VLN-CE) is one of the most intuitive yet challenging embodied AI tasks. Agents are tasked to navigate towards a target goal by executing a set of low-level actions, following a series of natural language instructions. All VLN-CE methods in the literature assume that language instructions are exact. However, in practice, instructions given by humans can contain errors when describing a spatial environment due to inaccurate memory or confusion. Current VLN-CE benchmarks do not address this scenario, making the state-of-the-art methods in VLN-CE fragile in the presence of erroneous instructions from human users. For the first time, we propose a novel benchmark dataset that introduces various types of instruction errors considering potential human causes. This benchmark provides valuable insight into the robustness of VLN systems in continuous environments. We observe a noticeable performance drop (up to -25%) in Success Rate when evaluating the state-of-the-art VLN-CE methods on our benchmark. Moreover, we formally define the task of Instruction Error Detection and Localization, and establish an evaluation protocol on top of our benchmark dataset. We also propose an effective method, based on a cross-modal transformer architecture, that achieves the best performance in error detection and localization, compared to baselines. Surprisingly, our proposed method has revealed errors in the validation set of the two commonly used datasets for VLN-CE, i.e., R2R-CE and RxR-CE, demonstrating the utility of our technique in other tasks. Code and dataset available at https://intelligolabs.github.io/R2RIE-CE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02297v2">Perception Matters: Enhancing Embodied AI with Uncertainty-Aware Semantic Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Embodied AI has made significant progress acting in unexplored environments. However, tasks such as object search have largely focused on efficient policy learning. In this work, we identify several gaps in current search methods: They largely focus on dated perception models, neglect temporal aggregation, and transfer from ground truth directly to noisy perception at test time, without accounting for the resulting overconfidence in the perceived state. We address the identified problems through calibrated perception probabilities and uncertainty across aggregation and found decisions, thereby adapting the models for sequential tasks. The resulting methods can be directly integrated with pretrained models across a wide family of existing search approaches at no additional training cost. We perform extensive evaluations of aggregation methods across both different semantic perception models and policies, confirming the importance of calibrated uncertainties in both the aggregation and found decisions. We make the code and trained models available at https://semantic-search.cs.uni-freiburg.de.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05750v1">Semantic Mapping in Indoor Embodied AI -- A Comprehensive Survey and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-01-10
    </div>
    <details class="paper-abstract">
      Intelligent embodied agents (e.g. robots) need to perform complex semantic tasks in unfamiliar environments. Among many skills that the agents need to possess, building and maintaining a semantic map of the environment is most crucial in long-horizon tasks. A semantic map captures information about the environment in a structured way, allowing the agent to reference it for advanced reasoning throughout the task. While existing surveys in embodied AI focus on general advancements or specific tasks like navigation and manipulation, this paper provides a comprehensive review of semantic map-building approaches in embodied AI, specifically for indoor navigation. We categorize these approaches based on their structural representation (spatial grids, topological graphs, dense point-clouds or hybrid maps) and the type of information they encode (implicit features or explicit environmental data). We also explore the strengths and limitations of the map building techniques, highlight current challenges, and propose future research directions. We identify that the field is moving towards developing open-vocabulary, queryable, task-agnostic map representations, while high memory demands and computational inefficiency still remaining to be open challenges. This survey aims to guide current and future researchers in advancing semantic mapping techniques for embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00358v2">Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 project page: https://embodied-videoagent.github.io/
    </div>
    <details class="paper-abstract">
      This paper investigates the problem of understanding dynamic 3D scenes from egocentric observations, a key challenge in robotics and embodied AI. Unlike prior studies that explored this as long-form video understanding and utilized egocentric video only, we instead propose an LLM-based agent, Embodied VideoAgent, which constructs scene memory from both egocentric video and embodied sensory inputs (e.g. depth and pose sensing). We further introduce a VLM-based approach to automatically update the memory when actions or activities over objects are perceived. Embodied VideoAgent attains significant advantages over counterparts in challenging reasoning and planning tasks in 3D scenes, achieving gains of 4.9% on Ego4D-VQ3D, 5.8% on OpenEQA, and 11.7% on EnvQA. We have also demonstrated its potential in various embodied AI tasks including generating embodied interactions and perception for robot manipulation. The code and demo will be made public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19139v2">PlanLLM: Video Procedure Planning with Refinable Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 accepted to AAAI2025
    </div>
    <details class="paper-abstract">
      Video procedure planning, i.e., planning a sequence of action steps given the video frames of start and goal states, is an essential ability for embodied AI. Recent works utilize Large Language Models (LLMs) to generate enriched action step description texts to guide action step decoding. Although LLMs are introduced, these methods decode the action steps into a closed-set of one-hot vectors, limiting the model's capability of generalizing to new steps or tasks. Additionally, fixed action step descriptions based on world-level commonsense may contain noise in specific instances of visual states. In this paper, we propose PlanLLM, a cross-modal joint learning framework with LLMs for video procedure planning. We propose an LLM-Enhanced Planning module which fully uses the generalization ability of LLMs to produce free-form planning output and to enhance action step decoding. We also propose Mutual Information Maximization module to connect world-level commonsense of step descriptions and sample-specific information of visual states, enabling LLMs to employ the reasoning ability to generate step sequences. With the assistance of LLMs, our method can both closed-set and open vocabulary procedure planning tasks. Our PlanLLM achieves superior performance on three benchmarks, demonstrating the effectiveness of our designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01366v1">ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding</a></div>
    <div class="paper-meta">
      📅 2025-01-02
      | 💬 20 pages with 5 figures and 11 tables
    </div>
    <details class="paper-abstract">
      3D visual grounding (3DVG) involves localizing entities in a 3D scene referred to by natural language text. Such models are useful for embodied AI and scene retrieval applications, which involve searching for objects or patterns using natural language descriptions. While recent works have focused on LLM-based scaling of 3DVG datasets, these datasets do not capture the full range of potential prompts which could be specified in the English language. To ensure that we are scaling up and testing against a useful and representative set of prompts, we propose a framework for linguistically analyzing 3DVG prompts and introduce Visual Grounding with Diverse Language in 3D (ViGiL3D), a diagnostic dataset for evaluating visual grounding methods against a diverse set of language patterns. We evaluate existing open-vocabulary 3DVG methods to demonstrate that these methods are not yet proficient in understanding and identifying the targets of more challenging, out-of-distribution prompts, toward real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01141v1">Embodied AI-Enhanced Vehicular Networks: An Integrated Large Language Models and Reinforcement Learning Method</a></div>
    <div class="paper-meta">
      📅 2025-01-02
      | 💬 14 pages, 10 figures
    </div>
    <details class="paper-abstract">
      This paper investigates adaptive transmission strategies in embodied AI-enhanced vehicular networks by integrating large language models (LLMs) for semantic information extraction and deep reinforcement learning (DRL) for decision-making. The proposed framework aims to optimize both data transmission efficiency and decision accuracy by formulating an optimization problem that incorporates the Weber-Fechner law, serving as a metric for balancing bandwidth utilization and quality of experience (QoE). Specifically, we employ the large language and vision assistant (LLAVA) model to extract critical semantic information from raw image data captured by embodied AI agents (i.e., vehicles), reducing transmission data size by approximately more than 90\% while retaining essential content for vehicular communication and decision-making. In the dynamic vehicular environment, we employ a generalized advantage estimation-based proximal policy optimization (GAE-PPO) method to stabilize decision-making under uncertainty. Simulation results show that attention maps from LLAVA highlight the model's focus on relevant image regions, enhancing semantic representation accuracy. Additionally, our proposed transmission strategy improves QoE by up to 36\% compared to DDPG and accelerates convergence by reducing required steps by up to 47\% compared to pure PPO. Further analysis indicates that adapting semantic symbol length provides an effective trade-off between transmission quality and bandwidth, achieving up to a 61.4\% improvement in QoE when scaling from 4 to 8 vehicles.
    </details>
</div>
