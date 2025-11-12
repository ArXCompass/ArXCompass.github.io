# embodied ai - 2023_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.16237v1">Object Motion Guided Human Motion Synthesis</a></div>
    <div class="paper-meta">
      📅 2023-09-28
      | 💬 SIGGRAPH Asia 2023
    </div>
    <details class="paper-abstract">
      Modeling human behaviors in contextual environments has a wide range of applications in character animation, embodied AI, VR/AR, and robotics. In real-world scenarios, humans frequently interact with the environment and manipulate various objects to complete daily tasks. In this work, we study the problem of full-body human motion synthesis for the manipulation of large-sized objects. We propose Object MOtion guided human MOtion synthesis (OMOMO), a conditional diffusion framework that can generate full-body manipulation behaviors from only the object motion. Since naively applying diffusion models fails to precisely enforce contact constraints between the hands and the object, OMOMO learns two separate denoising processes to first predict hand positions from object motion and subsequently synthesize full-body poses based on the predicted hand positions. By employing the hand positions as an intermediate representation between the two denoising processes, we can explicitly enforce contact constraints, resulting in more physically plausible manipulation motions. With the learned model, we develop a novel system that captures full-body human manipulation motions by simply attaching a smartphone to the object being manipulated. Through extensive experiments, we demonstrate the effectiveness of our proposed pipeline and its ability to generalize to unseen objects. Additionally, as high-quality human-object interaction datasets are scarce, we collect a large-scale dataset consisting of 3D object geometry, object motion, and human motion. Our dataset contains human-object interaction motion for 15 objects, with a total duration of approximately 10 hours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.14880v2">SGAligner : 3D Scene Alignment with Scene Graphs</a></div>
    <div class="paper-meta">
      📅 2023-09-26
      | 💬 Accepted at ICCV 2023
    </div>
    <details class="paper-abstract">
      Building 3D scene graphs has recently emerged as a topic in scene representation for several embodied AI applications to represent the world in a structured and rich manner. With their increased use in solving downstream tasks (eg, navigation and room rearrangement), can we leverage and recycle them for creating 3D maps of environments, a pivotal step in agent operation? We focus on the fundamental problem of aligning pairs of 3D scene graphs whose overlap can range from zero to partial and can contain arbitrary changes. We propose SGAligner, the first method for aligning pairs of 3D scene graphs that is robust to in-the-wild scenarios (ie, unknown overlap -- if any -- and changes in the environment). We get inspired by multi-modality knowledge graphs and use contrastive learning to learn a joint, multi-modal embedding space. We evaluate on the 3RScan dataset and further showcase that our method can be used for estimating the transformation between pairs of 3D scenes. Since benchmarks for these tasks are missing, we create them on this dataset. The code, benchmark, and trained models are available on the project website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.00923v2">Sonicverse: A Multisensory Simulation Platform for Embodied Household Agents that See and Hear</a></div>
    <div class="paper-meta">
      📅 2023-09-16
      | 💬 In ICRA 2023. Project page: https://ai.stanford.edu/~rhgao/sonicverse/. Code: https://github.com/StanfordVL/sonicverse. Gao and Li contributed equally to this work and are in alphabetical order
    </div>
    <details class="paper-abstract">
      Developing embodied agents in simulation has been a key research topic in recent years. Exciting new tasks, algorithms, and benchmarks have been developed in various simulators. However, most of them assume deaf agents in silent environments, while we humans perceive the world with multiple senses. We introduce Sonicverse, a multisensory simulation platform with integrated audio-visual simulation for training household agents that can both see and hear. Sonicverse models realistic continuous audio rendering in 3D environments in real-time. Together with a new audio-visual VR interface that allows humans to interact with agents with audio, Sonicverse enables a series of embodied AI tasks that need audio-visual perception. For semantic audio-visual navigation in particular, we also propose a new multi-task learning model that achieves state-of-the-art performance. In addition, we demonstrate Sonicverse's realism via sim-to-real transfer, which has not been achieved by other simulators: an agent trained in Sonicverse can successfully perform audio-visual navigation in real-world environments. Sonicverse is available at: https://github.com/StanfordVL/Sonicverse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2305.15021v2">EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought</a></div>
    <div class="paper-meta">
      📅 2023-09-15
    </div>
    <details class="paper-abstract">
      Embodied AI is a crucial frontier in robotics, capable of planning and executing action sequences for robots to accomplish long-horizon tasks in physical environments. In this work, we introduce EmbodiedGPT, an end-to-end multi-modal foundation model for embodied AI, empowering embodied agents with multi-modal understanding and execution capabilities. To achieve this, we have made the following efforts: (i) We craft a large-scale embodied planning dataset, termed EgoCOT. The dataset consists of carefully selected videos from the Ego4D dataset, along with corresponding high-quality language instructions. Specifically, we generate a sequence of sub-goals with the "Chain of Thoughts" mode for effective embodied planning. (ii) We introduce an efficient training approach to EmbodiedGPT for high-quality plan generation, by adapting a 7B large language model (LLM) to the EgoCOT dataset via prefix tuning. (iii) We introduce a paradigm for extracting task-related features from LLM-generated planning queries to form a closed loop between high-level planning and low-level control. Extensive experiments show the effectiveness of EmbodiedGPT on embodied tasks, including embodied planning, embodied control, visual captioning, and visual question answering. Notably, EmbodiedGPT significantly enhances the success rate of the embodied control task by extracting more effective features. It has achieved a remarkable 1.6 times increase in success rate on the Franka Kitchen benchmark and a 1.3 times increase on the Meta-World benchmark, compared to the BLIP-2 baseline fine-tuned with the Ego4D dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.15021v2">EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought</a></div>
    <div class="paper-meta">
      📅 2023-09-13
    </div>
    <details class="paper-abstract">
      Embodied AI is a crucial frontier in robotics, capable of planning and executing action sequences for robots to accomplish long-horizon tasks in physical environments. In this work, we introduce EmbodiedGPT, an end-to-end multi-modal foundation model for embodied AI, empowering embodied agents with multi-modal understanding and execution capabilities. To achieve this, we have made the following efforts: (i) We craft a large-scale embodied planning dataset, termed EgoCOT. The dataset consists of carefully selected videos from the Ego4D dataset, along with corresponding high-quality language instructions. Specifically, we generate a sequence of sub-goals with the "Chain of Thoughts" mode for effective embodied planning. (ii) We introduce an efficient training approach to EmbodiedGPT for high-quality plan generation, by adapting a 7B large language model (LLM) to the EgoCOT dataset via prefix tuning. (iii) We introduce a paradigm for extracting task-related features from LLM-generated planning queries to form a closed loop between high-level planning and low-level control. Extensive experiments show the effectiveness of EmbodiedGPT on embodied tasks, including embodied planning, embodied control, visual captioning, and visual question answering. Notably, EmbodiedGPT significantly enhances the success rate of the embodied control task by extracting more effective features. It has achieved a remarkable 1.6 times increase in success rate on the Franka Kitchen benchmark and a 1.3 times increase on the Meta-World benchmark, compared to the BLIP-2 baseline fine-tuned with the Ego4D dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.15514v2">International Governance of Civilian AI: A Jurisdictional Certification Approach</a></div>
    <div class="paper-meta">
      📅 2023-09-11
    </div>
    <details class="paper-abstract">
      This report describes trade-offs in the design of international governance arrangements for civilian artificial intelligence (AI) and presents one approach in detail. This approach represents the extension of a standards, licensing, and liability regime to the global level. We propose that states establish an International AI Organization (IAIO) to certify state jurisdictions (not firms or AI projects) for compliance with international oversight standards. States can give force to these international standards by adopting regulations prohibiting the import of goods whose supply chains embody AI from non-IAIO-certified jurisdictions. This borrows attributes from models of existing international organizations, such as the International Civilian Aviation Organization (ICAO), the International Maritime Organization (IMO), and the Financial Action Task Force (FATF). States can also adopt multilateral controls on the export of AI product inputs, such as specialized hardware, to non-certified jurisdictions. Indeed, both the import and export standards could be required for certification. As international actors reach consensus on risks of and minimum standards for advanced AI, a jurisdictional certification regime could mitigate a broad range of potential harms, including threats to public safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2304.09399v3">Designing a realistic peer-like embodied conversational agent for supporting children's storytelling</a></div>
    <div class="paper-meta">
      📅 2023-09-04
      | 💬 6 pages with 2 figures. The paper has been peer-reviewed and presented at the "CHI 2023 Workshop on Child-centred AI Design: Definition, Operation and Considerations, April 23, 2023, Hamburg, Germany
    </div>
    <details class="paper-abstract">
      Advances in artificial intelligence have facilitated the use of large language models (LLMs) and AI-generated synthetic media in education, which may inspire HCI researchers to develop technologies, in particular, embodied conversational agents (ECAs) to simulate the kind of scaffolding children might receive from a human partner. In this paper, we will propose a design prototype of a peer-like ECA named STARie that integrates multiple AI models - GPT-3, Speech Synthesis (Real-time Voice Cloning), VOCA (Voice Operated Character Animation), and FLAME (Faces Learned with an Articulated Model and Expressions) that aims to support narrative production in collaborative storytelling, specifically for children aged 4-8. However, designing a child-centered ECA raises concerns about age appropriateness, children privacy, gender choices of ECAs, and the uncanny valley effect. Thus, this paper will also discuss considerations and ethical concerns that must be taken into account when designing such an ECA. This proposal offers insights into the potential use of AI-generated synthetic media in child-centered AI design and how peer-like AI embodiment may support children\textquotesingle s storytelling.
    </details>
</div>
