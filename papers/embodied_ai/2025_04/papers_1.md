# embodied ai - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02247v2">WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Object Goal Navigation-requiring an agent to locate a specific object in an unseen environment-remains a core challenge in embodied AI. Although recent progress in Vision-Language Model (VLM)-based agents has demonstrated promising perception and decision-making abilities through prompting, none has yet established a fully modular world model design that reduces risky and costly interactions with the environment by predicting the future state of the world. We introduce WMNav, a novel World Model-based Navigation framework powered by Vision-Language Models (VLMs). It predicts possible outcomes of decisions and builds memories to provide feedback to the policy module. To retain the predicted state of the environment, WMNav proposes the online maintained Curiosity Value Map as part of the world model memory to provide dynamic configuration for navigation policy. By decomposing according to a human-like thinking process, WMNav effectively alleviates the impact of model hallucination by making decisions based on the feedback difference between the world model plan and observation. To further boost efficiency, we implement a two-stage action proposer strategy: broad exploration followed by precise localization. Extensive evaluation on HM3D and MP3D validates WMNav surpasses existing zero-shot benchmarks in both success rate and exploration efficiency (absolute improvement: +3.2% SR and +3.2% SPL on HM3D, +13.5% SR and +1.1% SPL on MP3D). Project page: https://b0b8k1ng.github.io/WMNav/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11218v2">3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 The first large-scale 3D Gaussians Affordance Reasoning Benchmark
    </div>
    <details class="paper-abstract">
      3D affordance reasoning is essential in associating human instructions with the functional regions of 3D objects, facilitating precise, task-oriented manipulations in embodied AI. However, current methods, which predominantly depend on sparse 3D point clouds, exhibit limited generalizability and robustness due to their sensitivity to coordinate variations and the inherent sparsity of the data. By contrast, 3D Gaussian Splatting (3DGS) delivers high-fidelity, real-time rendering with minimal computational overhead by representing scenes as dense, continuous distributions. This positions 3DGS as a highly effective approach for capturing fine-grained affordance details and improving recognition accuracy. Nevertheless, its full potential remains largely untapped due to the absence of large-scale, 3DGS-specific affordance datasets. To overcome these limitations, we present 3DAffordSplat, the first large-scale, multi-modal dataset tailored for 3DGS-based affordance reasoning. This dataset includes 23,677 Gaussian instances, 8,354 point cloud instances, and 6,631 manually annotated affordance labels, encompassing 21 object categories and 18 affordance types. Building upon this dataset, we introduce AffordSplatNet, a novel model specifically designed for affordance reasoning using 3DGS representations. AffordSplatNet features an innovative cross-modal structure alignment module that exploits structural consistency priors to align 3D point cloud and 3DGS representations, resulting in enhanced affordance recognition accuracy. Extensive experiments demonstrate that the 3DAffordSplat dataset significantly advances affordance learning within the 3DGS domain, while AffordSplatNet consistently outperforms existing methods across both seen and unseen settings, highlighting its robust generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11218v1">3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 The first large-scale 3D Gaussians Affordance Reasoning Benchmark
    </div>
    <details class="paper-abstract">
      3D affordance reasoning is essential in associating human instructions with the functional regions of 3D objects, facilitating precise, task-oriented manipulations in embodied AI. However, current methods, which predominantly depend on sparse 3D point clouds, exhibit limited generalizability and robustness due to their sensitivity to coordinate variations and the inherent sparsity of the data. By contrast, 3D Gaussian Splatting (3DGS) delivers high-fidelity, real-time rendering with minimal computational overhead by representing scenes as dense, continuous distributions. This positions 3DGS as a highly effective approach for capturing fine-grained affordance details and improving recognition accuracy. Nevertheless, its full potential remains largely untapped due to the absence of large-scale, 3DGS-specific affordance datasets. To overcome these limitations, we present 3DAffordSplat, the first large-scale, multi-modal dataset tailored for 3DGS-based affordance reasoning. This dataset includes 23,677 Gaussian instances, 8,354 point cloud instances, and 6,631 manually annotated affordance labels, encompassing 21 object categories and 18 affordance types. Building upon this dataset, we introduce AffordSplatNet, a novel model specifically designed for affordance reasoning using 3DGS representations. AffordSplatNet features an innovative cross-modal structure alignment module that exploits structural consistency priors to align 3D point cloud and 3DGS representations, resulting in enhanced affordance recognition accuracy. Extensive experiments demonstrate that the 3DAffordSplat dataset significantly advances affordance learning within the 3DGS domain, while AffordSplatNet consistently outperforms existing methods across both seen and unseen settings, highlighting its robust generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08306v4">Reasoning in visual navigation of end-to-end trained agents: a dynamical systems approach</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Progress in Embodied AI has made it possible for end-to-end-trained agents to navigate in photo-realistic environments with high-level reasoning and zero-shot or language-conditioned behavior, but benchmarks are still dominated by simulation. In this work, we focus on the fine-grained behavior of fast-moving real robots and present a large-scale experimental study involving \numepisodes{} navigation episodes in a real environment with a physical robot, where we analyze the type of reasoning emerging from end-to-end training. In particular, we study the presence of realistic dynamics which the agent learned for open-loop forecasting, and their interplay with sensing. We analyze the way the agent uses latent memory to hold elements of the scene structure and information gathered during exploration. We probe the planning capabilities of the agent, and find in its memory evidence for somewhat precise plans over a limited horizon. Furthermore, we show in a post-hoc analysis that the value function learned by the agent relates to long-term planning. Put together, our experiments paint a new picture on how using tools from computer vision and sequential decision making have led to new capabilities in robotics and control. An interactive tool is available at europe.naverlabs.com/research/publications/reasoning-in-visual-navigation-of-end-to-end-trained-agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13201v1">Concept Enhancement Engineering: A Lightweight and Efficient Robust Defense Against Jailbreak Attacks in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Embodied Intelligence (EI) systems integrated with large language models (LLMs) face significant security risks, particularly from jailbreak attacks that manipulate models into generating harmful outputs or executing unsafe physical actions. Traditional defense strategies, such as input filtering and output monitoring, often introduce high computational overhead or interfere with task performance in real-time embodied scenarios. To address these challenges, we propose Concept Enhancement Engineering (CEE), a novel defense framework that leverages representation engineering to enhance the safety of embodied LLMs by dynamically steering their internal activations. CEE operates by (1) extracting multilingual safety patterns from model activations, (2) constructing control directions based on safety-aligned concept subspaces, and (3) applying subspace concept rotation to reinforce safe behavior during inference. Our experiments demonstrate that CEE effectively mitigates jailbreak attacks while maintaining task performance, outperforming existing defense methods in both robustness and efficiency. This work contributes a scalable and interpretable safety mechanism for embodied AI, bridging the gap between theoretical representation engineering and practical security applications. Our findings highlight the potential of latent-space interventions as a viable defense paradigm against emerging adversarial threats in physically grounded AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10414v1">HUMOTO: A 4D Dataset of Mocap Human Object Interactions</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 19 pages, 15 figures
    </div>
    <details class="paper-abstract">
      We present Human Motions with Objects (HUMOTO), a high-fidelity dataset of human-object interactions for motion generation, computer vision, and robotics applications. Featuring 736 sequences (7,875 seconds at 30 fps), HUMOTO captures interactions with 63 precisely modeled objects and 72 articulated parts. Our innovations include a scene-driven LLM scripting pipeline creating complete, purposeful tasks with natural progression, and a mocap-and-camera recording setup to effectively handle occlusions. Spanning diverse activities from cooking to outdoor picnics, HUMOTO preserves both physical accuracy and logical task flow. Professional artists rigorously clean and verify each sequence, minimizing foot sliding and object penetrations. We also provide benchmarks compared to other datasets. HUMOTO's comprehensive full-body motion and simultaneous multi-object interactions address key data-capturing challenges and provide opportunities to advance realistic human-object interaction modeling across research domains with practical applications in animation, robotics, and embodied AI systems. Project: https://jiaxin-lu.github.io/humoto/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09927v1">Efficient Task-specific Conditional Diffusion Policies: Shortcut Model Acceleration and SO(3) Optimization</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 Accepted to CVPR 2025 Workshop on 2nd MEIS
    </div>
    <details class="paper-abstract">
      Imitation learning, particularly Diffusion Policies based methods, has recently gained significant traction in embodied AI as a powerful approach to action policy generation. These models efficiently generate action policies by learning to predict noise. However, conventional Diffusion Policy methods rely on iterative denoising, leading to inefficient inference and slow response times, which hinder real-time robot control. To address these limitations, we propose a Classifier-Free Shortcut Diffusion Policy (CF-SDP) that integrates classifier-free guidance with shortcut-based acceleration, enabling efficient task-specific action generation while significantly improving inference speed. Furthermore, we extend diffusion modeling to the SO(3) manifold in shortcut model, defining the forward and reverse processes in its tangent space with an isotropic Gaussian distribution. This ensures stable and accurate rotational estimation, enhancing the effectiveness of diffusion-based control. Our approach achieves nearly 5x acceleration in diffusion inference compared to DDIM-based Diffusion Policy while maintaining task performance. Evaluations both on the RoboTwin simulation platform and real-world scenarios across various tasks demonstrate the superiority of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09587v1">GeoNav: Empowering MLLMs with Explicit Geospatial Reasoning Abilities for Language-Goal Aerial Navigation</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Language-goal aerial navigation is a critical challenge in embodied AI, requiring UAVs to localize targets in complex environments such as urban blocks based on textual specification. Existing methods, often adapted from indoor navigation, struggle to scale due to limited field of view, semantic ambiguity among objects, and lack of structured spatial reasoning. In this work, we propose GeoNav, a geospatially aware multimodal agent to enable long-range navigation. GeoNav operates in three phases-landmark navigation, target search, and precise localization-mimicking human coarse-to-fine spatial strategies. To support such reasoning, it dynamically builds two different types of spatial memory. The first is a global but schematic cognitive map, which fuses prior textual geographic knowledge and embodied visual cues into a top-down, annotated form for fast navigation to the landmark region. The second is a local but delicate scene graph representing hierarchical spatial relationships between blocks, landmarks, and objects, which is used for definite target localization. On top of this structured representation, GeoNav employs a spatially aware, multimodal chain-of-thought prompting mechanism to enable multimodal large language models with efficient and interpretable decision-making across stages. On the CityNav urban navigation benchmark, GeoNav surpasses the current state-of-the-art by up to 12.53% in success rate and significantly improves navigation efficiency, even in hard-level tasks. Ablation studies highlight the importance of each module, showcasing how geospatial representations and coarse-to-fine reasoning enhance UAV navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08581v1">FMLGS: Fast Multilevel Language Embedded Gaussians for Part-level Interactive Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      The semantically interactive radiance field has long been a promising backbone for 3D real-world applications, such as embodied AI to achieve scene understanding and manipulation. However, multi-granularity interaction remains a challenging task due to the ambiguity of language and degraded quality when it comes to queries upon object components. In this work, we present FMLGS, an approach that supports part-level open-vocabulary query within 3D Gaussian Splatting (3DGS). We propose an efficient pipeline for building and querying consistent object- and part-level semantics based on Segment Anything Model 2 (SAM2). We designed a semantic deviation strategy to solve the problem of language ambiguity among object parts, which interpolates the semantic features of fine-grained targets for enriched information. Once trained, we can query both objects and their describable parts using natural language. Comparisons with other state-of-the-art methods prove that our method can not only better locate specified part-level targets, but also achieve first-place performance concerning both speed and accuracy, where FMLGS is 98 x faster than LERF, 4 x faster than LangSplat and 2.5 x faster than LEGaussians. Meanwhile, we further integrate FMLGS as a virtual agent that can interactively navigate through 3D scenes, locate targets, and respond to user demands through a chat interface, which demonstrates the potential of our work to be further expanded and applied in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11858v2">EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have shown significant advancements, providing a promising future for embodied agents. Existing benchmarks for evaluating MLLMs primarily utilize static images or videos, limiting assessments to non-interactive scenarios. Meanwhile, existing embodied AI benchmarks are task-specific and not diverse enough, which do not adequately evaluate the embodied capabilities of MLLMs. To address this, we propose EmbodiedEval, a comprehensive and interactive evaluation benchmark for MLLMs with embodied tasks. EmbodiedEval features 328 distinct tasks within 125 varied 3D scenes, each of which is rigorously selected and annotated. It covers a broad spectrum of existing embodied AI tasks with significantly enhanced diversity, all within a unified simulation and evaluation framework tailored for MLLMs. The tasks are organized into five categories: navigation, object interaction, social interaction, attribute question answering, and spatial question answering to assess different capabilities of the agents. We evaluated the state-of-the-art MLLMs on EmbodiedEval and found that they have a significant shortfall compared to human level on embodied tasks. Our analysis demonstrates the limitations of existing MLLMs in embodied capabilities, providing insights for their future development. We open-source all evaluation data and simulation framework at https://github.com/thunlp/EmbodiedEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06736v5">MrSteve: Instruction-Following Agents in Minecraft with What-Where-When Memory</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Significant advances have been made in developing general-purpose embodied AI in environments like Minecraft through the adoption of LLM-augmented hierarchical approaches. While these approaches, which combine high-level planners with low-level controllers, show promise, low-level controllers frequently become performance bottlenecks due to repeated failures. In this paper, we argue that the primary cause of failure in many low-level controllers is the absence of an episodic memory system. To address this, we introduce MrSteve (Memory Recall Steve), a novel low-level controller equipped with Place Event Memory (PEM), a form of episodic memory that captures what, where, and when information from episodes. This directly addresses the main limitation of the popular low-level controller, Steve-1. Unlike previous models that rely on short-term memory, PEM organizes spatial and event-based data, enabling efficient recall and navigation in long-horizon tasks. Additionally, we propose an Exploration Strategy and a Memory-Augmented Task Solving Framework, allowing agents to alternate between exploration and task-solving based on recalled events. Our approach significantly improves task-solving and exploration efficiency compared to existing methods. We will release our code and demos on the project page: https://sites.google.com/view/mr-steve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23765v2">STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      The use of Multimodal Large Language Models (MLLMs) as an end-to-end solution for Embodied AI and Autonomous Driving has become a prevailing trend. While MLLMs have been extensively studied for visual semantic understanding tasks, their ability to perform precise and quantitative spatial-temporal understanding in real-world applications remains largely unexamined, leading to uncertain prospects. To evaluate models' Spatial-Temporal Intelligence, we introduce STI-Bench, a benchmark designed to evaluate MLLMs' spatial-temporal understanding through challenging tasks such as estimating and predicting the appearance, pose, displacement, and motion of objects. Our benchmark encompasses a wide range of robot and vehicle operations across desktop, indoor, and outdoor scenarios. The extensive experiments reveals that the state-of-the-art MLLMs still struggle in real-world spatial-temporal understanding, especially in tasks requiring precise distance estimation and motion analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16310v4">Functionality understanding and segmentation in 3D scenes</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 CVPR 2025 Highlight. Camera ready version. 20 pages, 12 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Understanding functionalities in 3D scenes involves interpreting natural language descriptions to locate functional interactive objects, such as handles and buttons, in a 3D environment. Functionality understanding is highly challenging, as it requires both world knowledge to interpret language and spatial perception to identify fine-grained objects. For example, given a task like 'turn on the ceiling light', an embodied AI agent must infer that it needs to locate the light switch, even though the switch is not explicitly mentioned in the task description. To date, no dedicated methods have been developed for this problem. In this paper, we introduce Fun3DU, the first approach designed for functionality understanding in 3D scenes. Fun3DU uses a language model to parse the task description through Chain-of-Thought reasoning in order to identify the object of interest. The identified object is segmented across multiple views of the captured scene by using a vision and language model. The segmentation results from each view are lifted in 3D and aggregated into the point cloud using geometric information. Fun3DU is training-free, relying entirely on pre-trained models. We evaluate Fun3DU on SceneFun3D, the most recent and only dataset to benchmark this task, which comprises over 3000 task descriptions on 230 scenes. Our method significantly outperforms state-of-the-art open-vocabulary 3D segmentation approaches. Project page: https://tev-fbk.github.io/fun3du/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22869v2">SIGHT: Single-Image Conditioned Generation of Hand Trajectories for Hand-Object Interaction</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      We introduce a novel task of generating realistic and diverse 3D hand trajectories given a single image of an object, which could be involved in a hand-object interaction scene or pictured by itself. When humans grasp an object, appropriate trajectories naturally form in our minds to use it for specific tasks. Hand-object interaction trajectory priors can greatly benefit applications in robotics, embodied AI, augmented reality and related fields. However, synthesizing realistic and appropriate hand trajectories given a single object or hand-object interaction image is a highly ambiguous task, requiring to correctly identify the object of interest and possibly even the correct interaction among many possible alternatives. To tackle this challenging problem, we propose the SIGHT-Fusion system, consisting of a curated pipeline for extracting visual features of hand-object interaction details from egocentric videos involving object manipulation, and a diffusion-based conditional motion generation model processing the extracted features. We train our method given video data with corresponding hand trajectory annotations, without supervision in the form of action labels. For the evaluation, we establish benchmarks utilizing the first-person FPHAB and HOI4D datasets, testing our method against various baselines and using multiple metrics. We also introduce task simulators for executing the generated hand trajectories and reporting task success rates as an additional metric. Experiments show that our method generates more appropriate and realistic hand trajectories than baselines and presents promising generalization capability on unseen objects. The accuracy of the generated hand trajectories is confirmed in a physics simulation setting, showcasing the authenticity of the created sequences and their applicability in downstream uses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03629v1">SeGuE: Semantic Guided Exploration for Mobile Robots</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 6 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      The rise of embodied AI applications has enabled robots to perform complex tasks which require a sophisticated understanding of their environment. To enable successful robot operation in such settings, maps must be constructed so that they include semantic information, in addition to geometric information. In this paper, we address the novel problem of semantic exploration, whereby a mobile robot must autonomously explore an environment to fully map both its structure and the semantic appearance of features. We develop a method based on next-best-view exploration, where potential poses are scored based on the semantic features visible from that pose. We explore two alternative methods for sampling potential views and demonstrate the effectiveness of our framework in both simulation and physical experiments. Automatic creation of high-quality semantic maps can enable robots to better understand and interact with their environments and enable future embodied AI applications to be more easily deployed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17735v5">3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Constructing compact and informative 3D scene representations is essential for effective embodied exploration and reasoning, especially in complex environments over extended periods. Existing representations, such as object-centric 3D scene graphs, oversimplify spatial relationships by modeling scenes as isolated objects with restrictive textual relationships, making it difficult to address queries requiring nuanced spatial understanding. Moreover, these representations lack natural mechanisms for active exploration and memory management, hindering their application to lifelong autonomy. In this work, we propose 3D-Mem, a novel 3D scene memory framework for embodied agents. 3D-Mem employs informative multi-view images, termed Memory Snapshots, to represent the scene and capture rich visual information of explored regions. It further integrates frontier-based exploration by introducing Frontier Snapshots-glimpses of unexplored areas-enabling agents to make informed decisions by considering both known and potential new information. To support lifelong memory in active exploration settings, we present an incremental construction pipeline for 3D-Mem, as well as a memory retrieval technique for memory management. Experimental results on three benchmarks demonstrate that 3D-Mem significantly enhances agents' exploration and reasoning capabilities in 3D environments, highlighting its potential for advancing applications in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03153v1">MORAL: A Multimodal Reinforcement Learning Framework for Decision Making in Autonomous Laboratories</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 9 pages, 14 figures and 3 tables
    </div>
    <details class="paper-abstract">
      We propose MORAL (a multimodal reinforcement learning framework for decision making in autonomous laboratories) that enhances sequential decision-making in autonomous robotic laboratories through the integration of visual and textual inputs. Using the BridgeData V2 dataset, we generate fine-tuned image captions with a pretrained BLIP-2 vision-language model and combine them with visual features through an early fusion strategy. The fused representations are processed using Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents. Experimental results demonstrate that multimodal agents achieve a 20% improvement in task completion rates and significantly outperform visual-only and textual-only baselines after sufficient training. Compared to transformer-based and recurrent multimodal RL models, our approach achieves superior performance in cumulative reward and caption quality metrics (BLEU, METEOR, ROUGE-L). These results highlight the impact of semantically aligned language cues in enhancing agent learning efficiency and generalization. The proposed framework contributes to the advancement of multimodal reinforcement learning and embodied AI systems in dynamic, real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03800v1">Decision SpikeFormer: Spike-Driven Transformer for Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 This work has been accepted to CVPR 2025
    </div>
    <details class="paper-abstract">
      Offline reinforcement learning (RL) enables policy training solely on pre-collected data, avoiding direct environment interaction - a crucial benefit for energy-constrained embodied AI applications. Although Artificial Neural Networks (ANN)-based methods perform well in offline RL, their high computational and energy demands motivate exploration of more efficient alternatives. Spiking Neural Networks (SNNs) show promise for such tasks, given their low power consumption. In this work, we introduce DSFormer, the first spike-driven transformer model designed to tackle offline RL via sequence modeling. Unlike existing SNN transformers focused on spatial dimensions for vision tasks, we develop Temporal Spiking Self-Attention (TSSA) and Positional Spiking Self-Attention (PSSA) in DSFormer to capture the temporal and positional dependencies essential for sequence modeling in RL. Additionally, we propose Progressive Threshold-dependent Batch Normalization (PTBN), which combines the benefits of LayerNorm and BatchNorm to preserve temporal dependencies while maintaining the spiking nature of SNNs. Comprehensive results in the D4RL benchmark show DSFormer's superiority over both SNN and ANN counterparts, achieving 78.4% energy savings, highlighting DSFormer's advantages not only in energy efficiency but also in competitive performance. Code and models are public at https://wei-nijuan.github.io/DecisionSpikeFormer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19901v2">TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Synthesizing diverse and physically plausible Human-Scene Interactions (HSI) is pivotal for both computer animation and embodied AI. Despite encouraging progress, current methods mainly focus on developing separate controllers, each specialized for a specific interaction task. This significantly hinders the ability to tackle a wide variety of challenging HSI tasks that require the integration of multiple skills, e.g., sitting down while carrying an object. To address this issue, we present TokenHSI, a single, unified transformer-based policy capable of multi-skill unification and flexible adaptation. The key insight is to model the humanoid proprioception as a separate shared token and combine it with distinct task tokens via a masking mechanism. Such a unified policy enables effective knowledge sharing across skills, thereby facilitating the multi-task training. Moreover, our policy architecture supports variable length inputs, enabling flexible adaptation of learned skills to new scenarios. By training additional task tokenizers, we can not only modify the geometries of interaction targets but also coordinate multiple skills to address complex tasks. The experiments demonstrate that our approach can significantly improve versatility, adaptability, and extensibility in various HSI tasks. Website: https://liangpan99.github.io/TokenHSI/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00954v1">IDMR: Towards Instance-Driven Precise Visual Correspondence in Multimodal Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Multimodal retrieval systems are becoming increasingly vital for cutting-edge AI technologies, such as embodied AI and AI-driven digital content industries. However, current multimodal retrieval tasks lack sufficient complexity and demonstrate limited practical application value. It spires us to design Instance-Driven Multimodal Image Retrieval (IDMR), a novel task that requires models to retrieve images containing the same instance as a query image while matching a text-described scenario. Unlike existing retrieval tasks focused on global image similarity or category-level matching, IDMR demands fine-grained instance-level consistency across diverse contexts. To benchmark this capability, we develop IDMR-bench using real-world object tracking and first-person video data. Addressing the scarcity of training data, we propose a cross-domain synthesis method that creates 557K training samples by cropping objects from standard detection datasets. Our Multimodal Large Language Model (MLLM) based retrieval model, trained on 1.2M samples, outperforms state-of-the-art approaches on both traditional benchmarks and our zero-shot IDMR-bench. Experimental results demonstrate previous models' limitations in instance-aware retrieval and highlight the potential of MLLM for advanced retrieval applications. The whole training dataset, codes and models, with wide ranges of sizes, are available at https://github.com/BwLiu01/IDMR.
    </details>
</div>
