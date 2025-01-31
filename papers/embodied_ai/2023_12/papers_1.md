# embodied ai - 2023_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01824v2">Mini-BEHAVIOR: A Procedurally Generated Benchmark for Long-horizon Decision-Making in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-12-27
    </div>
    <details class="paper-abstract">
      We present Mini-BEHAVIOR, a novel benchmark for embodied AI that challenges agents to use reasoning and decision-making skills to solve complex activities that resemble everyday human challenges. The Mini-BEHAVIOR environment is a fast, realistic Gridworld environment that offers the benefits of rapid prototyping and ease of use while preserving a symbolic level of physical realism and complexity found in complex embodied AI benchmarks. We introduce key features such as procedural generation, to enable the creation of countless task variations and support open-ended learning. Mini-BEHAVIOR provides implementations of various household tasks from the original BEHAVIOR benchmark, along with starter code for data collection and reinforcement learning agent training. In essence, Mini-BEHAVIOR offers a fast, open-ended benchmark for evaluating decision-making and planning solutions in embodied AI. It serves as a user-friendly entry point for research and facilitates the evaluation and development of solutions, simplifying their assessment and development while advancing the field of embodied AI. Code is publicly available at https://github.com/StanfordVL/mini_behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.16170v1">EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-12-26
      | 💬 A multi-modal, ego-centric 3D perception dataset and benchmark for holistic 3D scene understanding. Project page: http://tai-wang.github.io/embodiedscan
    </div>
    <details class="paper-abstract">
      In the realm of computer vision and robotics, embodied agents are expected to explore their environment and carry out human instructions. This necessitates the ability to fully understand 3D scenes given their first-person observations and contextualize them into language for interaction. However, traditional research focuses more on scene-level input and output setups from a global view. To address the gap, we introduce EmbodiedScan, a multi-modal, ego-centric 3D perception dataset and benchmark for holistic 3D scene understanding. It encompasses over 5k scans encapsulating 1M ego-centric RGB-D views, 1M language prompts, 160k 3D-oriented boxes spanning over 760 categories, some of which partially align with LVIS, and dense semantic occupancy with 80 common categories. Building upon this database, we introduce a baseline framework named Embodied Perceptron. It is capable of processing an arbitrary number of multi-modal inputs and demonstrates remarkable 3D perception capabilities, both within the two series of benchmarks we set up, i.e., fundamental 3D perception tasks and language-grounded tasks, and in the wild. Codes, datasets, and benchmarks will be available at https://github.com/OpenRobotLab/EmbodiedScan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.06141v5">Active Semantic Localization with Graph Neural Embedding</a></div>
    <div class="paper-meta">
      📅 2023-12-26
      | 💬 ACPR2023 (extended version)
    </div>
    <details class="paper-abstract">
      Semantic localization, i.e., robot self-localization with semantic image modality, is critical in recently emerging embodied AI applications (e.g., point-goal navigation, object-goal navigation, vision language navigation) and topological mapping applications (e.g., graph neural SLAM, ego-centric topological map). However, most existing works on semantic localization focus on passive vision tasks without viewpoint planning, or rely on additional rich modalities (e.g., depth measurements). Thus, the problem is largely unsolved. In this work, we explore a lightweight, entirely CPU-based, domain-adaptive semantic localization framework, called graph neural localizer. Our approach is inspired by two recently emerging technologies: (1) Scene graph, which combines the viewpoint- and appearance- invariance of local and global features; (2) Graph neural network, which enables direct learning/recognition of graph data (i.e., non-vector data). Specifically, a graph convolutional neural network is first trained as a scene graph classifier for passive vision, and then its knowledge is transferred to a reinforcement-learning planner for active vision. Experiments on two scenarios, self-supervised learning and unsupervised domain adaptation, using a photo-realistic Habitat simulator validate the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.10531v3">GenPose: Generative Category-level Object Pose Estimation via Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2023-12-25
    </div>
    <details class="paper-abstract">
      Object pose estimation plays a vital role in embodied AI and computer vision, enabling intelligent agents to comprehend and interact with their surroundings. Despite the practicality of category-level pose estimation, current approaches encounter challenges with partially observed point clouds, known as the multihypothesis issue. In this study, we propose a novel solution by reframing categorylevel object pose estimation as conditional generative modeling, departing from traditional point-to-point regression. Leveraging score-based diffusion models, we estimate object poses by sampling candidates from the diffusion model and aggregating them through a two-step process: filtering out outliers via likelihood estimation and subsequently mean-pooling the remaining candidates. To avoid the costly integration process when estimating the likelihood, we introduce an alternative method that trains an energy-based model from the original score-based model, enabling end-to-end likelihood estimation. Our approach achieves state-of-the-art performance on the REAL275 dataset, surpassing 50% and 60% on strict 5d2cm and 5d5cm metrics, respectively. Furthermore, our method demonstrates strong generalizability to novel categories sharing similar symmetric properties without fine-tuning and can readily adapt to object pose tracking tasks, yielding comparable results to the current state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09337v1">Promptable Behaviors: Personalizing Multi-Objective Rewards from Human Preferences</a></div>
    <div class="paper-meta">
      📅 2023-12-14
    </div>
    <details class="paper-abstract">
      Customizing robotic behaviors to be aligned with diverse human preferences is an underexplored challenge in the field of embodied AI. In this paper, we present Promptable Behaviors, a novel framework that facilitates efficient personalization of robotic agents to diverse human preferences in complex environments. We use multi-objective reinforcement learning to train a single policy adaptable to a broad spectrum of preferences. We introduce three distinct methods to infer human preferences by leveraging different types of interactions: (1) human demonstrations, (2) preference feedback on trajectory comparisons, and (3) language instructions. We evaluate the proposed method in personalized object-goal navigation and flee navigation tasks in ProcTHOR and RoboTHOR, demonstrating the ability to prompt agent behaviors to satisfy human preferences in various scenarios. Project page: https://promptable-behaviors.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08611v1">UniTeam: Open Vocabulary Mobile Manipulation Challenge</a></div>
    <div class="paper-meta">
      📅 2023-12-14
    </div>
    <details class="paper-abstract">
      This report introduces our UniTeam agent - an improved baseline for the "HomeRobot: Open Vocabulary Mobile Manipulation" challenge. The challenge poses problems of navigation in unfamiliar environments, manipulation of novel objects, and recognition of open-vocabulary object classes. This challenge aims to facilitate cross-cutting research in embodied AI using recent advances in machine learning, computer vision, natural language, and robotics. In this work, we conducted an exhaustive evaluation of the provided baseline agent; identified deficiencies in perception, navigation, and manipulation skills; and improved the baseline agent's performance. Notably, enhancements were made in perception - minimizing misclassifications; navigation - preventing infinite loop commitments; picking - addressing failures due to changing object visibility; and placing - ensuring accurate positioning for successful object placement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07886v1">Modality Plug-and-Play: Elastic Modality Adaptation in Multimodal LLMs for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-12-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are capable of reasoning over diverse input data modalities through pre-trained encoders. However, the growing diversity of input data modalities prevents incorporating all modalities into LLMs, especially when LLMs are deployed on resource-constrained edge devices for embodied AI applications. Instead, a better option is to adaptively involve only the useful modalities at runtime, depending on the current environmental contexts and task requirements. For such modality adaptation, existing work adopts fixed connections between encoders and the LLM's input layer, leading to high training cost at runtime and ineffective cross-modal interaction. In this paper, we address these limitations by presenting mPnP-LLM, a new technique that allows fully elastic, automated and prompt runtime modality adaptation, by connecting unimodal encoders to a flexible set of last LLM blocks and making such latent connections fully trainable at runtime. Experiments over the nuScenes-QA dataset show that mPnP-LLM can achieve up to 3.7x FLOPs reduction and 30% GPU memory usage reduction, while retaining on-par accuracy with the existing schemes. Under the same compute budget, mPnP-LLM improves the task accuracy by up to 4% compared to the best existing scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.05822v1">Toward Open-ended Embodied Tasks Solving</a></div>
    <div class="paper-meta">
      📅 2023-12-10
    </div>
    <details class="paper-abstract">
      Empowering embodied agents, such as robots, with Artificial Intelligence (AI) has become increasingly important in recent years. A major challenge is task open-endedness. In practice, robots often need to perform tasks with novel goals that are multifaceted, dynamic, lack a definitive "end-state", and were not encountered during training. To tackle this problem, this paper introduces \textit{Diffusion for Open-ended Goals} (DOG), a novel framework designed to enable embodied AI to plan and act flexibly and dynamically for open-ended task goals. DOG synergizes the generative prowess of diffusion models with state-of-the-art, training-free guidance techniques to adaptively perform online planning and control. Our evaluations demonstrate that DOG can handle various kinds of novel task goals not seen during training, in both maze navigation and robot control problems. Our work sheds light on enhancing embodied AI's adaptability and competency in tackling open-ended goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.10069v1">Understanding Representations Pretrained with Auxiliary Losses for Embodied Agent Planning</a></div>
    <div class="paper-meta">
      📅 2023-12-06
    </div>
    <details class="paper-abstract">
      Pretrained representations from large-scale vision models have boosted the performance of downstream embodied policy learning. We look to understand whether additional self-supervised pretraining on exploration trajectories can build on these general-purpose visual representations to better support embodied planning in realistic environments. We evaluated four common auxiliary losses in embodied AI, two hindsight-based losses, and a standard imitation learning loss, by pretraining the agent's visual compression module and state belief representations with each objective and using CLIP as a representative visual backbone. The learned representations are then frozen for downstream multi-step evaluation on two goal-directed tasks. Surprisingly, we find that imitation learning on these exploration trajectories out-performs all other auxiliary losses even despite the exploration trajectories being dissimilar from the downstream tasks. This suggests that imitation of exploration may be ''all you need'' for building powerful planning representations. Additionally, we find that popular auxiliary losses can benefit from simple modifications to improve their support for downstream planning ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.01097v1">Planning as In-Painting: A Diffusion-Based Embodied Task Planning Framework for Environments under Uncertainty</a></div>
    <div class="paper-meta">
      📅 2023-12-02
    </div>
    <details class="paper-abstract">
      Task planning for embodied AI has been one of the most challenging problems where the community does not meet a consensus in terms of formulation. In this paper, we aim to tackle this problem with a unified framework consisting of an end-to-end trainable method and a planning algorithm. Particularly, we propose a task-agnostic method named 'planning as in-painting'. In this method, we use a Denoising Diffusion Model (DDM) for plan generation, conditioned on both language instructions and perceptual inputs under partially observable environments. Partial observation often leads to the model hallucinating the planning. Therefore, our diffusion-based method jointly models both state trajectory and goal estimation to improve the reliability of the generated plan, given the limited available information at each step. To better leverage newly discovered information along the plan execution for a higher success rate, we propose an on-the-fly planning algorithm to collaborate with the diffusion-based planner. The proposed framework achieves promising performances in various embodied AI tasks, including vision-language navigation, object manipulation, and task planning in a photorealistic virtual environment. The code is available at: https://github.com/joeyy5588/planning-as-inpainting.
    </details>
</div>
