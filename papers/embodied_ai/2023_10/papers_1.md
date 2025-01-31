# embodied ai - 2023_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11772v2">Neural Foundations of Mental Simulation: Future Prediction of Latent Representations on Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2023-10-25
      | 💬 20 pages, 10 figures, NeurIPS 2023 Camera Ready Version (spotlight)
    </div>
    <details class="paper-abstract">
      Humans and animals have a rich and flexible understanding of the physical world, which enables them to infer the underlying dynamical trajectories of objects and events, plausible future states, and use that to plan and anticipate the consequences of actions. However, the neural mechanisms underlying these computations are unclear. We combine a goal-driven modeling approach with dense neurophysiological data and high-throughput human behavioral readouts to directly impinge on this question. Specifically, we construct and evaluate several classes of sensory-cognitive networks to predict the future state of rich, ethologically-relevant environments, ranging from self-supervised end-to-end models with pixel-wise or object-centric objectives, to models that future predict in the latent space of purely static image-based or dynamic video-based pretrained foundation models. We find strong differentiation across these model classes in their ability to predict neural and behavioral data both within and across diverse environments. In particular, we find that neural responses are currently best predicted by models trained to predict the future state of their environment in the latent space of pretrained foundation models optimized for dynamic scenes in a self-supervised manner. Notably, models that future predict in the latent space of video foundation models that are optimized to support a diverse range of sensorimotor tasks, reasonably match both human behavioral error patterns and neural dynamics across all environmental scenarios that we were able to test. Overall, these findings suggest that the neural mechanisms and behaviors of primate mental simulation are thus far most consistent with being optimized to future predict on dynamic, reusable visual representations that are useful for Embodied AI more generally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13398v1">OpenAnnotate3D: Open-Vocabulary Auto-Labeling System for Multi-modal 3D Data</a></div>
    <div class="paper-meta">
      📅 2023-10-20
      | 💬 The source code will be released at https://github.com/Fudan-ProjectTitan/OpenAnnotate3D
    </div>
    <details class="paper-abstract">
      In the era of big data and large models, automatic annotating functions for multi-modal data are of great significance for real-world AI-driven applications, such as autonomous driving and embodied AI. Unlike traditional closed-set annotation, open-vocabulary annotation is essential to achieve human-level cognition capability. However, there are few open-vocabulary auto-labeling systems for multi-modal 3D data. In this paper, we introduce OpenAnnotate3D, an open-source open-vocabulary auto-labeling system that can automatically generate 2D masks, 3D masks, and 3D bounding box annotations for vision and point cloud data. Our system integrates the chain-of-thought capabilities of Large Language Models (LLMs) and the cross-modality capabilities of vision-language models (VLMs). To the best of our knowledge, OpenAnnotate3D is one of the pioneering works for open-vocabulary multi-modal 3D auto-labeling. We conduct comprehensive evaluations on both public and in-house real-world datasets, which demonstrate that the system significantly improves annotation efficiency compared to manual annotation while providing accurate open-vocabulary auto-annotating results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13724v1">Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots</a></div>
    <div class="paper-meta">
      📅 2023-10-19
      | 💬 Project page: http://aihabitat.org/habitat3
    </div>
    <details class="paper-abstract">
      We present Habitat 3.0: a simulation platform for studying collaborative human-robot tasks in home environments. Habitat 3.0 offers contributions across three dimensions: (1) Accurate humanoid simulation: addressing challenges in modeling complex deformable bodies and diversity in appearance and motion, all while ensuring high simulation speed. (2) Human-in-the-loop infrastructure: enabling real human interaction with simulated robots via mouse/keyboard or a VR interface, facilitating evaluation of robot policies with human input. (3) Collaborative tasks: studying two collaborative tasks, Social Navigation and Social Rearrangement. Social Navigation investigates a robot's ability to locate and follow humanoid avatars in unseen environments, whereas Social Rearrangement addresses collaboration between a humanoid and robot while rearranging a scene. These contributions allow us to study end-to-end learned and heuristic baselines for human-robot collaboration in-depth, as well as evaluate them with humans in the loop. Our experiments demonstrate that learned robot policies lead to efficient task completion when collaborating with unseen humanoid agents and human partners that might exhibit behaviors that the robot has not seen before. Additionally, we observe emergent behaviors during collaborative task execution, such as the robot yielding space when obstructing a humanoid agent, thereby allowing the effective completion of the task by the humanoid agent. Furthermore, our experiments using the human-in-the-loop tool demonstrate that our automated evaluation with humanoids can provide an indication of the relative ordering of different policies when evaluated with real human collaborators. Habitat 3.0 unlocks interesting new features in simulators for Embodied AI, and we hope it paves the way for a new frontier of embodied human-AI interaction capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.06125v3">Learning Hierarchical Interactive Multi-Object Search for Mobile Manipulation</a></div>
    <div class="paper-meta">
      📅 2023-10-19
      | 💬 11 pages, 6 figures, Accepted for publication in RA-L. Code and Models: http://himos.cs.uni-freiburg.de/
    </div>
    <details class="paper-abstract">
      Existing object-search approaches enable robots to search through free pathways, however, robots operating in unstructured human-centered environments frequently also have to manipulate the environment to their needs. In this work, we introduce a novel interactive multi-object search task in which a robot has to open doors to navigate rooms and search inside cabinets and drawers to find target objects. These new challenges require combining manipulation and navigation skills in unexplored environments. We present HIMOS, a hierarchical reinforcement learning approach that learns to compose exploration, navigation, and manipulation skills. To achieve this, we design an abstract high-level action space around a semantic map memory and leverage the explored environment as instance navigation points. We perform extensive experiments in simulation and the real world that demonstrate that, with accurate perception, the decision making of HIMOS effectively transfers to new environments in a zero-shot manner. It shows robustness to unseen subpolicies, failures in their execution, and different robot kinematics. These capabilities open the door to a wide range of downstream tasks across embodied AI and real-world use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2109.10493v3">Benchmarking Augmentation Methods for Learning Robust Navigation Agents: the Winning Entry of the 2021 iGibson Challenge</a></div>
    <div class="paper-meta">
      📅 2023-10-12
    </div>
    <details class="paper-abstract">
      Recent advances in deep reinforcement learning and scalable photorealistic simulation have led to increasingly mature embodied AI for various visual tasks, including navigation. However, while impressive progress has been made for teaching embodied agents to navigate static environments, much less progress has been made on more dynamic environments that may include moving pedestrians or movable obstacles. In this study, we aim to benchmark different augmentation techniques for improving the agent's performance in these challenging environments. We show that adding several dynamic obstacles into the scene during training confers significant improvements in test-time generalization, achieving much higher success rates than baseline agents. We find that this approach can also be combined with image augmentation methods to achieve even higher success rates. Additionally, we show that this approach is also more robust to sim-to-sim transfer than image augmentation methods. Finally, we demonstrate the effectiveness of this dynamic obstacle augmentation approach by using it to train an agent for the 2021 iGibson Challenge at CVPR, where it achieved 1st place for Interactive Navigation. Video link: https://www.youtube.com/watch?v=HxUX2HeOSE4
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2301.03213v5">EgoTracks: A Long-term Egocentric Visual Object Tracking Dataset</a></div>
    <div class="paper-meta">
      📅 2023-10-01
    </div>
    <details class="paper-abstract">
      Visual object tracking is a key component to many egocentric vision problems. However, the full spectrum of challenges of egocentric tracking faced by an embodied AI is underrepresented in many existing datasets; these tend to focus on relatively short, third-person videos. Egocentric video has several distinguishing characteristics from those commonly found in past datasets: frequent large camera motions and hand interactions with objects commonly lead to occlusions or objects exiting the frame, and object appearance can change rapidly due to widely different points of view, scale, or object states. Embodied tracking is also naturally long-term, and being able to consistently (re-)associate objects to their appearances and disappearances over as long as a lifetime is critical. Previous datasets under-emphasize this re-detection problem, and their "framed" nature has led to adoption of various spatiotemporal priors that we find do not necessarily generalize to egocentric video. We thus introduce EgoTracks, a new dataset for long-term egocentric visual object tracking. Sourced from the Ego4D dataset, this new dataset presents a significant challenge to recent state-of-the-art single-object tracking models, which we find score poorly on traditional tracking metrics for our new dataset, compared to popular benchmarks. We further show improvements that can be made to a STARK tracker to significantly increase its performance on egocentric data, resulting in a baseline model we call EgoSTARK. We publicly release our annotations and benchmark, hoping our dataset leads to further advancements in tracking.
    </details>
</div>
