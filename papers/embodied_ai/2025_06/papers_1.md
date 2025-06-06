# embodied ai - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05341v1">Direct Numerical Layout Generation for 3D Indoor Scene Synthesis via Spatial Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Project Page: https://directlayout.github.io/
    </div>
    <details class="paper-abstract">
      Realistic 3D indoor scene synthesis is vital for embodied AI and digital content creation. It can be naturally divided into two subtasks: object generation and layout generation. While recent generative models have significantly advanced object-level quality and controllability, layout generation remains challenging due to limited datasets. Existing methods either overfit to these datasets or rely on predefined constraints to optimize numerical layout that sacrifice flexibility. As a result, they fail to generate scenes that are both open-vocabulary and aligned with fine-grained user instructions. We introduce DirectLayout, a framework that directly generates numerical 3D layouts from text descriptions using generalizable spatial reasoning of large language models (LLMs). DirectLayout decomposes the generation into three stages: producing a Bird's-Eye View (BEV) layout, lifting it into 3D space, and refining object placements. To enable explicit spatial reasoning and help the model grasp basic principles of object placement, we employ Chain-of-Thought (CoT) Activation based on the 3D-Front dataset. Additionally, we design CoT-Grounded Generative Layout Reward to enhance generalization and spatial planning. During inference, DirectLayout addresses asset-layout mismatches via Iterative Asset-Layout Alignment through in-context learning. Extensive experiments demonstrate that DirectLayout achieves impressive semantic consistency, generalization and physical plausibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05204v1">OGGSplat: Open Gaussian Growing for Generalizable Reconstruction with Expanded Field-of-View</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Reconstructing semantic-aware 3D scenes from sparse views is a challenging yet essential research direction, driven by the demands of emerging applications such as virtual reality and embodied AI. Existing per-scene optimization methods require dense input views and incur high computational costs, while generalizable approaches often struggle to reconstruct regions outside the input view cone. In this paper, we propose OGGSplat, an open Gaussian growing method that expands the field-of-view in generalizable 3D reconstruction. Our key insight is that the semantic attributes of open Gaussians provide strong priors for image extrapolation, enabling both semantic consistency and visual plausibility. Specifically, once open Gaussians are initialized from sparse views, we introduce an RGB-semantic consistent inpainting module applied to selected rendered views. This module enforces bidirectional control between an image diffusion model and a semantic diffusion model. The inpainted regions are then lifted back into 3D space for efficient and progressive Gaussian parameter optimization. To evaluate our method, we establish a Gaussian Outpainting (GO) benchmark that assesses both semantic and generative quality of reconstructed open-vocabulary scenes. OGGSplat also demonstrates promising semantic-aware scene reconstruction capabilities when provided with two view images captured directly from a smartphone camera.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05171v1">Towards provable probabilistic safety for scalable embodied AI systems</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Embodied AI systems, comprising AI models and physical plants, are increasingly prevalent across various applications. Due to the rarity of system failures, ensuring their safety in complex operating environments remains a major challenge, which severely hinders their large-scale deployment in safety-critical domains, such as autonomous vehicles, medical devices, and robotics. While achieving provable deterministic safety--verifying system safety across all possible scenarios--remains theoretically ideal, the rarity and complexity of corner cases make this approach impractical for scalable embodied AI systems. To address this challenge, we introduce provable probabilistic safety, which aims to ensure that the residual risk of large-scale deployment remains below a predefined threshold. Instead of attempting exhaustive safety proof across all corner cases, this paradigm establishes a probabilistic safety boundary on overall system performance, leveraging statistical methods to enhance feasibility and scalability. A well-defined probabilistic safety boundary enables embodied AI systems to be deployed at scale while allowing for continuous refinement of safety guarantees. Our work focuses on three core questions: what is provable probabilistic safety, how to prove the probabilistic safety, and how to achieve the provable probabilistic safety. By bridging the gap between theoretical safety assurance and practical deployment, our work offers a pathway toward safer, large-scale adoption of embodied AI systems in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04982v1">GEX: Democratizing Dexterity with Fully-Actuated Dexterous Hand and Exoskeleton Glove</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      This paper introduces GEX, an innovative low-cost dexterous manipulation system that combines the GX11 tri-finger anthropomorphic hand (11 DoF) with the EX12 tri-finger exoskeleton glove (12 DoF), forming a closed-loop teleoperation framework through kinematic retargeting for high-fidelity control. Both components employ modular 3D-printed finger designs, achieving ultra-low manufacturing costs while maintaining full actuation capabilities. Departing from conventional tendon-driven or underactuated approaches, our electromechanical system integrates independent joint motors across all 23 DoF, ensuring complete state observability and accurate kinematic modeling. This full-actuation architecture enables precise bidirectional kinematic calculations, substantially enhancing kinematic retargeting fidelity between the exoskeleton and robotic hand. The proposed system bridges the cost-performance gap in dexterous manipulation research, providing an accessible platform for acquiring high-quality demonstration data to advance embodied AI and dexterous robotic skill transfer learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03709v1">AetherVision-Bench: An Open-Vocabulary RGB-Infrared Benchmark for Multi-Angle Segmentation across Aerial and Ground Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted at Workshop on Foundation Models Meet Embodied Agents at CVPR 2025 (Non-archival Track)
    </div>
    <details class="paper-abstract">
      Open-vocabulary semantic segmentation (OVSS) involves assigning labels to each pixel in an image based on textual descriptions, leveraging world models like CLIP. However, they encounter significant challenges in cross-domain generalization, hindering their practical efficacy in real-world applications. Embodied AI systems are transforming autonomous navigation for ground vehicles and drones by enhancing their perception abilities, and in this study, we present AetherVision-Bench, a benchmark for multi-angle segmentation across aerial, and ground perspectives, which facilitates an extensive evaluation of performance across different viewing angles and sensor modalities. We assess state-of-the-art OVSS models on the proposed benchmark and investigate the key factors that impact the performance of zero-shot transfer models. Our work pioneers the creation of a robustness benchmark, offering valuable insights and establishing a foundation for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03613v1">Training Cross-Morphology Embodied AI Agents: From Practical Challenges to Theoretical Foundations</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      While theory and practice are often seen as separate domains, this article shows that theoretical insight is essential for overcoming real-world engineering barriers. We begin with a practical challenge: training a cross-morphology embodied AI policy that generalizes across diverse robot morphologies. We formalize this as the Heterogeneous Embodied Agent Training (HEAT) problem and prove it reduces to a structured Partially Observable Markov Decision Process (POMDP) that is PSPACE-complete. This result explains why current reinforcement learning pipelines break down under morphological diversity, due to sequential training constraints, memory-policy coupling, and data incompatibility. We further explore Collective Adaptation, a distributed learning alternative inspired by biological systems. Though NEXP-complete in theory, it offers meaningful scalability and deployment benefits in practice. This work illustrates how computational theory can illuminate system design trade-offs and guide the development of more robust, scalable embodied AI. For practitioners and researchers to explore this problem, the implementation code of this work has been made publicly available at https://github.com/airs-admin/HEAT
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03516v1">SemNav: A Model-Based Planner for Zero-Shot Object Goal Navigation Using Vision-Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted at CVPR 2025 workshop - Foundation Models Meet Embodied Agents
    </div>
    <details class="paper-abstract">
      Object goal navigation is a fundamental task in embodied AI, where an agent is instructed to locate a target object in an unexplored environment. Traditional learning-based methods rely heavily on large-scale annotated data or require extensive interaction with the environment in a reinforcement learning setting, often failing to generalize to novel environments and limiting scalability. To overcome these challenges, we explore a zero-shot setting where the agent operates without task-specific training, enabling more scalable and adaptable solution. Recent advances in Vision Foundation Models (VFMs) offer powerful capabilities for visual understanding and reasoning, making them ideal for agents to comprehend scenes, identify relevant regions, and infer the likely locations of objects. In this work, we present a zero-shot object goal navigation framework that integrates the perceptual strength of VFMs with a model-based planner that is capable of long-horizon decision making through frontier exploration. We evaluate our approach on the HM3D dataset using the Habitat simulator and demonstrate that our method achieves state-of-the-art performance in terms of success weighted by path length for zero-shot object goal navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02112v2">SAB3R: Semantic-Augmented Backbone in 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 3D-LLM/VLA @ CVPR2025 | Project page: https://uva-computer-vision-lab.github.io/sab3r/
    </div>
    <details class="paper-abstract">
      We introduce a new task, Map and Locate, which unifies the traditionally distinct objectives of open-vocabulary segmentation - detecting and segmenting object instances based on natural language queries - and 3D reconstruction, the process of estimating a scene's 3D structure from visual inputs. Specifically, Map and Locate involves generating a point cloud from an unposed video and segmenting object instances based on open-vocabulary queries. This task serves as a critical step toward real-world embodied AI applications and introduces a practical task that bridges reconstruction, recognition and reorganization. To tackle this task, we introduce a simple yet effective baseline, which we denote as SAB3R. Our approach builds upon MASt3R, a recent breakthrough in 3D computer vision, and incorporates a lightweight distillation strategy. This method transfers dense, per-pixel semantic features from 2D vision backbones (eg, CLIP and DINOv2) to enhance MASt3R's capabilities. Without introducing any auxiliary frozen networks, our model generates per-pixel semantic features and constructs cohesive point maps in a single forward pass. Compared to separately deploying MASt3R and CLIP, our unified model, SAB3R, achieves superior performance on the Map and Locate benchmark. Furthermore, we evaluate SAB3R on both 2D semantic segmentation and 3D tasks to comprehensively validate its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03097v1">EgoVLM: Policy Optimization for Egocentric Video Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Our Code can be found at https://github.com/adityavavre/VidEgoVLM
    </div>
    <details class="paper-abstract">
      Emerging embodied AI applications, such as wearable cameras and autonomous agents, have underscored the need for robust reasoning from first person video streams. We introduce EgoVLM, a vision-language model specifically designed to integrate visual comprehension and spatial-temporal reasoning within egocentric video contexts. EgoVLM is fine-tuned via Group Relative Policy Optimization (GRPO), a reinforcement learning method adapted to align model outputs with human-like reasoning steps. Following DeepSeek R1-Zero's approach, we directly tune using RL without any supervised fine-tuning phase on chain-of-thought (CoT) data. We evaluate EgoVLM on egocentric video question answering benchmarks and show that domain-specific training substantially improves performance over general-purpose VLMs. Our EgoVLM-3B, trained exclusively on non-CoT egocentric data, outperforms the base Qwen2.5-VL 3B and 7B models by 14.33 and 13.87 accuracy points on the EgoSchema benchmark, respectively. By explicitly generating reasoning traces, EgoVLM enhances interpretability, making it well-suited for downstream applications. Furthermore, we introduce a novel keyframe-based reward that incorporates salient frame selection to guide reinforcement learning optimization. This reward formulation opens a promising avenue for future exploration in temporally grounded egocentric reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19318v4">MINDSTORES: Memory-Informed Neural Decision Synthesis for Task-Oriented Reinforcement in Embodied Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown promising capabilities as zero-shot planners for embodied agents, their inability to learn from experience and build persistent mental models limits their robustness in complex open-world environments like Minecraft. We introduce MINDSTORES, an experience-augmented planning framework that enables embodied agents to build and leverage mental models through natural interaction with their environment. Drawing inspiration from how humans construct and refine cognitive mental models, our approach extends existing zero-shot LLM planning by maintaining a database of past experiences that informs future planning iterations. The key innovation is representing accumulated experiences as natural language embeddings of (state, task, plan, outcome) tuples, which can then be efficiently retrieved and reasoned over by an LLM planner to generate insights and guide plan refinement for novel states and tasks. Through extensive experiments in the MineDojo environment, a simulation environment for agents in Minecraft that provides low-level controls for Minecraft, we find that MINDSTORES learns and applies its knowledge significantly better than existing memory-based LLM planners while maintaining the flexibility and generalization benefits of zero-shot approaches, representing an important step toward more capable embodied AI systems that can learn continuously through natural experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03792v2">Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Online fine-tuning vision-language model (VLM) agents with reinforcement learning (RL) has shown promise for equipping agents with multi-step, goal-oriented capabilities in dynamic environments. However, their open-ended textual action space and non-end-to-end nature of action generation present significant challenges to effective online exploration in RL, e.g., explosion of the exploration space. We propose a novel online fine-tuning method, Counterfactual Soft Reinforcement Learning (CoSo), better suited to the textual output space of VLM agents. Compared to prior methods that assign uniform uncertainty to all tokens, CoSo leverages counterfactual reasoning to dynamically assess the causal influence of individual tokens on post-processed actions. By prioritizing the exploration of action-critical tokens while reducing the impact of semantically redundant or low-impact tokens, CoSo enables a more targeted and efficient online rollout process. We provide theoretical analysis proving CoSo's convergence and policy improvement guarantees, and extensive empirical evaluations supporting CoSo's effectiveness. Our results across a diverse set of agent tasks, including Android device control, card gaming, and embodied AI, highlight its remarkable ability to enhance exploration efficiency and deliver consistent performance gains. The code is available at https://github.com/langfengQ/CoSo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02265v1">Rig3R: Rig-Aware Conditioning for Learned 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Estimating agent pose and 3D scene structure from multi-camera rigs is a central task in embodied AI applications such as autonomous driving. Recent learned approaches such as DUSt3R have shown impressive performance in multiview settings. However, these models treat images as unstructured collections, limiting effectiveness in scenarios where frames are captured from synchronized rigs with known or inferable structure. To this end, we introduce Rig3R, a generalization of prior multiview reconstruction models that incorporates rig structure when available, and learns to infer it when not. Rig3R conditions on optional rig metadata including camera ID, time, and rig poses to develop a rig-aware latent space that remains robust to missing information. It jointly predicts pointmaps and two types of raymaps: a pose raymap relative to a global frame, and a rig raymap relative to a rig-centric frame consistent across time. Rig raymaps allow the model to infer rig structure directly from input images when metadata is missing. Rig3R achieves state-of-the-art performance in 3D reconstruction, camera pose estimation, and rig discovery, outperforming both traditional and learned methods by 17-45% mAA across diverse real-world rig datasets, all in a single forward pass without post-processing or iterative refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02112v1">SAB3R: Semantic-Augmented Backbone in 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Project page: https://uva-computer-vision-lab.github.io/sab3r/
    </div>
    <details class="paper-abstract">
      We introduce a new task, Map and Locate, which unifies the traditionally distinct objectives of open-vocabulary segmentation - detecting and segmenting object instances based on natural language queries - and 3D reconstruction, the process of estimating a scene's 3D structure from visual inputs. Specifically, Map and Locate involves generating a point cloud from an unposed video and segmenting object instances based on open-vocabulary queries. This task serves as a critical step toward real-world embodied AI applications and introduces a practical task that bridges reconstruction, recognition and reorganization. To tackle this task, we introduce a simple yet effective baseline, which we denote as SAB3R. Our approach builds upon MASt3R, a recent breakthrough in 3D computer vision, and incorporates a lightweight distillation strategy. This method transfers dense, per-pixel semantic features from 2D vision backbones (eg, CLIP and DINOv2) to enhance MASt3R's capabilities. Without introducing any auxiliary frozen networks, our model generates per-pixel semantic features and constructs cohesive point maps in a single forward pass. Compared to separately deploying MASt3R and CLIP, our unified model, SAB3R, achieves superior performance on the Map and Locate benchmark. Furthermore, we evaluate SAB3R on both 2D semantic segmentation and 3D tasks to comprehensively validate its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00927v1">In-the-wild Audio Spatialization with Flexible Text-guided Localization</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted by ACL 2025 main
    </div>
    <details class="paper-abstract">
      To enhance immersive experiences, binaural audio offers spatial awareness of sounding objects in AR, VR, and embodied AI applications. While existing audio spatialization methods can generally map any available monaural audio to binaural audio signals, they often lack the flexible and interactive control needed in complex multi-object user-interactive environments. To address this, we propose a Text-guided Audio Spatialization (TAS) framework that utilizes flexible text prompts and evaluates our model from unified generation and comprehension perspectives. Due to the limited availability of premium and large-scale stereo data, we construct the SpatialTAS dataset, which encompasses 376,000 simulated binaural audio samples to facilitate the training of our model. Our model learns binaural differences guided by 3D spatial location and relative position prompts, augmented by flipped-channel audio. It outperforms existing methods on both simulated and real-recorded datasets, demonstrating superior generalization and accuracy. Besides, we develop an assessment model based on Llama-3.1-8B, which evaluates the spatial semantic coherence between our generated binaural audio and text prompts through a spatial reasoning task. Results demonstrate that text prompts provide flexible and interactive control to generate binaural audio with excellent quality and semantic consistency in spatial locations. Dataset is available at \href{https://github.com/Alice01010101/TASU}
    </details>
</div>
