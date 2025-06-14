# embodied ai - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09680v2">Object-Centric Latent Action Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted by Workshop on World Models at ICLR 2025
    </div>
    <details class="paper-abstract">
      Leveraging vast amounts of unlabeled internet video data for embodied AI is currently bottlenecked by the lack of action labels and the presence of action-correlated visual distractors. Although recent latent action policy optimization (LAPO) has shown promise in inferring proxy-action labels from visual observations, its performance degrades significantly when distractors are present. To address this limitation, we propose a novel object-centric latent action learning framework that centers on objects rather than pixels. We leverage self-supervised object-centric pretraining to disentangle action-related and distracting dynamics. This allows LAPO to focus on task-relevant interactions, resulting in more robust proxy-action labels, enabling better imitation learning and efficient adaptation of the agent with just a few action-labeled trajectories. We evaluated our method in eight visually complex tasks across the Distracting Control Suite (DCS) and Distracting MetaWorld (DMW). Our results show that object-centric pretraining mitigates the negative effects of distractors by 50%, as measured by downstream task performance: average return (DCS) and success rate (DMW).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00379v5">Latent Action Learning Requires Supervision in the Presence of Distractors</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 ICML 2025, Poster, Project Page: https://laom.dunnolab.ai/, Source code: https://github.com/dunnolab/laom
    </div>
    <details class="paper-abstract">
      Recently, latent action learning, pioneered by Latent Action Policies (LAPO), have shown remarkable pre-training efficiency on observation-only data, offering potential for leveraging vast amounts of video available on the web for embodied AI. However, prior work has focused on distractor-free data, where changes between observations are primarily explained by ground-truth actions. Unfortunately, real-world videos contain action-correlated distractors that may hinder latent action learning. Using Distracting Control Suite (DCS) we empirically investigate the effect of distractors on latent action learning and demonstrate that LAPO struggle in such scenario. We propose LAOM, a simple LAPO modification that improves the quality of latent actions by 8x, as measured by linear probing. Importantly, we show that providing supervision with ground-truth actions, as few as 2.5% of the full dataset, during latent action learning improves downstream performance by 4.2x on average. Our findings suggest that integrating supervision during Latent Action Models (LAM) training is critical in the presence of distractors, challenging the conventional pipeline of first learning LAM and only then decoding from latent to ground-truth actions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10600v1">EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Constructing a physically realistic and accurately scaled simulated 3D world is crucial for the training and evaluation of embodied intelligence tasks. The diversity, realism, low cost accessibility and affordability of 3D data assets are critical for achieving generalization and scalability in embodied AI. However, most current embodied intelligence tasks still rely heavily on traditional 3D computer graphics assets manually created and annotated, which suffer from high production costs and limited realism. These limitations significantly hinder the scalability of data driven approaches. We present EmbodiedGen, a foundational platform for interactive 3D world generation. It enables the scalable generation of high-quality, controllable and photorealistic 3D assets with accurate physical properties and real-world scale in the Unified Robotics Description Format (URDF) at low cost. These assets can be directly imported into various physics simulation engines for fine-grained physical control, supporting downstream tasks in training and evaluation. EmbodiedGen is an easy-to-use, full-featured toolkit composed of six key modules: Image-to-3D, Text-to-3D, Texture Generation, Articulated Object Generation, Scene Generation and Layout Generation. EmbodiedGen generates diverse and interactive 3D worlds composed of generative 3D assets, leveraging generative AI to address the challenges of generalization and evaluation to the needs of embodied intelligence related research. Code is available at https://horizonrobotics.github.io/robot_lab/embodied_gen/index.html.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09839v1">OctoNav: Towards Generalist Embodied Navigation</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 31 pages, 25 figures
    </div>
    <details class="paper-abstract">
      Embodied navigation stands as a foundation pillar within the broader pursuit of embodied AI. However, previous navigation research is divided into different tasks/capabilities, e.g., ObjNav, ImgNav and VLN, where they differ in task objectives and modalities, making datasets and methods are designed individually. In this work, we take steps toward generalist navigation agents, which can follow free-form instructions that include arbitrary compounds of multi-modal and multi-capability. To achieve this, we propose a large-scale benchmark and corresponding method, termed OctoNav-Bench and OctoNav-R1. Specifically, OctoNav-Bench features continuous environments and is constructed via a designed annotation pipeline. We thoroughly craft instruction-trajectory pairs, where instructions are diverse in free-form with arbitrary modality and capability. Also, we construct a Think-Before-Action (TBA-CoT) dataset within OctoNav-Bench to provide the thinking process behind actions. For OctoNav-R1, we build it upon MLLMs and adapt it to a VLA-type model, which can produce low-level actions solely based on 2D visual observations. Moreover, we design a Hybrid Training Paradigm (HTP) that consists of three stages, i.e., Action-/TBA-SFT, Nav-GPRO, and Online RL stages. Each stage contains specifically designed learning policies and rewards. Importantly, for TBA-SFT and Nav-GRPO designs, we are inspired by the OpenAI-o1 and DeepSeek-R1, which show impressive reasoning ability via thinking-before-answer. Thus, we aim to investigate how to achieve thinking-before-action in the embodied navigation field, to improve model's reasoning ability toward generalists. Specifically, we propose TBA-SFT to utilize the TBA-CoT dataset to fine-tune the model as a cold-start phrase and then leverage Nav-GPRO to improve its thinking ability. Finally, OctoNav-R1 shows superior performance compared with previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09623v1">Analytic Task Scheduler: Recursive Least Squares Based Method for Continual Learning in Embodied Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Embodied foundation models are crucial for Artificial Intelligence (AI) interacting with the physical world by integrating multi-modal inputs, such as proprioception, vision and language, to understand human intentions and generate actions to control robots. While these models demonstrate strong generalization and few-shot learning capabilities, they face significant challenges in continually acquiring new skills without forgetting previously learned skills, a problem known as catastrophic forgetting. To address this issue, we propose the Analytic Task Scheduler (ATS), a novel framework for continual learning in embodied foundation models. ATS consists of a task-specific model library, where each model is fine-tuned independently on a single task, and an analytic scheduler trained using recursive least squares (RLS) to learn the mapping between language instructions and task-specific models. This architecture enables accurate task recognition and dynamic model selection while fundamentally avoiding parameter interference across tasks. The scheduler updates its parameters incrementally using only statistics (autocorrelation and cross-correlation matrices), enabling forgetting-resistant learning without the need to revisit historical data. We validate ATS on a real-world robot platform (RM65B), demonstrating superior resistance to forgetting and strong adaptability to task variations. The results highlight ATS as an effective, scalable, and deployable solution for continual learning in embodied foundation models operating in complex, dynamic environments. Our code will be available at https://github.com/MIAA-Embodied-AI/AnalyticTaskScheduler
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19789v2">What Can RL Bring to VLA Generalization? An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Large Vision-Language Action (VLA) models have shown significant potential for embodied AI. However, their predominant training via supervised fine-tuning (SFT) limits generalization due to susceptibility to compounding errors under distribution shifts. Reinforcement learning (RL) offers a path to overcome these limitations by optimizing for task objectives via trial-and-error, yet a systematic understanding of its specific generalization benefits for VLAs compared to SFT is lacking. To address this, our study introduces a comprehensive benchmark for evaluating VLA generalization and systematically investigates the impact of RL fine-tuning across diverse visual, semantic, and execution dimensions. Our extensive experiments reveal that RL fine-tuning, particularly with PPO, significantly enhances generalization in semantic understanding and execution robustness over SFT, while maintaining comparable visual robustness. We identify PPO as a more effective RL algorithm for VLAs than LLM-derived methods like DPO and GRPO. We also develop a simple recipe for efficient PPO training on VLAs, and demonstrate its practical utility for improving VLA generalization. The project page is at https://rlvla.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10172v1">A Navigation Framework Utilizing Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) presents a complex challenge in embodied AI, requiring agents to interpret natural language instructions and navigate through visually rich, unfamiliar environments. Recent advances in large vision-language models (LVLMs), such as CLIP and Flamingo, have significantly improved multimodal understanding but introduced new challenges related to computational cost and real-time deployment. In this project, we propose a modular, plug-and-play navigation framework that decouples vision-language understanding from action planning. By integrating a frozen vision-language model, Qwen2.5-VL-7B-Instruct, with lightweight planning logic, we aim to achieve flexible, fast, and adaptable navigation without extensive model fine-tuning. Our framework leverages prompt engineering, structured history management, and a two-frame visual input strategy to enhance decision-making continuity across navigation steps. We evaluate our system on the Room-to-Room benchmark within the VLN-CE setting using the Matterport3D dataset and Habitat-Lab simulation environment. Although our initial results reveal challenges in generalizing to unseen environments under strict evaluation settings, our modular approach lays a foundation for scalable and efficient navigation systems, highlighting promising directions for future improvement through enhanced environmental priors and expanded multimodal input integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09049v1">VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-10
      | 💬 Project page: https://faceong.github.io/VIKI-R/
    </div>
    <details class="paper-abstract">
      Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08334v1">Generalizable Articulated Object Reconstruction from Casually Captured RGBD Videos</a></div>
    <div class="paper-meta">
      📅 2025-06-10
      | 💬 Project website can be found at https://3dlg-hcvc.github.io/video2articulation/
    </div>
    <details class="paper-abstract">
      Articulated objects are prevalent in daily life. Understanding their kinematic structure and reconstructing them have numerous applications in embodied AI and robotics. However, current methods require carefully captured data for training or inference, preventing practical, scalable, and generalizable reconstruction of articulated objects. We focus on reconstruction of an articulated object from a casually captured RGBD video shot with a hand-held camera. A casually captured video of an interaction with an articulated object is easy to acquire at scale using smartphones. However, this setting is quite challenging, as the object and camera move simultaneously and there are significant occlusions as the person interacts with the object. To tackle these challenges, we introduce a coarse-to-fine framework that infers joint parameters and segments movable parts of the object from a dynamic RGBD video. To evaluate our method under this new setting, we build a 20$\times$ larger synthetic dataset of 784 videos containing 284 objects across 11 categories. We compare our approach with existing methods that also take video as input. Experiments show that our method can reconstruct synthetic and real articulated objects across different categories from dynamic RGBD videos, outperforming existing methods significantly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08006v1">Dreamland: Controllable World Creation with Simulator and Generative Models</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Project Page: https://metadriverse.github.io/dreamland/
    </div>
    <details class="paper-abstract">
      Large-scale video generative models can synthesize diverse and realistic visual content for dynamic world creation, but they often lack element-wise controllability, hindering their use in editing scenes and training embodied AI agents. We propose Dreamland, a hybrid world generation framework combining the granular control of a physics-based simulator and the photorealistic content output of large-scale pretrained generative models. In particular, we design a layered world abstraction that encodes both pixel-level and object-level semantics and geometry as an intermediate representation to bridge the simulator and the generative model. This approach enhances controllability, minimizes adaptation cost through early alignment with real-world distributions, and supports off-the-shelf use of existing and future pretrained generative models. We further construct a D3Sim dataset to facilitate the training and evaluation of hybrid generation pipelines. Experiments demonstrate that Dreamland outperforms existing baselines with 50.8% improved image quality, 17.9% stronger controllability, and has great potential to enhance embodied agent training. Code and data will be made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20220v2">DINeMo: Learning Neural Mesh Models with no 3D Annotations</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Accepted to 3rd Workshop on Compositional 3D Vision at CVPR 2025 (C3DV)
    </div>
    <details class="paper-abstract">
      Category-level 3D/6D pose estimation is a crucial step towards comprehensive 3D scene understanding, which would enable a broad range of applications in robotics and embodied AI. Recent works explored neural mesh models that approach a range of 2D and 3D tasks from an analysis-by-synthesis perspective. Despite the largely enhanced robustness to partial occlusion and domain shifts, these methods depended heavily on 3D annotations for part-contrastive learning, which confines them to a narrow set of categories and hinders efficient scaling. In this work, we present DINeMo, a novel neural mesh model that is trained with no 3D annotations by leveraging pseudo-correspondence obtained from large visual foundation models. We adopt a bidirectional pseudo-correspondence generation method, which produce pseudo correspondence utilize both local appearance features and global context information. Experimental results on car datasets demonstrate that our DINeMo outperforms previous zero- and few-shot 3D pose estimation by a wide margin, narrowing the gap with fully-supervised methods by 67.3%. Our DINeMo also scales effectively and efficiently when incorporating more unlabeled images during training, which demonstrate the advantages over supervised learning methods that rely on 3D annotations. Our project page is available at https://analysis-by-synthesis.github.io/DINeMo/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07570v1">LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Automatic indoor layout generation has attracted increasing attention due to its potential in interior design, virtual environment construction, and embodied AI. Existing methods fall into two categories: prompt-driven approaches that leverage proprietary LLM services (e.g., GPT APIs) and learning-based methods trained on layout data upon diffusion-based models. Prompt-driven methods often suffer from spatial inconsistency and high computational costs, while learning-based methods are typically constrained by coarse relational graphs and limited datasets, restricting their generalization to diverse room categories. In this paper, we revisit LLM-based indoor layout generation and present 3D-SynthPlace, a large-scale dataset that combines synthetic layouts generated via a 'GPT synthesize, Human inspect' pipeline, upgraded from the 3D-Front dataset. 3D-SynthPlace contains nearly 17,000 scenes, covering four common room types -- bedroom, living room, kitchen, and bathroom -- enriched with diverse objects and high-level spatial annotations. We further introduce OptiScene, a strong open-source LLM optimized for indoor layout generation, fine-tuned based on our 3D-SynthPlace dataset through our two-stage training. For the warum-up stage I, we adopt supervised fine-tuning (SFT), which is taught to first generate high-level spatial descriptions then conditionally predict concrete object placements. For the reinforcing stage II, to better align the generated layouts with human design preferences, we apply multi-turn direct preference optimization (DPO), which significantly improving layout quality and generation success rates. Extensive experiments demonstrate that OptiScene outperforms traditional prompt-driven and learning-based baselines. Moreover, OptiScene shows promising potential in interactive tasks such as scene editing and robot navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07286v1">Multi-Step Guided Diffusion for Image Restoration on Edge Devices: Toward Lightweight Perception in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 Accepted in CVPR 2025 Embodied AI Workshop
    </div>
    <details class="paper-abstract">
      Diffusion models have shown remarkable flexibility for solving inverse problems without task-specific retraining. However, existing approaches such as Manifold Preserving Guided Diffusion (MPGD) apply only a single gradient update per denoising step, limiting restoration fidelity and robustness, especially in embedded or out-of-distribution settings. In this work, we introduce a multistep optimization strategy within each denoising timestep, significantly enhancing image quality, perceptual accuracy, and generalization. Our experiments on super-resolution and Gaussian deblurring demonstrate that increasing the number of gradient updates per step improves LPIPS and PSNR with minimal latency overhead. Notably, we validate this approach on a Jetson Orin Nano using degraded ImageNet and a UAV dataset, showing that MPGD, originally trained on face datasets, generalizes effectively to natural and aerial scenes. Our findings highlight MPGD's potential as a lightweight, plug-and-play restoration module for real-time visual perception in embodied AI agents such as drones and mobile robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04292v5">DaDu-Corki: Algorithm-Architecture Co-Design for Embodied AI-powered Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Embodied AI robots have the potential to fundamentally improve the way human beings live and manufacture. Continued progress in the burgeoning field of using large language models to control robots depends critically on an efficient computing substrate, and this trend is strongly evident in manipulation tasks. In particular, today's computing systems for embodied AI robots for manipulation tasks are designed purely based on the interest of algorithm developers, where robot actions are divided into a discrete frame basis. Such an execution pipeline creates high latency and energy consumption. This paper proposes \textsc{Corki}\xspace, an algorithm-architecture co-design framework for real-time embodied AI-powered robotic manipulation applications. We aim to decouple LLM inference, robotic control, and data communication in the embodied AI robots' compute pipeline. Instead of predicting action for one single frame, \textsc{Corki}\xspace predicts the trajectory for the near future to reduce the frequency of LLM inference. The algorithm is coupled with a hardware that accelerates transforming trajectory into actual torque signals used to control robots and an execution pipeline that parallels data communication with computation. \textsc{Corki}\xspace largely reduces LLM inference frequency by up to $5.1\times$, resulting in up to $5.9\times$ speed up. The success rate improvement can be up to 13.9\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05651v1">Hallucinate, Ground, Repeat: A Framework for Generalized Visual Relationship Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 22 pages, 9 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Understanding relationships between objects is central to visual intelligence, with applications in embodied AI, assistive systems, and scene understanding. Yet, most visual relationship detection (VRD) models rely on a fixed predicate set, limiting their generalization to novel interactions. A key challenge is the inability to visually ground semantically plausible, but unannotated, relationships hypothesized from external knowledge. This work introduces an iterative visual grounding framework that leverages large language models (LLMs) as structured relational priors. Inspired by expectation-maximization (EM), our method alternates between generating candidate scene graphs from detected objects using an LLM (expectation) and training a visual model to align these hypotheses with perceptual evidence (maximization). This process bootstraps relational understanding beyond annotated data and enables generalization to unseen predicates. Additionally, we introduce a new benchmark for open-world VRD on Visual Genome with 21 held-out predicates and evaluate under three settings: seen, unseen, and mixed. Our model outperforms LLM-only, few-shot, and debiased baselines, achieving mean recall (mR@50) of 15.9, 13.1, and 11.7 on predicate classification on these three sets. These results highlight the promise of grounded LLM priors for scalable open-world visual understanding.
    </details>
</div>
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
