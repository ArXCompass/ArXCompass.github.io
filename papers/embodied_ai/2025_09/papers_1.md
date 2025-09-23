# embodied ai - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17430v1">EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 16 pages, 18 figures, paper accepted at ICCV, 2025
    </div>
    <details class="paper-abstract">
      The field of Embodied AI predominantly relies on simulation for training and evaluation, often using either fully synthetic environments that lack photorealism or high-fidelity real-world reconstructions captured with expensive hardware. As a result, sim-to-real transfer remains a major challenge. In this paper, we introduce EmbodiedSplat, a novel approach that personalizes policy training by efficiently capturing the deployment environment and fine-tuning policies within the reconstructed scenes. Our method leverages 3D Gaussian Splatting (GS) and the Habitat-Sim simulator to bridge the gap between realistic scene capture and effective training environments. Using iPhone-captured deployment scenes, we reconstruct meshes via GS, enabling training in settings that closely approximate real-world conditions. We conduct a comprehensive analysis of training strategies, pre-training datasets, and mesh reconstruction techniques, evaluating their impact on sim-to-real predictivity in real-world scenarios. Experimental results demonstrate that agents fine-tuned with EmbodiedSplat outperform both zero-shot baselines pre-trained on large-scale real-world datasets (HM3D) and synthetically generated datasets (HSSD), achieving absolute success rate improvements of 20\% and 40\% on real-world Image Navigation task. Moreover, our approach yields a high sim-vs-real correlation (0.87--0.97) for the reconstructed meshes, underscoring its effectiveness in adapting policies to diverse environments with minimal effort. Project page: https://gchhablani.github.io/embodied-splat
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09585v2">Elevating Visual Perception in Multimodal LLMs with Auxiliary Embedding Distillation</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Project Page: https://praeclarumjj3.github.io/visper_lm/
    </div>
    <details class="paper-abstract">
      In recent times, the standard practice for developing MLLMs is to feed features from vision encoder(s) into the LLM and train with natural language supervision. This approach often causes models to lean towards language comprehension and undermine the rich visual perception signals present in the data, which are critical for tasks involving spatial reasoning in the domain of embodied AI and robotics. Is it possible to optimize both at the same time? In this work, we propose VisPer-LM, the first approach that infuses visual perception knowledge from expert vision encoders into the LLM's (of an MLLM) hidden representations. We start by investigating MLLMs trained solely with natural language supervision and identify a positive correlation between the quality of visual representations within these models and their downstream performance. Given this insight, we formulate the objective during the pretraining stage in MLLMs as a coupled optimization of predictive visual embedding and next (text) token prediction. Moreover, through extensive probing, we observe improved visual representation quality due to embedding optimization, underscoring the effectiveness of our probing setup. We demonstrate that our VisPer-LM outperforms the single and multi-encoder baselines, proving our approach's superiority over explicitly feeding the corresponding features to the LLM. In particular, VisPer-LM boosts performance by an average margin of up to 2.5% on various benchmarks, with a notable improvement of 8.7% on the Depth task in CV-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16176v1">Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      We present Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories (ACDC), an autonomous drone cinematography system driven by natural language communication between human directors and drones. The main limitation of previous drone cinematography workflows is that they require manual selection of waypoints and view angles based on predefined human intent, which is labor-intensive and yields inconsistent performance. In this paper, we propose employing large language models (LLMs) and vision foundation models (VFMs) to convert free-form natural language prompts directly into executable indoor UAV video tours. Specifically, our method comprises a vision-language retrieval pipeline for initial waypoint selection, a preference-based Bayesian optimization framework that refines poses using aesthetic feedback, and a motion planner that generates safe quadrotor trajectories. We validate ACDC through both simulation and hardware-in-the-loop experiments, demonstrating that it robustly produces professional-quality footage across diverse indoor scenes without requiring expertise in robotics or cinematography. These results highlight the potential of embodied AI agents to close the loop from open-vocabulary dialogue to real-world autonomous aerial cinematography.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07570v2">OptiScene: LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Automatic indoor layout generation has attracted increasing attention due to its potential in interior design, virtual environment construction, and embodied AI. Existing methods fall into two categories: prompt-driven approaches that leverage proprietary LLM services (e.g., GPT APIs) and learning-based methods trained on layout data upon diffusion-based models. Prompt-driven methods often suffer from spatial inconsistency and high computational costs, while learning-based methods are typically constrained by coarse relational graphs and limited datasets, restricting their generalization to diverse room categories. In this paper, we revisit LLM-based indoor layout generation and present 3D-SynthPlace, a large-scale dataset that combines synthetic layouts generated via a 'GPT synthesize, Human inspect' pipeline, upgraded from the 3D-Front dataset. 3D-SynthPlace contains nearly 17,000 scenes, covering four common room types -- bedroom, living room, kitchen, and bathroom -- enriched with diverse objects and high-level spatial annotations. We further introduce OptiScene, a strong open-source LLM optimized for indoor layout generation, fine-tuned based on our 3D-SynthPlace dataset through our two-stage training. For the warum-up stage I, we adopt supervised fine-tuning (SFT), which is taught to first generate high-level spatial descriptions then conditionally predict concrete object placements. For the reinforcing stage II, to better align the generated layouts with human design preferences, we apply multi-turn direct preference optimization (DPO), which significantly improving layout quality and generation success rates. Extensive experiments demonstrate that OptiScene outperforms traditional prompt-driven and learning-based baselines. Moreover, OptiScene shows promising potential in interactive tasks such as scene editing and robot navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16815v2">ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 NeurIPS 2025. Project page: https://jasper0314-huang.github.io/thinkact-vla/
    </div>
    <details class="paper-abstract">
      Vision-language-action (VLA) reasoning tasks require agents to interpret multimodal instructions, perform long-horizon planning, and act adaptively in dynamic environments. Existing approaches typically train VLA models in an end-to-end fashion, directly mapping inputs to actions without explicit reasoning, which hinders their ability to plan over multiple steps or adapt to complex task variations. In this paper, we propose ThinkAct, a dual-system framework that bridges high-level reasoning with low-level action execution via reinforced visual latent planning. ThinkAct trains a multimodal LLM to generate embodied reasoning plans guided by reinforcing action-aligned visual rewards based on goal completion and trajectory consistency. These reasoning plans are compressed into a visual plan latent that conditions a downstream action model for robust action execution on target environments. Extensive experiments on embodied reasoning and robot manipulation benchmarks demonstrate that ThinkAct enables few-shot adaptation, long-horizon planning, and self-correction behaviors in complex embodied AI tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14687v1">RealMirror: A Comprehensive, Open-Source Vision-Language-Action Platform for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      The emerging field of Vision-Language-Action (VLA) for humanoid robots faces several fundamental challenges, including the high cost of data acquisition, the lack of a standardized benchmark, and the significant gap between simulation and the real world. To overcome these obstacles, we propose RealMirror, a comprehensive, open-source embodied AI VLA platform. RealMirror builds an efficient, low-cost data collection, model training, and inference system that enables end-to-end VLA research without requiring a real robot. To facilitate model evolution and fair comparison, we also introduce a dedicated VLA benchmark for humanoid robots, featuring multiple scenarios, extensive trajectories, and various VLA models. Furthermore, by integrating generative models and 3D Gaussian Splatting to reconstruct realistic environments and robot models, we successfully demonstrate zero-shot Sim2Real transfer, where models trained exclusively on simulation data can perform tasks on a real robot seamlessly, without any fine-tuning. In conclusion, with the unification of these critical components, RealMirror provides a robust framework that significantly accelerates the development of VLA models for humanoid robots. Project page: https://terminators2025.github.io/RealMirror.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15333v1">Emulating Human-like Adaptive Vision for Efficient and Flexible Machine Visual Perception</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Human vision is highly adaptive, efficiently sampling intricate environments by sequentially fixating on task-relevant regions. In contrast, prevailing machine vision models passively process entire scenes at once, resulting in excessive resource demands scaling with spatial-temporal input resolution and model size, yielding critical limitations impeding both future advancements and real-world application. Here we introduce AdaptiveNN, a general framework aiming to drive a paradigm shift from 'passive' to 'active, adaptive' vision models. AdaptiveNN formulates visual perception as a coarse-to-fine sequential decision-making process, progressively identifying and attending to regions pertinent to the task, incrementally combining information across fixations, and actively concluding observation when sufficient. We establish a theory integrating representation learning with self-rewarding reinforcement learning, enabling end-to-end training of the non-differentiable AdaptiveNN without additional supervision on fixation locations. We assess AdaptiveNN on 17 benchmarks spanning 9 tasks, including large-scale visual recognition, fine-grained discrimination, visual search, processing images from real driving and medical scenarios, language-driven embodied AI, and side-by-side comparisons with humans. AdaptiveNN achieves up to 28x inference cost reduction without sacrificing accuracy, flexibly adapts to varying task demands and resource budgets without retraining, and provides enhanced interpretability via its fixation patterns, demonstrating a promising avenue toward efficient, flexible, and interpretable computer vision. Furthermore, AdaptiveNN exhibits closely human-like perceptual behaviors in many cases, revealing its potential as a valuable tool for investigating visual cognition. Code is available at https://github.com/LeapLabTHU/AdaptiveNN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15273v1">Embodied Arena: A Comprehensive, Unified, and Evolving Evaluation Platform for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 32 pages, 5 figures, Embodied Arena Technical Report
    </div>
    <details class="paper-abstract">
      Embodied AI development significantly lags behind large foundation models due to three critical challenges: (1) lack of systematic understanding of core capabilities needed for Embodied AI, making research lack clear objectives; (2) absence of unified and standardized evaluation systems, rendering cross-benchmark evaluation infeasible; and (3) underdeveloped automated and scalable acquisition methods for embodied data, creating critical bottlenecks for model scaling. To address these obstacles, we present Embodied Arena, a comprehensive, unified, and evolving evaluation platform for Embodied AI. Our platform establishes a systematic embodied capability taxonomy spanning three levels (perception, reasoning, task execution), seven core capabilities, and 25 fine-grained dimensions, enabling unified evaluation with systematic research objectives. We introduce a standardized evaluation system built upon unified infrastructure supporting flexible integration of 22 diverse benchmarks across three domains (2D/3D Embodied Q&A, Navigation, Task Planning) and 30+ advanced models from 20+ worldwide institutes. Additionally, we develop a novel LLM-driven automated generation pipeline ensuring scalable embodied evaluation data with continuous evolution for diversity and comprehensiveness. Embodied Arena publishes three real-time leaderboards (Embodied Q&A, Navigation, Task Planning) with dual perspectives (benchmark view and capability view), providing comprehensive overviews of advanced model capabilities. Especially, we present nine findings summarized from the evaluation results on the leaderboards of Embodied Arena. This helps to establish clear research veins and pinpoint critical research problems, thereby driving forward progress in the field of Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12989v1">PANORAMA: The Rise of Omnidirectional Vision in the Embodied AI Era</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 This paper presents a draft overview of the emerging field of omnidirectional vision in the context of embodied AI
    </div>
    <details class="paper-abstract">
      Omnidirectional vision, using 360-degree vision to understand the environment, has become increasingly critical across domains like robotics, industrial inspection, and environmental monitoring. Compared to traditional pinhole vision, omnidirectional vision provides holistic environmental awareness, significantly enhancing the completeness of scene perception and the reliability of decision-making. However, foundational research in this area has historically lagged behind traditional pinhole vision. This talk presents an emerging trend in the embodied AI era: the rapid development of omnidirectional vision, driven by growing industrial demand and academic interest. We highlight recent breakthroughs in omnidirectional generation, omnidirectional perception, omnidirectional understanding, and related datasets. Drawing on insights from both academia and industry, we propose an ideal panoramic system architecture in the embodied AI era, PANORAMA, which consists of four key subsystems. Moreover, we offer in-depth opinions related to emerging trends and cross-community impacts at the intersection of panoramic vision and embodied AI, along with the future roadmap and open challenges. This overview synthesizes state-of-the-art advancements and outlines challenges and opportunities for future research in building robust, general-purpose omnidirectional AI systems in the embodied AI era.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12129v2">Embodied Navigation Foundation Model</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Project Page: https://pku-epic.github.io/NavFoM-Web/
    </div>
    <details class="paper-abstract">
      Navigation is a fundamental capability in embodied AI, representing the intelligence required to perceive and interact within physical environments following language instructions. Despite significant progress in large Vision-Language Models (VLMs), which exhibit remarkable zero-shot performance on general vision-language tasks, their generalization ability in embodied navigation remains largely confined to narrow task settings and embodiment-specific architectures. In this work, we introduce a cross-embodiment and cross-task Navigation Foundation Model (NavFoM), trained on eight million navigation samples that encompass quadrupeds, drones, wheeled robots, and vehicles, and spanning diverse tasks such as vision-and-language navigation, object searching, target tracking, and autonomous driving. NavFoM employs a unified architecture that processes multimodal navigation inputs from varying camera configurations and navigation horizons. To accommodate diverse camera setups and temporal horizons, NavFoM incorporates identifier tokens that embed camera view information of embodiments and the temporal context of tasks. Furthermore, to meet the demands of real-world deployment, NavFoM controls all observation tokens using a dynamically adjusted sampling strategy under a limited token length budget. Extensive evaluations on public benchmarks demonstrate that our model achieves state-of-the-art or highly competitive performance across multiple navigation tasks and embodiments without requiring task-specific fine-tuning. Additional real-world experiments further confirm the strong generalization capability and practical applicability of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12129v1">Embodied Navigation Foundation Model</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Project Page: https://pku-epic.github.io/NavFoM-Web/
    </div>
    <details class="paper-abstract">
      Navigation is a fundamental capability in embodied AI, representing the intelligence required to perceive and interact within physical environments following language instructions. Despite significant progress in large Vision-Language Models (VLMs), which exhibit remarkable zero-shot performance on general vision-language tasks, their generalization ability in embodied navigation remains largely confined to narrow task settings and embodiment-specific architectures. In this work, we introduce a cross-embodiment and cross-task Navigation Foundation Model (NavFoM), trained on eight million navigation samples that encompass quadrupeds, drones, wheeled robots, and vehicles, and spanning diverse tasks such as vision-and-language navigation, object searching, target tracking, and autonomous driving. NavFoM employs a unified architecture that processes multimodal navigation inputs from varying camera configurations and navigation horizons. To accommodate diverse camera setups and temporal horizons, NavFoM incorporates identifier tokens that embed camera view information of embodiments and the temporal context of tasks. Furthermore, to meet the demands of real-world deployment, NavFoM controls all observation tokens using a dynamically adjusted sampling strategy under a limited token length budget. Extensive evaluations on public benchmarks demonstrate that our model achieves state-of-the-art or highly competitive performance across multiple navigation tasks and embodiments without requiring task-specific fine-tuning. Additional real-world experiments further confirm the strong generalization capability and practical applicability of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10123v2">Learning Precise Affordances from Egocentric Videos for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      Affordance, defined as the potential actions that an object offers, is crucial for embodied AI agents. For example, such knowledge directs an agent to grasp a knife by the handle for cutting or by the blade for safe handover. While existing approaches have made notable progress, affordance research still faces three key challenges: data scarcity, poor generalization, and real-world deployment. Specifically, there is a lack of large-scale affordance datasets with precise segmentation maps, existing models struggle to generalize across different domains or novel object and affordance classes, and little work demonstrates deployability in real-world scenarios. In this work, we address these issues by proposing a complete affordance learning system that (1) takes in egocentric videos and outputs precise affordance annotations without human labeling, (2) leverages geometric information and vision foundation models to improve generalization, and (3) introduces a framework that facilitates affordance-oriented robotic manipulation such as tool grasping and robot-to-human tool handover. Experimental results show that our model surpasses the state-of-the-art by 13.8% in mIoU, and the framework achieves 77.1% successful grasping among 179 trials, including evaluations on seen, unseen classes, and cluttered scenes. Project page: https://reagan1311.github.io/affgrasp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11895v1">Integrating Prior Observations for Incremental 3D Scene Graph Prediction</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted at 24th International Conference on Machine Learning and Applications (ICMLA'25)
    </div>
    <details class="paper-abstract">
      3D semantic scene graphs (3DSSG) provide compact structured representations of environments by explicitly modeling objects, attributes, and relationships. While 3DSSGs have shown promise in robotics and embodied AI, many existing methods rely mainly on sensor data, not integrating further information from semantically rich environments. Additionally, most methods assume access to complete scene reconstructions, limiting their applicability in real-world, incremental settings. This paper introduces a novel heterogeneous graph model for incremental 3DSSG prediction that integrates additional, multi-modal information, such as prior observations, directly into the message-passing process. Utilizing multiple layers, the model flexibly incorporates global and local scene representations without requiring specialized modules or full scene reconstructions. We evaluate our approach on the 3DSSG dataset, showing that GNNs enriched with multi-modal information such as semantic embeddings (e.g., CLIP) and prior observations offer a scalable and generalizable solution for complex, real-world environments. The full source code of the presented architecture will be made available at https://github.com/m4renz/incremental-scene-graph-prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03842v2">INGRID: Intelligent Generative Robotic Design Using Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into robotic systems has accelerated progress in embodied artificial intelligence, yet current approaches remain constrained by existing robotic architectures, particularly serial mechanisms. This hardware dependency fundamentally limits the scope of robotic intelligence. Here, we present INGRID (Intelligent Generative Robotic Design), a framework that enables the automated design of parallel robotic mechanisms through deep integration with reciprocal screw theory and kinematic synthesis methods. We decompose the design challenge into four progressive tasks: constraint analysis, kinematic joint generation, chain construction, and complete mechanism design. INGRID demonstrates the ability to generate novel parallel mechanisms with both fixed and variable mobility, discovering kinematic configurations not previously documented in the literature. We validate our approach through three case studies demonstrating how INGRID assists users in designing task-specific parallel robots based on desired mobility requirements. By bridging the gap between mechanism theory and machine learning, INGRID enables researchers without specialized robotics training to create custom parallel mechanisms, thereby decoupling advances in robotic intelligence from hardware constraints. This work establishes a foundation for mechanism intelligence, where AI systems actively design robotic hardware, potentially transforming the development of embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02029v5">RoboBrain 2.0 Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model, featuring a heterogeneous architecture with a vision encoder and a language model. Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models. In particular, it supports key real-world embodied AI capabilities, including spatial understanding (e.g., affordance prediction, spatial referring, trajectory forecasting) and temporal decision-making (e.g., closed-loop interaction, multi-agent long-horizon planning, and scene graph updating). This report details the model architecture, data construction, multi-stage training strategies, infrastructure and practical applications. We hope RoboBrain 2.0 advances embodied AI research and serves as a practical step toward building generalist embodied agents. The code, checkpoint and benchmark are available at https://superrobobrain.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10884v1">Nav-R1: Reasoning and Navigation in Embodied Scenes</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Embodied navigation requires agents to integrate perception, reasoning, and action for robust interaction in complex 3D environments. Existing approaches often suffer from incoherent and unstable reasoning traces that hinder generalization across diverse environments, and difficulty balancing long-horizon semantic reasoning with low-latency control for real-time navigation. To address these challenges, we propose Nav-R1, an embodied foundation model that unifies reasoning in embodied environments. We first construct Nav-CoT-110K, a large-scale dataset of step-by-step Chains-of-Thought (CoT) for embodied tasks, which enables cold-start initialization with structured reasoning. Building on this foundation, we design a GRPO-based reinforcement learning framework with three complementary rewards: format, understanding, and navigation, to improve structural adherence, semantic grounding, and path fidelity. Furthermore, we introduce a Fast-in-Slow reasoning paradigm, decoupling deliberate semantic reasoning from low-latency reactive control for efficient yet coherent navigation. Extensive evaluations on embodied AI benchmarks demonstrate that Nav-R1 consistently outperforms strong baselines, with over 8% average improvement in reasoning and navigation performance. Real-world deployment on a mobile robot further validates its robustness under limited onboard resources. Code: https://github.com/AIGeeksGroup/Nav-R1. Website: https://aigeeksgroup.github.io/Nav-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10813v1">InternScenes: A Large-scale Simulatable Indoor Scene Dataset with Realistic Layouts</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      The advancement of Embodied AI heavily relies on large-scale, simulatable 3D scene datasets characterized by scene diversity and realistic layouts. However, existing datasets typically suffer from limitations in data scale or diversity, sanitized layouts lacking small items, and severe object collisions. To address these shortcomings, we introduce \textbf{InternScenes}, a novel large-scale simulatable indoor scene dataset comprising approximately 40,000 diverse scenes by integrating three disparate scene sources, real-world scans, procedurally generated scenes, and designer-created scenes, including 1.96M 3D objects and covering 15 common scene types and 288 object classes. We particularly preserve massive small items in the scenes, resulting in realistic and complex layouts with an average of 41.5 objects per region. Our comprehensive data processing pipeline ensures simulatability by creating real-to-sim replicas for real-world scans, enhances interactivity by incorporating interactive objects into these scenes, and resolves object collisions by physical simulations. We demonstrate the value of InternScenes with two benchmark applications: scene layout generation and point-goal navigation. Both show the new challenges posed by the complex and realistic layouts. More importantly, InternScenes paves the way for scaling up the model training for both tasks, making the generation and navigation in such complex scenes possible. We commit to open-sourcing the data, models, and benchmarks to benefit the whole community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11528v6">LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 CoRL 2025
    </div>
    <details class="paper-abstract">
      Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09560v1">Boosting Embodied AI Agents through Perception-Generation Disaggregation and Asynchronous Pipeline Execution</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Embodied AI systems operate in dynamic environments, requiring seamless integration of perception and generation modules to process high-frequency input and output demands. Traditional sequential computation patterns, while effective in ensuring accuracy, face significant limitations in achieving the necessary "thinking" frequency for real-world applications. In this work, we present Auras, an algorithm-system co-designed inference framework to optimize the inference frequency of embodied AI agents. Auras disaggregates the perception and generation and provides controlled pipeline parallelism for them to achieve high and stable throughput. Faced with the data staleness problem that appears when the parallelism is increased, Auras establishes a public context for perception and generation to share, thereby promising the accuracy of embodied agents. Experimental results show that Auras improves throughput by 2.54x on average while achieving 102.7% of the original accuracy, demonstrating its efficacy in overcoming the constraints of sequential computation and providing high throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02359v3">Attribution Regularization for Multimodal Paradigms</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      Multimodal machine learning has gained significant attention in recent years due to its potential for integrating information from multiple modalities to enhance learning and decision-making processes. However, it is commonly observed that unimodal models outperform multimodal models, despite the latter having access to richer information. Additionally, the influence of a single modality often dominates the decision-making process, resulting in suboptimal performance. This research project aims to address these challenges by proposing a novel regularization term that encourages multimodal models to effectively utilize information from all modalities when making decisions. The focus of this project lies in the video-audio domain, although the proposed regularization technique holds promise for broader applications in embodied AI research, where multiple modalities are involved. By leveraging this regularization term, the proposed approach aims to mitigate the issue of unimodal dominance and improve the performance of multimodal machine learning systems. Through extensive experimentation and evaluation, the effectiveness and generalizability of the proposed technique will be assessed. The findings of this research project have the potential to significantly contribute to the advancement of multimodal machine learning and facilitate its application in various domains, including multimedia analysis, human-computer interaction, and embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12312v4">Visuospatial Cognitive Assistant</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 31 pages, 10 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06951v2">F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Homepage: https://aopolin-lv.github.io/F1-VLA/
    </div>
    <details class="paper-abstract">
      Executing language-conditioned tasks in dynamic visual environments remains a central challenge in embodied AI. Existing Vision-Language-Action (VLA) models predominantly adopt reactive state-to-action mappings, often leading to short-sighted behaviors and poor robustness in dynamic scenes. In this paper, we introduce F1, a pretrained VLA framework which integrates the visual foresight generation into decision-making pipeline. F1 adopts a Mixture-of-Transformer architecture with dedicated modules for perception, foresight generation, and control, thereby bridging understanding, generation, and actions. At its core, F1 employs a next-scale prediction mechanism to synthesize goal-conditioned visual foresight as explicit planning targets. By forecasting plausible future visual states, F1 reformulates action generation as a foresight-guided inverse dynamics problem, enabling actions that implicitly achieve visual goals. To endow F1 with robust and generalizable capabilities, we propose a three-stage training recipe on an extensive dataset comprising over 330k trajectories across 136 diverse tasks. This training scheme enhances modular reasoning and equips the model with transferable visual foresight, which is critical for complex and dynamic environments. Extensive evaluations on real-world tasks and simulation benchmarks demonstrate F1 consistently outperforms existing approaches, achieving substantial gains in both task success rate and generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06951v1">F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Executing language-conditioned tasks in dynamic visual environments remains a central challenge in embodied AI. Existing Vision-Language-Action (VLA) models predominantly adopt reactive state-to-action mappings, often leading to short-sighted behaviors and poor robustness in dynamic scenes. In this paper, we introduce F1, a pretrained VLA framework which integrates the visual foresight generation into decision-making pipeline. F1 adopts a Mixture-of-Transformer architecture with dedicated modules for perception, foresight generation, and control, thereby bridging understanding, generation, and actions. At its core, F1 employs a next-scale prediction mechanism to synthesize goal-conditioned visual foresight as explicit planning targets. By forecasting plausible future visual states, F1 reformulates action generation as a foresight-guided inverse dynamics problem, enabling actions that implicitly achieve visual goals. To endow F1 with robust and generalizable capabilities, we propose a three-stage training recipe on an extensive dataset comprising over 330k trajectories across 136 diverse tasks. This training scheme enhances modular reasoning and equips the model with transferable visual foresight, which is critical for complex and dynamic environments. Extensive evaluations on real-world tasks and simulation benchmarks demonstrate F1 consistently outperforms existing approaches, achieving substantial gains in both task success rate and generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05263v2">LatticeWorld: A Multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Recent research has been increasingly focusing on developing 3D world models that simulate complex real-world scenarios. World models have found broad applications across various domains, including embodied AI, autonomous driving, entertainment, etc. A more realistic simulation with accurate physics will effectively narrow the sim-to-real gap and allow us to gather rich information about the real world conveniently. While traditional manual modeling has enabled the creation of virtual 3D scenes, modern approaches have leveraged advanced machine learning algorithms for 3D world generation, with most recent advances focusing on generative methods that can create virtual worlds based on user instructions. This work explores such a research direction by proposing LatticeWorld, a simple yet effective 3D world generation framework that streamlines the industrial production pipeline of 3D environments. LatticeWorld leverages lightweight LLMs (LLaMA-2-7B) alongside the industry-grade rendering engine (e.g., Unreal Engine 5) to generate a dynamic environment. Our proposed framework accepts textual descriptions and visual instructions as multimodal inputs and creates large-scale 3D interactive worlds with dynamic agents, featuring competitive multi-agent interaction, high-fidelity physics simulation, and real-time rendering. We conduct comprehensive experiments to evaluate LatticeWorld, showing that it achieves superior accuracy in scene layout generation and visual fidelity. Moreover, LatticeWorld achieves over a $90\times$ increase in industrial production efficiency while maintaining high creative quality compared with traditional manual production methods. Our demo video is available at https://youtu.be/8VWZXpERR18
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11844v3">Generative World Explorer</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Website: generative-world-explorer.github.io
    </div>
    <details class="paper-abstract">
      Planning with partial observation is a central challenge in embodied AI. A majority of prior works have tackled this challenge by developing agents that physically explore their environment to update their beliefs about the world state. In contrast, humans can $\textit{imagine}$ unseen parts of the world through a mental exploration and $\textit{revise}$ their beliefs with imagined observations. Such updated beliefs can allow them to make more informed decisions, without necessitating the physical exploration of the world at all times. To achieve this human-like ability, we introduce the $\textit{Generative World Explorer (Genex)}$, an egocentric world exploration framework that allows an agent to mentally explore a large-scale 3D world (e.g., urban scenes) and acquire imagined observations to update its belief. This updated belief will then help the agent to make a more informed decision at the current step. To train $\textit{Genex}$, we create a synthetic urban scene dataset, Genex-DB. Our experimental results demonstrate that (1) $\textit{Genex}$ can generate high-quality and consistent observations during long-horizon exploration of a large virtual physical world and (2) the beliefs updated with the generated observations can inform an existing decision-making model (e.g., an LLM agent) to make better plans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06266v1">Spatial Reasoning with Vision-Language Models in Ego-Centric Multi-View Scenes</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Understanding 3D spatial relationships remains a major limitation of current Vision-Language Models (VLMs). Prior work has addressed this issue by creating spatial question-answering (QA) datasets based on single images or indoor videos. However, real-world embodied AI agents such as robots and self-driving cars typically rely on ego-centric, multi-view observations. To this end, we introduce Ego3D-Bench, a new benchmark designed to evaluate the spatial reasoning abilities of VLMs using ego-centric, multi-view outdoor data. Ego3D-Bench comprises over 8,600 QA pairs, created with significant involvement from human annotators to ensure quality and diversity. We benchmark 16 SOTA VLMs, including GPT-4o, Gemini1.5-Pro, InternVL3, and Qwen2.5-VL. Our results reveal a notable performance gap between human level scores and VLM performance, highlighting that current VLMs still fall short of human level spatial understanding. To bridge this gap, we propose Ego3D-VLM, a post-training framework that enhances 3D spatial reasoning of VLMs. Ego3D-VLM generates cognitive map based on estimated global 3D coordinates, resulting in 12% average improvement on multi-choice QA and 56% average improvement on absolute distance estimation. Ego3D-VLM is modular and can be integrated with any existing VLM. Together, Ego3D-Bench and Ego3D-VLM offer valuable tools for advancing toward human level spatial understanding in real-world, multi-view environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05263v1">LatticeWorld: A Multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Recent research has been increasingly focusing on developing 3D world models that simulate complex real-world scenarios. World models have found broad applications across various domains, including embodied AI, autonomous driving, entertainment, etc. A more realistic simulation with accurate physics will effectively narrow the sim-to-real gap and allow us to gather rich information about the real world conveniently. While traditional manual modeling has enabled the creation of virtual 3D scenes, modern approaches have leveraged advanced machine learning algorithms for 3D world generation, with most recent advances focusing on generative methods that can create virtual worlds based on user instructions. This work explores such a research direction by proposing LatticeWorld, a simple yet effective 3D world generation framework that streamlines the industrial production pipeline of 3D environments. LatticeWorld leverages lightweight LLMs (LLaMA-2-7B) alongside the industry-grade rendering engine (e.g., Unreal Engine 5) to generate a dynamic environment. Our proposed framework accepts textual descriptions and visual instructions as multimodal inputs and creates large-scale 3D interactive worlds with dynamic agents, featuring competitive multi-agent interaction, high-fidelity physics simulation, and real-time rendering. We conduct comprehensive experiments to evaluate LatticeWorld, showing that it achieves superior accuracy in scene layout generation and visual fidelity. Moreover, LatticeWorld achieves over a $90\times$ increase in industrial production efficiency while maintaining high creative quality compared with traditional manual production methods. Our demo video is available at https://youtu.be/8VWZXpERR18
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11191v2">Multi-Modal Multi-Task (M3T) Federated Foundation Models for Embodied AI: Potentials and Challenges for Edge Integration</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 Accepted for Publication in IEEE Internet of Things Magazine, 2025
    </div>
    <details class="paper-abstract">
      As embodied AI systems become increasingly multi-modal, personalized, and interactive, they must learn effectively from diverse sensory inputs, adapt continually to user preferences, and operate safely under resource and privacy constraints. These challenges expose a pressing need for machine learning models capable of swift, context-aware adaptation while balancing model generalization and personalization. Here, two methods emerge as suitable candidates, each offering parts of these capabilities: multi-modal multi-task foundation models (M3T-FMs) provide a pathway toward generalization across tasks and modalities, whereas federated learning (FL) offers the infrastructure for distributed, privacy-preserving model updates and user-level model personalization. However, when used in isolation, each of these approaches falls short of meeting the complex and diverse capability requirements of real-world embodied AI environments. In this vision paper, we introduce multi-modal multi-task federated foundation models (M3T-FFMs) for embodied AI, a new paradigm that unifies the strengths of M3T-FMs with the privacy-preserving distributed training nature of FL, enabling intelligent systems at the wireless edge. We collect critical deployment dimensions of M3T-FFMs in embodied AI ecosystems under a unified framework, which we name "EMBODY": Embodiment heterogeneity, Modality richness and imbalance, Bandwidth and compute constraints, On-device continual learning, Distributed control and autonomy, and Yielding safety, privacy, and personalization. For each, we identify concrete challenges and envision actionable research directions. We also present an evaluation framework for deploying M3T-FFMs in embodied AI systems, along with the associated trade-offs. Finally, we present a prototype implementation of M3T-FFMs and evaluate their energy and latency performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02761v2">Plan Verification for LLM-Based Embodied Task Completion Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17832v2">HLG: Comprehensive 3D Room Construction via Hierarchical Layout Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Realistic 3D indoor scene generation is crucial for virtual reality, interior design, embodied intelligence, and scene understanding. While existing methods have made progress in coarse-scale furniture arrangement, they struggle to capture fine-grained object placements, limiting the realism and utility of generated environments. This gap hinders immersive virtual experiences and detailed scene comprehension for embodied AI applications. To address these issues, we propose Hierarchical Layout Generation (HLG), a novel method for fine-grained 3D scene generation. HLG is the first to adopt a coarse-to-fine hierarchical approach, refining scene layouts from large-scale furniture placement to intricate object arrangements. Specifically, our fine-grained layout alignment module constructs a hierarchical layout through vertical and horizontal decoupling, effectively decomposing complex 3D indoor scenes into multiple levels of granularity. Additionally, our trainable layout optimization network addresses placement issues, such as incorrect positioning, orientation errors, and object intersections, ensuring structurally coherent and physically plausible scene generation. We demonstrate the effectiveness of our approach through extensive experiments, showing superior performance in generating realistic indoor scenes compared to existing methods. This work advances the field of scene generation and opens new possibilities for applications requiring detailed 3D environments. We will release our code upon publication to encourage future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03956v1">World Model Implanting for Test-time Adaptation of Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      In embodied AI, a persistent challenge is enabling agents to robustly adapt to novel domains without requiring extensive data collection or retraining. To address this, we present a world model implanting framework (WorMI) that combines the reasoning capabilities of large language models (LLMs) with independently learned, domain-specific world models through test-time composition. By allowing seamless implantation and removal of the world models, the embodied agent's policy achieves and maintains cross-domain adaptability. In the WorMI framework, we employ a prototype-based world model retrieval approach, utilizing efficient trajectory-based abstract representation matching, to incorporate relevant models into test-time composition. We also develop a world-wise compound attention method that not only integrates the knowledge from the retrieved world models but also aligns their intermediate representations with the reasoning model's representation within the agent's policy. This framework design effectively fuses domain-specific knowledge from multiple world models, ensuring robust adaptation to unseen domains. We evaluate our WorMI on the VirtualHome and ALFWorld benchmarks, demonstrating superior zero-shot and few-shot performance compared to several LLM-based approaches across a range of unseen domains. These results highlight the frameworks potential for scalable, real-world deployment in embodied agent scenarios where adaptability and data efficiency are essential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03842v1">INGRID: Intelligent Generative Robotic Design Using Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into robotic systems has accelerated progress in embodied artificial intelligence, yet current approaches remain constrained by existing robotic architectures, particularly serial mechanisms. This hardware dependency fundamentally limits the scope of robotic intelligence. Here, we present INGRID (Intelligent Generative Robotic Design), a framework that enables the automated design of parallel robotic mechanisms through deep integration with reciprocal screw theory and kinematic synthesis methods. We decompose the design challenge into four progressive tasks: constraint analysis, kinematic joint generation, chain construction, and complete mechanism design. INGRID demonstrates the ability to generate novel parallel mechanisms with both fixed and variable mobility, discovering kinematic configurations not previously documented in the literature. We validate our approach through three case studies demonstrating how INGRID assists users in designing task-specific parallel robots based on desired mobility requirements. By bridging the gap between mechanism theory and machine learning, INGRID enables researchers without specialized robotics training to create custom parallel mechanisms, thereby decoupling advances in robotic intelligence from hardware constraints. This work establishes a foundation for mechanism intelligence, where AI systems actively design robotic hardware, potentially transforming the development of embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00117v2">Embodied AI: Emerging Risks and Opportunities for Policy Action</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      The field of embodied AI (EAI) is rapidly advancing. Unlike virtual AI, EAI systems can exist in, learn from, reason about, and act in the physical world. With recent advances in AI models and hardware, EAI systems are becoming increasingly capable across wider operational domains. While EAI systems can offer many benefits, they also pose significant risks, including physical harm from malicious use, mass surveillance, as well as economic and societal disruption. These risks require urgent attention from policymakers, as existing policies governing industrial robots and autonomous vehicles are insufficient to address the full range of concerns EAI systems present. To help address this issue, this paper makes three contributions. First, we provide a taxonomy of the physical, informational, economic, and social risks EAI systems pose. Second, we analyze policies in the US, EU, and UK to assess how existing frameworks address these risks and to identify critical gaps. We conclude by offering policy recommendations for the safe and beneficial deployment of EAI systems, such as mandatory testing and certification schemes, clarified liability frameworks, and strategies to manage EAI's potentially transformative economic and societal impacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03383v1">ANNIE: Be Careful of Your Robots</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      The integration of vision-language-action (VLA) models into embodied AI (EAI) robots is rapidly advancing their ability to perform complex, long-horizon tasks in humancentric environments. However, EAI systems introduce critical security risks: a compromised VLA model can directly translate adversarial perturbations on sensory input into unsafe physical actions. Traditional safety definitions and methodologies from the machine learning community are no longer sufficient. EAI systems raise new questions, such as what constitutes safety, how to measure it, and how to design effective attack and defense mechanisms in physically grounded, interactive settings. In this work, we present the first systematic study of adversarial safety attacks on embodied AI systems, grounded in ISO standards for human-robot interactions. We (1) formalize a principled taxonomy of safety violations (critical, dangerous, risky) based on physical constraints such as separation distance, velocity, and collision boundaries; (2) introduce ANNIEBench, a benchmark of nine safety-critical scenarios with 2,400 video-action sequences for evaluating embodied safety; and (3) ANNIE-Attack, a task-aware adversarial framework with an attack leader model that decomposes long-horizon goals into frame-level perturbations. Our evaluation across representative EAI models shows attack success rates exceeding 50% across all safety categories. We further demonstrate sparse and adaptive attack strategies and validate the real-world impact through physical robot experiments. These results expose a previously underexplored but highly consequential attack surface in embodied AI systems, highlighting the urgent need for security-driven defenses in the physical AI era. Code is available at https://github.com/RLCLab/Annie.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00917v3">A Survey: Learning Embodied Intelligence from Physical Simulators and World Models</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 Update with recent progresses. 49pages, 25figures, 6tables, github repository avalible in https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey
    </div>
    <details class="paper-abstract">
      The pursuit of artificial general intelligence (AGI) has placed embodied intelligence at the forefront of robotics research. Embodied intelligence focuses on agents capable of perceiving, reasoning, and acting within the physical world. Achieving robust embodied intelligence requires not only advanced perception and control, but also the ability to ground abstract cognition in real-world interactions. Two foundational technologies, physical simulators and world models, have emerged as critical enablers in this quest. Physical simulators provide controlled, high-fidelity environments for training and evaluating robotic agents, allowing safe and efficient development of complex behaviors. In contrast, world models empower robots with internal representations of their surroundings, enabling predictive planning and adaptive decision-making beyond direct sensory input. This survey systematically reviews recent advances in learning embodied AI through the integration of physical simulators and world models. We analyze their complementary roles in enhancing autonomy, adaptability, and generalization in intelligent robots, and discuss the interplay between external simulation and internal modeling in bridging the gap between simulated training and real-world deployment. By synthesizing current progress and identifying open challenges, this survey aims to provide a comprehensive perspective on the path toward more capable and generalizable embodied AI systems. We also maintain an active repository that contains up-to-date literature and open-source projects at https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02761v1">Plan Verification for LLM-Based Embodied Task Completion Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02164v1">Omnidirectional Spatial Modeling from Correlated Panoramas</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Omnidirectional scene understanding is vital for various downstream applications, such as embodied AI, autonomous driving, and immersive environments, yet remains challenging due to geometric distortion and complex spatial relations in 360{\deg} imagery. Existing omnidirectional methods achieve scene understanding within a single frame while neglecting cross-frame correlated panoramas. To bridge this gap, we introduce \textbf{CFpano}, the \textbf{first} benchmark dataset dedicated to cross-frame correlated panoramas visual question answering in the holistic 360{\deg} scenes. CFpano consists of over 2700 images together with over 8000 question-answer pairs, and the question types include both multiple choice and open-ended VQA. Building upon our CFpano, we further present \methodname, a multi-modal large language model (MLLM) fine-tuned with Group Relative Policy Optimization (GRPO) and a set of tailored reward functions for robust and consistent reasoning with cross-frame correlated panoramas. Benchmark experiments with existing MLLMs are conducted with our CFpano. The experimental results demonstrate that \methodname achieves state-of-the-art performance across both multiple-choice and open-ended VQA tasks, outperforming strong baselines on all major reasoning categories (\textbf{+5.37\%} in overall performance). Our analyses validate the effectiveness of GRPO and establish a new benchmark for panoramic scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13073v2">Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Project Page: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
    </div>
    <details class="paper-abstract">
      Robotic manipulation, a key frontier in robotics and embodied AI, requires precise motor control and multimodal understanding, yet traditional rule-based methods fail to scale or generalize in unstructured, novel environments. In recent years, Vision-Language-Action (VLA) models, built upon Large Vision-Language Models (VLMs) pretrained on vast image-text datasets, have emerged as a transformative paradigm. This survey provides the first systematic, taxonomy-oriented review of large VLM-based VLA models for robotic manipulation. We begin by clearly defining large VLM-based VLA models and delineating two principal architectural paradigms: (1) monolithic models, encompassing single-system and dual-system designs with differing levels of integration; and (2) hierarchical models, which explicitly decouple planning from execution via interpretable intermediate representations. Building on this foundation, we present an in-depth examination of large VLM-based VLA models: (1) integration with advanced domains, including reinforcement learning, training-free optimization, learning from human videos, and world model integration; (2) synthesis of distinctive characteristics, consolidating architectural traits, operational strengths, and the datasets and benchmarks that support their development; (3) identification of promising directions, including memory mechanisms, 4D perception, efficient adaptation, multi-agent cooperation, and other emerging capabilities. This survey consolidates recent advances to resolve inconsistencies in existing taxonomies, mitigate research fragmentation, and fill a critical gap through the systematic integration of studies at the intersection of large VLMs and robotic manipulation. We provide a regularly updated project page to document ongoing progress: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01547v1">FGO-SLAM: Enhancing Gaussian SLAM with Globally Consistent Opacity Radiance Field</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 ICRA 2025
    </div>
    <details class="paper-abstract">
      Visual SLAM has regained attention due to its ability to provide perceptual capabilities and simulation test data for Embodied AI. However, traditional SLAM methods struggle to meet the demands of high-quality scene reconstruction, and Gaussian SLAM systems, despite their rapid rendering and high-quality mapping capabilities, lack effective pose optimization methods and face challenges in geometric reconstruction. To address these issues, we introduce FGO-SLAM, a Gaussian SLAM system that employs an opacity radiance field as the scene representation to enhance geometric mapping performance. After initial pose estimation, we apply global adjustment to optimize camera poses and sparse point cloud, ensuring robust tracking of our approach. Additionally, we maintain a globally consistent opacity radiance field based on 3D Gaussians and introduce depth distortion and normal consistency terms to refine the scene representation. Furthermore, after constructing tetrahedral grids, we identify level sets to directly extract surfaces from 3D Gaussians. Results across various real-world and large-scale synthetic datasets demonstrate that our method achieves state-of-the-art tracking accuracy and mapping performance.
    </details>
</div>
