# embodied ai - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00839v1">TransNormal: Dense Visual Semantics for Diffusion-based Transparent Object Normal Estimation</a></div>
    <div class="paper-meta">
      📅 2026-01-31
      | 💬 Project Page: https://longxiang-ai.github.io/TransNormal
    </div>
    <details class="paper-abstract">
      Monocular normal estimation for transparent objects is critical for laboratory automation, yet it remains challenging due to complex light refraction and reflection. These optical properties often lead to catastrophic failures in conventional depth and normal sensors, hindering the deployment of embodied AI in scientific environments. We propose TransNormal, a novel framework that adapts pre-trained diffusion priors for single-step normal regression. To handle the lack of texture in transparent surfaces, TransNormal integrates dense visual semantics from DINOv3 via a cross-attention mechanism, providing strong geometric cues. Furthermore, we employ a multi-task learning objective and wavelet-based regularization to ensure the preservation of fine-grained structural details. To support this task, we introduce TransNormal-Synthetic, a physics-based dataset with high-fidelity normal maps for transparent labware. Extensive experiments demonstrate that TransNormal significantly outperforms state-of-the-art methods: on the ClearGrasp benchmark, it reduces mean error by 24.4% and improves 11.25° accuracy by 22.8%; on ClearPose, it achieves a 15.2% reduction in mean error. The code and dataset will be made publicly available at https://longxiang-ai.github.io/TransNormal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00551v1">APEX: A Decoupled Memory-based Explorer for Asynchronous Aerial Object Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2026-01-31
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Aerial Object Goal Navigation, a challenging frontier in Embodied AI, requires an Unmanned Aerial Vehicle (UAV) agent to autonomously explore, reason, and identify a specific target using only visual perception and language description. However, existing methods struggle with the memorization of complex spatial representations in aerial environments, reliable and interpretable action decision-making, and inefficient exploration and information gathering. To address these challenges, we introduce \textbf{APEX} (Aerial Parallel Explorer), a novel hierarchical agent designed for efficient exploration and target acquisition in complex aerial settings. APEX is built upon a modular, three-part architecture: 1) Dynamic Spatio-Semantic Mapping Memory, which leverages the zero-shot capability of a Vision-Language Model (VLM) to dynamically construct high-resolution 3D Attraction, Exploration, and Obstacle maps, serving as an interpretable memory mechanism. 2) Action Decision Module, trained with reinforcement learning, which translates this rich spatial understanding into a fine-grained and robust control policy. 3) Target Grounding Module, which employs an open-vocabulary detector to achieve definitive and generalizable target identification. All these components are integrated into a hierarchical, asynchronous, and parallel framework, effectively bypassing the VLM's inference latency and boosting the agent's proactivity in exploration. Extensive experiments show that APEX outperforms the previous state of the art by +4.2\% SR and +2.8\% SPL on challenging UAV-ON benchmarks, demonstrating its superior efficiency and the effectiveness of its hierarchical asynchronous design. Our source code is provided in \href{https://github.com/4amGodvzx/apex}{GitHub}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00500v1">Inject Once Survive Later: Backdooring Vision-Language-Action Models to Persist Through Downstream Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-31
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have become foundational to modern embodied AI systems. By integrating visual perception, language understanding, and action planning, they enable general-purpose task execution across diverse environments. Despite their importance, the security of VLA models remains underexplored -- particularly in the context of backdoor attacks, which pose realistic threats in physical-world deployments. While recent methods attempt to inject backdoors into VLA models, these backdoors are easily erased during downstream adaptation, as user-side fine-tuning with clean data significantly alters model parameters, rendering them impractical for real-world applications. To address these challenges, we propose INFUSE (INjection into Fine-tUne-inSensitive modulEs), the first backdoor attack framework for VLA base models that remains effective even with arbitrary user fine-tuning. INFUSE begins by analyzing parameter sensitivity across diverse fine-tuning scenarios to identify modules that remain largely unchanged -- the fine-tune-insensitive modules. It then injects backdoors into these stable modules while freezing the rest, ensuring malicious behavior persists after extensive user fine-tuning. Comprehensive experiments across multiple VLA architectures demonstrate INFUSE's effectiveness. After user-side fine-tuning, INFUSE maintains mean attack success rates of 91.0% on simulation environments and 79.8% on real-world robot tasks, substantially surpassing BadVLA (38.8% and 36.6%, respectively), while preserving clean-task performance comparable to standard models. These results uncover a critical threat: backdoors implanted before distribution can persist through fine-tuning and remain effective at deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00494v1">SRL Proxemics: Spatial Guidelines for Supernumerary Robotic Limbs in Near-Body Interactions</a></div>
    <div class="paper-meta">
      📅 2026-01-31
      | 💬 Accepted to CHI 2026. Version of Record: DOI 10.1145/3772318.3790532
    </div>
    <details class="paper-abstract">
      Wearable supernumerary robotic limbs (SRLs) sit at the intersection of human augmentation and embodied AI, transforming into extensions of the human body. However, their movements within the intimate near-body space raise unresolved challenges for perceived safety, user control, and trust. In this paper, we present results from a Wizard-of-Oz study (n=18), where participants completed near-body collaboration tasks with SRLs to explore these challenges. We collected qualitative data through think-aloud protocols and semi-structured interviews, complemented by physiological signals and post-task ratings. Findings indicate that greater autonomy did not inherently enhance perceived safety or trust. Instead, participants identified near-body zones and paired them with clear coordination rules. They also expressed expectations for how different arm components should behave, shaping preferences around autonomy, perceived safety, and trust. Building on these insights, we introduce SRL Proxemics, a zone- and segment-level design framework showing that autonomy is not monolithic: perceived safety hinges on spatially calibrated, legible behaviors, not higher autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20835v2">Open-Vocabulary Functional 3D Human-Scene Interaction Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Generating 3D humans that functionally interact with 3D scenes remains an open problem with applications in embodied AI, robotics, and interactive content creation. The key challenge involves reasoning about both the semantics of functional elements in 3D scenes and the 3D human poses required to achieve functionality-aware interaction. Unfortunately, existing methods typically lack explicit reasoning over object functionality and the corresponding human-scene contact, resulting in implausible or functionally incorrect interactions. In this work, we propose FunHSI, a training-free, functionality-driven framework that enables functionally correct human-scene interactions from open-vocabulary task prompts. Given a task prompt, FunHSI performs functionality-aware contact reasoning to identify functional scene elements, reconstruct their 3D geometry, and model high-level interactions via a contact graph. We then leverage vision-language models to synthesize a human performing the task in the image and estimate proposed 3D body and hand poses. Finally, the proposed 3D body configuration is refined via stage-wise optimization to ensure physical plausibility and functional correctness. In contrast to existing methods, FunHSI not only synthesizes more plausible general 3D interactions, such as "sitting on a sofa'', while supporting fine-grained functional human-scene interactions, e.g., "increasing the room temperature''. Extensive experiments demonstrate that FunHSI consistently generates functionally correct and physically plausible human-scene interactions across diverse indoor and outdoor scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23065v1">EAG-PT: Emission-Aware Gaussians and Path Tracing for Indoor Scene Reconstruction and Editing</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 project page: https://eag-pt.github.io
    </div>
    <details class="paper-abstract">
      Recent reconstruction methods based on radiance field such as NeRF and 3DGS reproduce indoor scenes with high visual fidelity, but break down under scene editing due to baked illumination and the lack of explicit light transport. In contrast, physically based inverse rendering relies on mesh representations and path tracing, which enforce correct light transport but place strong requirements on geometric fidelity, becoming a practical bottleneck for real indoor scenes. In this work, we propose Emission-Aware Gaussians and Path Tracing (EAG-PT), aiming for physically based light transport with a unified 2D Gaussian representation. Our design is based on three cores: (1) using 2D Gaussians as a unified scene representation and transport-friendly geometry proxy that avoids reconstructed mesh, (2) explicitly separating emissive and non-emissive components during reconstruction for further scene editing, and (3) decoupling reconstruction from final rendering by using efficient single-bounce optimization and high-quality multi-bounce path tracing after scene editing. Experiments on synthetic and real indoor scenes show that EAG-PT produces more natural and physically consistent renders after editing than radiant scene reconstructions, while preserving finer geometric detail and avoiding mesh-induced artifacts compared to mesh-based inverse path tracing. These results suggest promising directions for future use in interior design, XR content creation, and embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22948v1">Alignment among Language, Vision and Action Representations</a></div>
    <div class="paper-meta">
      📅 2026-01-30
    </div>
    <details class="paper-abstract">
      A fundamental question in cognitive science and AI concerns whether different learning modalities: language, vision, and action, give rise to distinct or shared internal representations. Traditional views assume that models trained on different data types develop specialized, non-transferable representations. However, recent evidence suggests unexpected convergence: models optimized for distinct tasks may develop similar representational geometries. We investigate whether this convergence extends to embodied action learning by training a transformer-based agent to execute goal-directed behaviors in response to natural language instructions. Using behavioral cloning on the BabyAI platform, we generated action-grounded language embeddings shaped exclusively by sensorimotor control requirements. We then compared these representations with those extracted from state-of-the-art large language models (LLaMA, Qwen, DeepSeek, BERT) and vision-language models (CLIP, BLIP). Despite substantial differences in training data, modality, and objectives, we observed robust cross-modal alignment. Action representations aligned strongly with decoder-only language models and BLIP (precision@15: 0.70-0.73), approaching the alignment observed among language models themselves. Alignment with CLIP and BERT was significantly weaker. These findings indicate that linguistic, visual, and action representations converge toward partially shared semantic structures, supporting modality-independent semantic organization and highlighting potential for cross-domain transfer in embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22046v2">PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 Project page: https://city-super.github.io/PLANING/
    </div>
    <details class="paper-abstract">
      Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of PLANING make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00288v1">TimeBlind: A Spatio-Temporal Compositionality Benchmark for Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-30
      | 💬 For code and data, see https://baiqi-li.github.io/timeblind_project/
    </div>
    <details class="paper-abstract">
      Fine-grained spatio-temporal understanding is essential for video reasoning and embodied AI. Yet, while Multimodal Large Language Models (MLLMs) master static semantics, their grasp of temporal dynamics remains brittle. We present TimeBlind, a diagnostic benchmark for compositional spatio-temporal understanding. Inspired by cognitive science, TimeBlind categorizes fine-grained temporal understanding into three levels: recognizing atomic events, characterizing event properties, and reasoning about event interdependencies. Unlike benchmarks that conflate recognition with temporal reasoning, TimeBlind leverages a minimal-pairs paradigm: video pairs share identical static visual content but differ solely in temporal structure, utilizing complementary questions to neutralize language priors. Evaluating over 20 state-of-the-art MLLMs (e.g., GPT-5, Gemini 3 Pro) on 600 curated instances (2400 video-question pairs), reveals that the Instance Accuracy (correctly distinguishing both videos in a pair) of the best performing MLLM is only 48.2%, far below the human performance (98.2%). These results demonstrate that even frontier models rely heavily on static visual shortcuts rather than genuine temporal logic, positioning TimeBlind as a vital diagnostic tool for next-generation video understanding. Dataset and code are available at https://baiqi-li.github.io/timeblind_project/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22046v1">PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of \modelname~make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21570v1">EmboCoach-Bench: Benchmarking AI Agents on Developing Embodied Robots</a></div>
    <div class="paper-meta">
      📅 2026-01-29
      | 💬 37 pages, 13 figures
    </div>
    <details class="paper-abstract">
      The field of Embodied AI is witnessing a rapid evolution toward general-purpose robotic systems, fueled by high-fidelity simulation and large-scale data collection. However, this scaling capability remains severely bottlenecked by a reliance on labor-intensive manual oversight from intricate reward shaping to hyperparameter tuning across heterogeneous backends. Inspired by LLMs' success in software automation and science discovery, we introduce \textsc{EmboCoach-Bench}, a benchmark evaluating the capacity of LLM agents to autonomously engineer embodied policies. Spanning 32 expert-curated RL and IL tasks, our framework posits executable code as the universal interface. We move beyond static generation to assess a dynamic closed-loop workflow, where agents leverage environment feedback to iteratively draft, debug, and optimize solutions, spanning improvements from physics-informed reward design to policy architectures such as diffusion policies. Extensive evaluations yield three critical insights: (1) autonomous agents can qualitatively surpass human-engineered baselines by 26.5\% in average success rate; (2) agentic workflow with environment feedback effectively strengthens policy development and substantially narrows the performance gap between open-source and proprietary models; and (3) agents exhibit self-correction capabilities for pathological engineering cases, successfully resurrecting task performance from near-total failures through iterative simulation-in-the-loop debugging. Ultimately, this work establishes a foundation for self-evolving embodied intelligence, accelerating the paradigm shift from labor-intensive manual tuning to scalable, autonomous engineering in embodied AI field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13421v2">Virtuous Machines: Towards Artificial General Science</a></div>
    <div class="paper-meta">
      📅 2026-01-29
    </div>
    <details class="paper-abstract">
      Artificial intelligence systems are transforming scientific discovery by accelerating specific research tasks, from protein structure prediction to materials design, yet remain confined to narrow domains requiring substantial human oversight. The exponential growth of scientific literature and increasing domain specialisation constrain researchers' capacity to synthesise knowledge across disciplines and develop unifying theories, motivating exploration of more general-purpose AI systems for science. Here we show that a domain-agnostic, agentic AI Scientist system can independently navigate the scientific workflow - from hypothesis generation through data collection to manuscript preparation. The system autonomously designed and executed three psychological studies on visual working memory, mental rotation, and imagery vividness, executed one new online data collection with 288 participants, developed analysis pipelines through 8-hour+ continuous coding sessions, and produced completed manuscripts. The results demonstrate the capability of AI scientific discovery pipelines to conduct non-trivial research with theoretical reasoning and methodological rigour comparable to experienced researchers, though with limitations in conceptual nuance and theoretical interpretation. This is a step toward embodied AI that can test hypotheses through real-world experiments, accelerating discovery by autonomously exploring regions of scientific space that human cognitive and resource constraints might otherwise leave unexplored. It raises important questions about the nature of scientific understanding and the attribution of scientific credit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20835v1">Open-Vocabulary Functional 3D Human-Scene Interaction Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Generating 3D humans that functionally interact with 3D scenes remains an open problem with applications in embodied AI, robotics, and interactive content creation. The key challenge involves reasoning about both the semantics of functional elements in 3D scenes and the 3D human poses required to achieve functionality-aware interaction. Unfortunately, existing methods typically lack explicit reasoning over object functionality and the corresponding human-scene contact, resulting in implausible or functionally incorrect interactions. In this work, we propose FunHSI, a training-free, functionality-driven framework that enables functionally correct human-scene interactions from open-vocabulary task prompts. Given a task prompt, FunHSI performs functionality-aware contact reasoning to identify functional scene elements, reconstruct their 3D geometry, and model high-level interactions via a contact graph. We then leverage vision-language models to synthesize a human performing the task in the image and estimate proposed 3D body and hand poses. Finally, the proposed 3D body configuration is refined via stage-wise optimization to ensure physical plausibility and functional correctness. In contrast to existing methods, FunHSI not only synthesizes more plausible general 3D interactions, such as "sitting on a sofa'', while supporting fine-grained functional human-scene interactions, e.g., "increasing the room temperature''. Extensive experiments demonstrate that FunHSI consistently generates functionally correct and physically plausible human-scene interactions across diverse indoor and outdoor scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20742v1">Compression Tells Intelligence: Visual Coding, Visual Token Technology, and the Unification</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      "Compression Tells Intelligence", is supported by research in artificial intelligence, particularly concerning (multimodal) large language models (LLMs/MLLMs), where compression efficiency often correlates with improved model performance and capabilities. For compression, classical visual coding based on traditional information theory has developed over decades, achieving great success with numerous international industrial standards widely applied in multimedia (e.g., image/video) systems. Except that, the recent emergingvisual token technology of generative multi-modal large models also shares a similar fundamental objective like visual coding: maximizing semantic information fidelity during the representation learning while minimizing computational cost. Therefore, this paper provides a comprehensive overview of two dominant technique families first -- Visual Coding and Vision Token Technology -- then we further unify them from the aspect of optimization, discussing the essence of compression efficiency and model performance trade-off behind. Next, based on the proposed unified formulation bridging visual coding andvisual token technology, we synthesize bidirectional insights of themselves and forecast the next-gen visual codec and token techniques. Last but not least, we experimentally show a large potential of the task-oriented token developments in the more practical tasks like multimodal LLMs (MLLMs), AI-generated content (AIGC), and embodied AI, as well as shedding light on the future possibility of standardizing a general token technology like the traditional codecs (e.g., H.264/265) with high efficiency for a wide range of intelligent tasks in a unified and effective manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20377v1">RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Accurate material identification plays a crucial role in embodied AI systems, enabling a wide range of applications. However, current vision-based solutions are limited by the inherent constraints of optical sensors, while radio-frequency (RF) approaches, which can reveal intrinsic material properties, have received growing attention. Despite this progress, RF-based material identification remains hindered by the lack of large-scale public datasets and the limited benchmarking of learning-based approaches. In this work, we present RF-MatID, the first open-source, large-scale, wide-band, and geometry-diverse RF dataset for fine-grained material identification. RF-MatID includes 16 fine-grained categories grouped into 5 superclasses, spanning a broad frequency range from 4 to 43.5 GHz, and comprises 142k samples in both frequency- and time-domain representations. The dataset systematically incorporates controlled geometry perturbations, including variations in incidence angle and stand-off distance. We further establish a multi-setting, multi-protocol benchmark by evaluating state-of-the-art deep learning models, assessing both in-distribution performance and out-of-distribution robustness under cross-angle and cross-distance shifts. The 5 frequency-allocation protocols enable systematic frequency- and region-level analysis, thereby facilitating real-world deployment. RF-MatID aims to enable reproducible research, accelerate algorithmic advancement, foster cross-domain robustness, and support the development of real-world application in RF-based material identification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20503v2">Embodied AI with Foundation Models for Mobile Service Robots: A Systematic Review</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 v2: Expanded systematic review; resubmitted to Robotics
    </div>
    <details class="paper-abstract">
      Rapid advancements in foundation models, including Large Language Models, Vision-Language Models, Multimodal Large Language Models, and Vision-Language-Action Models, have opened new avenues for embodied AI in mobile service robotics. By combining foundation models with the principles of embodied AI, where intelligent systems perceive, reason, and act through physical interaction, mobile service robots can achieve more flexible understanding, adaptive behavior, and robust task execution in dynamic real-world environments. Despite this progress, embodied AI for mobile service robots continues to face fundamental challenges related to the translation of natural language instructions into executable robot actions, multimodal perception in human-centered environments, uncertainty estimation for safe decision-making, and computational constraints for real-time onboard deployment. In this paper, we present the first systematic review focused specifically on the integration of foundation models in mobile service robotics. We analyze how recent advances in foundation models address these core challenges through language-conditioned control, multimodal sensor fusion, uncertainty-aware reasoning, and efficient model scaling. We further examine real-world applications in domestic assistance, healthcare, and service automation, highlighting how foundation models enable context-aware, socially responsive, and generalizable robot behaviors. Beyond technical considerations, we discuss ethical, societal, and human-interaction implications associated with deploying foundation model-enabled service robots in human environments. Finally, we outline future research directions emphasizing reliability and lifelong adaptation, privacy-aware and resource-constrained deployment, and governance and human-in-the-loop frameworks required for safe, scalable, and trustworthy mobile service robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21124v1">PhaseCoder: Microphone Geometry-Agnostic Spatial Audio Understanding for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Current multimodal LLMs process audio as a mono stream, ignoring the rich spatial information essential for embodied AI. Existing spatial audio models, conversely, are constrained to fixed microphone geometries, preventing deployment across diverse devices. We present PhaseCoder, a transformer-only spatial audio encoder that is agnostic to microphone geometry. PhaseCoder takes raw multichannel audio and microphone coordinates as inputs to perform localization and produces robust spatial embeddings. We demonstrate that Gemma 3n LLM can be fine-tuned to reason over "Spatial Audio Tokens" produced by PhaseCoder. We show our encoder achieves state-of-the-art results on microphone-invariant localization benchmarks and, for the first time, enables an LLM to perform complex spatial reasoning and targeted transcription tasks from an arbitrary microphone array.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19233v1">UniMGS: Unifying Mesh and 3D Gaussian Splatting with Single-Pass Rasterization and Proxy-Based Deformation</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 conference
    </div>
    <details class="paper-abstract">
      Joint rendering and deformation of mesh and 3D Gaussian Splatting (3DGS) have significant value as both representa tions offer complementary advantages for graphics applica tions. However, due to differences in representation and ren dering pipelines, existing studies render meshes and 3DGS separately, making it difficult to accurately handle occlusions and transparency. Moreover, the deformed 3DGS still suffers from visual artifacts due to the sensitivity to the topology quality of the proxy mesh. These issues pose serious obsta cles to the joint use of 3DGS and meshes, making it diffi cult to adapt 3DGS to conventional mesh-oriented graphics pipelines. We propose UniMGS, the first unified framework for rasterizing mesh and 3DGS in a single-pass anti-aliased manner, with a novel binding strategy for 3DGS deformation based on proxy mesh. Our key insight is to blend the col ors of both triangle and Gaussian fragments by anti-aliased α-blending in a single pass, achieving visually coherent re sults with precise handling of occlusion and transparency. To improve the visual appearance of the deformed 3DGS, our Gaussian-centric binding strategy employs a proxy mesh and spatially associates Gaussians with the mesh faces, signifi cantly reducing rendering artifacts. With these two compo nents, UniMGS enables the visualization and manipulation of 3D objects represented by mesh or 3DGS within a unified framework, opening up new possibilities in embodied AI, vir tual reality, and gaming. We will release our source code to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18733v1">Advances and Innovations in the Multi-Agent Robotic System (MARS) Challenge</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 MARS Challenge @ NeurIPS 2025 Workshop on Space in Vision, Language, and Embodied AI. Challenge page: https://mars-eai.github.io/MARS-Challenge-Webpage/
    </div>
    <details class="paper-abstract">
      Recent advancements in multimodal large language models and vision-languageaction models have significantly driven progress in Embodied AI. As the field transitions toward more complex task scenarios, multi-agent system frameworks are becoming essential for achieving scalable, efficient, and collaborative solutions. This shift is fueled by three primary factors: increasing agent capabilities, enhancing system efficiency through task delegation, and enabling advanced human-agent interactions. To address the challenges posed by multi-agent collaboration, we propose the Multi-Agent Robotic System (MARS) Challenge, held at the NeurIPS 2025 Workshop on SpaVLE. The competition focuses on two critical areas: planning and control, where participants explore multi-agent embodied planning using vision-language models (VLMs) to coordinate tasks and policy execution to perform robotic manipulation in dynamic environments. By evaluating solutions submitted by participants, the challenge provides valuable insights into the design and coordination of embodied multi-agent systems, contributing to the future development of advanced collaborative AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18323v1">TC-IDM: Grounding Video Generation for Executable Zero-shot Robot Motion</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The vision-language-action (VLA) paradigm has enabled powerful robotic control by leveraging vision-language models, but its reliance on large-scale, high-quality robot data limits its generalization. Generative world models offer a promising alternative for general-purpose embodied AI, yet a critical gap remains between their pixel-level plans and physically executable actions. To this end, we propose the Tool-Centric Inverse Dynamics Model (TC-IDM). By focusing on the tool's imagined trajectory as synthesized by the world model, TC-IDM establishes a robust intermediate representation that bridges the gap between visual planning and physical control. TC-IDM extracts the tool's point cloud trajectories via segmentation and 3D motion estimation from generated videos. Considering diverse tool attributes, our architecture employs decoupled action heads to project these planned trajectories into 6-DoF end-effector motions and corresponding control signals. This plan-and-translate paradigm not only supports a wide range of end-effectors but also significantly improves viewpoint invariance. Furthermore, it exhibits strong generalization capabilities across long-horizon and out-of-distribution tasks, including interacting with deformable objects. In real-world evaluations, the world model with TC-IDM achieves an average success rate of 61.11 percent, with 77.7 percent on simple tasks and 38.46 percent on zero-shot deformable object tasks. It substantially outperforms end-to-end VLA-style baselines and other inverse dynamics models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17657v1">SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Contrastive Language-Image Pre-training (CLIP) has accomplished extraordinary success for semantic understanding but inherently struggles to perceive geometric structure. Existing methods attempt to bridge this gap by querying CLIP with textual prompts, a process that is often indirect and inefficient. This paper introduces a fundamentally different approach using a dual-pathway decoder. We present SPACE-CLIP, an architecture that unlocks and interprets latent geometric knowledge directly from a frozen CLIP vision encoder, completely bypassing the text encoder and its associated textual prompts. A semantic pathway interprets high-level features, dynamically conditioned on global context using feature-wise linear modulation (FiLM). In addition, a structural pathway extracts fine-grained spatial details from early layers. These complementary streams are hierarchically fused, enabling a robust synthesis of semantic context and precise geometry. Extensive experiments on the KITTI benchmark show that SPACE-CLIP dramatically outperforms previous CLIP-based methods. Our ablation studies validate that the synergistic fusion of our dual pathways is critical to this success. SPACE-CLIP offers a new, efficient, and architecturally elegant blueprint for repurposing large-scale vision models. The proposed method is not just a standalone depth estimator, but a readily integrable spatial perception module for the next generation of embodied AI systems, such as vision-language-action (VLA) models. Our model is available at https://github.com/taewan2002/space-clip
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10046v3">SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Conference: NeurIPS 2025 (main)
    </div>
    <details class="paper-abstract">
      Recent advances in foundation models have shown promising results in developing generalist robotics that can perform diverse tasks in open-ended scenarios given multimodal inputs. However, current work has been mainly focused on indoor, household scenarios. In this work, we present SimWorld-Robotics~(SWR), a simulation platform for embodied AI in large-scale, photorealistic urban environments. Built on Unreal Engine 5, SWR procedurally generates unlimited photorealistic urban scenes populated with dynamic elements such as pedestrians and traffic systems, surpassing prior urban simulations in realism, complexity, and scalability. It also supports multi-robot control and communication. With these key features, we build two challenging robot benchmarks: (1) a multimodal instruction-following task, where a robot must follow vision-language navigation instructions to reach a destination in the presence of pedestrians and traffic; and (2) a multi-agent search task, where two robots must communicate to cooperatively locate and meet each other. Unlike existing benchmarks, these two new benchmarks comprehensively evaluate a wide range of critical robot capacities in realistic scenarios, including (1) multimodal instructions grounding, (2) 3D spatial reasoning in large environments, (3) safe, long-range navigation with people and traffic, (4) multi-robot collaboration, and (5) grounded communication. Our experimental results demonstrate that state-of-the-art models, including vision-language models (VLMs), struggle with our tasks, lacking robust perception, reasoning, and planning abilities necessary for urban environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10046v2">SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Conference: NeurIPS 2025 (main)
    </div>
    <details class="paper-abstract">
      Recent advances in foundation models have shown promising results in developing generalist robotics that can perform diverse tasks in open-ended scenarios given multimodal inputs. However, current work has been mainly focused on indoor, household scenarios. In this work, we present SimWorld-Robotics~(SWR), a simulation platform for embodied AI in large-scale, photorealistic urban environments. Built on Unreal Engine 5, SWR procedurally generates unlimited photorealistic urban scenes populated with dynamic elements such as pedestrians and traffic systems, surpassing prior urban simulations in realism, complexity, and scalability. It also supports multi-robot control and communication. With these key features, we build two challenging robot benchmarks: (1) a multimodal instruction-following task, where a robot must follow vision-language navigation instructions to reach a destination in the presence of pedestrians and traffic; and (2) a multi-agent search task, where two robots must communicate to cooperatively locate and meet each other. Unlike existing benchmarks, these two new benchmarks comprehensively evaluate a wide range of critical robot capacities in realistic scenarios, including (1) multimodal instructions grounding, (2) 3D spatial reasoning in large environments, (3) safe, long-range navigation with people and traffic, (4) multi-robot collaboration, and (5) grounded communication. Our experimental results demonstrate that state-of-the-art models, including vision-language models (VLMs), struggle with our tasks, lacking robust perception, reasoning, and planning abilities necessary for urban environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09049v3">VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Accepted by NeurIPS 2025 Track on Datasets and Benchmarks. Project page: https://faceong.github.io/VIKI-R/
    </div>
    <details class="paper-abstract">
      Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09954v2">The Spatial Blindspot of Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Work done as part of the EleutherAI SOAR Program
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) have advanced rapidly, but their ability to capture spatial relationships remains a blindspot. Current VLMs are typically built with contrastive language-image pretraining (CLIP) style image encoders. The training recipe often flattens images into 1D patch sequences, discarding the 2D structure necessary for spatial reasoning. We argue that this lack of spatial awareness is a missing dimension in VLM design and a bottleneck for applications requiring spatial grounding, such as robotics and embodied AI. To address this, we investigate (i) image encoders trained with alternative objectives and (ii) 2D positional encodings. Our experiments show that these architectural choices can lead to improved spatial reasoning on several benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15282v1">Rethinking Video Generation Model for the Embodied World</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Github: https://github.com/DAGroup-PKU/ReVidgen/ Project website: https://dagroup-pku.github.io/ReVidgen.github.io/
    </div>
    <details class="paper-abstract">
      Video generation models have significantly advanced embodied intelligence, unlocking new possibilities for generating diverse robot data that capture perception, reasoning, and action in the physical world. However, synthesizing high-quality videos that accurately reflect real-world robotic interactions remains challenging, and the lack of a standardized benchmark limits fair comparisons and progress. To address this gap, we introduce a comprehensive robotics benchmark, RBench, designed to evaluate robot-oriented video generation across five task domains and four distinct embodiments. It assesses both task-level correctness and visual fidelity through reproducible sub-metrics, including structural consistency, physical plausibility, and action completeness. Evaluation of 25 representative models highlights significant deficiencies in generating physically realistic robot behaviors. Furthermore, the benchmark achieves a Spearman correlation coefficient of 0.96 with human evaluations, validating its effectiveness. While RBench provides the necessary lens to identify these deficiencies, achieving physical realism requires moving beyond evaluation to address the critical shortage of high-quality training data. Driven by these insights, we introduce a refined four-stage data pipeline, resulting in RoVid-X, the largest open-source robotic dataset for video generation with 4 million annotated video clips, covering thousands of tasks and enriched with comprehensive physical property annotations. Collectively, this synergistic ecosystem of evaluation and data establishes a robust foundation for rigorous assessment and scalable training of video models, accelerating the evolution of embodied AI toward general intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14140v1">CREATE: Cross-Layer Resilience Characterization and Optimization for Efficient yet Reliable Embodied AI Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 18 pages, 21 figures. Accepted by ASPLOS 2026
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (AI) has recently attracted significant attention as it bridges AI with the physical world. Modern embodied AI systems often combine a Large Language Model (LLM)-based planner for high-level task planning and a reinforcement learning (RL)-based controller for low-level action generation, enabling embodied agents to tackle complex tasks in real-world environments. However, deploying embodied agents remains challenging due to their high computation requirements, especially for battery-powered local devices. Although techniques like lowering operating voltage can improve energy efficiency, they can introduce bit errors and result in task failures. In this work, we propose CREATE, a general design principle that leverages heterogeneous resilience at different layers for synergistic energy-reliability co-optimization. For the first time, we conduct a comprehensive error injection study on modern embodied AI systems and observe an inherent but heterogeneous fault tolerance. Building upon these insights, we develop an anomaly detection and clearance mechanism at the circuit level to eliminate outlier errors. At the model level, we propose a weight-rotation-enhanced planning algorithm to improve the fault tolerance of the LLM-based planner. Furthermore, we introduce an application-level technique, autonomy-adaptive voltage scaling, to dynamically adjust the operating voltage of the controllers. The voltage scaling circuit is co-designed to enable online voltage adjustment. Extensive experiments demonstrate that without compromising task quality, CREATE achieves 40.6% computational energy savings on average over nominal-voltage baselines and 35.0% over prior-art techniques. This further leads to 29.5% to 37.3% chip-level energy savings and approximately a 15% to 30% improvement in battery life.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13945v1">Efficient Coordination with the System-Level Shared State: An Embodied-AI Native Modular Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      As Embodied AI systems move from research prototypes to real world deployments, they tend to evolve rapidly while remaining reliable under workload changes and partial failures. In practice, many deployments are only partially decoupled: middleware moves messages, but shared context and feedback semantics are implicit, causing interface drift, cross-module interference, and brittle recovery at scale. We present ANCHOR, a modular framework that makes decoupling and robustness explicit system-level primitives. ANCHOR separates (i) Canonical Records, an evolvable contract for the standardized shared state, from (ii) a communication bus for many-to-many dissemination and feedback-oriented coordination, forming an inspectable end-to-end loop. We validate closed-loop feasibility on a de-identified workflow instantiation, characterize latency distributions under varying payload sizes and publish rates, and demonstrate automatic stream resumption after hard crashes and restarts even with shared-memory loss. Overall, ANCHOR turns ad-hoc integration glue into explicit contracts, enabling controlled degradation under load and self-healing recovery for scalable deployment of closed-loop AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.09680v3">Object-Centric Latent Action Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted by AAAI 2026 (Oral). Source code: https://github.com/dunnolab/object-centric-lapo
    </div>
    <details class="paper-abstract">
      Leveraging vast amounts of unlabeled internet video data for embodied AI is currently bottlenecked by the lack of action labels and the presence of action-correlated visual distractors. Although recent latent action policy optimization (LAPO) has shown promise in inferring proxy action labels from visual observations, its performance degrades significantly when distractors are present. To address this limitation, we propose a novel object-centric latent action learning framework that centers on objects rather than pixels. We leverage self-supervised object-centric pretraining to disentangle the movement of the agent and distracting background dynamics. This allows LAPO to focus on task-relevant interactions, resulting in more robust proxy-action labels, enabling better imitation learning and efficient adaptation of the agent with just a few action-labeled trajectories. We evaluated our method in eight visually complex tasks across the Distracting Control Suite (DCS) and Distracting MetaWorld (DMW). Our results show that object-centric pretraining mitigates the negative effects of distractors by 50%, as measured by downstream task performance: average return (DCS) and success rate (DMW).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13556v1">LogicEnvGen: Task-Logic Driven Generation of Diverse Simulated Environments for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 19 pages, 15 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Simulated environments play an essential role in embodied AI, functionally analogous to test cases in software engineering. However, existing environment generation methods often emphasize visual realism (e.g., object diversity and layout coherence), overlooking a crucial aspect: logical diversity from the testing perspective. This limits the comprehensive evaluation of agent adaptability and planning robustness in distinct simulated environments. To bridge this gap, we propose LogicEnvGen, a novel method driven by Large Language Models (LLMs) that adopts a top-down paradigm to generate logically diverse simulated environments as test cases for agents. Given an agent task, LogicEnvGen first analyzes its execution logic to construct decision-tree-structured behavior plans and then synthesizes a set of logical trajectories. Subsequently, it adopts a heuristic algorithm to refine the trajectory set, reducing redundant simulation. For each logical trajectory, which represents a potential task situation, LogicEnvGen correspondingly instantiates a concrete environment. Notably, it employs constraint solving for physical plausibility. Furthermore, we introduce LogicEnvEval, a novel benchmark comprising four quantitative metrics for environment evaluation. Experimental results verify the lack of logical diversity in baselines and demonstrate that LogicEnvGen achieves 1.04-2.61x greater diversity, significantly improving the performance in revealing agent faults by 4.00%-68.00%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14352v1">RoboBrain 2.5: Depth in Sight, Time in Mind</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 37 pages, 13 figures, Technical Report
    </div>
    <details class="paper-abstract">
      We introduce RoboBrain 2.5, a next-generation embodied AI foundation model that advances general perception, spatial reasoning, and temporal modeling through extensive training on high-quality spatiotemporal supervision. Building upon its predecessor, RoboBrain 2.5 introduces two major capability upgrades. Specifically, it unlocks Precise 3D Spatial Reasoning by shifting from 2D pixel-relative grounding to depth-aware coordinate prediction and absolute metric constraint comprehension, generating complete 3D manipulation traces as ordered keypoint sequences under physical constraints. Complementing this spatial precision, the model establishes Dense Temporal Value Estimation that provides dense, step-aware progress prediction and execution state understanding across varying viewpoints, producing stable feedback signals for downstream learning. Together, these upgrades extend the framework toward more physically grounded and execution-aware embodied intelligence for complex, fine-grained manipulation. The code and checkpoints are available at project website: https://superrobobrain.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14339v1">CityCube: Benchmarking Cross-view Spatial Reasoning on Vision-Language Models in Urban Environments</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Cross-view spatial reasoning is essential for embodied AI, underpinning spatial understanding, mental simulation and planning in complex environments. Existing benchmarks primarily emphasize indoor or street settings, overlooking the unique challenges of open-ended urban spaces characterized by rich semantics, complex geometries, and view variations. To address this, we introduce CityCube, a systematic benchmark designed to probe cross-view reasoning capabilities of current VLMs in urban settings. CityCube integrates four viewpoint dynamics to mimic camera movements and spans a wide spectrum of perspectives from multiple platforms, e.g., vehicles, drones and satellites. For a comprehensive assessment, it features 5,022 meticulously annotated multi-view QA pairs categorized into five cognitive dimensions and three spatial relation expressions. A comprehensive evaluation of 33 VLMs reveals a significant performance disparity with humans: even large-scale models struggle to exceed 54.1% accuracy, remaining 34.2% below human performance. By contrast, small-scale fine-tuned VLMs achieve over 60.0% accuracy, highlighting the necessity of our benchmark. Further analyses indicate the task correlations and fundamental cognitive disparity between VLMs and human-like reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.14093v6">A Survey on Vision-Language-Action Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Project page: https://github.com/yueen-ma/Awesome-VLA
    </div>
    <details class="paper-abstract">
      Embodied AI is widely recognized as a cornerstone of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models -- referred to as vision-language-action models (VLAs) -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. The recent proliferation of VLAs necessitates a comprehensive survey to capture the rapidly evolving landscape. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing VLA-based control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges facing VLAs and outline promising future directions in embodied AI. A curated repository associated with this survey is available at: https://github.com/yueen-ma/Awesome-VLA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11421v1">The Great March 100: 100 Detail-oriented Tasks for Evaluating Embodied AI Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Recently, with the rapid development of robot learning and imitation learning, numerous datasets and methods have emerged. However, these datasets and their task designs often lack systematic consideration and principles. This raises important questions: Do the current datasets and task designs truly advance the capabilities of robotic agents? Do evaluations on a few common tasks accurately reflect the differentiated performance of various methods proposed by different teams and evaluated on different tasks? To address these issues, we introduce the Great March 100 (\textbf{GM-100}) as the first step towards a robot learning Olympics. GM-100 consists of 100 carefully designed tasks that cover a wide range of interactions and long-tail behaviors, aiming to provide a diverse and challenging set of tasks to comprehensively evaluate the capabilities of robotic agents and promote diversity and complexity in robot dataset task designs. These tasks are developed through systematic analysis and expansion of existing task designs, combined with insights from human-object interaction primitives and object affordances. We collect a large amount of trajectory data on different robotic platforms and evaluate several baseline models. Experimental results demonstrate that the GM-100 tasks are 1) feasible to execute and 2) sufficiently challenging to effectively differentiate the performance of current VLA models. Our data and code are available at https://rhos.ai/research/gm-100.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05810v2">SceneFoundry: Generating Interactive Infinite 3D Worlds</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The ability to automatically generate large-scale, interactive, and physically realistic 3D environments is crucial for advancing robotic learning and embodied intelligence. However, existing generative approaches often fail to capture the functional complexity of real-world interiors, particularly those containing articulated objects with movable parts essential for manipulation and navigation. This paper presents SceneFoundry, a language-guided diffusion framework that generates apartment-scale 3D worlds with functionally articulated furniture and semantically diverse layouts for robotic training. From natural language prompts, an LLM module controls floor layout generation, while diffusion-based posterior sampling efficiently populates the scene with articulated assets from large-scale 3D repositories. To ensure physical usability, SceneFoundry employs differentiable guidance functions to regulate object quantity, prevent articulation collisions, and maintain sufficient walkable space for robotic navigation. Extensive experiments demonstrate that our framework generates structurally valid, semantically coherent, and functionally interactive environments across diverse scene types and conditions, enabling scalable embodied AI research. project page: https://anc891203.github.io/SceneFoundry-Demo/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08355v2">Semantic Misalignment in Vision-Language Models under Perceptual Degradation</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 10 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are increasingly deployed in autonomous driving and embodied AI systems, where reliable perception is critical for safe semantic reasoning and decision-making. While recent VLMs demonstrate strong performance on multimodal benchmarks, their robustness to realistic perception degradation remains poorly understood. In this work, we systematically study semantic misalignment in VLMs under controlled degradation of upstream visual perception, using semantic segmentation on the Cityscapes dataset as a representative perception module. We introduce perception-realistic corruptions that induce only moderate drops in conventional segmentation metrics, yet observe severe failures in downstream VLM behavior, including hallucinated object mentions, omission of safety-critical entities, and inconsistent safety judgments. To quantify these effects, we propose a set of language-level misalignment metrics that capture hallucination, critical omission, and safety misinterpretation, and analyze their relationship with segmentation quality across multiple contrastive and generative VLMs. Our results reveal a clear disconnect between pixel-level robustness and multimodal semantic reliability, highlighting a critical limitation of current VLM-based systems and motivating the need for evaluation frameworks that explicitly account for perception uncertainty in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09954v1">The Spatial Blindspot of Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) have advanced rapidly, but their ability to capture spatial relationships remains a blindspot. Current VLMs are typically built with contrastive language-image pretraining (CLIP) style image encoders. The training recipe often flattens images into 1D patch sequences, discarding the 2D structure necessary for spatial reasoning. We argue that this lack of spatial awareness is a missing dimension in VLM design and a bottleneck for applications requiring spatial grounding, such as robotics and embodied AI. To address this, we investigate (i) image encoders trained with alternative objectives and (ii) 2D positional encodings. Our experiments show that these architectural choices can lead to improved spatial reasoning on several benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09697v1">Efficient Camera-Controlled Video Generation of Static Scenes via Sparse Diffusion and 3D Rendering</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Project page: https://ayushtewari.com/projects/srender/
    </div>
    <details class="paper-abstract">
      Modern video generative models based on diffusion models can produce very realistic clips, but they are computationally inefficient, often requiring minutes of GPU time for just a few seconds of video. This inefficiency poses a critical barrier to deploying generative video in applications that require real-time interactions, such as embodied AI and VR/AR. This paper explores a new strategy for camera-conditioned video generation of static scenes: using diffusion-based generative models to generate a sparse set of keyframes, and then synthesizing the full video through 3D reconstruction and rendering. By lifting keyframes into a 3D representation and rendering intermediate views, our approach amortizes the generation cost across hundreds of frames while enforcing geometric consistency. We further introduce a model that predicts the optimal number of keyframes for a given camera trajectory, allowing the system to adaptively allocate computation. Our final method, SRENDER, uses very sparse keyframes for simple trajectories and denser ones for complex camera motion. This results in video generation that is more than 40 times faster than the diffusion-based baseline in generating 20 seconds of video, while maintaining high visual fidelity and temporal stability, offering a practical path toward efficient and controllable video synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.19789v4">What Can RL Bring to VLA Generalization? An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Vision-Language Action (VLA) models have shown significant potential for embodied AI. However, their predominant training via supervised fine-tuning (SFT) limits generalization due to susceptibility to compounding errors under distribution shifts. Reinforcement learning (RL) offers a path to overcome these limitations by optimizing for task objectives via trial-and-error, yet a systematic understanding of its specific generalization benefits for VLAs compared to SFT is lacking. To address this, our study introduces a comprehensive benchmark for evaluating VLA generalization and systematically investigates the impact of RL fine-tuning across diverse visual, semantic, and execution dimensions. Our extensive experiments reveal that RL fine-tuning, particularly with PPO, significantly enhances generalization in semantic understanding and execution robustness over SFT, while maintaining comparable visual robustness. We identify PPO as a more effective RL algorithm for VLAs than LLM-derived methods like DPO and GRPO. We also develop a simple recipe for efficient PPO training on VLAs, and demonstrate its practical utility for improving VLA generalization. The project page is at https://rlvla.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08355v1">Semantic Misalignment in Vision-Language Models under Perceptual Degradation</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are increasingly deployed in autonomous driving and embodied AI systems, where reliable perception is critical for safe semantic reasoning and decision-making. While recent VLMs demonstrate strong performance on multimodal benchmarks, their robustness to realistic perception degradation remains poorly understood. In this work, we systematically study semantic misalignment in VLMs under controlled degradation of upstream visual perception, using semantic segmentation on the Cityscapes dataset as a representative perception module. We introduce perception-realistic corruptions that induce only moderate drops in conventional segmentation metrics, yet observe severe failures in downstream VLM behavior, including hallucinated object mentions, omission of safety-critical entities, and inconsistent safety judgments. To quantify these effects, we propose a set of language-level misalignment metrics that capture hallucination, critical omission, and safety misinterpretation, and analyze their relationship with segmentation quality across multiple contrastive and generative VLMs. Our results reveal a clear disconnect between pixel-level robustness and multimodal semantic reliability, highlighting a critical limitation of current VLM-based systems and motivating the need for evaluation frameworks that explicitly account for perception uncertainty in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07553v1">VirtualEnv: A Platform for Embodied AI Research</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to improve in reasoning and decision-making, there is a growing need for realistic and interactive environments where their abilities can be rigorously evaluated. We present VirtualEnv, a next-generation simulation platform built on Unreal Engine 5 that enables fine-grained benchmarking of LLMs in embodied and interactive scenarios. VirtualEnv supports rich agent-environment interactions, including object manipulation, navigation, and adaptive multi-agent collaboration, as well as game-inspired mechanics like escape rooms and procedurally generated environments. We provide a user-friendly API built on top of Unreal Engine, allowing researchers to deploy and control LLM-driven agents using natural language instructions. We integrate large-scale LLMs and vision-language models (VLMs), such as GPT-based models, to generate novel environments and structured tasks from multimodal inputs. Our experiments benchmark the performance of several popular LLMs across tasks of increasing complexity, analyzing differences in adaptability, planning, and multi-agent coordination. We also describe our methodology for procedural task generation, task validation, and real-time environment control. VirtualEnv is released as an open-source platform, we aim to advance research at the intersection of AI and gaming, enable standardized evaluation of LLMs in embodied AI settings, and pave the way for future developments in immersive simulations and interactive entertainment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01705v2">Explicit World Models for Reliable Human-Robot Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      This paper addresses the topic of robustness under sensing noise, ambiguous instructions, and human-robot interaction. We take a radically different tack to the issue of reliable embodied AI: instead of focusing on formal verification methods aimed at achieving model predictability and robustness, we emphasise the dynamic, ambiguous and subjective nature of human-robot interactions that requires embodied AI systems to perceive, interpret, and respond to human intentions in a manner that is consistent, comprehensible and aligned with human expectations. We argue that when embodied agents operate in human environments that are inherently social, multimodal, and fluid, reliability is contextually determined and only has meaning in relation to the goals and expectations of humans involved in the interaction. This calls for a fundamentally different approach to achieving reliable embodied AI that is centred on building and updating an accessible "explicit world model" representing the common ground between human and AI, that is used to align robot behaviours with human expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08876v1">The Semantic Lifecycle in Embodied AI: Acquisition, Representation and Storage via Foundation Models</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Semantic information in embodied AI is inherently multi-source and multi-stage, making it challenging to fully leverage for achieving stable perception-to-action loops in real-world environments. Early studies have combined manual engineering with deep neural networks, achieving notable progress in specific semantic-related embodied tasks. However, as embodied agents encounter increasingly complex environments and open-ended tasks, the demand for more generalizable and robust semantic processing capabilities has become imperative. Recent advances in foundation models (FMs) address this challenge through their cross-domain generalization abilities and rich semantic priors, reshaping the landscape of embodied AI research. In this survey, we propose the Semantic Lifecycle as a unified framework to characterize the evolution of semantic knowledge within embodied AI driven by foundation models. Departing from traditional paradigms that treat semantic processing as isolated modules or disjoint tasks, our framework offers a holistic perspective that captures the continuous flow and maintenance of semantic knowledge. Guided by this embodied semantic lifecycle, we further analyze and compare recent advances across three key stages: acquisition, representation, and storage. Finally, we summarize existing challenges and outline promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06573v1">QMAVIS: Long Video-Audio Understanding using Fusion of Large Multimodal Models</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Large Multimodal Models (LMMs) for video-audio understanding have traditionally been evaluated only on shorter videos of a few minutes long. In this paper, we introduce QMAVIS (Q Team-Multimodal Audio Video Intelligent Sensemaking), a novel long video-audio understanding pipeline built through a late fusion of LMMs, Large Language Models, and speech recognition models. QMAVIS addresses the gap in long-form video analytics, particularly for longer videos of a few minutes to beyond an hour long, opening up new potential applica- tions in sensemaking, video content analysis, embodied AI, etc. Quantitative experiments using QMAVIS demonstrated a 38.75% improvement over state-of-the-art video-audio LMMs like Vide- oLlaMA2 and InternVL2 on the VideoMME (with subtitles) dataset, which comprises long videos with audio information. Evaluations on other challenging video understanding datasets like PerceptionTest and EgoSchema saw up to 2% improvement, indicating competitive performance. Qualitative experiments also showed that QMAVIS is able to extract the nuances of different scenes in a long video audio content while understanding the overarching narrative. Ablation studies were also conducted to ascertain the impact of each component in the fusion pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05991v1">Open-Vocabulary 3D Instruction Ambiguity Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      In safety-critical domains, linguistic ambiguity can have severe consequences; a vague command like "Pass me the vial" in a surgical setting could lead to catastrophic errors. Yet, most embodied AI research overlooks this, assuming instructions are clear and focusing on execution rather than confirmation. To address this critical safety gap, we are the first to define Open-Vocabulary 3D Instruction Ambiguity Detection, a fundamental new task where a model must determine if a command has a single, unambiguous meaning within a given 3D scene. To support this research, we build Ambi3D, the large-scale benchmark for this task, featuring over 700 diverse 3D scenes and around 22k instructions. Our analysis reveals a surprising limitation: state-of-the-art 3D Large Language Models (LLMs) struggle to reliably determine if an instruction is ambiguous. To address this challenge, we propose AmbiVer, a two-stage framework that collects explicit visual evidence from multiple views and uses it to guide an vision-language model (VLM) in judging instruction ambiguity. Extensive experiments demonstrate the challenge of our task and the effectiveness of AmbiVer, paving the way for safer and more trustworthy embodied AI. Code and dataset available at https://jiayuding031020.github.io/ambi3d/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05810v1">SceneFoundry: Generating Interactive Infinite 3D Worlds</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The ability to automatically generate large-scale, interactive, and physically realistic 3D environments is crucial for advancing robotic learning and embodied intelligence. However, existing generative approaches often fail to capture the functional complexity of real-world interiors, particularly those containing articulated objects with movable parts essential for manipulation and navigation. This paper presents SceneFoundry, a language-guided diffusion framework that generates apartment-scale 3D worlds with functionally articulated furniture and semantically diverse layouts for robotic training. From natural language prompts, an LLM module controls floor layout generation, while diffusion-based posterior sampling efficiently populates the scene with articulated assets from large-scale 3D repositories. To ensure physical usability, SceneFoundry employs differentiable guidance functions to regulate object quantity, prevent articulation collisions, and maintain sufficient walkable space for robotic navigation. Extensive experiments demonstrate that our framework generates structurally valid, semantically coherent, and functionally interactive environments across diverse scene types and conditions, enabling scalable embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07855v1">An Empirical Study on Knowledge Transfer under Domain and Label Shifts in 3D LiDAR Point Clouds</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      For 3D perception systems to be practical in real-world applications -- from autonomous driving to embodied AI -- models must adapt to continuously evolving object definitions and sensor domains. Yet, research on continual and transfer learning in 3D point cloud perception remains underexplored compared to 2D vision -- particularly under simultaneous domain and label shifts. To address this gap, we propose the RObust Autonomous driving under Dataset shifts (ROAD) benchmark, a comprehensive evaluation suite for LiDAR-based object classification that explicitly accounts for domain shifts as well as three key forms of label evolution: class split, class expansion, and class insertion. Using large-scale datasets (Waymo, NuScenes, Argoverse2), we evaluate zero-shot transfer, linear probe, and CL, and analyze the impact of backbone architectures, training objectives, and CL methods. Our findings reveal limitations of existing approaches under realistic shifts and establish strong baselines for future research in robust 3D perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03470v2">Toward Maturity-Based Certification of Embodied AI: Quantifying Trustworthiness Through Measurement Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      We propose a maturity-based framework for certifying embodied AI systems through explicit measurement mechanisms. We argue that certifiable embodied AI requires structured assessment frameworks, quantitative scoring mechanisms, and methods for navigating multi-objective trade-offs inherent in trustworthiness evaluation. We demonstrate this approach using uncertainty quantification as an exemplar measurement mechanism and illustrate feasibility through an Uncrewed Aircraft System (UAS) detection case study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04137v1">Wow, wo, val! A Comprehensive Embodied World Model Evaluation Turing Test</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      As world models gain momentum in Embodied AI, an increasing number of works explore using video foundation models as predictive world models for downstream embodied tasks like 3D prediction or interactive generation. However, before exploring these downstream tasks, video foundation models still have two critical questions unanswered: (1) whether their generative generalization is sufficient to maintain perceptual fidelity in the eyes of human observers, and (2) whether they are robust enough to serve as a universal prior for real-world embodied agents. To provide a standardized framework for answering these questions, we introduce the Embodied Turing Test benchmark: WoW-World-Eval (Wow,wo,val). Building upon 609 robot manipulation data, Wow-wo-val examines five core abilities, including perception, planning, prediction, generalization, and execution. We propose a comprehensive evaluation protocol with 22 metrics to assess the models' generation ability, which achieves a high Pearson Correlation between the overall score and human preference (>0.93) and establishes a reliable foundation for the Human Turing Test. On Wow-wo-val, models achieve only 17.27 on long-horizon planning and at best 68.02 on physical consistency, indicating limited spatiotemporal consistency and physical reasoning. For the Inverse Dynamic Model Turing Test, we first use an IDM to evaluate the video foundation models' execution accuracy in the real world. However, most models collapse to $\approx$ 0% success, while WoW maintains a 40.74% success rate. These findings point to a noticeable gap between the generated videos and the real world, highlighting the urgency and necessity of benchmarking World Model in Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04266v1">State Backdoor: Towards Stealthy Real-world Poisoning Attack on Vision-Language-Action Model in State Space</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models are widely deployed in safety-critical embodied AI applications such as robotics. However, their complex multimodal interactions also expose new security vulnerabilities. In this paper, we investigate a backdoor threat in VLA models, where malicious inputs cause targeted misbehavior while preserving performance on clean data. Existing backdoor methods predominantly rely on inserting visible triggers into visual modality, which suffer from poor robustness and low insusceptibility in real-world settings due to environmental variability. To overcome these limitations, we introduce the State Backdoor, a novel and practical backdoor attack that leverages the robot arm's initial state as the trigger. To optimize trigger for insusceptibility and effectiveness, we design a Preference-guided Genetic Algorithm (PGA) that efficiently searches the state space for minimal yet potent triggers. Extensive experiments on five representative VLA models and five real-world tasks show that our method achieves over 90% attack success rate without affecting benign task performance, revealing an underexplored vulnerability in embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03136v1">Limited Linguistic Diversity in Embodied AI Datasets</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Language plays a critical role in Vision-Language-Action (VLA) models, yet the linguistic characteristics of the datasets used to train and evaluate these systems remain poorly documented. In this work, we present a systematic dataset audit of several widely used VLA corpora, aiming to characterize what kinds of instructions these datasets actually contain and how much linguistic variety they provide. We quantify instruction language along complementary dimensions-including lexical variety, duplication and overlap, semantic similarity, and syntactic complexity. Our analysis shows that many datasets rely on highly repetitive, template-like commands with limited structural variation, yielding a narrow distribution of instruction forms. We position these findings as descriptive documentation of the language signal available in current VLA training and evaluation data, intended to support more detailed dataset reporting, more principled dataset selection, and targeted curation or augmentation strategies that broaden language coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16853v2">ISCS: Parameter-Guided Feature Pruning for Resource-Constrained Embodied Perception</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Significant revision: The focus has been pivoted from learned image compression to embodied perception tasks. Experimental results and downstream applications have been updated to demonstrate the method's efficiency in split computing
    </div>
    <details class="paper-abstract">
      Prior studies in embodied AI consistently show that robust perception is critical for human-robot interaction, yet deploying high-fidelity visual models on resource-constrained agents remains challenging due to limited on-device computation power and transmission latency. Exploiting the redundancy in latent representations could improve system efficiency, yet existing approaches often rely on costly dataset-specific ablation tests or heavy entropy models unsuitable for real-time edge-robot collaboration. We propose a generalizable, dataset-agnostic method to identify and selectively transmit structure-critical channels in pretrained encoders. Instead of brute-force empirical evaluations, our approach leverages intrinsic parameter statistics-weight variances and biases-to estimate channel importance. This analysis reveals a consistent organizational structure, termed the Invariant Salient Channel Space (ISCS), where Salient-Core channels capture dominant structures while Salient-Auxiliary channels encode fine visual details. Building on ISCS, we introduce a deterministic static pruning strategy that enables lightweight split-computing. Experiments across different datasets demonstrate that our method achieves a deterministic, ultra-low latency pipeline by bypassing heavy entropy modeling. Our method reduces end-to-end latency, providing a critical speed-accuracy trade-off for resource-constrained human-aware embodied systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03470v1">Toward Maturity-Based Certification of Embodied AI: Quantifying Trustworthiness Through Measurement Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 5 pages, Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      We propose a maturity-based framework for certifying embodied AI systems through explicit measurement mechanisms. We argue that certifiable embodied AI requires structured assessment frameworks, quantitative scoring mechanisms, and methods for navigating multi-objective trade-offs inherent in trustworthiness evaluation. We demonstrate this approach using uncertainty quantification as an exemplar measurement mechanism and illustrate feasibility through an Uncrewed Aircraft System (UAS) detection case study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01705v1">Explicit World Models for Reliable Human-Robot Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      This paper addresses the topic of robustness under sensing noise, ambiguous instructions, and human-robot interaction. We take a radically different tack to the issue of reliable embodied AI: instead of focusing on formal verification methods aimed at achieving model predictability and robustness, we emphasise the dynamic, ambiguous and subjective nature of human-robot interactions that requires embodied AI systems to perceive, interpret, and respond to human intentions in a manner that is consistent, comprehensible and aligned with human expectations. We argue that when embodied agents operate in human environments that are inherently social, multimodal, and fluid, reliability is contextually determined and only has meaning in relation to the goals and expectations of humans involved in the interaction. This calls for a fundamentally different approach to achieving reliable embodied AI that is centred on building and updating an accessible "explicit world model" representing the common ground between human and AI, that is used to align robot behaviours with human expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10071v3">Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Post-challenge bug fix
    </div>
    <details class="paper-abstract">
      The 2025 BEHAVIOR Challenge is designed to rigorously track progress toward solving long-horizon tasks by physical agents in simulated environments. BEHAVIOR-1K focuses on everyday household tasks that people most want robots to assist with and these tasks introduce long-horizon mobile manipulation challenges in realistic settings, bridging the gap between current research and real-world, human-centric applications. This report presents our solution to the 2025 BEHAVIOR Challenge in a very close 2nd place and substantially outperforms the rest of the submissions. Building on $π_{0.5}$, we focus on systematically building our solution by studying the effects of training techniques and data. Through careful ablation studies, we reveal the scaling benefits in both the pre-training and post-training phases, leading to a validation Q-score of 0.345, significantly surpassing previous state-of-the-art performance. We summarize our practical lessons and design recommendations that we hope will provide actionable insights for the broader embodied AI community when adapting powerful foundation models to complex embodied scenarios. Project page: https://github.com/mli0603/openpi-comet
    </details>
</div>
