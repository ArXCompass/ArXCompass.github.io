# gaussian splatting - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03536v1">HumanDreamer-X: Photorealistic Single-image Human Avatars Reconstruction via Gaussian Restoration</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Project Page: https://humandreamer-x.github.io/
    </div>
    <details class="paper-abstract">
      Single-image human reconstruction is vital for digital human modeling applications but remains an extremely challenging task. Current approaches rely on generative models to synthesize multi-view images for subsequent 3D reconstruction and animation. However, directly generating multiple views from a single human image suffers from geometric inconsistencies, resulting in issues like fragmented or blurred limbs in the reconstructed models. To tackle these limitations, we introduce \textbf{HumanDreamer-X}, a novel framework that integrates multi-view human generation and reconstruction into a unified pipeline, which significantly enhances the geometric consistency and visual fidelity of the reconstructed 3D models. In this framework, 3D Gaussian Splatting serves as an explicit 3D representation to provide initial geometry and appearance priority. Building upon this foundation, \textbf{HumanFixer} is trained to restore 3DGS renderings, which guarantee photorealistic results. Furthermore, we delve into the inherent challenges associated with attention mechanisms in multi-view human generation, and propose an attention modulation strategy that effectively enhances geometric details identity consistency across multi-view. Experimental results demonstrate that our approach markedly improves generation and reconstruction PSNR quality metrics by 16.45% and 12.65%, respectively, achieving a PSNR of up to 25.62 dB, while also showing generalization capabilities on in-the-wild data and applicability to various human reconstruction backbone models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07494v2">ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3D-GS) has prevailed in novel view synthesis, achieving high fidelity and efficiency. However, it often struggles to capture rich details and complete geometry. Our analysis reveals that the 3D-GS densification operation lacks adaptiveness and faces a dilemma between geometry coverage and detail recovery. To address this, we introduce a novel densification operation, residual split, which adds a downscaled Gaussian as a residual. Our approach is capable of adaptively retrieving details and complementing missing geometry. To further support this method, we propose a pipeline named ResGS. Specifically, we integrate a Gaussian image pyramid for progressive supervision and implement a selection scheme that prioritizes the densification of coarse Gaussians over time. Extensive experiments demonstrate that our method achieves SOTA rendering quality. Consistent performance improvements can be achieved by applying our residual split on various 3D-GS variants, underscoring its versatility and potential for broader application in 3D-GS-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10620v2">PyTorchGeoNodes: Enabling Differentiable Shape Programs for 3D Shape Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Accepted at CVPR
    </div>
    <details class="paper-abstract">
      We propose PyTorchGeoNodes, a differentiable module for reconstructing 3D objects and their parameters from images using interpretable shape programs. Unlike traditional CAD model retrieval, shape programs allow reasoning about semantic parameters, editing, and a low memory footprint. Despite their potential, shape programs for 3D scene understanding have been largely overlooked. Our key contribution is enabling gradient-based optimization by parsing shape programs, or more precisely procedural models designed in Blender, into efficient PyTorch code. While there are many possible applications of our PyTochGeoNodes, we show that a combination of PyTorchGeoNodes with genetic algorithm is a method of choice to optimize both discrete and continuous shape program parameters for 3D reconstruction and understanding of 3D object parameters. Our modular framework can be further integrated with other reconstruction algorithms, and we demonstrate one such integration to enable procedural Gaussian splatting. Our experiments on the ScanNet dataset show that our method achieves accurate reconstructions while enabling, until now, unseen level of 3D scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03886v1">WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      We present WildGS-SLAM, a robust and efficient monocular RGB SLAM system designed to handle dynamic environments by leveraging uncertainty-aware geometric mapping. Unlike traditional SLAM systems, which assume static scenes, our approach integrates depth and uncertainty information to enhance tracking, mapping, and rendering performance in the presence of moving objects. We introduce an uncertainty map, predicted by a shallow multi-layer perceptron and DINOv2 features, to guide dynamic object removal during both tracking and mapping. This uncertainty map enhances dense bundle adjustment and Gaussian map optimization, improving reconstruction accuracy. Our system is evaluated on multiple datasets and demonstrates artifact-free view synthesis. Results showcase WildGS-SLAM's superior performance in dynamic environments compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02541v4">Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Accepted to TMLR 2025. Project page at https://video-3dgs-project.github.io/
    </div>
    <details class="paper-abstract">
      Recent advancements in zero-shot video diffusion models have shown promise for text-driven video editing, but challenges remain in achieving high temporal consistency. To address this, we introduce Video-3DGS, a 3D Gaussian Splatting (3DGS)-based video refiner designed to enhance temporal consistency in zero-shot video editors. Our approach utilizes a two-stage 3D Gaussian optimizing process tailored for editing dynamic monocular videos. In the first stage, Video-3DGS employs an improved version of COLMAP, referred to as MC-COLMAP, which processes original videos using a Masked and Clipped approach. For each video clip, MC-COLMAP generates the point clouds for dynamic foreground objects and complex backgrounds. These point clouds are utilized to initialize two sets of 3D Gaussians (Frg-3DGS and Bkg-3DGS) aiming to represent foreground and background views. Both foreground and background views are then merged with a 2D learnable parameter map to reconstruct full views. In the second stage, we leverage the reconstruction ability developed in the first stage to impose the temporal constraints on the video diffusion model. To demonstrate the efficacy of Video-3DGS on both stages, we conduct extensive experiments across two related tasks: Video Reconstruction and Video Editing. Video-3DGS trained with 3k iterations significantly improves video reconstruction quality (+3 PSNR, +7 PSNR increase) and training efficiency (x1.9, x4.5 times faster) over NeRF-based and 3DGS-based state-of-art methods on DAVIS dataset, respectively. Moreover, it enhances video editing by ensuring temporal consistency across 58 dynamic monocular videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08174v2">Object-Centric 2D Gaussian Splatting: Background Removal and Occlusion-Aware Pruning for Compact Object Models</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 ICPRAM 2025. Implementation details (no code): https://github.com/MarcelRogge/object-centric-2dgs
    </div>
    <details class="paper-abstract">
      Current Gaussian Splatting approaches are effective for reconstructing entire scenes but lack the option to target specific objects, making them computationally expensive and unsuitable for object-specific applications. We propose a novel approach that leverages object masks to enable targeted reconstruction, resulting in object-centric models. Additionally, we introduce an occlusion-aware pruning strategy to minimize the number of Gaussians without compromising quality. Our method reconstructs compact object models, yielding object-centric Gaussian and mesh representations that are up to 96% smaller and up to 71% faster to train compared to the baseline while retaining competitive quality. These representations are immediately usable for downstream applications such as appearance editing and physics simulation without additional processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01543v2">6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Efficient and accurate object pose estimation is an essential component for modern vision systems in many applications such as Augmented Reality, autonomous driving, and robotics. While research in model-based 6D object pose estimation has delivered promising results, model-free methods are hindered by the high computational load in rendering and inferring consistent poses of arbitrary objects in a live RGB-D video stream. To address this issue, we present 6DOPE-GS, a novel method for online 6D object pose estimation \& tracking with a single RGB-D camera by effectively leveraging advances in Gaussian Splatting. Thanks to the fast differentiable rendering capabilities of Gaussian Splatting, 6DOPE-GS can simultaneously optimize for 6D object poses and 3D object reconstruction. To achieve the necessary efficiency and accuracy for live tracking, our method uses incremental 2D Gaussian Splatting with an intelligent dynamic keyframe selection procedure to achieve high spatial object coverage and prevent erroneous pose updates. We also propose an opacity statistic-based pruning mechanism for adaptive Gaussian density control, to ensure training stability and efficiency. We evaluate our method on the HO3D and YCBInEOAT datasets and show that 6DOPE-GS matches the performance of state-of-the-art baselines for model-free simultaneous 6D pose tracking and reconstruction while providing a 5$\times$ speedup. We also demonstrate the method's suitability for live, dynamic object tracking and reconstruction in a real-world setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02437v1">MonoGS++: Fast and Accurate Monocular RGB Gaussian SLAM</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      We present MonoGS++, a novel fast and accurate Simultaneous Localization and Mapping (SLAM) method that leverages 3D Gaussian representations and operates solely on RGB inputs. While previous 3D Gaussian Splatting (GS)-based methods largely depended on depth sensors, our approach reduces the hardware dependency and only requires RGB input, leveraging online visual odometry (VO) to generate sparse point clouds in real-time. To reduce redundancy and enhance the quality of 3D scene reconstruction, we implemented a series of methodological enhancements in 3D Gaussian mapping. Firstly, we introduced dynamic 3D Gaussian insertion to avoid adding redundant Gaussians in previously well-reconstructed areas. Secondly, we introduced clarity-enhancing Gaussian densification module and planar regularization to handle texture-less areas and flat surfaces better. We achieved precise camera tracking results both on the synthetic Replica and real-world TUM-RGBD datasets, comparable to those of the state-of-the-art. Additionally, our method realized a significant 5.57x improvement in frames per second (fps) over the previous state-of-the-art, MonoGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01957v2">Toward Real-world BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 Accepted to CVPR'25. https://hcis-lab.github.io/GaussianLSS/
    </div>
    <details class="paper-abstract">
      Bird's-eye view (BEV) perception has gained significant attention because it provides a unified representation to fuse multiple view images and enables a wide range of down-stream autonomous driving tasks, such as forecasting and planning. Recent state-of-the-art models utilize projection-based methods which formulate BEV perception as query learning to bypass explicit depth estimation. While we observe promising advancements in this paradigm, they still fall short of real-world applications because of the lack of uncertainty modeling and expensive computational requirement. In this work, we introduce GaussianLSS, a novel uncertainty-aware BEV perception framework that revisits unprojection-based methods, specifically the Lift-Splat-Shoot (LSS) paradigm, and enhances them with depth un-certainty modeling. GaussianLSS represents spatial dispersion by learning a soft depth mean and computing the variance of the depth distribution, which implicitly captures object extents. We then transform the depth distribution into 3D Gaussians and rasterize them to construct uncertainty-aware BEV features. We evaluate GaussianLSS on the nuScenes dataset, achieving state-of-the-art performance compared to unprojection-based methods. In particular, it provides significant advantages in speed, running 2.5x faster, and in memory efficiency, using 0.3x less memory compared to projection-based methods, while achieving competitive performance with only a 0.4% IoU difference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02316v1">ConsDreamer: Advancing Multi-View Consistency for Zero-Shot Text-to-3D Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 13 pages, 11 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Recent advances in zero-shot text-to-3D generation have revolutionized 3D content creation by enabling direct synthesis from textual descriptions. While state-of-the-art methods leverage 3D Gaussian Splatting with score distillation to enhance multi-view rendering through pre-trained text-to-image (T2I) models, they suffer from inherent view biases in T2I priors. These biases lead to inconsistent 3D generation, particularly manifesting as the multi-face Janus problem, where objects exhibit conflicting features across views. To address this fundamental challenge, we propose ConsDreamer, a novel framework that mitigates view bias by refining both the conditional and unconditional terms in the score distillation process: (1) a View Disentanglement Module (VDM) that eliminates viewpoint biases in conditional prompts by decoupling irrelevant view components and injecting precise camera parameters; and (2) a similarity-based partial order loss that enforces geometric consistency in the unconditional term by aligning cosine similarities with azimuth relationships. Extensive experiments demonstrate that ConsDreamer effectively mitigates the multi-face Janus problem in text-to-3D generation, outperforming existing methods in both visual quality and consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00457v3">Distilling Multi-view Diffusion Models into 3D Generators</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      We introduce DD3G, a formulation that Distills a multi-view Diffusion model (MV-DM) into a 3D Generator using gaussian splatting. DD3G compresses and integrates extensive visual and spatial geometric knowledge from the MV-DM by simulating its ordinary differential equation (ODE) trajectory, ensuring the distilled generator generalizes better than those trained solely on 3D data. Unlike previous amortized optimization approaches, we align the MV-DM and 3D generator representation spaces to transfer the teacher's probabilistic flow to the student, thus avoiding inconsistencies in optimization objectives caused by probabilistic sampling. The introduction of probabilistic flow and the coupling of various attributes in 3D Gaussians introduce challenges in the generation process. To tackle this, we propose PEPD, a generator consisting of Pattern Extraction and Progressive Decoding phases, which enables efficient fusion of probabilistic flow and converts a single image into 3D Gaussians within 0.06 seconds. Furthermore, to reduce knowledge loss and overcome sparse-view supervision, we design a joint optimization objective that ensures the quality of generated samples through explicit supervision and implicit verification. Leveraging existing 2D generation models, we compile 120k high-quality RGBA images for distillation. Experiments on synthetic and public datasets demonstrate the effectiveness of our method. Our project is available at: https://qinbaigao.github.io/DD3G_project/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03059v1">Compressing 3D Gaussian Splatting by Noise-Substituted Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated remarkable effectiveness in 3D reconstruction, achieving high-quality results with real-time radiance field rendering. However, a key challenge is the substantial storage cost: reconstructing a single scene typically requires millions of Gaussian splats, each represented by 59 floating-point parameters, resulting in approximately 1~GB of memory. To address this challenge, we propose a compression method by building separate attribute codebooks and storing only discrete code indices. Specifically, we employ noise-substituted vector quantization technique to jointly train the codebooks and model features, ensuring consistency between gradient descent optimization and parameter discretization. Our method reduces the memory consumption efficiently (around $45\times$) while maintaining competitive reconstruction quality on standard 3D benchmark scenes. Experiments on different codebook sizes show the trade-off between compression ratio and image quality. Furthermore, the trained compressed model remains fully compatible with popular 3DGS viewers and enables faster rendering speed, making it well-suited for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12001v3">3D Gaussian Splatting against Moving Objects for High-Fidelity Street Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      The accurate reconstruction of dynamic street scenes is critical for applications in autonomous driving, augmented reality, and virtual reality. Traditional methods relying on dense point clouds and triangular meshes struggle with moving objects, occlusions, and real-time processing constraints, limiting their effectiveness in complex urban environments. While multi-view stereo and neural radiance fields have advanced 3D reconstruction, they face challenges in computational efficiency and handling scene dynamics. This paper proposes a novel 3D Gaussian point distribution method for dynamic street scene reconstruction. Our approach introduces an adaptive transparency mechanism that eliminates moving objects while preserving high-fidelity static scene details. Additionally, iterative refinement of Gaussian point distribution enhances geometric accuracy and texture representation. We integrate directional encoding with spatial position optimization to optimize storage and rendering efficiency, reducing redundancy while maintaining scene integrity. Experimental results demonstrate that our method achieves high reconstruction quality, improved rendering performance, and adaptability in large-scale dynamic environments. These contributions establish a robust framework for real-time, high-precision 3D reconstruction, advancing the practicality of dynamic scene modeling across multiple applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09563v2">Self-Calibrating Gaussian Splatting for Large Field of View Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 Project Page: https://denghilbert.github.io/self-cali/
    </div>
    <details class="paper-abstract">
      In this paper, we present a self-calibrating framework that jointly optimizes camera parameters, lens distortion and 3D Gaussian representations, enabling accurate and efficient scene reconstruction. In particular, our technique enables high-quality scene reconstruction from Large field-of-view (FOV) imagery taken with wide-angle lenses, allowing the scene to be modeled from a smaller number of images. Our approach introduces a novel method for modeling complex lens distortions using a hybrid network that combines invertible residual networks with explicit grids. This design effectively regularizes the optimization process, achieving greater accuracy than conventional camera models. Additionally, we propose a cubemap-based resampling strategy to support large FOV images without sacrificing resolution or introducing distortion artifacts. Our method is compatible with the fast rasterization of Gaussian Splatting, adaptable to a wide variety of camera lens distortion, and demonstrates state-of-the-art performance on both synthetic and real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02158v1">UAVTwin: Neural Digital Twins for UAVs using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      We present UAVTwin, a method for creating digital twins from real-world environments and facilitating data augmentation for training downstream models embedded in unmanned aerial vehicles (UAVs). Specifically, our approach focuses on synthesizing foreground components, such as various human instances in motion within complex scene backgrounds, from UAV perspectives. This is achieved by integrating 3D Gaussian Splatting (3DGS) for reconstructing backgrounds along with controllable synthetic human models that display diverse appearances and actions in multiple poses. To the best of our knowledge, UAVTwin is the first approach for UAV-based perception that is capable of generating high-fidelity digital twins based on 3DGS. The proposed work significantly enhances downstream models through data augmentation for real-world environments with multiple dynamic objects and significant appearance variations-both of which typically introduce artifacts in 3DGS-based modeling. To tackle these challenges, we propose a novel appearance modeling strategy and a mask refinement module to enhance the training of 3D Gaussian Splatting. We demonstrate the high quality of neural rendering by achieving a 1.23 dB improvement in PSNR compared to recent methods. Furthermore, we validate the effectiveness of data augmentation by showing a 2.5% to 13.7% improvement in mAP for the human detection task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20522v2">MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 CVPR 2025; Project page:https://maskgaussian.github.io/
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in novel view synthesis and real-time rendering, the high memory consumption due to the use of millions of Gaussians limits its practicality. To mitigate this issue, improvements have been made by pruning unnecessary Gaussians, either through a hand-crafted criterion or by using learned masks. However, these methods deterministically remove Gaussians based on a snapshot of the pruning moment, leading to sub-optimized reconstruction performance from a long-term perspective. To address this issue, we introduce MaskGaussian, which models Gaussians as probabilistic entities rather than permanently removing them, and utilize them according to their probability of existence. To achieve this, we propose a masked-rasterization technique that enables unused yet probabilistically existing Gaussians to receive gradients, allowing for dynamic assessment of their contribution to the evolving scene and adjustment of their probability of existence. Hence, the importance of Gaussians iteratively changes and the pruned Gaussians are selected diversely. Extensive experiments demonstrate the superiority of the proposed method in achieving better rendering quality with fewer Gaussians than previous pruning methods, pruning over 60% of Gaussians on average with only a 0.02 PSNR decline. Our code can be found at: https://github.com/kaikai23/MaskGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02045v1">WorldPrompter: Traversable Text-to-Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Scene-level 3D generation is a challenging research topic, with most existing methods generating only partial scenes and offering limited navigational freedom. We introduce WorldPrompter, a novel generative pipeline for synthesizing traversable 3D scenes from text prompts. We leverage panoramic videos as an intermediate representation to model the 360{\deg} details of a scene. WorldPrompter incorporates a conditional 360{\deg} panoramic video generator, capable of producing a 128-frame video that simulates a person walking through and capturing a virtual environment. The resulting video is then reconstructed as Gaussian splats by a fast feedforward 3D reconstructor, enabling a true walkable experience within the 3D scene. Experiments demonstrate that our panoramic video generation model achieves convincing view consistency across frames, enabling high-quality panoramic Gaussian splat reconstruction and facilitating traversal over an area of the scene. Qualitative and quantitative results also show it outperforms the state-of-the-art 360{\deg} video generators and 3D scene generation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01960v1">Diffusion-Guided Gaussian Splatting for Large-Scale Unconstrained 3D Reconstruction and Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 WACV ULTRRA Workshop 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) have achieved impressive results in real-time 3D reconstruction and novel view synthesis. However, these methods struggle in large-scale, unconstrained environments where sparse and uneven input coverage, transient occlusions, appearance variability, and inconsistent camera settings lead to degraded quality. We propose GS-Diff, a novel 3DGS framework guided by a multi-view diffusion model to address these limitations. By generating pseudo-observations conditioned on multi-view inputs, our method transforms under-constrained 3D reconstruction problems into well-posed ones, enabling robust optimization even with sparse data. GS-Diff further integrates several enhancements, including appearance embedding, monocular depth priors, dynamic object modeling, anisotropy regularization, and advanced rasterization techniques, to tackle geometric and photometric challenges in real-world settings. Experiments on four benchmarks demonstrate that GS-Diff consistently outperforms state-of-the-art baselines by significant margins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01844v1">BOGausS: Better Optimized Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) proposes an efficient solution for novel view synthesis. Its framework provides fast and high-fidelity rendering. Although less complex than other solutions such as Neural Radiance Fields (NeRF), there are still some challenges building smaller models without sacrificing quality. In this study, we perform a careful analysis of 3DGS training process and propose a new optimization methodology. Our Better Optimized Gaussian Splatting (BOGausS) solution is able to generate models up to ten times lighter than the original 3DGS with no quality degradation, thus significantly boosting the performance of Gaussian Splatting compared to the state of the art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09227v3">DreamScape: 3D Scene Creation via Gaussian Splatting joint Correlation Modeling</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Recent advances in text-to-3D creation integrate the potent prior of Diffusion Models from text-to-image generation into 3D domain. Nevertheless, generating 3D scenes with multiple objects remains challenging. Therefore, we present DreamScape, a method for generating 3D scenes from text. Utilizing Gaussian Splatting for 3D representation, DreamScape introduces 3D Gaussian Guide that encodes semantic primitives, spatial transformations and relationships from text using LLMs, enabling local-to-global optimization. Progressive scale control is tailored during local object generation, addressing training instability issue arising from simple blending in the global optimization stage. Collision relationships between objects are modeled at the global level to mitigate biases in LLMs priors, ensuring physical correctness. Additionally, to generate pervasive objects like rain and snow distributed extensively across the scene, we design specialized sparse initialization and densification strategy. Experiments demonstrate that DreamScape achieves state-of-the-art performance, enabling high-fidelity, controllable 3D scene generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01732v1">FIORD: A Fisheye Indoor-Outdoor Dataset with LIDAR Ground Truth for 3D Scene Reconstruction and Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 SCIA 2025
    </div>
    <details class="paper-abstract">
      The development of large-scale 3D scene reconstruction and novel view synthesis methods mostly rely on datasets comprising perspective images with narrow fields of view (FoV). While effective for small-scale scenes, these datasets require large image sets and extensive structure-from-motion (SfM) processing, limiting scalability. To address this, we introduce a fisheye image dataset tailored for scene reconstruction tasks. Using dual 200-degree fisheye lenses, our dataset provides full 360-degree coverage of 5 indoor and 5 outdoor scenes. Each scene has sparse SfM point clouds and precise LIDAR-derived dense point clouds that can be used as geometric ground-truth, enabling robust benchmarking under challenging conditions such as occlusions and reflections. While the baseline experiments focus on vanilla Gaussian Splatting and NeRF based Nerfacto methods, the dataset supports diverse approaches for scene reconstruction, novel view synthesis, and image-based rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01647v1">FlowR: Flowing from Sparse to Dense 3D Reconstructions</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Project page is available at https://tobiasfshr.github.io/pub/flowr
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting enables high-quality novel view synthesis (NVS) at real-time frame rates. However, its quality drops sharply as we depart from the training views. Thus, dense captures are needed to match the high-quality expectations of some applications, e.g. Virtual Reality (VR). However, such dense captures are very laborious and expensive to obtain. Existing works have explored using 2D generative models to alleviate this requirement by distillation or generating additional training views. These methods are often conditioned only on a handful of reference input views and thus do not fully exploit the available 3D information, leading to inconsistent generation results and reconstruction artifacts. To tackle this problem, we propose a multi-view, flow matching model that learns a flow to connect novel view renderings from possibly sparse reconstructions to renderings that we expect from dense reconstructions. This enables augmenting scene captures with novel, generated views to improve reconstruction quality. Our model is trained on a novel dataset of 3.6M image pairs and can process up to 45 views at 540x960 resolution (91K tokens) on one H100 GPU in a single forward pass. Our pipeline consistently improves NVS in sparse- and dense-view scenarios, leading to higher-quality reconstructions than prior works across multiple, widely-used NVS benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01619v1">3DBonsai: Structure-Aware Bonsai Modeling Using Conditioned 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Accepted by ICME 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in text-to-3D generation have shown remarkable results by leveraging 3D priors in combination with 2D diffusion. However, previous methods utilize 3D priors that lack detailed and complex structural information, limiting them to generating simple objects and presenting challenges for creating intricate structures such as bonsai. In this paper, we propose 3DBonsai, a novel text-to-3D framework for generating 3D bonsai with complex structures. Technically, we first design a trainable 3D space colonization algorithm to produce bonsai structures, which are then enhanced through random sampling and point cloud augmentation to serve as the 3D Gaussian priors. We introduce two bonsai generation pipelines with distinct structural levels: fine structure conditioned generation, which initializes 3D Gaussians using a 3D structure prior to produce detailed and complex bonsai, and coarse structure conditioned generation, which employs a multi-view structure consistency module to align 2D and 3D structures. Moreover, we have compiled a unified 2D and 3D Chinese-style bonsai dataset. Our experimental results demonstrate that 3DBonsai significantly outperforms existing methods, providing a new benchmark for structure-aware 3D bonsai generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01559v1">RealityAvatar: Towards Realistic Loose Clothing Modeling in Animatable 3D Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Modeling animatable human avatars from monocular or multi-view videos has been widely studied, with recent approaches leveraging neural radiance fields (NeRFs) or 3D Gaussian Splatting (3DGS) achieving impressive results in novel-view and novel-pose synthesis. However, existing methods often struggle to accurately capture the dynamics of loose clothing, as they primarily rely on global pose conditioning or static per-frame representations, leading to oversmoothing and temporal inconsistencies in non-rigid regions. To address this, We propose RealityAvatar, an efficient framework for high-fidelity digital human modeling, specifically targeting loosely dressed avatars. Our method leverages 3D Gaussian Splatting to capture complex clothing deformations and motion dynamics while ensuring geometric consistency. By incorporating a motion trend module and a latentbone encoder, we explicitly model pose-dependent deformations and temporal variations in clothing behavior. Extensive experiments on benchmark datasets demonstrate the effectiveness of our approach in capturing fine-grained clothing deformations and motion-driven shape variations. Our method significantly enhances structural fidelity and perceptual quality in dynamic human reconstruction, particularly in non-rigid regions, while achieving better consistency across temporal frames.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01512v1">High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Recently single-view 3D generation via Gaussian splatting has emerged and developed quickly. They learn 3D Gaussians from 2D RGB images generated from pre-trained multi-view diffusion (MVD) models, and have shown a promising avenue for 3D generation through a single image. Despite the current progress, these methods still suffer from the inconsistency jointly caused by the geometric ambiguity in the 2D images, and the lack of structure of 3D Gaussians, leading to distorted and blurry 3D object generation. In this paper, we propose to fix these issues by GS-RGBN, a new RGBN-volume Gaussian Reconstruction Model designed to generate high-fidelity 3D objects from single-view images. Our key insight is a structured 3D representation can simultaneously mitigate the afore-mentioned two issues. To this end, we propose a novel hybrid Voxel-Gaussian representation, where a 3D voxel representation contains explicit 3D geometric information, eliminating the geometric ambiguity from 2D images. It also structures Gaussians during learning so that the optimization tends to find better local optima. Our 3D voxel representation is obtained by a fusion module that aligns RGB features and surface normal features, both of which can be estimated from 2D images. Extensive experiments demonstrate the superiority of our methods over prior works in terms of high-quality reconstruction results, robust generalization, and good efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01503v1">Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 CVPR 2025, project page: https://cuiziteng.github.io/Luminance_GS_web/
    </div>
    <details class="paper-abstract">
      Capturing high-quality photographs under diverse real-world lighting conditions is challenging, as both natural lighting (e.g., low-light) and camera exposure settings (e.g., exposure time) significantly impact image quality. This challenge becomes more pronounced in multi-view scenarios, where variations in lighting and image signal processor (ISP) settings across viewpoints introduce photometric inconsistencies. Such lighting degradations and view-dependent variations pose substantial challenges to novel view synthesis (NVS) frameworks based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). To address this, we introduce Luminance-GS, a novel approach to achieving high-quality novel view synthesis results under diverse challenging lighting conditions using 3DGS. By adopting per-view color matrix mapping and view-adaptive curve adjustments, Luminance-GS achieves state-of-the-art (SOTA) results across various lighting conditions -- including low-light, overexposure, and varying exposure -- while not altering the original 3DGS explicit representation. Compared to previous NeRF- and 3DGS-based baselines, Luminance-GS provides real-time rendering speed with improved reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01358v1">3D Gaussian Inverse Rendering with Approximated Global Illumination</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting shows great potential in reconstructing photo-realistic 3D scenes. However, these methods typically bake illumination into their representations, limiting their use for physically-based rendering and scene editing. Although recent inverse rendering approaches aim to decompose scenes into material and lighting components, they often rely on simplifying assumptions that fail when editing. We present a novel approach that enables efficient global illumination for 3D Gaussians Splatting through screen-space ray tracing. Our key insight is that a substantial amount of indirect light can be traced back to surfaces visible within the current view frustum. Leveraging this observation, we augment the direct shading computed by 3D Gaussians with Monte-Carlo screen-space ray-tracing to capture one-bounce indirect illumination. In this way, our method enables realistic global illumination without sacrificing the computational efficiency and editability benefits of 3D Gaussians. Through experiments, we show that the screen-space approximation we utilize allows for indirect illumination and supports real-time rendering and editing. Code, data, and models will be made available at our project page: https://wuzirui.github.io/gs-ssr.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06897v2">ActiveGAMER: Active GAussian Mapping through Efficient Rendering</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted to CVPR2025
    </div>
    <details class="paper-abstract">
      We introduce ActiveGAMER, an active mapping system that utilizes 3D Gaussian Splatting (3DGS) to achieve high-quality, real-time scene mapping and exploration. Unlike traditional NeRF-based methods, which are computationally demanding and restrict active mapping performance, our approach leverages the efficient rendering capabilities of 3DGS, allowing effective and efficient exploration in complex environments. The core of our system is a rendering-based information gain module that dynamically identifies the most informative viewpoints for next-best-view planning, enhancing both geometric and photometric reconstruction accuracy. ActiveGAMER also integrates a carefully balanced framework, combining coarse-to-fine exploration, post-refinement, and a global-local keyframe selection strategy to maximize reconstruction completeness and fidelity. Our system autonomously explores and reconstructs environments with state-of-the-art geometric and photometric accuracy and completeness, significantly surpassing existing approaches in both aspects. Extensive evaluations on benchmark datasets such as Replica and MP3D highlight ActiveGAMER's effectiveness in active mapping tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21442v2">RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      We consider the problem of adding dynamic rain effects to in-the-wild scenes in a physically-correct manner. Recent advances in scene modeling have made significant progress, with NeRF and 3DGS techniques emerging as powerful tools for reconstructing complex scenes. However, while effective for novel view synthesis, these methods typically struggle with challenging scene editing tasks, such as physics-based rain simulation. In contrast, traditional physics-based simulations can generate realistic rain effects, such as raindrops and splashes, but they often rely on skilled artists to carefully set up high-fidelity scenes. This process lacks flexibility and scalability, limiting its applicability to broader, open-world environments. In this work, we introduce RainyGS, a novel approach that leverages the strengths of both physics-based modeling and 3DGS to generate photorealistic, dynamic rain effects in open-world scenes with physical accuracy. At the core of our method is the integration of physically-based raindrop and shallow water simulation techniques within the fast 3DGS rendering framework, enabling realistic and efficient simulations of raindrop behavior, splashes, and reflections. Our method supports synthesizing rain effects at over 30 fps, offering users flexible control over rain intensity -- from light drizzles to heavy downpours. We demonstrate that RainyGS performs effectively for both real-world outdoor scenes and large-scale driving scenarios, delivering more photorealistic and physically-accurate rain effects compared to state-of-the-art methods. Project page can be found at https://pku-vcl-geometry.github.io/RainyGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19671v2">GaussianRoom: Improving 3D Gaussian Splatting with SDF Guidance and Monocular Cues for Indoor Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Embodied intelligence requires precise reconstruction and rendering to simulate large-scale real-world data. Although 3D Gaussian Splatting (3DGS) has recently demonstrated high-quality results with real-time performance, it still faces challenges in indoor scenes with large, textureless regions, resulting in incomplete and noisy reconstructions due to poor point cloud initialization and underconstrained optimization. Inspired by the continuity of signed distance field (SDF), which naturally has advantages in modeling surfaces, we propose a unified optimization framework that integrates neural signed distance fields (SDFs) with 3DGS for accurate geometry reconstruction and real-time rendering. This framework incorporates a neural SDF field to guide the densification and pruning of Gaussians, enabling Gaussians to model scenes accurately even with poor initialized point clouds. Simultaneously, the geometry represented by Gaussians improves the efficiency of the SDF field by piloting its point sampling. Additionally, we introduce two regularization terms based on normal and edge priors to resolve geometric ambiguities in textureless areas and enhance detail accuracy. Extensive experiments in ScanNet and ScanNet++ show that our method achieves state-of-the-art performance in both surface reconstruction and novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03526v2">Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Project website: https://research.nvidia.com/labs/toronto-ai/bullet-timer/
    </div>
    <details class="paper-abstract">
      Recent advancements in static feed-forward scene reconstruction have demonstrated significant progress in high-quality novel view synthesis. However, these models often struggle with generalizability across diverse environments and fail to effectively handle dynamic content. We present BTimer (short for BulletTimer), the first motion-aware feed-forward model for real-time reconstruction and novel view synthesis of dynamic scenes. Our approach reconstructs the full scene in a 3D Gaussian Splatting representation at a given target ('bullet') timestamp by aggregating information from all the context frames. Such a formulation allows BTimer to gain scalability and generalization by leveraging both static and dynamic scene datasets. Given a casual monocular dynamic video, BTimer reconstructs a bullet-time scene within 150ms while reaching state-of-the-art performance on both static and dynamic scene datasets, even compared with optimization-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24270v2">Visual Acoustic Fields</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Objects produce different sounds when hit, and humans can intuitively infer how an object might sound based on its appearance and material properties. Inspired by this intuition, we propose Visual Acoustic Fields, a framework that bridges hitting sounds and visual signals within a 3D space using 3D Gaussian Splatting (3DGS). Our approach features two key modules: sound generation and sound localization. The sound generation module leverages a conditional diffusion model, which takes multiscale features rendered from a feature-augmented 3DGS to generate realistic hitting sounds. Meanwhile, the sound localization module enables querying the 3D scene, represented by the feature-augmented 3DGS, to localize hitting positions based on the sound sources. To support this framework, we introduce a novel pipeline for collecting scene-level visual-sound sample pairs, achieving alignment between captured images, impact locations, and corresponding sounds. To the best of our knowledge, this is the first dataset to connect visual and acoustic signals in a 3D context. Extensive experiments on our dataset demonstrate the effectiveness of Visual Acoustic Fields in generating plausible impact sounds and accurately localizing impact sources. Our project page is at https://yuelei0428.github.io/projects/Visual-Acoustic-Fields/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10437v2">4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 CVPR 2025. Project Page: https://4d-langsplat.github.io
    </div>
    <details class="paper-abstract">
      Learning 4D language fields to enable time-sensitive, open-ended language queries in dynamic scenes is essential for many real-world applications. While LangSplat successfully grounds CLIP features into 3D Gaussian representations, achieving precision and efficiency in 3D static scenes, it lacks the ability to handle dynamic 4D fields as CLIP, designed for static image-text tasks, cannot capture temporal dynamics in videos. Real-world environments are inherently dynamic, with object semantics evolving over time. Building a precise 4D language field necessitates obtaining pixel-aligned, object-wise video features, which current vision models struggle to achieve. To address these challenges, we propose 4D LangSplat, which learns 4D language fields to handle time-agnostic or time-sensitive open-vocabulary queries in dynamic scenes efficiently. 4D LangSplat bypasses learning the language field from vision features and instead learns directly from text generated from object-wise video captions via Multimodal Large Language Models (MLLMs). Specifically, we propose a multimodal object-wise video prompting method, consisting of visual and text prompts that guide MLLMs to generate detailed, temporally consistent, high-quality captions for objects throughout a video. These captions are encoded using a Large Language Model into high-quality sentence embeddings, which then serve as pixel-aligned, object-specific feature supervision, facilitating open-vocabulary text queries through shared embedding spaces. Recognizing that objects in 4D scenes exhibit smooth transitions across states, we further propose a status deformable network to model these continuous changes over time effectively. Our results across multiple benchmarks demonstrate that 4D LangSplat attains precise and efficient results for both time-sensitive and time-agnostic open-vocabulary queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00370v2">Scalable Real2Sim: Physics-Aware Asset Generation Via Robotic Pick-and-Place Setups</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Website: https://scalable-real2sim.github.io/
    </div>
    <details class="paper-abstract">
      Simulating object dynamics from real-world perception shows great promise for digital twins and robotic manipulation but often demands labor-intensive measurements and expertise. We present a fully automated Real2Sim pipeline that generates simulation-ready assets for real-world objects through robotic interaction. Using only a robot's joint torque sensors and an external camera, the pipeline identifies visual geometry, collision geometry, and physical properties such as inertial parameters. Our approach introduces a general method for extracting high-quality, object-centric meshes from photometric reconstruction techniques (e.g., NeRF, Gaussian Splatting) by employing alpha-transparent training while explicitly distinguishing foreground occlusions from background subtraction. We validate the full pipeline through extensive experiments, demonstrating its effectiveness across diverse objects. By eliminating the need for manual intervention or environment modifications, our pipeline can be integrated directly into existing pick-and-place setups, enabling scalable and efficient dataset creation. Project page (with code and data): https://scalable-real2sim.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22876v2">VizFlyt: Perception-centric Pedagogical Framework For Autonomous Aerial Robots</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted at ICRA 2025. Projected Page: https://pear.wpi.edu/research/vizflyt.html
    </div>
    <details class="paper-abstract">
      Autonomous aerial robots are becoming commonplace in our lives. Hands-on aerial robotics courses are pivotal in training the next-generation workforce to meet the growing market demands. Such an efficient and compelling course depends on a reliable testbed. In this paper, we present VizFlyt, an open-source perception-centric Hardware-In-The-Loop (HITL) photorealistic testing framework for aerial robotics courses. We utilize pose from an external localization system to hallucinate real-time and photorealistic visual sensors using 3D Gaussian Splatting. This enables stress-free testing of autonomy algorithms on aerial robots without the risk of crashing into obstacles. We achieve over 100Hz of system update rate. Lastly, we build upon our past experiences of offering hands-on aerial robotics courses and propose a new open-source and open-hardware curriculum based on VizFlyt for the future. We test our framework on various course projects in real-world HITL experiments and present the results showing the efficacy of such a system and its large potential use cases. Code, datasets, hardware guides and demo videos are available at https://pear.wpi.edu/research/vizflyt.html
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00773v1">DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting (3DGS) has gained considerable attentions in the field of novel view synthesis due to its fast performance while yielding the excellent image quality. However, 3DGS in sparse-view settings (e.g., three-view inputs) often faces with the problem of overfitting to training views, which significantly drops the visual quality of novel view images. Many existing approaches have tackled this issue by using strong priors, such as 2D generative contextual information and external depth signals. In contrast, this paper introduces a prior-free method, so-called DropGaussian, with simple changes in 3D Gaussian splatting. Specifically, we randomly remove Gaussians during the training process in a similar way of dropout, which allows non-excluded Gaussians to have larger gradients while improving their visibility. This makes the remaining Gaussians to contribute more to the optimization process for rendering with sparse input views. Such simple operation effectively alleviates the overfitting problem and enhances the quality of novel view synthesis. By simply applying DropGaussian to the original 3DGS framework, we can achieve the competitive performance with existing prior-based 3DGS methods in sparse-view settings of benchmark datasets without any additional complexity. The code and model are publicly available at: https://github.com/DCVL-3D/DropGaussian release.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00763v1">UnIRe: Unsupervised Instance Decomposition for Dynamic Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Reconstructing and decomposing dynamic urban scenes is crucial for autonomous driving, urban planning, and scene editing. However, existing methods fail to perform instance-aware decomposition without manual annotations, which is crucial for instance-level scene editing.We propose UnIRe, a 3D Gaussian Splatting (3DGS) based approach that decomposes a scene into a static background and individual dynamic instances using only RGB images and LiDAR point clouds. At its core, we introduce 4D superpoints, a novel representation that clusters multi-frame LiDAR points in 4D space, enabling unsupervised instance separation based on spatiotemporal correlations. These 4D superpoints serve as the foundation for our decomposed 4D initialization, i.e., providing spatial and temporal initialization to train a dynamic 3DGS for arbitrary dynamic classes without requiring bounding boxes or object templates.Furthermore, we introduce a smoothness regularization strategy in both 2D and 3D space, further improving the temporal stability.Experiments on benchmark datasets show that our method outperforms existing methods in decomposed dynamic scene reconstruction while enabling accurate and flexible instance-level editing, making it a practical solution for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00665v1">Monocular and Generalizable Gaussian Talking Head Animation</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      In this work, we introduce Monocular and Generalizable Gaussian Talking Head Animation (MGGTalk), which requires monocular datasets and generalizes to unseen identities without personalized re-training. Compared with previous 3D Gaussian Splatting (3DGS) methods that requires elusive multi-view datasets or tedious personalized learning/inference, MGGtalk enables more practical and broader applications. However, in the absence of multi-view and personalized training data, the incompleteness of geometric and appearance information poses a significant challenge. To address these challenges, MGGTalk explores depth information to enhance geometric and facial symmetry characteristics to supplement both geometric and appearance features. Initially, based on the pixel-wise geometric information obtained from depth estimation, we incorporate symmetry operations and point cloud filtering techniques to ensure a complete and precise position parameter for 3DGS. Subsequently, we adopt a two-stage strategy with symmetric priors for predicting the remaining 3DGS parameters. We begin by predicting Gaussian parameters for the visible facial regions of the source image. These parameters are subsequently utilized to improve the prediction of Gaussian parameters for the non-visible regions. Extensive experiments demonstrate that MGGTalk surpasses previous state-of-the-art methods, achieving superior performance across various metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00525v1">Robust LiDAR-Camera Calibration with 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted in IEEE Robotics and Automation Letters. Code available at: https://github.com/ShuyiZhou495/RobustCalibration
    </div>
    <details class="paper-abstract">
      LiDAR-camera systems have become increasingly popular in robotics recently. A critical and initial step in integrating the LiDAR and camera data is the calibration of the LiDAR-camera system. Most existing calibration methods rely on auxiliary target objects, which often involve complex manual operations, whereas targetless methods have yet to achieve practical effectiveness. Recognizing that 2D Gaussian Splatting (2DGS) can reconstruct geometric information from camera image sequences, we propose a calibration method that estimates LiDAR-camera extrinsic parameters using geometric constraints. The proposed method begins by reconstructing colorless 2DGS using LiDAR point clouds. Subsequently, we update the colors of the Gaussian splats by minimizing the photometric loss. The extrinsic parameters are optimized during this process. Additionally, we address the limitations of the photometric loss by incorporating the reprojection and triangulation losses, thereby enhancing the calibration robustness and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00437v1">ADGaussian: Generalizable Gaussian Splatting for Autonomous Driving with Multi-modal Inputs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 The project page can be found at https://maggiesong7.github.io/research/ADGaussian/
    </div>
    <details class="paper-abstract">
      We present a novel approach, termed ADGaussian, for generalizable street scene reconstruction. The proposed method enables high-quality rendering from single-view input. Unlike prior Gaussian Splatting methods that primarily focus on geometry refinement, we emphasize the importance of joint optimization of image and depth features for accurate Gaussian prediction. To this end, we first incorporate sparse LiDAR depth as an additional input modality, formulating the Gaussian prediction process as a joint learning framework of visual information and geometric clue. Furthermore, we propose a multi-modal feature matching strategy coupled with a multi-scale Gaussian decoding model to enhance the joint refinement of multi-modal features, thereby enabling efficient multi-modal Gaussian learning. Extensive experiments on two large-scale autonomous driving datasets, Waymo and KITTI, demonstrate that our ADGaussian achieves state-of-the-art performance and exhibits superior zero-shot generalization capabilities in novel-view shifting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00387v1">Scene4U: Hierarchical Layered 3D Scene Reconstruction from Single Panoramic Image for Your Immerse Exploration</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 CVPR 2025, 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The reconstruction of immersive and realistic 3D scenes holds significant practical importance in various fields of computer vision and computer graphics. Typically, immersive and realistic scenes should be free from obstructions by dynamic objects, maintain global texture consistency, and allow for unrestricted exploration. The current mainstream methods for image-driven scene construction involves iteratively refining the initial image using a moving virtual camera to generate the scene. However, previous methods struggle with visual discontinuities due to global texture inconsistencies under varying camera poses, and they frequently exhibit scene voids caused by foreground-background occlusions. To this end, we propose a novel layered 3D scene reconstruction framework from panoramic image, named Scene4U. Specifically, Scene4U integrates an open-vocabulary segmentation model with a large language model to decompose a real panorama into multiple layers. Then, we employs a layered repair module based on diffusion model to restore occluded regions using visual cues and depth information, generating a hierarchical representation of the scene. The multi-layer panorama is then initialized as a 3D Gaussian Splatting representation, followed by layered optimization, which ultimately produces an immersive 3D scene with semantic and structural consistency that supports free exploration. Scene4U outperforms state-of-the-art method, improving by 24.24% in LPIPS and 24.40% in BRISQUE, while also achieving the fastest training speed. Additionally, to demonstrate the robustness of Scene4U and allow users to experience immersive scenes from various landmarks, we build WorldVista3D dataset for 3D scene reconstruction, which contains panoramic images of globally renowned sites. The implementation code and dataset will be released at https://github.com/LongHZ140516/Scene4U .
    </details>
</div>
