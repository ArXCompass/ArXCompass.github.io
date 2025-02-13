# gaussian splatting - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01521v2">MiraGe: Editable 2D Images using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Implicit Neural Representations (INRs) approximate discrete data through continuous functions and are commonly used for encoding 2D images. Traditional image-based INRs employ neural networks to map pixel coordinates to RGB values, capturing shapes, colors, and textures within the network's weights. Recently, GaussianImage has been proposed as an alternative, using Gaussian functions instead of neural networks to achieve comparable quality and compression. Such a solution obtains a quality and compression ratio similar to classical INR models but does not allow image modification. In contrast, our work introduces a novel method, MiraGe, which uses mirror reflections to perceive 2D images in 3D space and employs flat-controlled Gaussians for precise 2D image editing. Our approach improves the rendering quality and allows realistic image modifications, including human-inspired perception of photos in the 3D world. Thanks to modeling images in 3D space, we obtain the illusion of 3D-based modification in 2D images. We also show that our Gaussian representation can be easily combined with a physics engine to produce physics-based modification of 2D images. Consequently, MiraGe allows for better quality than the standard approach and natural modification of 2D images
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09591v3">3D Gaussian Splatting as Markov Chain Monte Carlo</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting has recently become popular for neural rendering, current methods rely on carefully engineered cloning and splitting strategies for placing Gaussians, which can lead to poor-quality renderings, and reliance on a good initialization. In this work, we rethink the set of 3D Gaussians as a random sample drawn from an underlying probability distribution describing the physical representation of the scene-in other words, Markov Chain Monte Carlo (MCMC) samples. Under this view, we show that the 3D Gaussian updates can be converted as Stochastic Gradient Langevin Dynamics (SGLD) updates by simply introducing noise. We then rewrite the densification and pruning strategies in 3D Gaussian Splatting as simply a deterministic state transition of MCMC samples, removing these heuristics from the framework. To do so, we revise the 'cloning' of Gaussians into a relocalization scheme that approximately preserves sample probability. To encourage efficient use of Gaussians, we introduce a regularizer that promotes the removal of unused Gaussians. On various standard evaluation scenes, we show that our method provides improved rendering quality, easy control over the number of Gaussians, and robustness to initialization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18862v3">WeatherGS: 3D Scene Reconstruction in Adverse Weather Conditions via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for 3D scene reconstruction, but still suffers from complex outdoor environments, especially under adverse weather. This is because 3DGS treats the artifacts caused by adverse weather as part of the scene and will directly reconstruct them, largely reducing the clarity of the reconstructed scene. To address this challenge, we propose WeatherGS, a 3DGS-based framework for reconstructing clear scenes from multi-view images under different weather conditions. Specifically, we explicitly categorize the multi-weather artifacts into the dense particles and lens occlusions that have very different characters, in which the former are caused by snowflakes and raindrops in the air, and the latter are raised by the precipitation on the camera lens. In light of this, we propose a dense-to-sparse preprocess strategy, which sequentially removes the dense particles by an Atmospheric Effect Filter (AEF) and then extracts the relatively sparse occlusion masks with a Lens Effect Detector (LED). Finally, we train a set of 3D Gaussians by the processed images and generated masks for excluding occluded areas, and accurately recover the underlying clear scene by Gaussian splatting. We conduct a diverse and challenging benchmark to facilitate the evaluation of 3D reconstruction under complex weather scenarios. Extensive experiments on this benchmark demonstrate that our WeatherGS consistently produces high-quality, clean scenes across various weather scenarios, outperforming existing state-of-the-art methods. See project page:https://jumponthemoon.github.io/weather-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08085v1">Interactive Holographic Visualization for 3D Facial Avatar</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Traditional methods for visualizing dynamic human expressions, particularly in medical training, often rely on flat-screen displays or static mannequins, which have proven inefficient for realistic simulation. In response, we propose a platform that leverages a 3D interactive facial avatar capable of displaying non-verbal feedback, including pain signals. This avatar is projected onto a stereoscopic, view-dependent 3D display, offering a more immersive and realistic simulated patient experience for pain assessment practice. However, there is no existing solution that dynamically predicts and projects interactive 3D facial avatars in real-time. To overcome this, we emphasize the need for a 3D display projection system that can project the facial avatar holographically, allowing users to interact with the avatar from any viewpoint. By incorporating 3D Gaussian Splatting (3DGS) and real-time view-dependent calibration, we significantly improve the training environment for accurate pain recognition and assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01846v2">UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 https://aashishrai3799.github.io/uvgs
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated superior quality in modeling 3D objects and scenes. However, generating 3DGS remains challenging due to their discrete, unstructured, and permutation-invariant nature. In this work, we present a simple yet effective method to overcome these challenges. We utilize spherical mapping to transform 3DGS into a structured 2D representation, termed UVGS. UVGS can be viewed as multi-channel images, with feature dimensions as a concatenation of Gaussian attributes such as position, scale, color, opacity, and rotation. We further find that these heterogeneous features can be compressed into a lower-dimensional (e.g., 3-channel) shared feature space using a carefully designed multi-branch network. The compressed UVGS can be treated as typical RGB images. Remarkably, we discover that typical VAEs trained with latent diffusion models can directly generalize to this new representation without additional training. Our novel representation makes it effortless to leverage foundational 2D models, such as diffusion models, to directly model 3DGS. Additionally, one can simply increase the 2D UV resolution to accommodate more Gaussians, making UVGS a scalable solution compared to typical 3D backbones. This approach immediately unlocks various novel generation applications of 3DGS by inherently utilizing the already developed superior 2D generation capabilities. In our experiments, we demonstrate various unconditional, conditional generation, and inpainting applications of 3DGS based on diffusion models, which were previously non-trivial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07754v1">MeshSplats: Mesh-Based Rendering with Gaussian Splatting Initialization</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) is a recent and pivotal technique in 3D computer graphics. GS-based algorithms almost always bypass classical methods such as ray tracing, which offers numerous inherent advantages for rendering. For example, ray tracing is able to handle incoherent rays for advanced lighting effects, including shadows and reflections. To address this limitation, we introduce MeshSplats, a method which converts GS to a mesh-like format. Following the completion of training, MeshSplats transforms Gaussian elements into mesh faces, enabling rendering using ray tracing methods with all their associated benefits. Our model can be utilized immediately following transformation, yielding a mesh of slightly reduced quality without additional training. Furthermore, we can enhance the reconstruction quality through the application of a dedicated optimization algorithm that operates on mesh faces rather than Gaussian components. The efficacy of our method is substantiated by experimental results, underscoring its extensive applications in computer graphics and image processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07615v1">Flow Distillation Sampling: Regularizing 3D Gaussians with Pre-trained Matching Priors</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has achieved excellent rendering quality with fast training and rendering speed. However, its optimization process lacks explicit geometric constraints, leading to suboptimal geometric reconstruction in regions with sparse or no observational input views. In this work, we try to mitigate the issue by incorporating a pre-trained matching prior to the 3DGS optimization process. We introduce Flow Distillation Sampling (FDS), a technique that leverages pre-trained geometric knowledge to bolster the accuracy of the Gaussian radiance field. Our method employs a strategic sampling technique to target unobserved views adjacent to the input views, utilizing the optical flow calculated from the matching model (Prior Flow) to guide the flow analytically calculated from the 3DGS geometry (Radiance Flow). Comprehensive experiments in depth rendering, mesh reconstruction, and novel view synthesis showcase the significant advantages of FDS over state-of-the-art methods. Additionally, our interpretive experiments and analysis aim to shed light on the effects of FDS on geometric accuracy and rendering quality, potentially providing readers with insights into its performance. Project page: https://nju-3dv.github.io/projects/fds
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12886v2">EdgeGaussians -- 3D Edge Mapping via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 To appear in the proceedings of WACV 2025
    </div>
    <details class="paper-abstract">
      With their meaningful geometry and their omnipresence in the 3D world, edges are extremely useful primitives in computer vision. 3D edges comprise of lines and curves, and methods to reconstruct them use either multi-view images or point clouds as input. State-of-the-art image-based methods first learn a 3D edge point cloud then fit 3D edges to it. The edge point cloud is obtained by learning a 3D neural implicit edge field from which the 3D edge points are sampled on a specific level set (0 or 1). However, such methods present two important drawbacks: i) it is not realistic to sample points on exact level sets due to float imprecision and training inaccuracies. Instead, they are sampled within a range of levels so the points do not lie accurately on the 3D edges and require further processing. ii) Such implicit representations are computationally expensive and require long training times. In this paper, we address these two limitations and propose a 3D edge mapping that is simpler, more efficient, and preserves accuracy. Our method learns explicitly the 3D edge points and their edge direction hence bypassing the need for point sampling. It casts a 3D edge point as the center of a 3D Gaussian and the edge direction as the principal axis of the Gaussian. Such a representation has the advantage of being not only geometrically meaningful but also compatible with the efficient training optimization defined in Gaussian Splatting. Results show that the proposed method produces edges as accurate and complete as the state-of-the-art while being an order of magnitude faster. Code is released at https://github.com/kunalchelani/EdgeGaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12255v4">HAC++: Towards 100X Compression of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Project Page: https://yihangchen-ee.github.io/project_hac++/ Code: https://github.com/YihangChen-ee/HAC-plus. This paper is a journal extension of HAC at arXiv:2403.14530 (ECCV 2024)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising framework for novel view synthesis, boasting rapid rendering speed with high fidelity. However, the substantial Gaussians and their associated attributes necessitate effective compression techniques. Nevertheless, the sparse and unorganized nature of the point cloud of Gaussians (or anchors in our paper) presents challenges for compression. To achieve a compact size, we propose HAC++, which leverages the relationships between unorganized anchors and a structured hash grid, utilizing their mutual information for context modeling. Additionally, HAC++ captures intra-anchor contextual relationships to further enhance compression performance. To facilitate entropy coding, we utilize Gaussian distributions to precisely estimate the probability of each quantized attribute, where an adaptive quantization module is proposed to enable high-precision quantization of these attributes for improved fidelity restoration. Moreover, we incorporate an adaptive masking strategy to eliminate invalid Gaussians and anchors. Overall, HAC++ achieves a remarkable size reduction of over 100X compared to vanilla 3DGS when averaged on all datasets, while simultaneously improving fidelity. It also delivers more than 20X size reduction compared to Scaffold-GS. Our code is available at https://github.com/YihangChen-ee/HAC-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04843v2">PoI: Pixel of Interest for Novel View Synthesis Assisted Scene Coordinate Regression</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      The task of estimating camera poses can be enhanced through novel view synthesis techniques such as NeRF and Gaussian Splatting to increase the diversity and extension of training data. However, these techniques often produce rendered images with issues like blurring and ghosting, which compromise their reliability. These issues become particularly pronounced for Scene Coordinate Regression (SCR) methods, which estimate 3D coordinates at the pixel level. To mitigate the problems associated with unreliable rendered images, we introduce a novel filtering approach, which selectively extracts well-rendered pixels while discarding the inferior ones. This filter simultaneously measures the SCR model's real-time reprojection loss and gradient during training. Building on this filtering technique, we also develop a new strategy to improve scene coordinate regression using sparse inputs, drawing on successful applications of sparse input techniques in novel view synthesis. Our experimental results validate the effectiveness of our method, demonstrating state-of-the-art performance on indoor and outdoor datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08508v3">BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      We present billboard Splatting (BBSplat) - a novel approach for 3D scene representation based on textured geometric primitives. BBSplat represents the scene as a set of optimizable textured planar primitives with learnable RGB textures and alpha-maps to control their shape. BBSplat primitives can be used in any Gaussian Splatting pipeline as drop-in replacements for Gaussians. The proposed primitives close the rendering quality gap between 2D and 3D Gaussian Splatting (GS), preserving the accurate mesh extraction ability of 2D primitives. Our novel regularization term encourages textures to have a sparser structure, unlocking an efficient compression that leads to a reduction in the storage space of the model. Our experiments show the efficiency of BBSplat on standard datasets of real indoor and outdoor scenes such as Tanks&Temples, DTU, and Mip-NeRF-360.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11394v3">DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Score distillation sampling (SDS) has emerged as an effective framework in text-driven 3D editing tasks, leveraging diffusion models for 3D-consistent editing. However, existing SDS-based 3D editing methods suffer from long training times and produce low-quality results. We identify that the root cause of this performance degradation is \textit{their conflict with the sampling dynamics of diffusion models}. Addressing this conflict allows us to treat SDS as a diffusion reverse process for 3D editing via sampling from data space. In contrast, existing methods naively distill the score function using diffusion models. From these insights, we propose DreamCatalyst, a novel framework that considers these sampling dynamics in the SDS framework. Specifically, we devise the optimization process of our DreamCatalyst to approximate the diffusion reverse process in editing tasks, thereby aligning with diffusion sampling dynamics. As a result, DreamCatalyst successfully reduces training time and improves editing quality. Our method offers two modes: (1) a fast mode that edits Neural Radiance Fields (NeRF) scenes approximately 23 times faster than current state-of-the-art NeRF editing methods, and (2) a high-quality mode that produces superior results about 8 times faster than these methods. Notably, our high-quality mode outperforms current state-of-the-art NeRF editing methods in terms of both speed and quality. DreamCatalyst also surpasses the state-of-the-art 3D Gaussian Splatting (3DGS) editing methods, establishing itself as an effective and model-agnostic 3D editing solution. See more extensive results on our project page: https://dream-catalyst.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05769v2">Digital Twin Buildings: 3D Modeling, GIS Integration, and Visual Descriptions Using Gaussian Splatting, ChatGPT/Deepseek, and Google Maps Platform</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 -Fixed minor typo
    </div>
    <details class="paper-abstract">
      Urban digital twins are virtual replicas of cities that use multi-source data and data analytics to optimize urban planning, infrastructure management, and decision-making. Towards this, we propose a framework focused on the single-building scale. By connecting to cloud mapping platforms such as Google Map Platforms APIs, by leveraging state-of-the-art multi-agent Large Language Models data analysis using ChatGPT(4o) and Deepseek-V3/R1, and by using our Gaussian Splatting-based mesh extraction pipeline, our Digital Twin Buildings framework can retrieve a building's 3D model, visual descriptions, and achieve cloud-based mapping integration with large language model-based data analytics using a building's address, postal code, or geographic coordinates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07840v1">TranSplat: Surface Embedding-guided 3D Gaussian Splatting for Transparent Object Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Transparent object manipulation remains a sig- nificant challenge in robotics due to the difficulty of acquiring accurate and dense depth measurements. Conventional depth sensors often fail with transparent objects, resulting in in- complete or erroneous depth data. Existing depth completion methods struggle with interframe consistency and incorrectly model transparent objects as Lambertian surfaces, leading to poor depth reconstruction. To address these challenges, we propose TranSplat, a surface embedding-guided 3D Gaussian Splatting method tailored for transparent objects. TranSplat uses a latent diffusion model to generate surface embeddings that provide consistent and continuous representations, making it robust to changes in viewpoint and lighting. By integrating these surface embeddings with input RGB images, TranSplat effectively captures the complexities of transparent surfaces, enhancing the splatting of 3D Gaussians and improving depth completion. Evaluations on synthetic and real-world transpar- ent object benchmarks, as well as robot grasping tasks, show that TranSplat achieves accurate and dense depth completion, demonstrating its effectiveness in practical applications. We open-source synthetic dataset and model: https://github. com/jeongyun0609/TranSplat
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06519v1">SIREN: Semantic, Initialization-Free Registration of Multi-Robot Gaussian Splatting Maps</a></div>
    <div class="paper-meta">
      📅 2025-02-10
    </div>
    <details class="paper-abstract">
      We present SIREN for registration of multi-robot Gaussian Splatting (GSplat) maps, with zero access to camera poses, images, and inter-map transforms for initialization or fusion of local submaps. To realize these capabilities, SIREN harnesses the versatility and robustness of semantics in three critical ways to derive a rigorous registration pipeline for multi-robot GSplat maps. First, SIREN utilizes semantics to identify feature-rich regions of the local maps where the registration problem is better posed, eliminating the need for any initialization which is generally required in prior work. Second, SIREN identifies candidate correspondences between Gaussians in the local maps using robust semantic features, constituting the foundation for robust geometric optimization, coarsely aligning 3D Gaussian primitives extracted from the local maps. Third, this key step enables subsequent photometric refinement of the transformation between the submaps, where SIREN leverages novel-view synthesis in GSplat maps along with a semantics-based image filter to compute a high-accuracy non-rigid transformation for the generation of a high-fidelity fused map. We demonstrate the superior performance of SIREN compared to competing baselines across a range of real-world datasets, and in particular, across the most widely-used robot hardware platforms, including a manipulator, drone, and quadruped. In our experiments, SIREN achieves about 90x smaller rotation errors, 300x smaller translation errors, and 44x smaller scale errors in the most challenging scenes, where competing methods struggle. We will release the code and provide a link to the project page after the review process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01895v2">EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 Website: https://sites.google.com/view/enerverse
    </div>
    <details class="paper-abstract">
      We introduce EnerVerse, a generative robotics foundation model that constructs and interprets embodied spaces. EnerVerse employs an autoregressive video diffusion framework to predict future embodied spaces from instructions, enhanced by a sparse context memory for long-term reasoning. To model the 3D robotics world, we propose Free Anchor Views (FAVs), a multi-view video representation offering flexible, task-adaptive perspectives to address challenges like motion ambiguity and environmental constraints. Additionally, we present EnerVerse-D, a data engine pipeline combining the generative model with 4D Gaussian Splatting, forming a self-reinforcing data loop to reduce the sim-to-real gap. Leveraging these innovations, EnerVerse translates 4D world representations into physical actions via a policy head (EnerVerse-A), enabling robots to execute task instructions. EnerVerse-A achieves state-of-the-art performance in both simulation and real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14823v2">LapisGS: Layered Progressive 3D Gaussian Splatting for Adaptive Streaming</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 3DV 2025; Project Page: https://yuang-ian.github.io/lapisgs/ ; Code: https://github.com/nus-vv-streams/lapis-gs
    </div>
    <details class="paper-abstract">
      The rise of Extended Reality (XR) requires efficient streaming of 3D online worlds, challenging current 3DGS representations to adapt to bandwidth-constrained environments. This paper proposes LapisGS, a layered 3DGS that supports adaptive streaming and progressive rendering. Our method constructs a layered structure for cumulative representation, incorporates dynamic opacity optimization to maintain visual fidelity, and utilizes occupancy maps to efficiently manage Gaussian splats. This proposed model offers a progressive representation supporting a continuous rendering quality adapted for bandwidth-aware streaming. Extensive experiments validate the effectiveness of our approach in balancing visual fidelity with the compactness of the model, with up to 50.71% improvement in SSIM, 286.53% improvement in LPIPS with 23% of the original model size, and shows its potential for bandwidth-adapted 3D streaming and rendering applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07007v1">Grounding Creativity in Physics: A Brief Survey of Physical Priors in AIGC</a></div>
    <div class="paper-meta">
      📅 2025-02-10
    </div>
    <details class="paper-abstract">
      Recent advancements in AI-generated content have significantly improved the realism of 3D and 4D generation. However, most existing methods prioritize appearance consistency while neglecting underlying physical principles, leading to artifacts such as unrealistic deformations, unstable dynamics, and implausible objects interactions. Incorporating physics priors into generative models has become a crucial research direction to enhance structural integrity and motion realism. This survey provides a review of physics-aware generative methods, systematically analyzing how physical constraints are integrated into 3D and 4D generation. First, we examine recent works in incorporating physical priors into static and dynamic 3D generation, categorizing methods based on representation types, including vision-based, NeRF-based, and Gaussian Splatting-based approaches. Second, we explore emerging techniques in 4D generation, focusing on methods that model temporal dynamics with physical simulations. Finally, we conduct a comparative analysis of major methods, highlighting their strengths, limitations, and suitability for different materials and motion dynamics. By presenting an in-depth analysis of physics-grounded AIGC, this survey aims to bridge the gap between generative models and physical realism, providing insights that inspire future research in physically consistent content generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09981v3">Controllable Text-to-3D Generation via Surface-Aligned Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 3DV-2025
    </div>
    <details class="paper-abstract">
      While text-to-3D and image-to-3D generation tasks have received considerable attention, one important but under-explored field between them is controllable text-to-3D generation, which we mainly focus on in this work. To address this task, 1) we introduce Multi-view ControlNet (MVControl), a novel neural network architecture designed to enhance existing pre-trained multi-view diffusion models by integrating additional input conditions, such as edge, depth, normal, and scribble maps. Our innovation lies in the introduction of a conditioning module that controls the base diffusion model using both local and global embeddings, which are computed from the input condition images and camera poses. Once trained, MVControl is able to offer 3D diffusion guidance for optimization-based 3D generation. And, 2) we propose an efficient multi-stage 3D generation pipeline that leverages the benefits of recent large reconstruction models and score distillation algorithm. Building upon our MVControl architecture, we employ a unique hybrid diffusion guidance method to direct the optimization process. In pursuit of efficiency, we adopt 3D Gaussians as our representation instead of the commonly used implicit representations. We also pioneer the use of SuGaR, a hybrid representation that binds Gaussians to mesh triangle faces. This approach alleviates the issue of poor geometry in 3D Gaussians and enables the direct sculpting of fine-grained geometry on the mesh. Extensive experiments demonstrate that our method achieves robust generalization and enables the controllable generation of high-quality 3D content. Project page: https://lizhiqi49.github.io/MVControl/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05769v1">Digital Twin Buildings: 3D Modeling, GIS Integration, and Visual Descriptions Using Gaussian Splatting, ChatGPT/Deepseek, and Google Maps Platforms</a></div>
    <div class="paper-meta">
      📅 2025-02-09
    </div>
    <details class="paper-abstract">
      Urban digital twins are virtual replicas of cities that use multi-source data and data analytics to optimize urban planning, infrastructure management, and decision-making. Towards this, we propose a framework focused on the single-building scale. By connecting to cloud mapping platforms such as Google Map Platforms APIs, by leveraging state-of-the-art multi-agent Large Language Models data analysis using ChatGPT(4o) and Deepseek-V3/R1, and by using our Gaussian Splatting-based mesh extraction pipeline, our Digital Twin Buildings framework can retrieve a building's 3D model, visual descriptions, and achieve cloud-based mapping integration with large language model-based data analytics using a building's address, postal code, or geographic coordinates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05752v1">PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Robots require high-fidelity reconstructions of their environment for effective operation. Such scene representations should be both, geometrically accurate and photorealistic to support downstream tasks. While this can be achieved by building distance fields from range sensors and radiance fields from cameras, the scalable incremental mapping of both fields consistently and at the same time with high quality remains challenging. In this paper, we propose a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact point-based implicit neural map. By enforcing geometric consistency between these fields, we achieve mutual improvements by exploiting both modalities. We devise a LiDAR-visual SLAM system called PINGS using the proposed map representation and evaluate it on several challenging large-scale datasets. Experimental results demonstrate that PINGS can incrementally build globally consistent distance and radiance fields encoded with a compact set of neural points. Compared to the state-of-the-art methods, PINGS achieves superior photometric and geometric rendering at novel views by leveraging the constraints from the distance field. Furthermore, by utilizing dense photometric cues and multi-view consistency from the radiance field, PINGS produces more accurate distance fields, leading to improved odometry estimation and mesh reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10588v2">DEGAS: Detailed Expressions on Full-Body Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 3DV 2025
    </div>
    <details class="paper-abstract">
      Although neural rendering has made significant advances in creating lifelike, animatable full-body and head avatars, incorporating detailed expressions into full-body avatars remains largely unexplored. We present DEGAS, the first 3D Gaussian Splatting (3DGS)-based modeling method for full-body avatars with rich facial expressions. Trained on multiview videos of a given subject, our method learns a conditional variational autoencoder that takes both the body motion and facial expression as driving signals to generate Gaussian maps in the UV layout. To drive the facial expressions, instead of the commonly used 3D Morphable Models (3DMMs) in 3D head avatars, we propose to adopt the expression latent space trained solely on 2D portrait images, bridging the gap between 2D talking faces and 3D avatars. Leveraging the rendering capability of 3DGS and the rich expressiveness of the expression latent space, the learned avatars can be reenacted to reproduce photorealistic rendering images with subtle and accurate facial expressions. Experiments on an existing dataset and our newly proposed dataset of full-body talking avatars demonstrate the efficacy of our method. We also propose an audio-driven extension of our method with the help of 2D talking faces, opening new possibilities for interactive AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05409v1">Vision-in-the-loop Simulation for Deep Monocular Pose Estimation of UAV in Ocean Environment</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 8 pages, 15 figures, conference
    </div>
    <details class="paper-abstract">
      This paper proposes a vision-in-the-loop simulation environment for deep monocular pose estimation of a UAV operating in an ocean environment. Recently, a deep neural network with a transformer architecture has been successfully trained to estimate the pose of a UAV relative to the flight deck of a research vessel, overcoming several limitations of GPS-based approaches. However, validating the deep pose estimation scheme in an actual ocean environment poses significant challenges due to the limited availability of research vessels and the associated operational costs. To address these issues, we present a photo-realistic 3D virtual environment leveraging recent advancements in Gaussian splatting, a novel technique that represents 3D scenes by modeling image pixels as Gaussian distributions in 3D space, creating a lightweight and high-quality visual model from multiple viewpoints. This approach enables the creation of a virtual environment integrating multiple real-world images collected in situ. The resulting simulation enables the indoor testing of flight maneuvers while verifying all aspects of flight software, hardware, and the deep monocular pose estimation scheme. This approach provides a cost-effective solution for testing and validating the autonomous flight of shipboard UAVs, specifically focusing on vision-based control and estimation algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05176v1">AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360° Unbounded Scene Inpainting</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 Project page: https://kkennethwu.github.io/aurafusion360/
    </div>
    <details class="paper-abstract">
      Three-dimensional scene inpainting is crucial for applications from virtual reality to architectural visualization, yet existing methods struggle with view consistency and geometric accuracy in 360{\deg} unbounded scenes. We present AuraFusion360, a novel reference-based method that enables high-quality object removal and hole filling in 3D scenes represented by Gaussian Splatting. Our approach introduces (1) depth-aware unseen mask generation for accurate occlusion identification, (2) Adaptive Guided Depth Diffusion, a zero-shot method for accurate initial point placement without requiring additional training, and (3) SDEdit-based detail enhancement for multi-view coherence. We also introduce 360-USID, the first comprehensive dataset for 360{\deg} unbounded scene inpainting with ground truth. Extensive experiments demonstrate that AuraFusion360 significantly outperforms existing methods, achieving superior perceptual quality while maintaining geometric accuracy across dramatic viewpoint changes. See our project page for video results and the dataset at https://kkennethwu.github.io/aurafusion360/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18672v2">Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 Visit our project page at https://quyans.github.io/Drag-Your-Gaussian
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D scene editing have been propelled by the rapid development of generative models. Existing methods typically utilize generative models to perform text-guided editing on 3D representations, such as 3D Gaussian Splatting (3DGS). However, these methods are often limited to texture modifications and fail when addressing geometric changes, such as editing a character's head to turn around. Moreover, such methods lack accurate control over the spatial position of editing results, as language struggles to precisely describe the extent of edits. To overcome these limitations, we introduce DYG, an effective 3D drag-based editing method for 3D Gaussian Splatting. It enables users to conveniently specify the desired editing region and the desired dragging direction through the input of 3D masks and pairs of control points, thereby enabling precise control over the extent of editing. DYG integrates the strengths of the implicit triplane representation to establish the geometric scaffold of the editing results, effectively overcoming suboptimal editing outcomes caused by the sparsity of 3DGS in the desired editing regions. Additionally, we incorporate a drag-based Latent Diffusion Model into our method through the proposed Drag-SDS loss function, enabling flexible, multi-view consistent, and fine-grained editing. Extensive experiments demonstrate that DYG conducts effective drag-based editing guided by control point prompts, surpassing other baselines in terms of editing effect and quality, both qualitatively and quantitatively. Visit our project page at https://quyans.github.io/Drag-Your-Gaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05040v1">GaussRender: Learning 3D Occupancy with Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Understanding the 3D geometry and semantics of driving scenes is critical for developing of safe autonomous vehicles. While 3D occupancy models are typically trained using voxel-based supervision with standard losses (e.g., cross-entropy, Lovasz, dice), these approaches treat voxel predictions independently, neglecting their spatial relationships. In this paper, we propose GaussRender, a plug-and-play 3D-to-2D reprojection loss that enhances voxel-based supervision. Our method projects 3D voxel representations into arbitrary 2D perspectives and leverages Gaussian splatting as an efficient, differentiable rendering proxy of voxels, introducing spatial dependencies across projected elements. This approach improves semantic and geometric consistency, handles occlusions more efficiently, and requires no architectural modifications. Extensive experiments on multiple benchmarks (SurroundOcc-nuScenes, Occ3D-nuScenes, SSCBench-KITTI360) demonstrate consistent performance gains across various 3D occupancy models (TPVFormer, SurroundOcc, Symphonies), highlighting the robustness and versatility of our framework. The code is available at https://github.com/valeoai/GaussRender.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12255v3">HAC++: Towards 100X Compression of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 Project Page: https://yihangchen-ee.github.io/project_hac++/ Code: https://github.com/YihangChen-ee/HAC-plus. This paper is a journal extension of HAC at arXiv:2403.14530 (ECCV 2024)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising framework for novel view synthesis, boasting rapid rendering speed with high fidelity. However, the substantial Gaussians and their associated attributes necessitate effective compression techniques. Nevertheless, the sparse and unorganized nature of the point cloud of Gaussians (or anchors in our paper) presents challenges for compression. To achieve a compact size, we propose HAC++, which leverages the relationships between unorganized anchors and a structured hash grid, utilizing their mutual information for context modeling. Additionally, HAC++ captures intra-anchor contextual relationships to further enhance compression performance. To facilitate entropy coding, we utilize Gaussian distributions to precisely estimate the probability of each quantized attribute, where an adaptive quantization module is proposed to enable high-precision quantization of these attributes for improved fidelity restoration. Moreover, we incorporate an adaptive masking strategy to eliminate invalid Gaussians and anchors. Overall, HAC++ achieves a remarkable size reduction of over 100X compared to vanilla 3DGS when averaged on all datasets, while simultaneously improving fidelity. It also delivers more than 20X size reduction compared to Scaffold-GS. Our code is available at https://github.com/YihangChen-ee/HAC-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04981v1">OccGS: Zero-shot 3D Occupancy Reconstruction with Semantic and Geometric-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Obtaining semantic 3D occupancy from raw sensor data without manual annotations remains an essential yet challenging task. While prior works have approached this as a perception prediction problem, we formulate it as scene-aware 3D occupancy reconstruction with geometry and semantics. In this work, we propose OccGS, a novel 3D Occupancy reconstruction framework utilizing Semantic and Geometric-Aware Gaussian Splatting in a zero-shot manner. Leveraging semantics extracted from vision-language models and geometry guided by LiDAR points, OccGS constructs Semantic and Geometric-Aware Gaussians from raw multisensor data. We also develop a cumulative Gaussian-to-3D voxel splatting method for reconstructing occupancy from the Gaussians. OccGS performs favorably against self-supervised methods in occupancy prediction, achieving comparable performance to fully supervised approaches and achieving state-of-the-art performance on zero-shot semantic 3D occupancy estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04843v1">PoI: Pixel of Interest for Novel View Synthesis Assisted Scene Coordinate Regression</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      The task of estimating camera poses can be enhanced through novel view synthesis techniques such as NeRF and Gaussian Splatting to increase the diversity and extension of training data. However, these techniques often produce rendered images with issues like blurring and ghosting, which compromise their reliability. These issues become particularly pronounced for Scene Coordinate Regression (SCR) methods, which estimate 3D coordinates at the pixel level. To mitigate the problems associated with unreliable rendered images, we introduce a novel filtering approach, which selectively extracts well-rendered pixels while discarding the inferior ones. This filter simultaneously measures the SCR model's real-time reprojection loss and gradient during training. Building on this filtering technique, we also develop a new strategy to improve scene coordinate regression using sparse inputs, drawing on successful applications of sparse input techniques in novel view synthesis. Our experimental results validate the effectiveness of our method, demonstrating state-of-the-art performance on indoor and outdoor datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04734v1">SC-OmniGS: Self-Calibrating Omnidirectional Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 Accepted to ICLR 2025, Project Page: http://www.chenyingshu.com/sc-omnigs/
    </div>
    <details class="paper-abstract">
      360-degree cameras streamline data collection for radiance field 3D reconstruction by capturing comprehensive scene data. However, traditional radiance field methods do not address the specific challenges inherent to 360-degree images. We present SC-OmniGS, a novel self-calibrating omnidirectional Gaussian splatting system for fast and accurate omnidirectional radiance field reconstruction using 360-degree images. Rather than converting 360-degree images to cube maps and performing perspective image calibration, we treat 360-degree images as a whole sphere and derive a mathematical framework that enables direct omnidirectional camera pose calibration accompanied by 3D Gaussians optimization. Furthermore, we introduce a differentiable omnidirectional camera model in order to rectify the distortion of real-world data for performance enhancement. Overall, the omnidirectional camera intrinsic model, extrinsic poses, and 3D Gaussians are jointly optimized by minimizing weighted spherical photometric loss. Extensive experiments have demonstrated that our proposed SC-OmniGS is able to recover a high-quality radiance field from noisy camera poses or even no pose prior in challenging scenarios characterized by wide baselines and non-object-centric configurations. The noticeable performance gain in the real-world dataset captured by consumer-grade omnidirectional cameras verifies the effectiveness of our general omnidirectional camera model in reducing the distortion of 360-degree images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04630v1">High-Speed Dynamic 3D Imaging with Sensor Fusion Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Capturing and reconstructing high-speed dynamic 3D scenes has numerous applications in computer graphics, vision, and interdisciplinary fields such as robotics, aerodynamics, and evolutionary biology. However, achieving this using a single imaging modality remains challenging. For instance, traditional RGB cameras suffer from low frame rates, limited exposure times, and narrow baselines. To address this, we propose a novel sensor fusion approach using Gaussian splatting, which combines RGB, depth, and event cameras to capture and reconstruct deforming scenes at high speeds. The key insight of our method lies in leveraging the complementary strengths of these imaging modalities: RGB cameras capture detailed color information, event cameras record rapid scene changes with microsecond resolution, and depth cameras provide 3D scene geometry. To unify the underlying scene representation across these modalities, we represent the scene using deformable 3D Gaussians. To handle rapid scene movements, we jointly optimize the 3D Gaussian parameters and their temporal deformation fields by integrating data from all three sensor modalities. This fusion enables efficient, high-quality imaging of fast and complex scenes, even under challenging conditions such as low light, narrow baselines, or rapid motion. Experiments on synthetic and real datasets captured with our prototype sensor fusion setup demonstrate that our method significantly outperforms state-of-the-art techniques, achieving noticeable improvements in both rendering fidelity and structural accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18311v2">Neural Surface Priors for Editable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      In computer graphics and vision, recovering easily modifiable scene appearance from image data is crucial for applications such as content creation. We introduce a novel method that integrates 3D Gaussian Splatting with an implicit surface representation, enabling intuitive editing of recovered scenes through mesh manipulation. Starting with a set of input images and camera poses, our approach reconstructs the scene surface using a neural signed distance field. This neural surface acts as a geometric prior guiding the training of Gaussian Splatting components, ensuring their alignment with the scene geometry. To facilitate editing, we encode the visual and geometric information into a lightweight triangle soup proxy. Edits applied to the mesh extracted from the neural surface propagate seamlessly through this intermediate structure to update the recovered appearance. Unlike previous methods relying on the triangle soup proxy representation, our approach supports a wider range of modifications and fully leverages the mesh topology, enabling a more flexible and intuitive editing process. The complete source code for this project can be accessed at: https://github.com/WJakubowska/NeuralSurfacePriors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02283v2">GP-GS: Gaussian Processes for Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 14 pages,11 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as an efficient photorealistic novel view synthesis method. However, its reliance on sparse Structure-from-Motion (SfM) point clouds consistently compromises the scene reconstruction quality. To address these limitations, this paper proposes a novel 3D reconstruction framework Gaussian Processes Gaussian Splatting (GP-GS), where a multi-output Gaussian Process model is developed to achieve adaptive and uncertainty-guided densification of sparse SfM point clouds. Specifically, we propose a dynamic sampling and filtering pipeline that adaptively expands the SfM point clouds by leveraging GP-based predictions to infer new candidate points from the input 2D pixels and depth maps. The pipeline utilizes uncertainty estimates to guide the pruning of high-variance predictions, ensuring geometric consistency and enabling the generation of dense point clouds. The densified point clouds provide high-quality initial 3D Gaussians to enhance reconstruction performance. Extensive experiments conducted on synthetic and real-world datasets across various scales validate the effectiveness and practicality of the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03228v1">GARAD-SLAM: 3D GAussian splatting for Real-time Anti Dynamic SLAM</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      The 3D Gaussian Splatting (3DGS)-based SLAM system has garnered widespread attention due to its excellent performance in real-time high-fidelity rendering. However, in real-world environments with dynamic objects, existing 3DGS-based SLAM systems often face mapping errors and tracking drift issues. To address these problems, we propose GARAD-SLAM, a real-time 3DGS-based SLAM system tailored for dynamic scenes. In terms of tracking, unlike traditional methods, we directly perform dynamic segmentation on Gaussians and map them back to the front-end to obtain dynamic point labels through a Gaussian pyramid network, achieving precise dynamic removal and robust tracking. For mapping, we impose rendering penalties on dynamically labeled Gaussians, which are updated through the network, to avoid irreversible erroneous removal caused by simple pruning. Our results on real-world datasets demonstrate that our method is competitive in tracking compared to baseline methods, generating fewer artifacts and higher-quality reconstructions in rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00860v3">Segment Any 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 AAAI-25. Project page: https://jumpat.github.io/SAGA
    </div>
    <details class="paper-abstract">
      This paper presents SAGA (Segment Any 3D GAussians), a highly efficient 3D promptable segmentation method based on 3D Gaussian Splatting (3D-GS). Given 2D visual prompts as input, SAGA can segment the corresponding 3D target represented by 3D Gaussians within 4 ms. This is achieved by attaching an scale-gated affinity feature to each 3D Gaussian to endow it a new property towards multi-granularity segmentation. Specifically, a scale-aware contrastive training strategy is proposed for the scale-gated affinity feature learning. It 1) distills the segmentation capability of the Segment Anything Model (SAM) from 2D masks into the affinity features and 2) employs a soft scale gate mechanism to deal with multi-granularity ambiguity in 3D segmentation through adjusting the magnitude of each feature channel according to a specified 3D physical scale. Evaluations demonstrate that SAGA achieves real-time multi-granularity segmentation with quality comparable to state-of-the-art methods. As one of the first methods addressing promptable segmentation in 3D-GS, the simplicity and effectiveness of SAGA pave the way for future advancements in this field. Our code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11085v3">GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-05
      | 💬 Accepted to ICLR2025. During the ICLR review process, we changed the name of our framework from GSLoc to GS-CPR (Camera Pose Refinement) according to the comments of reviewers. The project page is available at https://gsloc.active.vision
    </div>
    <details class="paper-abstract">
      We leverage 3D Gaussian Splatting (3DGS) as a scene representation and propose a novel test-time camera pose refinement (CPR) framework, GS-CPR. This framework enhances the localization accuracy of state-of-the-art absolute pose regression and scene coordinate regression methods. The 3DGS model renders high-quality synthetic images and depth maps to facilitate the establishment of 2D-3D correspondences. GS-CPR obviates the need for training feature extractors or descriptors by operating directly on RGB images, utilizing the 3D foundation model, MASt3R, for precise 2D matching. To improve the robustness of our model in challenging outdoor environments, we incorporate an exposure-adaptive module within the 3DGS framework. Consequently, GS-CPR enables efficient one-shot pose refinement given a single RGB query and a coarse initial pose estimation. Our proposed approach surpasses leading NeRF-based optimization methods in both accuracy and runtime across indoor and outdoor visual localization benchmarks, achieving new state-of-the-art accuracy on two indoor datasets. The project page is available at https://gsloc.active.vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13971v2">GS-LiDAR: Generating Realistic LiDAR Point Clouds with Panoramic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-05
    </div>
    <details class="paper-abstract">
      LiDAR novel view synthesis (NVS) has emerged as a novel task within LiDAR simulation, offering valuable simulated point cloud data from novel viewpoints to aid in autonomous driving systems. However, existing LiDAR NVS methods typically rely on neural radiance fields (NeRF) as their 3D representation, which incurs significant computational costs in both training and rendering. Moreover, NeRF and its variants are designed for symmetrical scenes, making them ill-suited for driving scenarios. To address these challenges, we propose GS-LiDAR, a novel framework for generating realistic LiDAR point clouds with panoramic Gaussian splatting. Our approach employs 2D Gaussian primitives with periodic vibration properties, allowing for precise geometric reconstruction of both static and dynamic elements in driving scenarios. We further introduce a novel panoramic rendering technique with explicit ray-splat intersection, guided by panoramic LiDAR supervision. By incorporating intensity and ray-drop spherical harmonic (SH) coefficients into the Gaussian primitives, we enhance the realism of the rendered point clouds. Extensive experiments on KITTI-360 and nuScenes demonstrate the superiority of our method in terms of quantitative metrics, visual quality, as well as training and rendering efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02283v1">GP-GS: Gaussian Processes for Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 14 pages,11 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as an efficient photorealistic novel view synthesis method. However, its reliance on sparse Structure-from-Motion (SfM) point clouds consistently compromises the scene reconstruction quality. To address these limitations, this paper proposes a novel 3D reconstruction framework Gaussian Processes Gaussian Splatting (GP-GS), where a multi-output Gaussian Process model is developed to achieve adaptive and uncertainty-guided densification of sparse SfM point clouds. Specifically, we propose a dynamic sampling and filtering pipeline that adaptively expands the SfM point clouds by leveraging GP-based predictions to infer new candidate points from the input 2D pixels and depth maps. The pipeline utilizes uncertainty estimates to guide the pruning of high-variance predictions, ensuring geometric consistency and enabling the generation of dense point clouds. The densified point clouds provide high-quality initial 3D Gaussians to enhance reconstruction performance. Extensive experiments conducted on synthetic and real-world datasets across various scales validate the effectiveness and practicality of the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11625v3">GaussNav: Gaussian Splatting for Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 journal
    </div>
    <details class="paper-abstract">
      In embodied vision, Instance ImageGoal Navigation (IIN) requires an agent to locate a specific object depicted in a goal image within an unexplored environment. The primary challenge of IIN arises from the need to recognize the target object across varying viewpoints while ignoring potential distractors. Existing map-based navigation methods typically use Bird's Eye View (BEV) maps, which lack detailed texture representation of a scene. Consequently, while BEV maps are effective for semantic-level visual navigation, they are struggling for instance-level tasks. To this end, we propose a new framework for IIN, Gaussian Splatting for Visual Navigation (GaussNav), which constructs a novel map representation based on 3D Gaussian Splatting (3DGS). The GaussNav framework enables the agent to memorize both the geometry and semantic information of the scene, as well as retain the textural features of objects. By matching renderings of similar objects with the target, the agent can accurately identify, ground, and navigate to the specified object. Our GaussNav framework demonstrates a significant performance improvement, with Success weighted by Path Length (SPL) increasing from 0.347 to 0.578 on the challenging Habitat-Matterport 3D (HM3D) dataset. The source code is publicly available at the link: https://github.com/XiaohanLei/GaussNav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01949v1">LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Recently, the field of text-guided 3D scene generation has garnered significant attention. High-quality generation that aligns with physical realism and high controllability is crucial for practical 3D scene applications. However, existing methods face fundamental limitations: (i) difficulty capturing complex relationships between multiple objects described in the text, (ii) inability to generate physically plausible scene layouts, and (iii) lack of controllability and extensibility in compositional scenes. In this paper, we introduce LayoutDreamer, a framework that leverages 3D Gaussian Splatting (3DGS) to facilitate high-quality, physically consistent compositional scene generation guided by text. Specifically, given a text prompt, we convert it into a directed scene graph and adaptively adjust the density and layout of the initial compositional 3D Gaussians. Subsequently, dynamic camera adjustments are made based on the training focal point to ensure entity-level generation quality. Finally, by extracting directed dependencies from the scene graph, we tailor physical and layout energy to ensure both realism and flexibility. Comprehensive experiments demonstrate that LayoutDreamer outperforms other compositional scene generation quality and semantic alignment methods. Specifically, it achieves state-of-the-art (SOTA) performance in the multiple objects generation metric of T3Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19282v2">Reflective Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 Accepted for ICLR 2025
    </div>
    <details class="paper-abstract">
      Novel view synthesis has experienced significant advancements owing to increasingly capable NeRF- and 3DGS-based methods. However, reflective object reconstruction remains challenging, lacking a proper solution to achieve real-time, high-quality rendering while accommodating inter-reflection. To fill this gap, we introduce a Reflective Gaussian splatting (Ref-Gaussian) framework characterized with two components: (I) Physically based deferred rendering that empowers the rendering equation with pixel-level material properties via formulating split-sum approximation; (II) Gaussian-grounded inter-reflection that realizes the desired inter-reflection function within a Gaussian splatting paradigm for the first time. To enhance geometry modeling, we further introduce material-aware normal propagation and an initial per-Gaussian shading stage, along with 2D Gaussian primitives. Extensive experiments on standard datasets demonstrate that Ref-Gaussian surpasses existing approaches in terms of quantitative metrics, visual quality, and compute efficiency. Further, we show that our method serves as a unified solution for both reflective and non-reflective scenes, going beyond the previous alternatives focusing on only reflective scenes. Also, we illustrate that Ref-Gaussian supports more applications such as relighting and editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08982v2">CityLoc: 6DoF Pose Distributional Localization for Text Descriptions in Large-Scale Scenes with Gaussian Representation</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Localizing textual descriptions within large-scale 3D scenes presents inherent ambiguities, such as identifying all traffic lights in a city. Addressing this, we introduce a method to generate distributions of camera poses conditioned on textual descriptions, facilitating robust reasoning for broadly defined concepts. Our approach employs a diffusion-based architecture to refine noisy 6DoF camera poses towards plausible locations, with conditional signals derived from pre-trained text encoders. Integration with the pretrained Vision-Language Model, CLIP, establishes a strong linkage between text descriptions and pose distributions. Enhancement of localization accuracy is achieved by rendering candidate poses using 3D Gaussian splatting, which corrects misaligned samples through visual reasoning. We validate our method's superiority by comparing it against standard distribution estimation methods across five large-scale datasets, demonstrating consistent outperformance. Code, datasets and more information will be publicly available at our project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12906v2">CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Recently, generalizable feed-forward methods based on 3D Gaussian Splatting have gained significant attention for their potential to reconstruct 3D scenes using finite resources. These approaches create a 3D radiance field, parameterized by per-pixel 3D Gaussian primitives, from just a few images in a single forward pass. However, unlike multi-view methods that benefit from cross-view correspondences, 3D scene reconstruction with a single-view image remains an underexplored area. In this work, we introduce CATSplat, a novel generalizable transformer-based framework designed to break through the inherent constraints in monocular settings. First, we propose leveraging textual guidance from a visual-language model to complement insufficient information from a single image. By incorporating scene-specific contextual details from text embeddings through cross-attention, we pave the way for context-aware 3D scene reconstruction beyond relying solely on visual cues. Moreover, we advocate utilizing spatial guidance from 3D point features toward comprehensive geometric understanding under single-view settings. With 3D priors, image features can capture rich structural insights for predicting 3D Gaussians without multi-view techniques. Extensive experiments on large-scale datasets demonstrate the state-of-the-art performance of CATSplat in single-view 3D scene reconstruction with high-quality novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01846v1">UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 https://aashishrai3799.github.io/uvgs
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated superior quality in modeling 3D objects and scenes. However, generating 3DGS remains challenging due to their discrete, unstructured, and permutation-invariant nature. In this work, we present a simple yet effective method to overcome these challenges. We utilize spherical mapping to transform 3DGS into a structured 2D representation, termed UVGS. UVGS can be viewed as multi-channel images, with feature dimensions as a concatenation of Gaussian attributes such as position, scale, color, opacity, and rotation. We further find that these heterogeneous features can be compressed into a lower-dimensional (e.g., 3-channel) shared feature space using a carefully designed multi-branch network. The compressed UVGS can be treated as typical RGB images. Remarkably, we discover that typical VAEs trained with latent diffusion models can directly generalize to this new representation without additional training. Our novel representation makes it effortless to leverage foundational 2D models, such as diffusion models, to directly model 3DGS. Additionally, one can simply increase the 2D UV resolution to accommodate more Gaussians, making UVGS a scalable solution compared to typical 3D backbones. This approach immediately unlocks various novel generation applications of 3DGS by inherently utilizing the already developed superior 2D generation capabilities. In our experiments, we demonstrate various unconditional, conditional generation, and inpainting applications of 3DGS based on diffusion models, which were previously non-trivial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01826v1">Scalable 3D Gaussian Splatting-Based RF Signal Spatial Propagation Modeling</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Effective network planning and sensing in wireless networks require resource-intensive site surveys for data collection. An alternative is Radio-Frequency (RF) signal spatial propagation modeling, which computes received signals given transceiver positions in a scene (e.g.s a conference room). We identify a fundamental trade-off between scalability and fidelity in the state-of-the-art method. To address this issue, we explore leveraging 3D Gaussian Splatting (3DGS), an advanced technique for the image synthesis of 3D scenes in real-time from arbitrary camera poses. By integrating domain-specific insights, we design three components for adapting 3DGS to the RF domain, including Gaussian-based RF scene representation, gradient-guided RF attribute learning, and RF-customized CUDA for ray tracing. Building on them, we develop RFSPM, an end-to-end framework for scalable RF signal Spatial Propagation Modeling. We evaluate RFSPM in four field studies and two applications across RFID, BLE, LoRa, and 5G, covering diverse frequencies, antennas, signals, and scenes. The results show that RFSPM matches the fidelity of the state-of-the-art method while reducing data requirements, training GPU-hours, and inference latency by up to 9.8\,$\times$, 18.6\,$\times$, and 84.4\,$\times$, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01536v1">VR-Robo: A Real-to-Sim-to-Real Framework for Visual Robot Navigation and Locomotion</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 Project Page: https://vr-robo.github.io/
    </div>
    <details class="paper-abstract">
      Recent success in legged robot locomotion is attributed to the integration of reinforcement learning and physical simulators. However, these policies often encounter challenges when deployed in real-world environments due to sim-to-real gaps, as simulators typically fail to replicate visual realism and complex real-world geometry. Moreover, the lack of realistic visual rendering limits the ability of these policies to support high-level tasks requiring RGB-based perception like ego-centric navigation. This paper presents a Real-to-Sim-to-Real framework that generates photorealistic and physically interactive "digital twin" simulation environments for visual navigation and locomotion learning. Our approach leverages 3D Gaussian Splatting (3DGS) based scene reconstruction from multi-view images and integrates these environments into simulations that support ego-centric visual perception and mesh-based physical interactions. To demonstrate its effectiveness, we train a reinforcement learning policy within the simulator to perform a visual goal-tracking task. Extensive experiments show that our framework achieves RGB-only sim-to-real policy transfer. Additionally, our framework facilitates the rapid adaptation of robot policies with effective exploration capability in complex new environments, highlighting its potential for applications in households and factories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01157v1">Radiant Foam: Real-Time Differentiable Ray Tracing</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Research on differentiable scene representations is consistently moving towards more efficient, real-time models. Recently, this has led to the popularization of splatting methods, which eschew the traditional ray-based rendering of radiance fields in favor of rasterization. This has yielded a significant improvement in rendering speeds due to the efficiency of rasterization algorithms and hardware, but has come at a cost: the approximations that make rasterization efficient also make implementation of light transport phenomena like reflection and refraction much more difficult. We propose a novel scene representation which avoids these approximations, but keeps the efficiency and reconstruction quality of splatting by leveraging a decades-old efficient volumetric mesh ray tracing algorithm which has been largely overlooked in recent computer vision research. The resulting model, which we name Radiant Foam, achieves rendering speed and quality comparable to Gaussian Splatting, without the constraints of rasterization. Unlike ray traced Gaussian models that use hardware ray tracing acceleration, our method requires no special hardware or APIs beyond the standard features of a programmable GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16619v3">Topology-Aware 3D Gaussian Splatting: Leveraging Persistent Homology for Optimized Structural Integrity</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has emerged as a crucial technique for representing discrete volumetric radiance fields. It leverages unique parametrization to mitigate computational demands in scene optimization. This work introduces Topology-Aware 3D Gaussian Splatting (Topology-GS), which addresses two key limitations in current approaches: compromised pixel-level structural integrity due to incomplete initial geometric coverage, and inadequate feature-level integrity from insufficient topological constraints during optimization. To overcome these limitations, Topology-GS incorporates a novel interpolation strategy, Local Persistent Voronoi Interpolation (LPVI), and a topology-focused regularization term based on persistent barcodes, named PersLoss. LPVI utilizes persistent homology to guide adaptive interpolation, enhancing point coverage in low-curvature areas while preserving topological structure. PersLoss aligns the visual perceptual similarity of rendered images with ground truth by constraining distances between their topological features. Comprehensive experiments on three novel-view synthesis benchmarks demonstrate that Topology-GS outperforms existing methods in terms of PSNR, SSIM, and LPIPS metrics, while maintaining efficient memory usage. This study pioneers the integration of topology with 3D-GS, laying the groundwork for future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06927v2">CULTURE3D: Cultural Landmarks and Terrain Dataset for 3D Applications</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      In this paper, we present a large-scale fine-grained dataset using high-resolution images captured from locations worldwide. Compared to existing datasets, our dataset offers a significantly larger size and includes a higher level of detail, making it uniquely suited for fine-grained 3D applications. Notably, our dataset is built using drone-captured aerial imagery, which provides a more accurate perspective for capturing real-world site layouts and architectural structures. By reconstructing environments with these detailed images, our dataset supports applications such as the COLMAP format for Gaussian Splatting and the Structure-from-Motion (SfM) method. It is compatible with widely-used techniques including SLAM, Multi-View Stereo, and Neural Radiance Fields (NeRF), enabling accurate 3D reconstructions and point clouds. This makes it a benchmark for reconstruction and segmentation tasks. The dataset enables seamless integration with multi-modal data, supporting a range of 3D applications, from architectural reconstruction to virtual tourism. Its flexibility promotes innovation, facilitating breakthroughs in 3D modeling and analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00654v1">EmoTalkingGaussian: Continuous Emotion-conditioned Talking Head Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting-based talking head synthesis has recently gained attention for its ability to render high-fidelity images with real-time inference speed. However, since it is typically trained on only a short video that lacks the diversity in facial emotions, the resultant talking heads struggle to represent a wide range of emotions. To address this issue, we propose a lip-aligned emotional face generator and leverage it to train our EmoTalkingGaussian model. It is able to manipulate facial emotions conditioned on continuous emotion values (i.e., valence and arousal); while retaining synchronization of lip movements with input audio. Additionally, to achieve the accurate lip synchronization for in-the-wild audio, we introduce a self-supervised learning method that leverages a text-to-speech network and a visual-audio synchronization network. We experiment our EmoTalkingGaussian on publicly available videos and have obtained better results than state-of-the-arts in terms of image quality (measured in PSNR, SSIM, LPIPS), emotion expression (measured in V-RMSE, A-RMSE, V-SA, A-SA, Emotion Accuracy), and lip synchronization (measured in LMD, Sync-E, Sync-C), respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00333v3">Gaussians on their Way: Wasserstein-Constrained 4D Gaussian Splatting with State-Space Modeling</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Dynamic scene rendering has taken a leap forward with the rise of 4D Gaussian Splatting, but there's still one elusive challenge: how to make 3D Gaussians move through time as naturally as they would in the real world, all while keeping the motion smooth and consistent. In this paper, we unveil a fresh approach that blends state-space modeling with Wasserstein geometry, paving the way for a more fluid and coherent representation of dynamic scenes. We introduce a State Consistency Filter that merges prior predictions with the current observations, enabling Gaussians to stay true to their way over time. We also employ Wasserstein distance regularization to ensure smooth, consistent updates of Gaussian parameters, reducing motion artifacts. Lastly, we leverage Wasserstein geometry to capture both translational motion and shape deformations, creating a more physically plausible model for dynamic scenes. Our approach guides Gaussians along their natural way in the Wasserstein space, achieving smoother, more realistic motion and stronger temporal coherence. Experimental results show significant improvements in rendering quality and efficiency, outperforming current state-of-the-art techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09740v2">Gaussian Splatting Visual MPC for Granular Media Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 project website https://weichengtseng.github.io/gs-granular-mani/
    </div>
    <details class="paper-abstract">
      Recent advancements in learned 3D representations have enabled significant progress in solving complex robotic manipulation tasks, particularly for rigid-body objects. However, manipulating granular materials such as beans, nuts, and rice, remains challenging due to the intricate physics of particle interactions, high-dimensional and partially observable state, inability to visually track individual particles in a pile, and the computational demands of accurate dynamics prediction. Current deep latent dynamics models often struggle to generalize in granular material manipulation due to a lack of inductive biases. In this work, we propose a novel approach that learns a visual dynamics model over Gaussian splatting representations of scenes and leverages this model for manipulating granular media via Model-Predictive Control. Our method enables efficient optimization for complex manipulation tasks on piles of granular media. We evaluate our approach in both simulated and real-world settings, demonstrating its ability to solve unseen planning tasks and generalize to new environments in a zero-shot transfer. We also show significant prediction and manipulation performance improvements compared to existing granular media manipulation methods.
    </details>
</div>
