# gaussian splatting - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30342v1">Uncertainty-driven 3D Gaussian Splatting Active Mapping via Anisotropic Visibility Field</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted to CVPR 2026. Project page https://gatech-rl2.github.io/GAVIS/
    </div>
    <details class="paper-abstract">
      We present Gaussian Splatting Anisotropic Visibility Field (GAVIS), a novel framework for uncertainty quantification and active mapping in 3DGS. Our key insight is that regions unseen from the training views yield unreliable predictions from the 3DGS. To address this, we introduce a principled and efficient method for quantifying the visibility field in 3DGS, defined as the anisotropic visibility of each particle with respect to the training views, and represented using spherical harmonics. The resulting visibility field is integrated into a Bayesian Network-based uncertainty-aware 3DGS rasterizer, enabling real-time (200 FPS) uncertainty quantification for synthesized views. Active mapping is further performed within a maximum information gain framework building on this formulation. Extensive experiments across diverse environments demonstrate that GAVIS consistently and significantly outperforms prior approaches in both accuracy and efficiency. Moreover, beyond standalone use, our method can be applied post-hoc to improve the performance of existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30328v1">Supercharging Thermal Gaussian Splatting with Depth Estimation</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 8 pages, 4 figures. Accepted and will be published in ISPRS proceedings (ISPRS Congress 2026)
    </div>
    <details class="paper-abstract">
      Efficient and robust 3D scene representation is crucial in autonomous driving, robotics, and related fields. While RGB images provide valuable content for 3D reconstruction, other modalities like thermal or depth can enable additional information on the environment. Lately, novel view synthesis methods like 3D Gaussian Splatting have started using multiple modalities to further boost their performance. But fusing or combining multimodal data can make the process slower and can bring in additional challenges. Therefore, our project aims to use single modality based on thermal infrared domain, by removing the reliance on visible light as much as possible. This single modality can be expected to be faster as it does not rely on multimodal data. We propose a method, Thermal-to-Depth Gaussian Splatting (TDg), that uses only thermal images and depth estimation in its architecture to derive the radiance fields. Our TDg method outperforms the MSMG (Multiple Single-Modal Gaussians) baseline in most cases on our test datasets, RGBT-Scenes and ThermalMix. On average, the rendering quality metrics such as learned perceptual image patch similarity (LPIPS), structural similarity index measure (SSIM), and peak signal-to-noise ratio (PSNR) of TDg are 1.12%, 0.034%, and 0.01% better than the baseline MSMG values. It also reduces the training time significantly, by 12 mins 47 secs (55% improvement). Overall, our method is successful in deriving these thermal radiance fields, which can ultimately have several applications, such as identifying heat sources critical in surveillance, search or rescue operations, and industrial inspections where temperature is widely used to monitor machines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30320v1">MonoPhysics: Estimating Geometry, Appearance, and Physical Parameters from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Existing inverse physics methods recover physical parameters from multi-view videos, where geometric constraints across views resolve scale and 3D structure. In monocular settings, however, such constraints are absent, leading to severe scale ambiguity, inaccurate geometry, and weak coupling between appearance optimization and physical simulation. We propose MonoPhysics, a framework for monocular inverse physics estimation of deformable objects using differentiable MPM simulation and 3D Gaussian Splatting, which jointly optimizes geometry, appearance, and physical parameters from a single camera view. We address these challenges through three visual-physical bridges: global scale alignment, physics-aware geometry refinement, and a differentiable position map, which together enable accurate optimization from monocular observations alone. We evaluate on Vid2Sim and our new dataset of elastic and plastic objects, showing that MonoPhysics outperforms existing baselines in monocular settings and achieves performance comparable to multi-view baselines using only a single camera. Our project page is available at https://daniel03c1.github.io/MonoPhysics/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30310v1">City-Mesh3R: Simulation-Ready City-Scale 3D Mesh Reconstruction from Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted to the USM3D Workshop Proceedings at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026 as an Oral Presentation. Project page: https://citymesh3r.github.io/
    </div>
    <details class="paper-abstract">
      City-scale 3D surface reconstruction from multiview images for downstream 3D simulation, poses highly challenging problems due to the scale and complexity of urban scenes. Existing city-scale 3D reconstruction methods based on NeRF, Gaussian Splatting etc. often fail to recover 3D meshes ready for simulation due to incomplete/missing geometry and irregular, noisy surfaces. Scaling existing small-scale 3D reconstruction methods to arbitrarily large urban scenes is highly infeasible due to their computational complexity. We present City-Mesh3R, a scalable framework for reconstructing watertight surface meshes directly from large unordered image collections. Unlike recent methods which use global sparse SfM point-cloud initialization followed by a distributed 3D dense reconstruction of large-scale scenes, our method follows an end-to-end images-to-mesh 3D reconstruction approach using a divide-and-conquer strategy. The sparse city map is reconstructed via topological image clustering, cluster-wise independent sparse SfM and map merging, without need for exhaustive image feature matching. Then this map is partitioned spatially to perform geometry-aware camera selection, followed by dense surface reconstruction and surface refinement using curvature-aware adaptive vertex density remeshing. These partition meshes are then stitched together to produce the global mesh of the city. The proposed end-to-end framework is evaluated on city-scale reconstruction datasets. As demonstrated by our qualitative and quantitative results, our proposed method yields high-fidelity watertight 3D meshes with regular geometry, capturing fine surface details, and is suitable for scaling to arbitrarily large scenes owing to the end-to-end processing in a distributed setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30268v1">PhyGenHOI: Physically-Aware 4D Generation of Dynamic Human-Object Interactions</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      We address the task of generating physically accurate and visually faithful 4D Human-Object Interaction (HOI). Given a static 3D human and target object represented as 3D Gaussian Splats (3DGS), our goal is to synthesize dynamic scenes where the human actively engages with the object through actions, such as punching or kicking, in accordance with a given input text. To this end, we introduce PhyGenHOI, a novel framework that couples generative human motion with an explicit physical object simulation. We model the human as a semantic agent driven by a Motion Diffusion Model (MDM) and the object as a physical agent simulated via the Material Point Method (MPM), utilizing 3D Gaussians as a unified, differentiable representation. We supervise their interaction through three coupled mechanisms: (1) A Windowed Attraction Loss that temporally synchronizes generative motion to intercept the object; (2) A Contact-Driven Re-simulation step that triggers physically consistent momentum transfer upon impact; and (3) A Masked Video-SDS objective that injects video-based priors to enhance contact fidelity. Experiments show PhyGenHOI generates physically consistent 4D HOI across diverse actions, humans, and objects, outperforming baselines. Project page and videos: https://omerbenishu.github.io/PhyGenHOI/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05155v3">Aes3D: Aesthetic Assessment in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) gains attention in immersive media and digital content creation, assessing the aesthetics of 3D scenes becomes important in helping creators build more visually compelling 3D content. However, existing evaluation methods for 3D scenes primarily emphasize reconstruction fidelity and perceptual realism, largely overlooking higher-level aesthetic attributes such as composition, harmony, and visual appeal. This limitation comes from two key challenges: (1) the absence of general 3DGS datasets with aesthetic annotations, and (2) the intrinsic nature of 3DGS as a low-level primitive representation, which makes it difficult to capture high-level aesthetic features. To address these challenges, we propose Aes3D, the first systematic framework for assessing the aesthetics of 3D neural rendering scenes. Aes3D includes Aesthetic3D, the first dataset dedicated to 3D scene aesthetic assessment, built on our proposed annotation strategy for 3D scene aesthetics. In addition, we present Aes3DGSNet, a lightweight model that directly predicts scene-level aesthetic scores from 3DGS representations. Notably, our model operates solely on 3D Gaussian primitives, eliminating the need for rendering multi-view images and thus reducing computational cost and hardware requirements. Through aesthetics-supervised learning on multi-view 3DGS scene representations, Aes3DGSNet effectively captures high-level aesthetic cues and accurately regresses aesthetic scores. Experimental results demonstrate that our approach achieves strong performance while maintaining a lightweight design, establishing a new benchmark for 3D scene aesthetic assessment. Code and datasets will be made available in a future version.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30065v1">Boosting Zero-Shot 3D Style Transfer with 2D Pre-trained Priors</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted by IEEE IVMSP2026
    </div>
    <details class="paper-abstract">
      In this work, we focus on zero-shot 3D style transfer that can generate multi-view consistent stylized views of the 3D scene given an arbitrary style image. We primarily tackle the issue of data scarcity in 3D style transfer, which arises when each model is trained on only a single scene, thereby limiting the number of available content images. This scarcity significantly hampers stylization performance, as model optimization relies on a sufficient number of content-style image pairs to provide supervisory signals. Our core idea is to integrate a decoder pre-trained on large-scale 2D image datasets into the 3D style transfer pipeline, thereby leveraging the prior knowledge encoded in the decoder from learning over numerous content-style image pairs. Our method combines feature Gaussian splatting and deferred stylization, enabling high-quality stylization with the data-sufficient decoder network while ensuring view consistency by unifying view-dependent operations into a view-invariant process. Experiments demonstrate that our Data-Sufficient StyleGaussian (DS-StyleGaussian) model outperforms existing zero-shot 3D style transfer methods in terms of visual quality across various datasets. This work also suggests that 2D pre-training can serve as a strong enhancement for 3D tasks, bridging the data gap between 2D and 3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29997v1">FRUC: Feedforward Dynamic Scene Reconstruction from Uncalibrated Collaborative Driving Views</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      We present FRUC, a feed-forward 3D Gaussian splatting framework for dynamic scene reconstruction from uncalibrated collaborative driving views. Existing multi-agent reconstruction frameworks are often hindered by rigid prerequisites, demanding precise spatial calibration and slow per-scene optimization. In this paper, we rethink this task by conceptualizing a distributed multi-vehicle network as a spatio-temporally unstructured ego-centric multi-camera system, where the core challenge lies in enhancing ego-centric occluded geometry through collaboration without degrading the ego's accurately observed visible geometry, while preserving reconstruction efficiency. For efficient reconstruction, FRUC is built upon a visual grounded geometric Transformer backbone to enable one-shot, calibration-free inference from a flexible number of multi-vehicle views. To achieve non-destructive geometric supplementation under uncalibrated cross-agent misalignment, FRUC first introduces an ego-centric causal occlusion field that explicitly derives occlusion evolution as latent priors by modeling agent-wise spatio-temporal correlations. Guided by these occlusion priors, it further formulates cross-agent integration as a deterministic residual denoising process via zero-initialized injection, turning challenging cross-agent fusion into bounded residual learning for robust collaborative blind-spot completion. Through extensive evaluations on the real-world V2XReal and UrbanIng-V2X datasets, FRUC is shown to be a new state-of-the-art for the scene reconstruction of dynamic collaborative driving environments, significantly outperforming existing methods in both rendering quality and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25975v2">F-RNG: Feed-Forward Relightable Neural Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      Capturing relightable 3D assets from real-world objects is a widely researched problem. Several per-scene optimization-based methods, based on 3D Gaussian splatting (3DGS), support relighting; however, they usually require dense input views, and their overfitting nature makes it difficult to generalize across scenes. Unlike per-scene optimization methods, generalized feed-forward models can directly reconstruct Gaussians from sparse input views. However, the resulting assets have baked-in illumination and cannot be easily used for relighting. In this paper, we present F-RNG, a feed-forward framework that directly generates relightable 3DGS assets from sparse-view inputs. Training such a model from scratch can require massive data and computing resources, and it is especially challenging to generate relightable assets in a feed-forward manner with acceptable cost. We develop F-RNG upon an existing large reconstruction model (LRM) to extract relightable representations, while also utilizing priors from an intrinsic decomposition model (IDM). Specifically, we first introduce a latent-interpolated fine-grained geometry synthesis to enhance the LRM's geometry representation. Second, we propose a prior-guided relightable appearance distillation to extract relightable neural representations by incorporating IDM priors. Finally, a universal neural renderer enables flexible and high-fidelity relighting. F-RNG requires neither re-training nor fine-tuning of the underlying LRMs, thus can automatically benefit from better LRMs and IDMs in the future. With only small networks that can be trained with affordable data and computational resources, F-RNG avoids the repetitive inference of large models under different light conditions. By comparison to the state-of-the-art LRM-based relighting method, F-RNG achieves ~25x faster relighting, as well as superior quality (~+2.0 dB).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29583v1">BitC-3DGS: High-Capacity 3D Gaussian Splatting Watermarking via Bit Compression</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      High-capacity watermarking is necessary for 3D Gaussian Splatting (3DGS) assets to embed rich information (e.g., ownership, provenance, and authentication codes), enabling reliable identification and integrity verification in large-scale 3D asset pipelines. Existing bit-to-token watermarking methods based on a pre-trained text encoder are limited to 77-bit messages due to CLIP's fixed 77-token context length, as tokens beyond this limit are unsupported by learned positional embeddings. To address this limitation, we introduce BitC-3DGS, a bit-compression framework that encodes multiple message bits per token. It employs a bit-compressed tokenization scheme that encodes multiple bits within the same chunk into a single semantic token. To enable recovery of the compressed information, it further introduces a dual-branch architecture for joint chunk decompression and bit decoding, along with a hard-message sampling strategy to improve combinatorial coverage during decoder training. Extensive experiments on the Blender and LLFF datasets demonstrate the effectiveness of BitC-3DGS for high-capacity watermarking, achieving high message recovery accuracy and rendering fidelity. For example, it supports 128-bit message capacity with recovery accuracy comparable to that of 64-bit messages in recent state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29549v1">Learning Representations from 3D Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 5 figures, 15 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a recent approach for scene rendering. Although primarily designed for view synthesis, its potential for scene understanding tasks remains underexplored. In this work, we conduct a comparative evaluation of various geometric deep learning architectures for the classification of 3D scenes represented using Gaussian Splatting. We benchmark point-based and graph-based models across both traditional point cloud datasets and dedicated Gaussian Splatting datasets. Scenes are embedded into latent representations, which are evaluated through end-to-end classification, linear probing, and clustering analysis. Our study provides insight into the suitability of different geometry-aware architectures and input feature configurations for learning effective 3D Gaussian Splat representations. The results highlight consistent differences between architectural families and reveal the impact of Gaussian-specific attributes on the quality of representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29452v1">Comparative evaluation of photogrammetric reconstruction methods and 3D Gaussian Splatting for road surface roughness analysis</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 accepted by RSMIP 2026
    </div>
    <details class="paper-abstract">
      Image-based 3D reconstruction offers a low-cost alternative to traditional sensor-based techniques for road surface assessment. This study compares four reconstruction pipelines--COLMAP, Meshroom, Metashape, and 3D Gaussian Splatting (3DGS)--to evaluate their ability to estimate road surface roughness from smartphone imagery. All point clouds were processed in CloudCompare using a consistent workflow involving orientation alignment, segmentation, normal estimation, and roughness computation at neighborhood radiuses of 0.2, 0.4, and 0.6 model units. The results show that COLMAP provides the highest sensitivity to micro-texture, while Meshroom yields balanced reconstructions with moderate roughness variation. Metashape produces the smoothest geometry due to its internal filtering, and 3DGS captures visible irregularities but exhibits higher noise and lower density. The comparison demonstrates that open-source pipelines are viable for relative roughness evaluation, offering a practical approach for low-cost pavement monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15151v3">Zero-shot CT Super-Resolution using Diffusion-based 2D Projection Priors and Signed 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 MICCAI 2026 early accepted
    </div>
    <details class="paper-abstract">
      Computed tomography (CT) is important in clinical diagnosis, but acquiring high-resolution (HR) CT is constrained by radiation exposure risks. While deep learning-based super-resolution (SR) methods have shown promise for reconstructing HR CT from low-resolution (LR) inputs, supervised approaches require paired datasets that are often unavailable. Zero-shot methods address this limitation by operating on single LR inputs; however, they frequently fail to recover fine structural details due to limited LR information within individual volumes. To overcome these limitations, we propose a novel zero-shot 3D CT SR framework that integrates diffusion-based upsampled 2D projection priors into the 3D reconstruction process. Specifically, our framework consists of two stages: (1) LR CT projection SR, training a diffusion model on abundant X-ray data to upsample LR projections, thereby enhancing the scarce information inherent in the LR inputs. (2) 3D CT volume reconstruction, using 3D Gaussian splatting with our novel Negative Alpha Blending (NAB-GS), which models positive and negative Gaussian densities to learn signed residuals between diffusion-generated HR and upsampled LR projections. Our framework demonstrates superior quantitative and qualitative performance on two public datasets, and expert evaluations present the framework's clinical potential at 4x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22069v2">TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Accepted at CVPR 2026, Project page: https://sandokim.github.io/twings/
    </div>
    <details class="paper-abstract">
      Novel view synthesis from sparse-view inputs poses a significant challenge in 3D computer vision, particularly for achieving high-quality scene reconstructions with limited viewpoints. We introduce TWINGS, a framework that enhances 3D Gaussian Splatting (3DGS) by directly addressing point sparsity. We employ Thin Plate Splines (TPS), a smooth non-rigid deformation model that minimizes bending energy to estimate a globally coherent warp from control-point correspondences, to align backprojected points from estimated depth with triangulated 3D control points, yielding calibrated backprojected points. By sampling these calibrated points near the control points, TWINGS provides a fast and geometrically accurate initialization for 3DGS, ultimately improving structural detail preservation and color fidelity in reconstructed scenes. Extensive experiments on DTU, LLFF, and Mip-NeRF360 demonstrate that TWINGS consistently outperforms existing methods, delivering detailed and accurate reconstructions under sparse-view scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29318v1">FreeForm: Reduced-Order Deformable Simulation from Particle-Based Skinning Eigenmodes</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 CVPR 2026, project website: https://research.nvidia.com/labs/sil/projects/freeform/
    </div>
    <details class="paper-abstract">
      We present a novel formulation for mesh-free, reduced-order simulation of deformable hyperelastic objects. Existing work in reduced-order elastodynamic simulation represents the input geometry by either meshes, which can be difficult to obtain due to challenges in scanning and triangulating complex shapes, or by neural fields that require per-shape optimization. We propose to adopt a Reproducing Kernel Particle Method (RKPM) representation, which enables the construction of reduced-order skinning weights by solving a generalized eigensystem on the Hessian matrix of the elastic energy. We demonstrate that this formulation not only leads to a 40x training speedup compared with the per-shape optimization of neural fields, but also achieves lower simulation error when evaluated against the converged results of finite element method. We show our simulation results on a wide variety of objects in different representations including meshes and Gaussian splats, as well as the application of our method in the downstream task of robot simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09632v3">X-GS: An Extensible Framework for Perceiving and Thinking via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods operate in isolation, focusing on specific domains. In this paper, we introduce X-GS, an extensible framework consisting of two major components. The X-GS-\textit{Perceiver} unifies a broad range of 3DGS techniques to enable real-time online SLAM with semantic distillation. The X-GS-\textit{Thinker} accommodates multimodal models, enabling them to seamlessly interface with the \textit{Perceiver} to complete downstream tasks. In our implementation of X-GS, the \textit{Perceiver} leverages the latest vision foundation models to improve online SLAM performance and employs three key mechanisms to accelerate semantic distillation. The \textit{Thinker} can be built upon both contrastive and generative vision-language models and utilizes the \textit{Perceiver}'s semantic Gaussian splats to unlock capabilities such as 3D visual grounding and scene captioning. Experimental results on diverse benchmarks demonstrate the efficiency and newly unlocked multimodal capabilities of the X-GS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29136v1">Eulerian Gaussian Splatting using Hashed Probability Pyramids</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 CVPR 2026. Project Page: https://euleriansplatting.github.io
    </div>
    <details class="paper-abstract">
      We introduce a probabilistic splat-based radiance field framework that retains the fast rasterization and test-time efficiency of 3D Gaussian Splatting (3DGS) while replacing heuristic primitive manipulation with gradient-based optimization of a volumetric probability density. Rather than relocating, splitting, or culling Gaussians via hand-tuned densification (e.g., ADC), we treat primitive locations as samples drawn from a persistent, learnable density. We instantiate this density using a novel, memory-efficient multi-scale hierarchical grid that enables end-to-end gradient-based optimization. To stabilize the optimization, we derive an unbiased gradient estimator with control variates that markedly reduces variance. By allowing probability mass to flow to where the loss demands, our framework eliminates brittle priors and naturally explores the volume, achieving state-of-the-art reconstruction quality on mip-NeRF 360 while preserving 3DGS-level rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19065v2">A Geometric Algebra-Informed 3DGS Framework for Wireless Channel Prediction</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Geometric Algebra-Informed 3D Gaussian Splatting (GAI-GS), a framework for wireless modeling that couples 3D Gaussian splatting with a geometric algebra-based attention mechanism to explicitly model ray-object interactions in complex propagation environments. GAI-GS encodes joint spatial-electromagnetic (EM) relations into token representations, enabling scene-level aggregation within a unified, end-to-end neural architecture. This design grounds wireless ray propagation in electromagnetic principles, allowing token interactions to model key effects such as multipath, attenuation, and reflection/diffraction. Through extensive evaluations on multiple real-world indoor datasets, GAI-GS consistently surpasses current baselines across various wireless tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.28237v1">POINav: Benchmarking and Enhancing Final-Meters Arrival in Real-World Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2026-05-27
      | 💬 25 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Real-world navigation is fundamentally driven by Points of Interest (POIs), yet reaching a precise POI remains a critical "final-meters" challenge. Existing Vision-Language Navigation (VLN) benchmarks of POI-goal navigation often suffer from coarse granularity or significant sim-to-real gaps due to generated scene. To bridge this gap, we present POINav-Bench, the first benchmark designed for closed-loop evaluation of real-world POI-goal navigation. It comprises 11 commercial areas reconstructed from real-world captures using 3D Gaussian Splatting (3DGS), covering 126,398 $m^{2}$ in total and spanning 163 distinct POIs. With traversability-aware annotations and reference trajectories, POINav-Bench enables high-fidelity evaluation of navigation agents in realistic, POI-rich real-world environments. Building on this, we propose the POINav Brain-Action Framework where a Brain module performs POI-grounded reasoning to guide an Action module in predicting continuous waypoints for real-world execution. We further curate the POINav-Dataset, containing 70K real-world signage-entrance pairs. Experiments show that our framework provides a viable path toward refining real-world POI-goal navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17354v5">PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling</a></div>
    <div class="paper-meta">
      📅 2026-05-27
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) enables real-time rendering, its training demands workstation-level compute and memory, making mobile deployment impractical under minute-scale time budgets and limited peak memory. We present PocketGS, a mobile scene modeling paradigm that enables on-device 3DGS training under these tightly coupled constraints while preserving high-fidelity reconstruction. PocketGS resolves the fundamental tension between training efficiency, memory compactness, and modeling quality through three co-designed operators: $\mathcal{G}$ builds geometry-faithful point-cloud priors; $\mathcal{I}$ injects local surface statistics to seed anisotropic Gaussians, thereby reducing early conditioning gaps; and $\mathcal{T}$ unrolls alpha compositing with cached intermediates and index-mapped gradient scattering for stable mobile backpropagation. Extensive experiments demonstrate that PocketGS outperforms the powerful mainstream workstation 3DGS baseline under mobile budgets, delivering high-quality reconstructions and enabling a fully on-device, practical capture-to-rendering workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20150v2">TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-26
      | 💬 Accepted to ICML 2026 as Spotlight. Website: https://sponge-lab.github.io/TideGS
    </div>
    <details class="paper-abstract">
      Training 3D Gaussian Splatting (3DGS) at billion-primitive scale is fundamentally memory-bound: each Gaussian primitive carries a large attribute vector, and the aggregate parameter table quickly exceeds GPU capacity, limiting prior systems to tens of millions of Gaussians on commodity single-GPU hardware. We observe that 3DGS training is inherently sparse and trajectory-conditioned: each iteration activates only the Gaussians visible from the current camera batch, so GPU memory can serve as a working-set cache rather than a persistent parameter store. Building on this insight, we introduce TideGS, an out-of-core training framework that manages parameters across an SSD-CPU-GPU hierarchy via three synergistic techniques: block-virtualized geometry for SSD-aligned spatial locality, a hierarchical asynchronous pipeline to overlap I/O with computation, and trajectory-adaptive differential streaming that transfers only incremental working-set deltas between iterations. Experiments show that TideGS enables training with over one billion Gaussians on a single 24 GB GPU while achieving the best reconstruction quality among evaluated single-GPU baselines on large-scale scenes, scaling beyond prior out-of-core baselines (e.g., approximately 100M Gaussians) and standard in-memory training (e.g., approximately 11M Gaussians).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15283v2">LuxRemix: Lighting Decomposition and Remixing for Indoor Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-26
      | 💬 CVPR 2026. Project page: https://luxremix.github.io
    </div>
    <details class="paper-abstract">
      We present a novel approach for interactive light editing in indoor scenes from a single multi-view scene capture. Our method leverages a generative image-based light decomposition model that factorizes complex indoor scene illumination into its constituent light sources. This factorization enables independent manipulation of individual light sources, specifically allowing control over their state (on/off), chromaticity, and intensity. We further introduce multi-view lighting harmonization to ensure consistent propagation of the lighting decomposition across all scene views. This is integrated into a relightable 3D Gaussian splatting representation, providing real-time interactive control over the individual light sources. Our results demonstrate highly photorealistic lighting decomposition and relighting outcomes across diverse indoor scenes. We evaluate our method on both synthetic and real-world datasets and provide a quantitative and qualitative comparison to state-of-the-art techniques. For video results and interactive demos, see https://luxremix.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12647v3">LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-26
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Recent 3D Gaussian Splatting (3DGS) methods have demonstrated the feasibility of self-driving scene reconstruction and novel view synthesis. However, most existing methods either rely solely on cameras or use LiDAR only for Gaussian initialization or depth supervision, while the rich scene information contained in point clouds, such as reflectance, and the complementarity between LiDAR and RGB have not been fully exploited, leading to degradation in challenging self-driving scenes, such as those with high ego-motion and complex lighting. To address these issues, we propose a robust and efficient LiDAR-reflectance-guided Salient Gaussian Splatting method (LR-SGS) for self-driving scenes, which introduces a structure-aware Salient Gaussian representation, initialized from geometric and reflectance feature points extracted from LiDAR and refined through a salient transform and improved density control to capture edge and planar structures. Furthermore, we calibrate LiDAR intensity into reflectance and attach it to each Gaussian as a lighting-invariant material channel, jointly aligned with RGB to enforce boundary consistency. Extensive experiments on the Waymo Open Dataset demonstrate that LR-SGS achieves superior reconstruction performance with fewer Gaussians and shorter training time. In particular, on Complex Lighting scenes, our method surpasses OmniRe by 1.18 dB PSNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26880v1">GScomp-QA: A Subjective Dataset for Quality Assessment of Compressed Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-26
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has emerged as an efficient representation for high-quality 3D reconstruction and novel view synthesis. However, its large model size poses challenges for storage and transmission. While several GS compression solutions have been proposed, their perceptual impact remains poorly understood due to the lack of dedicated evaluation datasets. To address this gap, this paper introduces GScomp-QA, a subjective quality assessment dataset for evaluating synthesis quality from compressed GS models. The dataset comprises 331 video stimuli from 13 real-world scenes, covering 9 state-of-the-art GS compression solutions. By using videos synthesized from uncompressed models as reference, GScomp-QA isolates compression-induced distortions from synthesis artifacts. A subjective study with 20 participants was conducted, providing reliable perceptual scores. Based on these data, GS compression solutions are evaluated through perceptual rate-distortion analysis. In addition, 18 objective quality metrics are evaluated, showing that they do not fully capture GS-specific distortions. GScomp-QA will be publicly available and provide a benchmark for evaluating GS compression solutions and supporting the development of quality metrics tailored to GS compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18866v2">FLUIDSPLAT: Reconstructing Physical Fields from Sparse Sensors via Gaussian Primitives</a></div>
    <div class="paper-meta">
      📅 2026-05-26
      | 💬 24 pages, 5 figures,preprint
    </div>
    <details class="paper-abstract">
      Reconstructing continuous flow fields from sparse surface-mounted sensors is central to aerodynamic design, flow control, and digital-twin instrumentation. Existing neural methods for this task typically encode sensor readings into implicit latent codes with little spatial interpretability and limited formal guidance on how representational capacity should scale with observation count. Inspired by 3D Gaussian Splatting, we introduce FLUIDSPLAT, a sensor-conditioned model that predicts K anisotropic Gaussian primitives forming a partition-of-unity scaffold, a spatially explicit and interpretable intermediate representation of the flow. For an idealized Gaussian primitive estimator, we prove an $O(K^{-s/d})$ approximation rate for fields with Sobolev smoothness $s$; incorporating $N$ noisy observations yields a squared-risk decomposition with bias $O(K^{-2s/d})$ and variance $O(σ^{2}K/N)$.Balancing the two yields $K^{*}\!\sim\!(N/σ^{2})^{d/(2s+d)}$: primitive count cannot grow freely under sparse sensing, revealing a variance bottleneck that motivates complementing the scaffold with a state-conditioned residual decoder. Across four benchmarks spanning 2D and 3D, FLUIDSPLAT achieves 11-28% error reduction over several strong baselines on cylinder flow, AirfRANS, FlowBench LDC-3D, and PhySense-Car 3D benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25029v2">ParkingWorld: End-to-End Autonomous Parking Reinforcement Learning from Corrective Experience in 3DGS Simulation</a></div>
    <div class="paper-meta">
      📅 2026-05-26
      | 💬 9 pages(including 1 page of Appendix), 6 figures. Will be submitted to RA-L 2026
    </div>
    <details class="paper-abstract">
      Autonomous parking demands precise low-speed maneuvering within narrow, cluttered, and highly constrained environments, where vehicles must navigate tight spaces while avoiding static obstacles and complex geometric boundaries. Unlike imitation learning, which typically requires massive volumes of high-quality expert demonstrations to converge to a stable policy and often suffers from limited generalization to unseen scenarios, traditional reinforcement learning (RL) methods face persistent challenges including excessive training overhead, inefficient exploration, and even failure to learn viable parking strategies in challenging settings. To address these limitations, this paper presents a correction-in-the-loop sample-efficient reinforcement learning (CIL-SERL) framework for end-to-end autonomous parking, which is entirely trained in a photorealistic 3D Gaussian Splatting (3DGS) parking simulator that enables high-fidelity digital reconstruction of real-world scenes. Inspired by error-correction notebooks used in learning practice, we design a novel multi-level replay buffer mechanism. These buffers hierarchically organize and store standard RL rollouts, human corrective interventions, failed exploration trajectories, and rollback-based correction segments in separate yet interconnected memory regions, facilitating structured sampling and targeted learning during training. The proposed framework is systematically evaluated in both the 3DGS simulation environment and a physical vehicle platform. Extensive experimental results demonstrate that our method achieves substantial improvements in parking success rate, operational efficiency, and safety performance across diverse scenarios, validating the effectiveness and practical applicability of the proposed CIL-SERL-based end-to-end autonomous parking solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26629v1">DelowlightSplat: Feed-Forward Gaussian Splatting for Lowlight 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-26
    </div>
    <details class="paper-abstract">
      Novel-view synthesis and 3D reconstruction from sparse posed images are central to robotics and AR/VR. Yet, feed-forward 3D Gaussian reconstruction fails under lowlight due to noise, color shifts, and unreliable correspondence. We propose DelowlightSplat, a lowlight-aware feed-forward Gaussian splatting framework for clean novel-view rendering. We build a controllable multi-view lowlight benchmark by degrading only context views while keeping target views clean. We introduce a lightweight Lowlight Adapter for residual enhancement to improve matchability, and couple it with cost-volume-based multi-view inference to directly predict clean 3D Gaussians. Experiments show that DelowlightSplat significantly outperforms previous feed-forward method and two-stage pipeline under lowlight conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.00648v2">DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-26
      | 💬 CVPR 2026 Highlight; Fix NSFC ID
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has enabled efficient 3D scene reconstruction from everyday images with real-time, high-fidelity rendering, greatly advancing VR/AR applications. Fisheye cameras, with their wider field of view (FOV), promise high-quality reconstructions from fewer inputs and have recently attracted much attention. However, since 3DGS relies on rasterization, most subsequent works involving fisheye camera inputs first undistort images before training, which introduces two problems: 1) Black borders at image edges cause information loss and negate the fisheye's large FOV advantage; 2) Undistortion's stretch-and-interpolate resampling spreads each pixel's value over a larger area, diluting detail density -- causes 3DGS overfitting these low-frequency zones, producing blur and floating artifacts. In this work, we integrate fisheye camera model into the original 3DGS framework, enabling native fisheye image input for training without preprocessing. Despite correct modeling, we observed that the reconstructed scenes still exhibit floaters at image edges: Distortion increases toward the periphery, and 3DGS's original per-iteration random-selecting-view optimization ignores the cross-view correlations of a Gaussian, leading to extreme shapes (e.g., oversized or elongated) that degrade reconstruction quality. To address this, we introduce a feature-overlap-driven cross-view joint optimization strategy that establishes consistent geometric and photometric constraints across views-a technique equally applicable to existing pinhole-camera-based pipelines. Our DirectFisheye-GS matches or surpasses state-of-the-art performance on public datasets. Project Page: https://yzxqh.github.io/DirectFisheye-GS/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26616v1">Gaussian-Voxel Duet: A Dual-Scaffolding Hybrid Representation for Fast and Accurate Monocular Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-26
      | 💬 27 pages, 14 figures
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting has achieved remarkable success in photorealistic novel view synthesis, its pursuit of fast and high-fidelity 3D reconstruction has long been constrained by a trade-off between geometric accuracy and optimization efficiency. Methods specialized in image rendering converge quickly at the cost of imperfect geometry caused by superfluous primitives overfitting training views, while methods integrating neural signed-distance field (SDF) for better geometry incur prohibitive training costs. In this paper, we attempt to strike a better trade-off by tethering scaffold-anchored Gaussians to a jointly optimized sparse voxel scaffold. This hybrid Gaussian-Voxel representation explicitly confines anchored Gaussians to a narrow band around surfaces defined by voxelized SDFs, which effectively improves representation efficiency and condenses floating Gaussians without sacrificing geometry quality. An implicit surface tethering loss further pulls individual Gaussian primitives closer to SDF-induced surfaces in a mutually regularized manner for improved reconstruction accuracy. Extensive experiments on diverse real-world indoor scenes from ScanNet++, ScanNetv2, and DeepBlending datasets demonstrate that our method achieves state-of-the-art surface reconstruction quality as well as superior novel view synthesis against leading baselines, while maintaining fast training convergence and real-time rendering. Code will be available at https://github.com/duzh11/VoxelGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26576v1">TrackRef3D: Multi-View Consistent Track-then-Label for Open-World Referring Segmentation in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-26
    </div>
    <details class="paper-abstract">
      Referring 3D Gaussian Splatting (R3DGS), which utilizes natural language for 3D object segmentation, has emerged as a crucial capability for embodied AI. However, existing methods typically rely on expensive per-scene manual annotation and per-view pseudo mask generation, which suffer from multi-view inconsistency and poor generalization to varying query specificities. To address this, we present TrackRef3D, a fully automatic pipeline that achieves open-world referring segmentation in 3D Gaussian Splatting (3DGS) without manual annotation by introducing a multi-view consistent track-then-label paradigm that fundamentally decouples object discovery from semantic grounding. Specifically, we propose a Trajectory-Aware Semantic Consensus Module (TSCM) which aggregates cross-view predictions via synonymous clustering and trajectory-aware voting to establish a canonical semantic identity, thereby ensuring multi-view consistency. Furthermore, we employ a visibility-aware description generation strategy to mitigate ambiguity and propose a Hybrid Training Strategy (HTS) that jointly optimizes coarse category semantics and fine-grained referential cues to ensure robustness under varying query specificities using a multi-positive contrastive objective. Extensive experiments on benchmarks demonstrate that TrackRef3D achieves state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26447v1">Underwater360: Reconstructing Underwater Scenes from Panoramic Images with Omnidirectional Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-26
    </div>
    <details class="paper-abstract">
      Underwater scene reconstruction is essential for immersive exploration of aquatic environments, yet remains challenging due to complex participating-media effects such as absorption and scattering, as well as the limited field of view (FoV) of conventional cameras. Although combining panoramic imaging with 3D Gaussian Splatting (3DGS) offers a promising direction for photorealistic underwater rendering, traditional 3DGS struggles with both spherical projection distortion and underwater medium degradation. In this paper, we propose \textbf{Underwater360}, a physics-informed omnidirectional 3DGS framework for underwater panoramic scene reconstruction. First, we introduce an Omnidirectional Gaussian Splatting module that performs ray casting directly in spherical camera space instead of relying on 2D projection approximations, thereby reducing geometric distortions under 360$^\circ$ FoV. Second, we design a physics-based appearance-medium modeling architecture with pose-conditioned appearance embeddings to explicitly decouple intrinsic scene radiance from depth-dependent backscatter and attenuation, enabling physically grounded scene appearance restoration. Finally, we establish a new panoramic underwater benchmark dataset containing both synthetic and real-world scenes. Extensive experiments demonstrate that Underwater360 achieves superior performance in underwater novel view synthesis and scene appearance restoration, delivering improved rendering quality and cross-view consistency in complex underwater environments. The code and datasets are released at https://github.com/SwcK423/Underwater360
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25909v1">R5DGS: Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints for Efficient Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-25
      | 💬 Code: https://github.com/be2rlab/r5dgs
    </div>
    <details class="paper-abstract">
      Reconstructing and predicting dynamic 3D scenes from multi-view videos is a foundational task for robotics, AR/VR, and digital twins. Recent physics-informed Gaussian Splatting methods achieve impressive future frame extrapolation but lack semantic awareness and suffer from large computational overhead. We introduce $\textbf{R5DGS}$, a framework that augments a physics-driven 4D Gaussian representation with compact Identity Encoding vectors, enabling precise Gaussian-to-object association. By constructing an offline CLIP-based object lookup table, we support open-vocabulary text prompting to retrieve and render object-specific Gaussians across arbitrary timestamps and viewpoints. Furthermore, we propose a rigid-body inference constraint that predicts and integrates physical dynamics exclusively for object centroids, propagating motion to associated Gaussians via relative transformations. This optimization yields a 11 FPS speedup during extrapolation without compromising trajectories plausibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25751v1">SplitAvatar: One-shot Head Avatar with Autoregressive Gaussian Splitting</a></div>
    <div class="paper-meta">
      📅 2026-05-25
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) provides an efficient method for high-quality scene reconstruction using anisotropic Gaussians. Recently, 3DGS-based methods have significantly improved the rendering quality of human avatars while enabling real-time performance. However, existing methods suffer from a magnitude mismatch in the number of Gaussians generated by image-based and 3DMM-based approaches. This discrepancy results in reconstructed expressions that lack fine-grained detail. In this paper, we introduce a novel method for reconstructing an animatable head avatar from a single image. We propose a Graph splitting network to progressively generate Gaussians from coarse to fine using an autoregressive architecture. To address the graph inconsistency caused by split Gaussians, we employ a mesh topology extension method to align the GNN's connectivity with the increased Gaussian count. Furthermore, we introduce a novel density control method that includes a gating mechanism that generates soft masks for Gaussians, preventing over-densification after the splitting operation. This allows for dynamic control over Gaussian density across different facial regions. For smooth and rapid training, we employ a delayed filtering strategy to avoid re-computing the graph topology during training. Experimental results demonstrate that our autoregressive structure effectively improves expression representation ability by progressively splitting Gaussians. This process, enabled by the GNN-guided splitting, synthesizes more precise facial details and achieves higher reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25563v1">CodecSplat: Ultra-Compact Latent Coding for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-25
    </div>
    <details class="paper-abstract">
      While feed-forward 3D Gaussian splatting reconstructs renderable Gaussian primitives from sparse context views without per-scene optimization, existing pipelines do not provide a compact scene representation for storage or transmission. A natural solution is to apply existing 3DGS compression methods to the generated Gaussian primitives. However, this approach operates on the final irregular 3D representation and is decoupled from the internal feature-to-Gaussian generation process, which limits compression efficiency. To address this, we introduce CodecSplat, an ultra-compact latent coding framework for feed-forward 3D Gaussian splatting. CodecSplat first encodes an intermediate 2D Gaussian-generation feature into an entropy-coded scene bitstream. At the decoder, the latent feature is reconstructed and used to predict depth and Gaussian parameters, which are then mapped to 3D Gaussian primitives. Note that, by integrating compression into the feed-forward Gaussian generation pipeline, CodecSplat avoids inefficient compression over irregular 3D Gaussian primitives and allows the codec to exploit the structured intermediate feature representation. We instantiate CodecSplat on a feed-forward Gaussian splatting backbone with depth-guided multi-view feature refinement and a hierarchical learned feature codec. On DL3DV and RealEstate10K datasets, CodecSplat achieves 23.56-26.36 dB and 24.76-27.05 dB PSNR with only 20.00-107.77 KiB and 3.37-12.51 KiB per scene, respectively. This is roughly one order of magnitude smaller than compressing feed-forward generated Gaussian primitives, while preserving controllable rate-distortion behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25373v1">Physics-Aware 3D Gaussian Editing for Driving Scene Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-25
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown great potential in autonomous driving simulation and data generation, enabling photorealistic reconstruction and flexible scene manipulation. However, existing 3DGS scene editing methods have limited support for road geometry editing (e.g., inserting speed humps or sunken roads), and generally do not couple such edits with plausible vehicle-road interaction dynamics. Such editing is essential for generating training data under extreme driving scenarios or evaluating system reliability under these road irregularities. Moreover, many optimization-based methods require minutes of per-edit refinement, while existing efficient alternatives mainly focus on appearance-level or object-level manipulation rather than physics-aware road irregularity editing. To address these limitations, we propose RoVES, a Road-and-Vehicle Editing System for physics-aware 3D Gaussian editing in driving scenes. RoVES enables single-image-driven road geometry insertion and couples the edited road profile with a 4-DOF half-car vehicle dynamics model to achieve physics-aware vehicle pose correction in vertical displacement and pitch. RoVES inserts road elements in a one-shot, optimization-free pipeline (1.84s), and the full pipeline (including color transfer and vehicle-dynamics-based pose correction) completes in 6.24s; it edits dynamic vehicles via pose editing and corrects poses frame-by-frame to approximate dynamics-consistent vertical displacement and pitch responses. Experiments on the Waymo dataset show that RoVES provides practical efficiency and competitive visual consistency for physics-aware driving scene generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.25345v1">Depth Peeling for High-Fidelity Gaussian-Enhanced Surfel Rendering</a></div>
    <div class="paper-meta">
      📅 2026-05-25
    </div>
    <details class="paper-abstract">
      Novel view synthesis has been significantly advanced by NeRFs and 3D Gaussian Splatting (3DGS), which require ordering volumetric samples or primitives for correct color blending. While the recent Gaussian-Enhanced Surfels (GES) enable high-performance, sort-free rendering, they suffer from aliasing artifacts and suboptimal reconstruction. To address these limitations, we propose DP-GES, a novel representation that augments opaque surfels with semi-transparent boundaries and leverages Depth Peeling to establish accurate per-pixel ordering. This design enables sort-free Gaussian splatting with correct transmittance modulation, effectively eliminating aliasing and popping artifacts while facilitating a fully differentiable joint optimization. Extensive experiments demonstrate that our method achieves superior reconstruction quality and compares favorably against state-of-the-art techniques across a wide range of scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.24964v1">ConFi-GS Confidence-Guided High-Frequency Injection for 3D Gaussian Splatting Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2026-05-24
    </div>
    <details class="paper-abstract">
      Reconstructing high-quality 3D scenes from low-resolution multi-view images remains challenging for 3D Gaussian Splatting (3DGS), because insufficient high-frequency observations often lead to blurred textures, weak boundaries, and view-inconsistent details. Existing approaches either apply super-resolution guidance uniformly or localize enhancement regions based mainly on geometric sampling. However, they typically do not distinguish between two fundamentally different questions: where additional detail is needed, and whether the corresponding candidate high-frequency content is reliable enough to be internalized into a multi-view consistent 3D representation. In this paper, we propose a reliability-aware frequency modeling framework for low-resolution 3DGS reconstruction. The framework first estimates a geometry-guided detail-demand prior to locate regions that are likely under-detailed under low-resolution supervision. It then computes a frequency-aware reliability map to determine whether candidate high-frequency details are structurally supported, spectrally unresolved, and cross-view stable. Combining these signals yields a detail-injection map that guides where super-resolved details should be introduced during optimization. Based on this map, we design a unified optimization scheme comprising spatially selective supervision, coarse-to-fine frequency regularization, and reliability-aware Gaussian densification. This scheme controls where reliable details are injected, when high-frequency supervision is activated, and how unresolved yet reliable details are internalized into the Gaussian representation. Experiments on multiple benchmarks show improved fidelity and perceptual quality while suppressing unstable or view-inconsistent details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22973v2">Scaling Up Occupancy-centric Driving Scene Generation: Dataset and Method</a></div>
    <div class="paper-meta">
      📅 2026-05-23
      | 💬 IEEE TPAMI
    </div>
    <details class="paper-abstract">
      Driving scene generation is a critical domain for autonomous driving, enabling downstream applications, including perception and planning evaluation. Occupancy-centric methods have recently achieved state-of-the-art results by offering consistent conditioning across frames and modalities; however, their performance heavily depends on annotated occupancy data, which still remains scarce. To overcome this limitation, we curate Nuplan-Occ, the largest semantic occupancy dataset to date, constructed from the widely used Nuplan benchmark. Its scale and diversity facilitate not only large-scale generative modeling but also autonomous driving downstream applications. Based on this dataset, we develop a unified framework that jointly synthesizes high-quality semantic occupancy, multi-view videos, and LiDAR point clouds. Our approach incorporates a spatio-temporal disentangled architecture to support high-fidelity spatial expansion and temporal forecasting of 4D dynamic occupancy. To bridge modal gaps, we further propose two novel techniques: a Gaussian splatting-based sparse point map rendering strategy that enhances multi-view video generation, and a sensor-aware embedding strategy that explicitly models LiDAR sensor properties for realistic multi-LiDAR simulation. Extensive experiments demonstrate that our method achieves superior generation fidelity and scalability compared to existing approaches, and validates its practical value in downstream tasks. Repo: https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/tree/v2
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.24304v1">ArtSplat: Feed-Forward Articulated 3D Gaussian Splatting from Sparse Multi-State Uncalibrated Views</a></div>
    <div class="paper-meta">
      📅 2026-05-23
    </div>
    <details class="paper-abstract">
      Articulated object reconstruction from sparse-view images is an ill-posed problem that requires simultaneous inference of geometry and underlying articulation structure. Existing methods for articulated object reconstruction based on NeRF and 3D Gaussian Splatting (3DGS) typically rely on dense views or strong priors (e.g., depth maps, joint types, predefined number of joints) and require costly per-object optimization. In this paper, we propose ArtSplat, the first feed-forward framework for articulated 3D Gaussian Splatting. It reconstructs both geometry and joint parameters from sparse multi-view images across multiple articulation states in a single forward pass. To address the challenges of single-pass articulated reconstruction, we introduce a per-pixel joint map representation that enables the integration of joint parameter estimation into the feed-forward pipeline. We further propose a Cross-State Attention (CSA) mechanism with state tokens, which effectively captures discrete motion across input states. Experiments on 68 articulated objects from PartNet-Mobility, including both single- and multi-joint configurations, demonstrate that ArtSplat achieves competitive performance in both geometry and joint estimation, while being over 400 times faster than baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.24290v1">RxGS: Receiver-Generalizable 3D Gaussian Splatting for Radio-Frequency Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-22
    </div>
    <details class="paper-abstract">
      Radio-frequency (RF) data synthesis predicts the received signal given transmitter and receiver positions, and is essential for wireless applications. Recent 3D Gaussian Splatting (3DGS)-based methods achieve efficient synthesis at any transmitter but only for a fixed receiver. Therefore, supporting $N$ receivers in one scene requires $N$ independent models and precludes prediction at unseen receivers. We present RxGS, which achieves receiver-generalizable synthesis within a single unified model. Our key insight is that scene geometry is receiver-independent while directional radiance is not: a first stage learns shared 3D Gaussian geometry, and a second stage freezes it and learns directional radiance conditioned on receiver position. A global conditioning branch captures shared receiver-dependent effects across the scene, while a local branch models per-scatterer variations from the receiver's geometry and occlusion. A multi-receiver CUDA rasterizer further batches rendering across all $N$ receivers. Evaluated across various RF datasets, RxGS matches or improves over per-receiver baselines with a single shared model and generalizes to receivers unseen during training within the scene, cutting training cost by up to $45\times$, inference cost by $7.6\times$, and storage by $N\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14180v2">Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere</a></div>
    <div class="paper-meta">
      📅 2026-05-22
    </div>
    <details class="paper-abstract">
      Radiance field methods (e.g. 3D Gaussian Splatting) have emerged as a powerful paradigm for novel view synthesis, yet their appearance modeling often relies on Spherical Harmonics (SH), which impose fundamental limitations. SH struggle with high-frequency signals, exhibit Gibbs ringing artifacts, and fail to capture specular reflections - a key component of realistic rendering. Although alternatives like spherical Gaussians offer improvements, they add significant optimization complexity. We propose Spherical Voronoi (SV) as a unified framework for appearance representation in 3D Gaussian Splatting. SV partitions the directional domain into learnable regions with smooth boundaries, providing an intuitive and stable parameterization for view-dependent effects. For diffuse appearance, SV achieves competitive results while keeping optimization simpler than existing alternatives. For reflections - where SH fail - we leverage SV as learnable reflection probes, taking reflected directions as input following principles from classical graphics. This formulation attains state-of-the-art results on synthetic and real-world datasets, demonstrating that SV offers a principled, efficient, and general solution for appearance modeling in explicit 3D representations. Project page: https://sphericalvoronoi.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22809v2">Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-05-22
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Robust training and validation of Autonomous Driving Systems (ADS) require massive, diverse datasets. Proprietary data collected by Autonomous Vehicle (AV) fleets, while high-fidelity, are limited in scale, diversity of sensor configurations, as well as geographic and long-tail-behavioral coverage. In contrast, in-the-wild data from sources like dashcams offers immense scale and diversity, capturing critical long-tail scenarios and novel environments. However, this unstructured, in-the-wild video data is incompatible with ADS expecting structured, multi-modal sensor inputs for validation and training. To bridge this data gap, we propose Sensor2Sensor, a novel generative modeling paradigm that translates in-the-wild monocular dashcam videos into a high-fidelity, multi-modal sensor suite (AV logs) comprising multi-view camera images and LiDAR point clouds. A core challenge is the lack of paired training data. We address this by converting real AV logs into dashcam-style videos via 4D Gaussian Splatting (4DGS) reconstruction and novel-view rendering. Sensor2Sensor then utilizes a diffusion architecture to perform the generative conversion. We perform comprehensive quantitative evaluations on the fidelity and realism of the generated sensor data. We demonstrate Sensor2Sensor's practical utility by converting challenging in-the-wild internet and dashcam footage into realistic, multi-modal data formats, further unlocking vast external data sources for AV development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.24114v1">COSY: Compositional 3DGS Synthesis for Disentangled Human Head Editing</a></div>
    <div class="paper-meta">
      📅 2026-05-22
    </div>
    <details class="paper-abstract">
      Recent 3D Gaussian Splatting (3DGS) GANs for human heads synthesize and render photorealistic 3D models in real-time and offer a vast variety in identity and appearance. However, controlling specific semantic attributes such as hair color or glasses remains challenging, as edits in the entangled latent space often induce unintended changes in identity or appearance. Although there are several methods that aim to disentangle the latent space post training by estimating directions that only modify certain features, these methods cannot guarantee complete disentanglement and often require pre-trained classifiers. In our approach, we propose a new generator architecture that synthesizes components, such as hair, skin, glasses, and torso, completely independently. This allows for changing the latent vector for one region while keeping the remaining parts fixed. Further, we achieve this separation using only sparse information such as the hair or skin color, eliminating the requirement of segmentation masks or geometric priors, often seen in prior work. To ensure matching shape and lighting conditions during editing, we allow minimal shared information via context tokens between the independent generators. These tokens even allow us to control the shape and light, without any prior annotation. Compared to existing works on GAN-based generation and editing, our method shows better disentanglement, more precise editing control, and competitive visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23845v1">Learning a Particle Dynamics Model with Real-world Videos</a></div>
    <div class="paper-meta">
      📅 2026-05-22
      | 💬 CVPR 2026 Findings
    </div>
    <details class="paper-abstract">
      Data-driven learning approaches for physics simulation, sometimes referred to as world models, have emerged as promising alternatives to traditional physics simulators due to their differentiable nature. Prior work has demonstrated impressive results in predicting the motions of rigid and non-rigid objects in complex scenes involving multiple interacting bodies. However, these models are typically trained in simulated environments because obtaining perfect state information such as complete scene point clouds and point correspondences over time is challenging in real-world settings. This reliance on synthetic data can limit their applicability when the sim-to-real gap is large. In this work, we aim to overcome these limitations by introducing a novel framework for training neural object dynamics models directly from unlabeled real-world videos. Specifically, we propose to learn a particle-based dynamics model compatible with a Gaussian splatting framework, which operates on dense particles derived from Gaussians (i.e., particles with scales and rotations) and predicts their position and rotation changes over time. The model is trained via rendering supervision, enabling learning from real-world videos without requiring particle-level labeled states. Our model operates directly on dense Gaussians without relying on heuristic subsampling anchor points. To enable this study, we also present a real-world dataset consisting of about 500 videos capturing diverse object interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14135v5">GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-05-22
      | 💬 https://ChaiYing1.github.io/projects/GAF/
    </div>
    <details class="paper-abstract">
      Accurate scene perception is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we adopt a V-4D-A framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing 4D modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF provides three interrelated outputs: reconstruction of the current scene, prediction of future frames, and estimation of init action via Gaussian motion. Furthermore, we employ an action-vision-aligned denoising framework, conditioned on a unified representation that combines the init action and the Gaussian perception, both generated by the GAF, to further obtain more precise actions. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR, +0.3864 SSIM and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average +7.3% success rate in robotic manipulation tasks over state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23672v1">RiGS: Rigid-aware 4D Gaussian Splatting from a Single Monocular Video</a></div>
    <div class="paper-meta">
      📅 2026-05-22
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular videos is a fundamental yet highly challenging task, as real-world motions often involve both long-term smooth transformations and short-term complex deformations. Existing methods either struggle to maintain temporal consistency or fail to capture high-frequency dynamics due to limited motion modeling capacity. In this work, we present Rigid-aware 4D Gaussian Splatting (RiGS), which simultaneously captures motions across multiple temporal scales. Specifically, RiGS introduces three types of Gaussian primitives: static, rigid, and transient, which represent static backgrounds, long-term low-frequency motions, and short-term high-frequency dynamics, respectively. An object-wise dynamic mask is proposed to aggregate long-range spatiotemporal motion information and guide the decomposition of static and dynamic regions. To jointly model motion across scales, rigid Gaussians are allowed to transition into transient Gaussians based on their temporal duration, and both are optimized under scene flow guidance, providing dense 3D motion supervision. Extensive experiments demonstrate that RiGS achieves state-of-the-art performance on novel view synthesis benchmarks. Code is available at \hyperlink{https://github.com/ladvu/RiGS}{https://github.com/ladvu/RiGS}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23602v1">GlowGS: Generative Semantic Feature Learning for 3D Gaussian Splatting in Nighttime Glow Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-22
      | 💬 Accepted by CVPR Findings 2026
    </div>
    <details class="paper-abstract">
      Existing 3DGS methods effectively render high-quality novel views in clear-day scenes. However, they struggle with night scenes, particularly in glow regions, due to the lack of structural features such as textures and edges, which are key cues for splatting-based reconstruction. To address this problem, we leverage a diffusion model and a Vision Foundation Model (VFM) to compensate for missing structural cues. Our method consists of two key novel ideas: semantic feature generation and novel-view semantic learning. First, semantic feature generation produces high-quality semantic features as implicit structural cues for novel views. Specifically, a diffusion model synthesizes novel views with unknown camera poses from training views, while a VFM evaluates their quality. Once high-quality novel views are identified, the VFM extracts robust features to construct the semantic feature bank. Second, novel-view semantic learning enables 3DGS to optimize rendered novel views without requiring ground truth. It achieves this by extracting semantic features from a rendered novel view, searching the feature bank for the most similar features, and minimizing their distance. This process enforces implicit structural constraints, ensuring semantically coherent, artifact-free rendered views. Extensive experiments demonstrate the effectiveness of our GlowGS in generating semantically accurate 3D views, showing significant improvements over existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06088v3">OpenGaFF: Open-Vocabulary Gaussian Feature Field with Codebook Attention</a></div>
    <div class="paper-meta">
      📅 2026-05-22
    </div>
    <details class="paper-abstract">
      Understanding open-vocabulary 3D scenes with Gaussian-based representations remains challenging due to fragmented and spatially inconsistent semantic predictions across multi-view observations. In this paper, we present OpenGaFF, a novel framework for open-vocabulary 3D scene understanding built upon 3D Gaussian Splatting. At the core of our method is a Gaussian Feature Field that models semantics as a continuous function of Gaussian geometry and appearance. By explicitly conditioning semantic predictions on geometric structure, this formulation strengthens the coupling between geometry and semantics, leading to improved spatial coherence across similar structures in 3D space. To further enforce object-level semantic consistency, we introduce a structured codebook that serves as a set of shared semantic primitives. Furthermore, a codebook-guided attention mechanism is proposed to retrieve language features via similarity matching between query embeddings and learned codebook entries, enabling robust open-vocabulary reasoning while reducing intra-object feature variance. Extensive experiments on standard 2D and 3D open-vocabulary benchmarks demonstrate that our method consistently outperforms prior approaches, achieving improved segmentation quality, stronger 3D semantic consistency and a semantically interpretable codebook that provides insight into the learned representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23287v1">LangFlash: Feed-forward 3D Language Gaussian Splatting from Sparse Unposed Images</a></div>
    <div class="paper-meta">
      📅 2026-05-22
      | 💬 CVPRF 2026
    </div>
    <details class="paper-abstract">
      We present LangFlash, a feed-forward framework for 3D Language Gaussian Splatting that reconstructs 3D scenes parameterized by Gaussian primitives enriched with language-aligned semantic features from sparse unposed multi-view images. Unlike optimization-based 3D methods, LangFlash directly predicts the geometry and semantics in a single forward pass, enabling low-latency 3D reconstruction and language-consistent scene understanding. To support large-scale training, we enriched the RealEstate10k dataset with coherent and dense semantic information for 3D semantic supervision. Furthermore, we propose a sparse semantic encoding scheme that combines a global semantic dictionary with locally varying per-primitive weights, preserving high-level linguistic information, while reducing representation complexity. Experimental results show that LangFlash achieves superior novel view synthesis and semantic consistency compared with previous methods. This study establishes a new paradigm for pose-free, language-grounded 3D scene reconstruction, advancing generalizable 3D vision and multimodal scene understanding. Demo is available at https://liylo.github.io/langflash.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22020v2">ForeSplat: Optimization-Aware Foresight for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-22
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting models offer fast single-pass reconstruction,but scaling them to match per-scene optimization quality is fundamentally hindered by the scarcity of large-scale 3D annotations. A practical compromise is predict-then-refine,where post-prediction optimization compensates for the limited capacity of the feed-forward network. However,standard feed-forward 3DGS is trained solely for zero-step rendering error,ignoring whether its output constitutes a good initialization for the downstream optimizer. We present ForeSplat,an optimization-aware training framework that equips feed-forward 3DGS models to produce initializations explicitly designed for rapid,effective refinement. By offloading part of the scene-modeling burden to the optimizer,ForeSplat substantially reduces the capacity pressure on the feed-forward model,making high-quality reconstruction feasible even with compact networks. At its core is MetaGrad,a lightweight multi-anchor meta-gradient training rule that bypasses costly higher-order differentiation through the 3DGS optimizer. MetaGrad unrolls a short inner-loop refinement trajectory,samples anchor states,and back-propagates aggregated first-order gradients to the prediction head as a surrogate optimization-aware signal. This fine-tuning adds no inference cost and enables high-quality reconstruction within seconds after a few refinement steps. We instantiate ForeSplat on diverse backbones,including AnySplat,Pi3X,and a distilled variant tailored for edge deployment. Across all tested architectures,a ForeSplat-trained initialization converges in fewer refinement steps and reaches a higher peak reconstruction quality than its vanilla counterpart,even fully converged. The framework consistently bridges the gap between amortized prediction and per-scene optimization,establishing a practical path toward lightweight,high-fidelity 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22809v1">Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      Robust training and validation of Autonomous Driving Systems (ADS) require massive, diverse datasets. Proprietary data collected by Autonomous Vehicle (AV) fleets, while high-fidelity, are limited in scale, diversity of sensor configurations, as well as geographic and long-tail-behavioral coverage. In contrast, in-the-wild data from sources like dashcams offers immense scale and diversity, capturing critical long-tail scenarios and novel environments. However, this unstructured, in-the-wild video data is incompatible with ADS expecting structured, multi-modal sensor inputs for validation and training. To bridge this data gap, we propose Sensor2Sensor, a novel generative modeling paradigm that translates in-the-wild monocular dashcam videos into a high-fidelity, multi-modal sensor suite (AV logs) comprising multi-view camera images and LiDAR point clouds. A core challenge is the lack of paired training data. We address this by converting real AV logs into dashcam-style videos via 4D Gaussian Splatting (4DGS) reconstruction and novel-view rendering. Sensor2Sensor then utilizes a diffusion architecture to perform the generative conversion. We perform comprehensive quantitative evaluations on the fidelity and realism of the generated sensor data. We demonstrate Sensor2Sensor's practical utility by converting challenging in-the-wild internet and dashcam footage into realistic, multi-modal data formats, further unlocking vast external data sources for AV development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02784v2">HumanSplatHMR: Closing the Loop Between Human Mesh Recovery and Gaussian Splatting Avatar</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Project page: https://scottyehengz.github.io/HumanSplat/
    </div>
    <details class="paper-abstract">
      Accurately recovering human pose and appearance from video is an essential component of scene reconstruction, with applications to motion capture, motion prediction, virtual reality, and digital twinning. Despite significant interest in building realistic human avatars from video, this paper demonstrates that existing methods do not accurately recover the 3D geometry of humans. ViT-based approaches are not consistently reliable and can overfit to 2D views, while NeRF- and Gaussian Splatting-based avatars treat pose and appearance separately, limiting rendering generalization to new poses. To resolve these shortcomings, this paper proposes HumanSplatHMR, a joint optimization framework that refines 3D human poses while simultaneously learning a high-fidelity avatar for novel-view and novel-pose synthesis. Our key insight is to close the loop between geometric pose estimation and differentiable rendering. Unlike prior human avatar methods that rely on accurate human pose obtained through motion capture systems or offline refinement, which are impractical in in-the-wild scenarios, our approach uses only human mesh estimates from a state-of-the-art human pose estimator to better reflect real-world conditions. Therefore, instead of using the human pose only as a deformation prior, HumanSplatHMR backpropagates photometric, segmentation, and depth losses through a differentiable renderer to the pose parameters and global position. This coupling refines the global 3D pose over time, improving accuracy and alignment while producing better renderings from novel views. Experiments show consistent improvements over pose recovery baselines that omit image-level refinement and avatar baselines that decouple pose estimation from avatar reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22536v1">SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have made rapid progress in spatial intelligence, yet existing spatial reasoning benchmarks largely assume pristine visual inputs and overlook the degradations that commonly occur in real-world deployment, such as motion blur, low light, adverse weather, lens distortion, and compression artifacts. This raises a fundamental question: how robust is the spatial intelligence of current MLLMs when visual observations are imperfect? To answer this question, we introduce SpaceDG, the first large-scale dataset for degradation-aware spatial understanding. It is constructed with a physically grounded degradation synthesis engine that embeds degradation formation process into 3D Gaussian Splatting (3DGS) rendering, enabling realistic simulation of nine degradation types. The resulting dataset contains approximately 1M QA pairs from nearly 1,000 indoor scenes. We further introduce SpaceDG-Bench, an human-verified benchmark with 1,102 questions spanning 11 reasoning categories and 9 visual degradation types, yielding over 10K VQA instances. Evaluating 25 open- and closed-source MLLMs reveals that visual degradations consistently and substantially impair spatial reasoning, exposing a critical robustness gap. Finally, we show that finetuning on SpaceDG markedly improves degradation robustness and can even surpass human performance under degraded conditions without any performance drop on clean images, highlighting the promise of degradation-aware training for robust spatial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22342v1">4D-GSW: Kinematic-Aware Spatio-Temporal Consistent Watermarking for 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 9 pages main paper, 7 figures, 18 pages in total
    </div>
    <details class="paper-abstract">
      While 4D Gaussian Splatting (4DGS) has revolutionized high-fidelity dynamic reconstruction, safeguarding the intellectual property of these assets remains an open challenge. Conventional steganographic techniques often neglect the underlying kinematic manifolds, triggering non-physical artifacts such as severe temporal flickering and "FVD collapse". To address this, we propose \textbf{4D-GSW}, a kinematic-aware watermarking framework designed to embed robust copyright information while preserving high spatio-temporal consistency. Unlike prior 4D steganography that primarily focuses on opacity-guided invisibility, our approach explicitly addresses the physical coherence of motion trajectories. We introduce a \textbf{Spatio-Temporal Curvature (STC)} metric to identify "Dynamic Instants," adaptively gating watermark gradient injection to shield critical motion manifolds from non-physical perturbations. To ensure global coherence across complex deformations, we formulate a joint \textbf{HMM-MRF energy minimization} model that synchronizes watermark phases within both temporal trajectories and spatial neighborhoods. Furthermore, an \textbf{anisotropic gradient routing} mechanism ensures that watermark embedding remains strictly decoupled from photometric reconstruction fidelity. Extensive experiments have demonstrated the superior performance of our method in robustly hiding watermarks while resisting various attacks and maintaining high rendering quality and spatiotemporal consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22190v1">No Pose, No Problem in 4D: Feed-Forward Dynamic Gaussians from Unposed Multi-View Videos</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 https://bralani.github.io/nopo4d_html/
    </div>
    <details class="paper-abstract">
      Recent feed-forward 3D gaussian splatting methods have made dramatic progress on individual aspects of 3D scene reconstruction, but no existing method jointly addresses dynamic content, multi-view input, and unknown camera poses in a single feed-forward pass. Methods that handle dynamics either require accurate camera poses or accept only monocular input; pose-free multi-view methods address only static scenes; and per-scene optimization methods bridge some of these gaps but at minutes-to-hours cost per scene. We introduce NoPo4D, the first feed-forward system that addresses this empty quadrant. Building on a pretrained geometry backbone and recent 4D Gaussian frameworks, NoPo4D introduces a velocity decomposition that splits Gaussian motion into per-pixel image-plane shifts and depth changes, allowing direct supervision from pseudo ground-truth optical flow on the 2D component. This sidesteps both the differentiable rendering that couples prior posed methods to pose accuracy and the 3D motion ground truth that prior pose-free methods require. The system is rounded out by a bidirectional motion encoder for cross-view and cross-frame feature aggregation, and view-dependent opacity that mitigates cross-view and cross-timestep Gaussian misalignments. On four multi-view dynamic benchmarks, NoPo4D consistently outperforms prior feed-forward baselines, and with an optional post-optimization stage surpasses per-scene optimization methods, while running orders of magnitude faster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22147v1">Flow-based Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      High-resolution remote sensing images (RSIs) are crucial for Earth observation applications, yet acquiring them is often limited by sensor constraints and costs. In recent years, generative super-resolution (SR) methods, particularly diffusion models, have made significant progress. However, they typically require slow iterative inference with 40--1000 steps and exhibit limited flexibility in continuous-scale SR settings. To address these issues, we propose FlowGS, a generative reconstruction framework for arbitrary-scale SR of RSIs. FlowGS models the high-frequency detail representations between high- and low-resolution images and learns a continuous probability flow from noise to detail priors via flow matching (FM) constrained by shortcut consistency, thereby reducing generative complexity and improving inference efficiency. Additionally, we employ 2D Gaussian splatting to construct a continuous feature field, thereby enabling flexible reconstruction at arbitrary query locations. Experimental results show that FlowGS delivers competitive perceptual quality compared with existing methods in both continuous-scale and fixed-scale SR settings, with substantially improved inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22069v1">TWINGS: Thin Plate Splines Warp-aligned Initialization for Sparse-View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted to CVPR 2025, Project page: https://sandokim.github.io/twings/
    </div>
    <details class="paper-abstract">
      Novel view synthesis from sparse-view inputs poses a significant challenge in 3D computer vision, particularly for achieving high-quality scene reconstructions with limited viewpoints. We introduce TWINGS, a framework that enhances 3D Gaussian Splatting (3DGS) by directly addressing point sparsity. We employ Thin Plate Splines (TPS), a smooth non-rigid deformation model that minimizes bending energy to estimate a globally coherent warp from control-point correspondences, to align backprojected points from estimated depth with triangulated 3D control points, yielding calibrated backprojected points. By sampling these calibrated points near the control points, TWINGS provides a fast and geometrically accurate initialization for 3DGS, ultimately improving structural detail preservation and color fidelity in reconstructed scenes. Extensive experiments on DTU, LLFF, and Mip-NeRF360 demonstrate that TWINGS consistently outperforms existing methods, delivering detailed and accurate reconstructions under sparse-view scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07287v2">SplatWeaver: Learning to Allocate Gaussian Primitives for Generalizable Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Project Page: https://yecongwan.github.io/SplatWeaver/
    </div>
    <details class="paper-abstract">
      Generalizable novel view synthesis aims to render unseen views from uncalibrated input images without requiring per-scene optimization. Recent feed-forward approaches based on 3D Gaussian Splatting have achieved promising efficiency and rendering quality. However, most of them assign a fixed number of Gaussians to each pixel or voxel, ignoring the spatially varying complexity of real-world scenes. Such uniform allocation often wastes Gaussian primitives in smooth regions while providing insufficient capacity for fine structures, complex geometry, and high-frequency details. This motivates us to predict region-dependent primitive cardinalities rather than impose a fixed primitive budget everywhere, enabling a more expressive 3D scene representation. Therefore, we propose SplatWeaver, a generalizable novel view synthesis framework that is able to dynamically allocate Gaussian primitives over different regions in a feed-forward manner. Specifically, SplatWeaver introduces cardinality Gaussian experts and a pixel-level routing scheme, wherein each expert specializes in producing a specific number of primitives from 0 to M, and the routing scheme coordinates these experts to adaptively determine how many Gaussian primitives should be allocated to each spatial location. Moreover, SplatWeaver incorporates a high-frequency prior with attendant guidance module and routing regularization to stabilize expert selection and promote complexity-aware allocation. By leveraging high-frequency cues, the routing process is encouraged to assign more Gaussian primitives to fine structures and textured regions, while suppressing redundancy in smooth areas. Extensive experiments across diverse scenarios show that SplatWeaver consistently outperforms state-of-the-art methods, delivering more faithful novel-view renderings with fewer Gaussian primitives. Project Page: https://yecongwan.github.io/SplatWeaver/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22020v1">ForeSplat: Optimization-Aware Foresight for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-21
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) models offer fast single-pass reconstruction,but scaling them to match per-scene optimization quality is fundamentally hindered by the scarcity of large-scale 3D annotations.A practical compromise is predict-then-refine,where post-prediction optimization compensates for the limited capacity of the feed-forward network.However,standard feed-forward 3DGS is trained solely for zero-step rendering error,ignoring whether its output constitutes a good initialization for the downstream optimizer.We present ForeSplat,an optimization-aware training framework that equips feed-forward 3DGS models to produce initializations explicitly designed for rapid,effective refinement.By offloading part of the scene-modeling burden to the optimizer,ForeSplat substantially reduces the capacity pressure on the feed-forward model,making high-quality reconstruction feasible even with compact networks.At its core is MetaGrad,a lightweight multi-anchor meta-gradient training rule that bypasses costly higher-order differentiation through the 3DGS optimizer.MetaGrad unrolls a short inner-loop refinement trajectory,samples anchor states,and back-propagates aggregated first-order gradients to the prediction head as a surrogate optimization-aware signal.This fine-tuning adds no inference cost and enables high-quality reconstruction within seconds after a few refinement steps.We instantiate ForeSplat on diverse backbones,including AnySplat,Pi3X,and a distilled variant tailored for edge deployment.Across all tested architectures,a ForeSplat-trained initialization converges in fewer refinement steps and reaches a higher peak reconstruction quality than its vanilla counterpart,even fully converged.The framework consistently bridges the gap between amortized prediction and per-scene optimization,establishing a practical path toward lightweight,high-fidelity 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01466v2">SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted as a Spotlight paper at ICML 2026; camera-ready version
    </div>
    <details class="paper-abstract">
      Although multi-modal learning has advanced point cloud completion, the theoretical mechanisms remain unclear. Recent works attribute success to the connection between modalities, yet we identify that standard hard projection severs this connection: projecting a sparse point cloud onto the image plane yields an extremely sparse support, which hinders visual prior propagation, a failure mode we term Cross-Modal Entropy Collapse. To address this practical limitation, we propose SplAttN, which replaces hard projection with Differentiable Gaussian Splatting to produce a dense, continuous image-plane representation. By reformulating projection as continuous density estimation, SplAttN avoids collapsed sparse support, facilitates gradient flow, and improves cross-modal connection learnability. Extensive experiments show that SplAttN achieves state-of-the-art performance on PCN and ShapeNet-55/34. Crucially, we utilize the real-world KITTI benchmark as a stress test for multi-modal reliance. Counter-factual evaluation reveals that while baselines degenerate into unimodal template retrievers insensitive to visual removal, SplAttN maintains a robust dependency on visual cues, validating that our method establishes an effective cross-modal connection. Code is available at https://github.com/zay002/SplAttN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21935v1">Learning to Evolve: Multi-modal Interactive Fields for Robust Humanoid Navigation in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2026-05-21
      | 💬 Accepted by Robotics: Science and Systems 2026
    </div>
    <details class="paper-abstract">
      Safe manipulation-oriented navigation for humanoid robots requires scene memory that remains reliable under locomotion-induced perceptual distortion, environmental changes, and interaction-level geometric safety constraints. Existing semantic mapping and scene-graph systems are difficult to deploy directly in this setting because they often assume stable camera trajectories, static environments, or coarse object geometry. We introduce the Multi-modal Interactive Field (MIF), a humanoid-oriented system that integrates confidence-aware semantic 3D Gaussian Splatting, discrepancy-triggered spatial memory updates, and task-driven geometric reconstruction within a closed-loop perception-adaptation pipeline. MIF couples three fields: an uncertainty-aware 3DGS Appearance Field that suppresses gait-induced blur, a Spatial Field that maintains topological memory, and a Geometry Field that supports Interaction Pose Safety (IPS) before manipulation. A discrepancy detection score is introduced to separate locomotion-induced false-positive changes from persistent changes and updates only locally inconsistent regions. On a Unitree-G1 humanoid in a real dynamic office, MIF improves relocation success in non-static environments from 12% to 94% compared with static scene-graph memory, while reducing semantic memory footprint by 91.4% through feature distillation for practical online operation. Project page and code: https://ziya-jiang.github.io/MIF-homepage/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17855v2">Accelerating 3D Gaussian Splatting using Tensor Cores</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a leading technique for real-time neural rendering and 3D scene reconstruction, but its rendering cost remains too high for many latency-sensitive scenarios. In particular, the rasterization stage in 3DGS dominates end-to-end rendering time, during which the renderer repeatedly evaluates each Gaussian's contribution to each covered pixel, making this stage compute-bound. At the same time, modern GPUs provide high-throughput Tensor Cores for low-precision matrix operations, yet existing 3DGS systems execute rasterization entirely on CUDA cores and leave Tensor Cores idle. We find that 3DGS rendering can be executed in FP16 with negligible quality degradation, suggesting a promising opportunity for Tensor Core acceleration. However, exploiting Tensor Cores for 3DGS is non-trivial because rasterization does not naturally match their execution model. Existing 3DGS rasterization is expressed as irregular per-pixel scalar operations, whereas Tensor Cores require dense, regular, and reuse-rich matrix workloads. Moreover, conventional tile-by-tile execution fails to exploit Gaussian reuse across neighboring tiles, resulting in repeated data loading and thus high data movement overhead. To this end, we present TensorGS, a 3DGS acceleration framework using Tensor Cores. TensorGS tensorizes the dominant rasterization computation into Tensor-Core-compatible matrix operations and introduces cross-tile grouping to improve Gaussian reuse, amortize overhead, and increase Tensor Core utilization. Experimental results show that TensorGS improves end-to-end rendering performance by 1.65$\times$ while preserving image quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21112v1">RCGDet3D: Rethinking 4D Radar-Camera Fusion-based 3D Object Detection with Enhanced Radar Feature Encoding</a></div>
    <div class="paper-meta">
      📅 2026-05-20
    </div>
    <details class="paper-abstract">
      4D automotive radar is indispensable for autonomous driving due to its low cost and robustness, yet its point cloud sparsity challenges 3D object detection. Existing 4D radar-camera fusion methods focus on complex fusion strategies, trading inference speed for marginal gains. This trade-off hinders real-time deployment due to heavy computation on dense feature maps. In contrast, feature extraction from sparse radar points is less time-consuming but remains under-explored. This work uncovers that simply enhancing radar feature extraction can achieve comparable or even higher performance than elaborate fusion modules, while maintaining real-time performance. Based on this finding, we propose RCGDet3D, which centers on radar feature encoding and simplifies multi-modal fusion. Its encoder inherits from the efficient Gaussian Splatting-based Point Gaussian Encoder (PGE) in RadarGaussianDet3D with two key improvements. First, the Ray-centric PGE (R-PGE) predicts Gaussian attributes in ray-aligned coordinate systems before unifying them to Bird's-Eye View (BEV) space, significantly improving geometric consistency and reducing learning difficulty by decoupling the coordinate transformation from representation learning. Second, a Semantic Injection (SI) module incorporates visual cues from images, producing more geometrically accurate and semantically enriched radar features. Experiments on View-of-Delft (VoD) and TJ4DRadSet show that RCGDet3D outperforms state-of-the-art methods in both accuracy and speed, setting a new benchmark for real-time deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.14978v2">E2GS: Event Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 7pages, Accepted at ICIP 2024
    </div>
    <details class="paper-abstract">
      Event cameras, known for their high dynamic range, absence of motion blur, and low energy usage, have recently found a wide range of applications thanks to these attributes. In the past few years, the field of event-based 3D reconstruction saw remarkable progress, with the Neural Radiance Field (NeRF) based approach demonstrating photorealistic view synthesis results. However, the volume rendering paradigm of NeRF necessitates extensive training and rendering times. In this paper, we introduce Event Enhanced Gaussian Splatting (E2GS), a novel method that incorporates event data into Gaussian Splatting, which has recently made significant advances in the field of novel view synthesis. Our E2GS effectively utilizes both blurry images and event data, significantly improving image deblurring and producing high-quality novel view synthesis. Our comprehensive experiments on both synthetic and real-world datasets demonstrate our E2GS can generate visually appealing renderings while offering faster training and rendering speed (140 FPS). Our code is available at https://github.com/deguchihiroyuki/E2GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20872v1">CAdam: Context-Adaptive Moment Estimation for 3D Gaussian Densification in Generative Distillation</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted to SIGGRAPH 2026 Conference Papers. 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Adaptive densification is the engine of 3D Gaussian Splatting (3DGS). However, when transposed to the optimization-based Generative Distillation paradigm, this reconstruction-native mechanism reveals fundamental limitations, resulting in inefficient representations cluttered with redundant primitives. We diagnose this failure as a Densification Dilemma stemming from the stochastic nature of generative guidance: the standard magnitude-based accumulation indiscriminately aggregates transient noise alongside geometric signals, making it difficult to strike a balance between over-densification and under-fitting. To resolve this, we introduce Context-Adaptive Moment Estimation (CAdam), a novel framework that reinterprets densification as a statistically grounded signal verification problem. CAdam leverages the first moment of gradients to exploit the interference principle, where stochastic fluctuations cancel out via destructive interference while consistent geometric drifts accumulate via constructive interference, effectively disentangling the underlying signal from the generative noise floor. This is further augmented by a quantile-based context awareness and an intrinsic Signal-to-Noise Ratio (SNR) gating mechanism, which ensure robust adaptation across optimization stages and enable the soft termination of densification. Extensive experiments across diverse objectives (SDS, ISM, VFDS) and strong generative 3DGS backbones show that CAdam reduces Gaussian count by 85%-97% relative to standard densification while preserving overall comparable perceptual quality. These results highlight signal-aware density control as a practical way to improve memory efficiency in optimization-based generative distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20820v1">AIR: Amortized Image Reconstruction Framework for Self-Supervised Feed-Forward 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 preprint version
    </div>
    <details class="paper-abstract">
      2D Gaussian splatting provides an efficient explicit representation for image reconstruction, but existing methods still require costly per-image iterative optimization or rely on handcrafted priors for primitive allocation. We present AIR, a self-supervised feed-forward framework that amortizes iterative Gaussian fitting into a single network pass, eliminating per-image test-time optimization. AIR adopts a stage-wise residual architecture that progressively predicts additional Gaussian primitives from reconstruction residuals, together with an explicit Stage Control mechanism that activates new primitives only in under-reconstructed regions. A Predict--Optimize--Distill training strategy stabilizes multi-stage prediction by distilling short-horizon optimized Gaussian increments back into the predictor. The stabilized predictor is then jointly finetuned across stages and equipped with an image-adaptive quantizer for compact Gaussian storage. Experiments on Kodak and DIV2K show that AIR achieves better reconstruction quality than representative Gaussian-based baselines while reducing encoding time to 160--300\,ms. Code: https://github.com/whoiszzj/AIR.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.13482v2">Improving 3D Gaussian Splatting Compression by Scene-Adaptive Lattice Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Accepted by IEEE TIP. Code available at https://github.com/hxu160/SALVQ
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is rapidly gaining popularity for its photorealistic rendering quality and real-time performance, but it generates massive amounts of data. Hence compressing 3DGS data is necessary for the cost effectiveness of 3DGS models. Recently, several anchor-based neural compression methods have been proposed, achieving good 3DGS compression performance. However, they all rely on uniform scalar quantization (USQ) due to its simplicity. A tantalizing question is whether more sophisticated quantizers can improve the current 3DGS compression methods with very little extra overhead and minimal change to the system. The answer is yes by replacing USQ with lattice vector quantization (LVQ). To better capture scene-specific characteristics, we optimize the lattice basis for each scene, improving LVQ's adaptability and R-D efficiency. This scene-adaptive LVQ (SALVQ) strikes a balance between the R-D efficiency of vector quantization and the low complexity of USQ. SALVQ can be seamlessly integrated into existing 3DGS compression architectures, enhancing their R-D performance with minimal modifications and computational overhead. Moreover, by scaling the lattice basis vectors, SALVQ can dynamically adjust lattice density, enabling a single model to accommodate multiple bit rate targets. This flexibility eliminates the need to train separate models for different compression levels, significantly reducing training time and memory consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20566v1">Conflict-Aware Active Perception and Control in 3D Gaussian Splatting Fields via Control Barrier Functions</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Project website: https://sircesoc.github.io/Conflict_Aware_Active_Perception/
    </div>
    <details class="paper-abstract">
      Active perception in uncertain environments requires robots to navigate safely while acquiring informative observations to reduce map uncertainty. These objectives inherently conflict, as informative viewpoints often lie near uncertain regions with higher collision risk. To address this challenge, we develop a conflict-aware active perception and control framework for robotic systems operating in environments represented by 3D Gaussian Splatting (3DGS). Safety is enforced using a Control Barrier Function (CBF) derived from an Average Value-at-Risk AV@R collision-risk metric that accounts for geometric uncertainty and guarantees forward invariance of a safe set. To improve perception, we propose a risk-aware Expected Information Gain (EIG) formulation for selecting the next-best-view and introduce perception barrier functions that align the camera orientation with the local information-ascent direction. To obtain a tractable formulation for these conflicting safety and perception objectives, we propose a unified safety-critical, perception-aware quadratic program that enforces safety as a hard constraint while relaxing perception constraints through slack variables. Simulation results demonstrate that the proposed method improves both safety and information acquisition compared to existing 3DGS-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20150v1">TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Accepted to ICML 2026 as Spotlight. Website: https://sponge-lab.github.io/TideGS
    </div>
    <details class="paper-abstract">
      Training 3D Gaussian Splatting (3DGS) at billion-primitive scale is fundamentally memory-bound: each Gaussian primitive carries a large attribute vector, and the aggregate parameter table quickly exceeds GPU capacity, limiting prior systems to tens of millions of Gaussians on commodity single-GPU hardware. We observe that 3DGS training is inherently sparse and trajectory-conditioned: each iteration activates only the Gaussians visible from the current camera batch, so GPU memory can serve as a working-set cache rather than a persistent parameter store. Building on this insight, we introduce TideGS, an out-of-core training framework that manages parameters across an SSD-CPU-GPU hierarchy via three synergistic techniques: block-virtualized geometry for SSD-aligned spatial locality, a hierarchical asynchronous pipeline to overlap I/O with computation, and trajectory-adaptive differential streaming that transfers only incremental working-set deltas between iterations. Experiments show that TideGS enables training with over one billion Gaussians on a single 24 GB GPU while achieving the best reconstruction quality among evaluated single-GPU baselines on large-scale scenes, scaling beyond prior out-of-core baselines (e.g., approximately 100M Gaussians) and standard in-memory training (e.g., approximately 11M Gaussians).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20044v1">OP2GS: Object-Aware 3D Gaussian Splatting with Dual-Opacity Primitives</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) provides an explicit and efficient scene representation, but its primitives lack inherent object-level identity, hindering downstream tasks such as open-vocabulary scene understanding. Existing methods typically address this by either distilling high-dimensional feature embeddings into Gaussians or by lifting 2D mask labels into 3D via heuristic refinement. However, feature-based approaches incur heavy storage and decoding overhead, while lifting-based pipelines remain vulnerable to label contamination: Gaussians necessary for appearance reconstruction often receive incorrect object labels during 2D-to-3D projection. We propose OP2GS, an object-aware Gaussian representation that augments each primitive with an explicit instance identity and a dedicated instance opacity $σ^{*}$ for object-mask rendering. The original opacity $σ$ remains responsible for visual reconstruction, while $σ^{*}$ models whether a Gaussian should contribute to a particular object mask. This dual-opacity formulation decouples visual existence from instance occupancy: mislabeled Gaussians can remain available for image rendering while becoming transparent in the object-mask branch. To learn this representation, we introduce a random object loss that optimizes the 1D instance occupancy field using the standard transmittance-based visibility of 3DGS. Semantic descriptors are then attached at the object level through multi-view aggregation, eliminating per-Gaussian feature storage. Compared with feature-training approaches, OP2GS achieves competitive open-vocabulary performance while significantly reducing computational overhead. Compared with training-free pipelines, it leverages physically consistent occupancy learning to resolve visibility ambiguities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19949v1">Feed-Forward Gaussian Splatting from Sparse Aerial Views</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Reconstructing large-scale urban scenes from sparse aerial views is a crucial yet challenging task. Due to biased top-down and shallow-oblique camera poses, sparse aerial captures exhibit strong evidence imbalance: roofs and open regions are repeatedly observed, while facades, distant buildings, and occluded structures receive little multi-view support. Existing feed-forward 3D Gaussian Splatting methods directly regress a deterministic representation from sparse inputs, but this often leads to ghosting, melted facades, and stretched textures. Recent pseudo-view and video-based generative reconstruction methods use additional supervision or generative priors. However, they often lack a clear separation between observed geometry and prior-driven content, which can lead to plausible but inconsistent structures. We propose AnyCity, an observation-grounded generative reconstruction framework for sparse aerial urban scenes. AnyCity first predicts an observation-supported geometry latent to anchor reliable structures, and then uses scaffold-conditioned aerial completion tokens to predict a gated residual update for weakly constrained content before Gaussian decoding. During training, dense-to-sparse distillation transfers structural cues from dense-view reconstruction, while an aerial-adapted video diffusion prior provides fine-grained urban appearance cues through gated token conditioning. Observation-preserving objectives keep the refined representation consistent with input-supported geometry. At inference time, AnyCity reconstructs the final 3D Gaussian scene from sparse aerial views in a single feed-forward pass, achieving coherent urban novel-view synthesis with second-level inference. Experiments on synthetic, aerial-domain, UAV-textured, and real-world scenes show consistent improvements over feed-forward baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19656v1">Cross-View Splatter: Feed-Forward View Synthesis with Georeferenced Images</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Submitted to CVPR 2026. 8 figures, 3 tables. Project page: https://nianticspatial.github.io/cross-view-splatter/
    </div>
    <details class="paper-abstract">
      We present Cross-View Splatter, a feed-forward method that predicts pixel-aligned Gaussian splats for outdoor scenes captured at ground level AND by satellite. Faithful reconstructions require good camera coverage, but ground imagery is time-consuming and hard to capture at scale for large outdoor scenes. Fortunately, satellite imagery can provide a global geometric prior that is easy to access via public APIs. Cross-View Splatter fuses orthorectified satellite views with GPS-tagged ground photos to predict Gaussian splats in a unified 3D coordinate frame. By aligning ground and bird's-eye feature representations, our model improves scene coverage and novel-view synthesis, compared to ground imagery alone. We train on curated georeferenced datasets and paired satellite-terrain data, mined from open mapping services. We evaluate our method on a new benchmark for novel-view synthesis with georeferenced imagery allowing comparison to prior state-of-the-art methods. Our code and data preparation will be available at https://nianticspatial.github.io/cross-view-splatter/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19600v1">FlyMirage: A Fully Automated Generation Pipeline for Diverse and Scalable UAV Flight Data via Generative World Model</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      In the field of Vision-Language Navigation (VLN), aerial datasets remain limited in their ability to combine scale, diversity, and realism, often relying on either costly real-world scenes or visually limited simulations. To address these challenges, we introduce FlyMirage, a highly scalable and fully automated data generation pipeline for aerial VLN. Our approach leverages large language models (LLM) as an environment designer to promote scene diversity, paired with a generative world model that instantiates these designs into high-fidelity 3D Gaussian Splatting (3DGS) scenes. To substantially reduce human labor and ensure the feasibility of flight data, FlyMirage automates scene exploration and semantic information acquisition, and further integrates a dynamically feasible planner for uncrewed aerial vehicle (UAV) trajectory generation. Utilizing this toolchain, we generate a large-scale, diverse, and photorealistic aerial VLN dataset, with dynamically feasible flying trajectories, designed to support the development of next-generation embodied navigation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19304v1">MMGS: 10$\times$ Compressed 3DGS through Optimal Transport Aggregation based on Multi-view Ranking</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has revolutionized 3D reconstruction, it suffers from significant overhead due to massive redundant primitives. Existing compression methods typically rely on local sampling or fixed pruning thresholds, which often struggle to balance redundancy reduction with high-fidelity rendering. To address this, we propose a novel framework that formulates Gaussian optimization as a global geometric distribution matching problem. Specifically, our approach integrates three components: (1) we introduce a multi-view 3D Gaussian contribution ranking mechanism that filters primitives using geometric consistency instead of local heuristics; (2) we propose a global Optimal Transport (OT)-based aggregation algorithm that merges redundant primitives while preserving the underlying geometry; and (3) we design an OT-based densification operator that maintains the Gaussian's distributional properties for stable optimization. Our approach achieves state-of-the-art rendering quality with only \textbf{10$\%$} primitives and \textbf{10$\times$} accelerated training speeds compared to vanilla 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17916v2">PanoWorld: A Generative Spatial World Model for Consistent Whole-House Panorama Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 17
    </div>
    <details class="paper-abstract">
      Generating a consistent whole-house VR tour from a floorplan and style reference requires both photorealistic panoramas and cross-view spatial coherence. Pure 2D generators produce appealing single panoramas but re-imagine geometry and materials when the viewpoint changes, whereas monolithic 3D generation becomes expensive and loses fine texture at multi-room scale. We introduce PanoWorld, a generative spatial world model that treats whole-house synthesis as autoregressive generation of node-based 360-degree panoramas, matching the discrete navigation used by real VR tour products. PanoWorld uses a floorplan-derived 3D shell as a global geometric proxy and a dynamic 3D Gaussian Splatting cache as renderable spatial memory. A feed-forward panoramic LRM designed for metric-scale multi-room 360-degree inputs lifts generated panoramas into local 3DGS updates, while Room-aware Group Attention suppresses cross-room feature interference. A topology-aware progressive caching strategy fuses these local updates without repeatedly reconstructing the full history. By decoupling shell-based geometry guidance from cache-rendered visual memory, PanoWorld preserves high-frequency 2D synthesis quality while improving cross-node layout and material consistency. The project link is https://jjrcn.github.io/PanoWorld-project-home/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19065v1">A Geometric Algebra-Informed 3D Gaussian Splatting Framework for Wireless Scene Representation</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Geometric Algebra-Informed 3D Gaussian Splatting (GAI-GS), a framework for wireless modeling that couples 3D Gaussian splatting with a geometric algebra-based attention mechanism to explicitly model ray-object interactions in complex propagation environments. GAI-GS encodes joint spatial-electromagnetic (EM) relations into token representations, enabling scene-level aggregation within a unified, end-to-end neural architecture. This design grounds wireless ray propagation in electromagnetic principles, allowing token interactions to model key effects such as multipath, attenuation, and reflection/diffraction. Through extensive evaluations on multiple real-world indoor datasets, GAI-GS consistently surpasses current baselines across various wireless tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09668v2">DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 Accepted by ICLR 2026. Project page: https://zju3dv.github.io/DiffWind/
    </div>
    <details class="paper-abstract">
      Modeling wind-driven object dynamics from video observations is highly challenging due to the invisibility and spatio-temporal variability of wind, as well as the complex deformations of objects. We present DiffWind, a physics-informed differentiable framework that unifies wind-object interaction modeling, video-based reconstruction, and forward simulation. Specifically, we represent wind as a grid-based physical field and objects as particle systems derived from 3D Gaussian Splatting, with their interaction modeled by the Material Point Method (MPM). To recover wind-driven object dynamics, we introduce a reconstruction framework that jointly optimizes the spatio-temporal wind force field and object motion through differentiable rendering and simulation. To ensure physical validity, we incorporate the Lattice Boltzmann Method (LBM) as a physics-informed constraint, enforcing compliance with fluid dynamics laws. Beyond reconstruction, our method naturally supports forward simulation under novel wind conditions and enables new applications such as wind retargeting. We further introduce WD-Objects, a dataset of synthetic and real-world wind-driven scenes. Extensive experiments demonstrate that our method significantly outperforms prior dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity, opening a new avenue for video-based wind-object interaction modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18334v1">3D Skew Gaussian Splatting with Any Camera Trajectory Visualization Engine</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has revolutionized real-time photorealistic view synthesis, its fundamental reliance on symmetric Gaussian distributions introduces visual artifacts that hinder accurate spatial data exploration. Specifically, symmetric kernels struggle to capture shape and color discontinuities , which cause blurriness and primitive redundancy that mislead human perception during visual analysis. To address these visualization barriers, we introduce 3D Skew Gaussian Splatting (3DSGS), a novel framework that significantly enhances the structural fidelity and compactness of explicit scene representations. Our key insight lies in extending the standard primitive to a general Skew Gaussian counterpart. This generalized primitive inherits the highly efficient rasterization properties of standard Gaussians while gaining intrinsic asymmetric modeling capabilities. We couple this with an enhanced opacity representation to better handle complex transparency, alongside a depth-aware densification strategy that intelligently manages primitive allocation. Furthermore, to make these advancements actionable for real-world visual analytics, we re-derive the CUDA rasterization pipeline to universally support both symmetric and skew Gaussians, integrating it into a decoupled, free-camera interactive visualization engine. Extensive experiments demonstrate that 3DSGS achieves superior rendering quality and structural compactness, particularly in regions with intricate details, while maintaining the real-time frame rates necessary for fluid interactive exploration. Supplementary derivations and visual results are available at \textbf{\textit{https://3d-skew-gs.github.io/}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18263v1">RT-Splatting: Joint Reflection-Transmission Modeling with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 CVPR 2026 Highlight, Project Page: https://sjj118.github.io/RT-Splatting/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables real-time novel view synthesis with high visual quality. However, existing methods struggle with semi-transparent specular surfaces that exhibit both complex reflections and clear transmission, often producing blurry reflections or overly occluded transmission. To address this, we present RT-Splatting, a framework that disentangles each Gaussian's geometric occupancy from its optical opacity. This factorization yields a unified surface-volume scene representation with a single set of Gaussian primitives. Our hybrid renderer interprets this representation both as a surface to capture high-frequency reflections and as a volume to preserve clear transmission. To mitigate the ambiguity in jointly optimizing reflection and transmission, we introduce Specular-Aware Gradient Gating, which suppresses misleading gradients from highly specular regions into the transmission branch, effectively reducing distracting floaters. Experiments on challenging semi-transparent scenes show that RT-Splatting achieves state-of-the-art performance, delivering high-fidelity reflections and clear transmission with real-time rendering. Moreover, our factorization naturally enables flexible scene editing. The project page is available at https://sjj118.github.io/RT-Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18252v1">GaussianZoom: Progressive Zoom-in Generative 3D Gaussian Splatting with Geometric and Semantic Guidance</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We introduce GaussianZoom, a generative zoom-in 3D reconstruction system with an iterative progressive framework that combines geometry-consistent scene modeling and multi-scale semantic reasoning to enable high-fidelity extreme zoom-in rendering from low-resolution inputs. To achieve this, we develop a novel multi-view consistent super-resolution module with depth-based feature warping and VLM-driven detail synthesis, ensuring accurate multi-view correspondence while enriching fine-scale appearance beyond the observed resolution. To support zooming across large magnification ranges, we further introduce a new expandable continuous Level-of-Detail hierarchy that dynamically modulates Gaussian visibility for smooth, alias-free cross-scale rendering. Experiments on Mip-NeRF360 and Tanks\&Temples demonstrate that GaussianZoom achieves superior perceptual quality, multi-view consistency, and robustness under extreme magnification, establishing a strong baseline for generative zoom-in 3D scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20155v2">GSCompleter: A Distillation-Free Plugin for Metric-Aware 3D Gaussian Splatting Completion in Seconds</a></div>
    <div class="paper-meta">
      📅 2026-05-18
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized high-fidelity neural rendering with its explicit representation and efficiency. However, reconstructing scenes from sparse viewpoints suffers from severe geometric voids and floaters due to limited coverage. Current scene completion methods typically rely on an iterative "Repair-then-Distill" paradigm, which is computationally intensive, prone to unstable optimization, and susceptible to overfitting. To address these limitations, we propose GSCompleter, a distillation-free plugin that shifts scene completion to a stable "Generate-then-Register" workflow. Specifically, GSCompleter synthesizes visually plausible 2D reference images and explicitly lifts them into 3D Gaussian primitives with a consistent metric scale via a robust Stereo-Anchor View Selection mechanism. These newly generated primitives are then seamlessly integrated into the global scene using a novel Ray-Constrained Registration strategy. By replacing unstable distillation with rapid geometric registration, GSCompleter exhibits superior 3DGS completion performance across three benchmarks, enhancing both quality and efficiency over various baselines and achieving new state-of-the-art (SOTA) results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10239v2">AdaptSplat: Adapting Vision Foundation Models for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-18
    </div>
    <details class="paper-abstract">
      This work explores a simple yet powerful lightweight adapter design for feed-forward 3D Gaussian Splatting (3DGS). Existing methods typically apply complex, architecture-specific designs on top of the generic pipeline of image feature extraction $\rightarrow$ multi-view interaction $\rightarrow$ feature decoding. However, constrained by the scale bottleneck of 3D training data and the low-pass filtering effect of deep networks, these methods still fall short in cross-domain generalization and high-frequency geometric fidelity. To address these problems, we propose AdaptSplat, which demonstrates that without complex component engineering, introducing a single adapter of only 1.5M parameters into the generic architecture is sufficient to achieve superior performance. Specifically, we design a lightweight Frequency-Preserving Adapter (FPA) that extracts direction-aware high-frequency structural priors from the shallow features of a powerful vision foundation model backbone, and seamlessly integrates them into the generic pipeline via high-frequency positional encodings and adaptive residual modulation. This effectively compensates for the high-frequency attenuation caused by over-smoothing in deep features, improving the fitting accuracy of Gaussian primitives on complex surfaces and sharp boundaries. Extensive experiments demonstrate that AdaptSplat achieves state-of-the-art feed-forward reconstruction performance on multiple standard benchmarks, with stable generalization across domains. Code available at: https://github.com/xmw666/AdaptSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17777v1">Efficient Sparse-to-Dense Visual Localization via Compact Gaussian Scene Representation and Accelerated Dense Pose Estimation</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 IEEE/CAA JAS 2026
    </div>
    <details class="paper-abstract">
      This letter presents LiteLoc, a novel and efficient localizer built on 3D Gaussian Splatting (3DGS). The previous state-of-the-art (SoTA) sparse-to-dense localizer, STDLoc, has shown remarkable localization capability but suffers from severe storage redundancy and computational latency. By revisiting its design decisions, we derive two simple yet highly effective improvements that cumulatively make LiteLoc much more efficient in both memory and computation, while also being easier to train. One key observation is that the color field, inherited directly from Feature 3DGS, is functionally useless for localization. Yet, its reconstruction of high-frequency photometric details necessitates excessive Gaussian primitives, resulting in a tightly coupled color-feature representation with significant memory overhead and sub-optimal feature field optimization. To resolve this, we propose a color-free decoupled feature field that constructs a compact Gaussian scene representation by retaining only task-essential feature attributes, thereby eliminating approximately 94% of redundant storage with no loss of localization-relevant information. We further find that the primary computational bottleneck lies in the dense Perspective-n-Point (PnP) solver, where most matches contribute saturated geometric constraints with diminishing accuracy gains. Accordingly, we propose a condensing strategy that distills dense matches into a subset of 5% representative matches, enabling a nearly 19-fold speedup in robust estimation with negligible performance drop. Extensive experiments show that LiteLoc surpasses STDLoc in multiple scenes with considerable efficiency benefits, opening up exciting prospects for latency-sensitive visual localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.00952v2">Decoupling Motion and Geometry in 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-16
    </div>
    <details class="paper-abstract">
      High-fidelity reconstruction of dynamic scenes is an important yet challenging problem. While recent 4D Gaussian Splatting (4DGS) has demonstrated the ability to model temporal dynamics, it couples Gaussian motion and geometric attributes within a single covariance formulation, which limits its expressiveness for complex motions and often leads to visual artifacts. To address this, we propose VeGaS, a novel velocity-based 4D Gaussian Splatting framework that decouples Gaussian motion and geometry. Specifically, we introduce a Galilean shearing matrix that explicitly incorporates time-varying velocity to flexibly model complex non-linear motions, while strictly isolating the effects of Gaussian motion from the geometry-related conditional Gaussian covariance. Furthermore, a Geometric Deformation Network is introduced to refine Gaussian shapes and orientations using spatio-temporal context and velocity cues, enhancing temporal geometric modeling. Extensive experiments on public datasets demonstrate that VeGaS achieves state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17011v1">Topo-GS: Continuous Volumetric Embedding of High-Dimensional Data via Topological Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-16
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Dimensionality reduction algorithms map high-dimensional data into visualizable 2D or 3D spaces, but traditionally rely on a discrete point-cloud paradigm. This discrete abstraction is susceptible to visual occlusion and artificial discontinuities, often failing to represent the continuous density of the underlying manifold. To address these limitations, we introduce Topo-GS, a framework that repurposes 3D Gaussian Splatting (3DGS) to cast multidimensional projection as a meshless volumetric reconstruction process. Instead of standard photometric losses, Topo-GS is driven by local geometric constraints. By solving orthogonal Procrustes targets, the optimization enforces an As-Rigid-As-Possible prior while explicitly aligning the spatial covariance of each Gaussian to the local tangent space. Recognizing that unrolling data of varying intrinsic dimensionalities requires distinct spatial treatments, we utilize a topology-aware strategy that tailors the loss formulation to preserve either continuous 1D trajectories or cohesive 2D surfaces. Quantitative and visual evaluations demonstrate that Topo-GS successfully transforms discrete scatter plots into continuous volumetric representations, where inherent projection distortions explicitly manifest as observable geometric variations, while preserving local topological fidelity comparable to discrete baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17002v1">A Single Atlas is All You Need: Decoder-Side Gaussian Splatting for Immersive Video</a></div>
    <div class="paper-meta">
      📅 2026-05-16
    </div>
    <details class="paper-abstract">
      Immersive video delivery is bottlenecked by pixel-rate constraints, making the transmission of high-resolution depth maps or explicit 3D volumetric data expensive. Decoder-Side Depth Estimation (DSDE) shifts depth computation to the client, but struggles with complex geometries, inter-view flickering, and non-Lambertian reflections. Conversely, 3D Gaussian Splatting (3DGS) offers state-of-the-art view synthesis, but transmitting splats (or their projected 2D maps) incurs prohibitive bandwidth costs and is poorly aligned with standard video codecs. We propose Decoder-Side Gaussian Splatting (DSGS), a framework that natively replaces the depth-estimation stage of DSDE with feed-forward 3DGS inference, optimizing volumetric scenes entirely on the decoder side from compressed textures and metadata. A central, counterintuitive finding is that lossy compression acts as an implicit low-pass filter stabilizing feed-forward splat prediction: compressed bitstreams exceed lossless quality while shrinking tenfold. Under extreme view sparsity (one 2D atlas comprising 4 input views), DSGS achieves a +5.79 dB BD-PSNR and +0.054 BD-SSIM gain over the DSDE anchor while reducing maximum inter-view Delta IV-PSNR from 17.2 dB to 6.4 dB, minimizing the domain shift between transmitted and virtual viewports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16925v1">P2GS: Physical Prior-guided Gaussian Splatting for Photometrically Consistent Urban Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-16
      | 💬 Accepted CVPR2026 main
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a powerful explicit representation enabling fast, high-fidelity rendering, making it a promising foundation for closed-loop simulators and perception models in autonomous driving. However, conventional 3DGS implicitly assumes consistent exposure and tone mapping across views. Real driving data violates this assumption due to heterogeneous camera pipelines and dynamic outdoor illumination, baking exposure discrepancies and sensor noise into the radiance field and producing artifacts and inconsistent illumination especially in static backgrounds crucial for realistic simulation. These issues are amplified in autonomous driving, where sparse viewpoints, varying exposures, and outdoor lighting interact, while prior work mainly targets dynamic-object reconstruction and overlooks cross-view photometric consistency. To address this limitation, we introduce P2GS, a physically consistent Gaussian Splatting framework that jointly decomposes a view-invariant linear HDR radiance field, per-view exposure scales, and tone-mapping functions from only LDR images without HDR supervision. P2GS employs a unified optimization strategy grounded in the physical image-formation process, enforcing relative-exposure consistency and HDR-domain radiance regularization. This yields a radiance field robust to inter-camera illumination differences while preserving the real-time efficiency of standard 3DGS. Experiments across real and simulated driving environments show that P2GS matches or surpasses prior methods in LDR reconstruction while providing substantially improved photometric consistency, reliable exposure normalization, and physically coherent illumination across diverse scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14009v2">GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-05-15
      | 💬 Published in: IEEE Robotics and Automation Letters ( Volume: 11, Issue: 2, February 2026)
    </div>
    <details class="paper-abstract">
      Autonomous drones capable of interpreting and executing high-level language instructions in unstructured environments remain a long-standing goal. Yet existing approaches are constrained by their dependence on hand-crafted skills, extensive parameter tuning, or computationally intensive models unsuitable for onboard use. We introduce GRaD-Nav++, a lightweight Vision-Language-Action (VLA) framework that runs fully onboard and follows natural-language commands in real time. Our policy is trained in a photorealistic 3D Gaussian Splatting (3DGS) simulator via Differentiable Reinforcement Learning (DiffRL), enabling efficient learning of low-level control from visual and linguistic inputs. At its core is a Mixture-of-Experts (MoE) action head, which adaptively routes computation to improve generalization while mitigating forgetting. In multi-task generalization experiments, GRaD-Nav++ achieves a success rate of 83% on trained tasks and 75% on unseen tasks in simulation. When deployed on real hardware, it attains 67% success on trained tasks and 50% on unseen ones. In multi-environment adaptation experiments, GRaD-Nav++ achieves an average success rate of 81% across diverse simulated environments and 67% across varied real-world settings. These results establish a new benchmark for fully onboard Vision-Language-Action (VLA) flight and demonstrate that compact, efficient models can enable reliable, language-guided navigation without relying on external infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06088v2">OpenGaFF: Open-Vocabulary Gaussian Feature Field with Codebook Attention</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      Understanding open-vocabulary 3D scenes with Gaussian-based representations remains challenging due to fragmented and spatially inconsistent semantic predictions across multi-view observations. In this paper, we present OpenGaFF, a novel framework for open-vocabulary 3D scene understanding built upon 3D Gaussian Splatting. At the core of our method is a Gaussian Feature Field that models semantics as a continuous function of Gaussian geometry and appearance. By explicitly conditioning semantic predictions on geometric structure, this formulation strengthens the coupling between geometry and semantics, leading to improved spatial coherence across similar structures in 3D space. To further enforce object-level semantic consistency, we introduce a structured codebook that serves as a set of shared semantic primitives. Furthermore, a codebook-guided attention mechanism is proposed to retrieve language features via similarity matching between query embeddings and learned codebook entries, enabling robust open-vocabulary reasoning while reducing intra-object feature variance. Extensive experiments on standard 2D and 3D open-vocabulary benchmarks demonstrate that our method consistently outperforms prior approaches, achieving improved segmentation quality, stronger 3D semantic consistency and a semantically interpretable codebook that provides insight into the learned representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16582v1">ArtMesh: Part-Aware Articulated Mesh Fields with Motion-Consistent Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      We present ArtMesh, a mesh-native method for reconstructing articulated objects explicitly as connected triangle meshes with per-part rigid motion from multi-view images in start and end states. Existing 3D Gaussian Splatting pipelines for articulated reconstruction inherit the unstructured point-based geometry of their splatting base, which provides no surface topology for reasoning about part boundaries or enforcing motion consistency along the object's connectivity. ArtMesh instead builds on a mesh-based differentiable rendering backbone, enabling part-aware dynamics to act directly on the structured topology. To make the topology compatible with articulation, we introduce part-aware restricted Delaunay remeshing, producing connected submeshes whose triangles do not cross semantic part boundaries. The dynamic mesh field then optimizes articulation using bidirectional Vertex-wise Motion Consistency on transported mesh vertices and Pixel-wise Motion Consistency on rendered RGB-D observations. We introduce Articulate-100, a new benchmark of 100 articulated objects spanning 16 PartNet-Mobility categories. On this benchmark, ArtMesh outperforms prior 3DGS-based pipelines in joint parameter estimation and part-level geometric reconstruction, with the largest gains on objects with many movable parts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16158v1">Smart target point control for Gaussian Splatting methods</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      Standard Gaussian splatting methods rely on heuristic densification and pruning to adaptively allocate primitives during training, and the resulting Gaussian count strongly influences both reconstruction quality and runtime. This makes comparisons across methods fragile: improvements can stem from higher representational capacity rather than algorithmic design. A common and naive workaround for this is hard-stopping or budgeting densification/pruning once a target count is reached, which biases training because different methods hit the cap at different times, yielding non-uniform densify/prune exposure across views and uneven point distributions. We propose a target point control scheme that preserves the standard densification window and cadence, but adjusts only the existing densification and opacity-culling hyper-parameters to track a quadratic target count trajectory. This quota-governor reaches the desired count by 15k iterations without abrupt cutoffs, ensuring that all methods and views receive equal densification and pruning cycles, enabling fairer, capacity-matched evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16065v1">Robust Prior-Guided Segmentation for Editable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-15
      | 💬 Accepted at IEEE International Conference on Image Processing 2026, 6 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) enables real-time 3D scene reconstruction but lacks robust segmentation for editing tasks such as object removal, extraction, and recoloring. Existing approaches that lift 2D segmentations to the 3D domain suffer from view inconsistencies and coarse masks. In this paper, we propose a novel framework that leverages the Segment Anything Model High Quality (SAM-HQ) to generate accurate 2D masks, addressing the limitations of the standard SAM in boundary fidelity and fine-structure preservation. To achieve robust 3D segmentation of any target object in a given scene, we introduce a prior-guided label reassignment method that assigns labels to 3D Gaussians by enforcing multiview consistency with learned priors. Our approach achieves state-of-the-art segmentation accuracy and enables interactive, real-time object editing while maintaining high visual fidelity. Qualitative results demonstrate superior boundary preservation and practical utility in Virtual Reality (VR) and robotics, advancing 3D scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16022v1">EndoGSim: Physics-Aware 4D Dynamic Endoscopic Scene Simulations via MLLM-Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-15
      | 💬 Early Accepted by MICCAI 2026
    </div>
    <details class="paper-abstract">
      In robot-assisted minimally invasive surgery, high-fidelity dynamic endoscopic scene reconstruction and simulation are crucial to enhancing downstream tasks and advancing surgical outcomes. However, existing methods primarily focus on visual reconstruction, lacking physics-based descriptions of the scene required for realistic simulation. We propose a unified framework that achieves physics-aware reconstruction and physical simulation of endoscopic scenes through Multi-modal Large Language Models (MLLMs)-guided Gaussian Splatting. Our approach utilizes 4D Gaussian Splatting (4DGS) integrated with pre-trained segmentation and depth estimation to represent deformable tissues and tools. To achieve automatic inference of physical properties, we introduce an object-wise material field that initializes material parameters via MLLM and refines them through a differentiable Material Point Method (MPM) under joint supervision from rendered images and optical flow. Validated on both open-source and in-house datasets, our framework achieves superior simulation fidelity and physical accuracy compared to state-of-the-art methods, underscoring its potential to advance robot-assisted surgical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28111v3">GSDrive: Reinforcing Driving Policies by Multi-mode Future Trajectory Probing with 3D Gaussian Splatting Environment</a></div>
    <div class="paper-meta">
      📅 2026-05-15
      | 💬 2nd version
    </div>
    <details class="paper-abstract">
      End-to-end (E2E) autonomous driving aims to directly map sensory observations to driving actions, but its real-world deployment is hindered by evolving data distributions and the high cost of continual annotation. While combining imitation learning (IL) and reinforcement learning (RL) is a common strategy for policy improvement, conventional RL training relies on delayed, event-based rewards, where policies learn only from catastrophic outcomes such as collisions, leading to premature convergence to suboptimal behaviors. To address these limitations, we propose GSDrive, a framework that uses a differentiable 3D Gaussian Splatting (3DGS) environment for future-aware trajectory probing and reward shaping in E2E driving. GSDrive first learns a multi-mode trajectory probe via IL and then uses RL to evaluate multiple candidate futures in the 3DGS environment, converting their simulated returns into dense shaping rewards for policy optimization. This yields a cyclic hybrid IL-RL training loop, where IL supplies structured future priors and RL provides interactive feedback for iterative refinement. Evaluated on the reconstructed nuScenes dataset, our method outperforms other simulation-based RL approaches in closed-loop experiments. Code is available at https://github.com/ZionGo6/GSDrive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15760v1">Learn2Splat: Extending the Horizon of Learned 3DGS Optimization</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) optimization is most commonly performed using standard optimizers (Adam, SGD). While stable across diverse scenes, standard optimizers are general-purpose and not tailored to the structure of the problem. In particular, they produce independent parameter updates that do not capture the structural and spatial relationships within a scene, leading to inefficient optimization and slow convergence. Recent works introduced learned optimizers that predict correlated updates informed by inter-parameter and inter-Gaussian dependencies. However, these methods are trained for a fixed number of optimization iterations and rely on manually scheduled learning rates to avoid degradation. In this paper, we introduce a learned optimizer for 3DGS that avoids degradation over extended optimization horizons without auxiliary mechanisms. To enable this, we propose a meta-learning scheme that extends the optimization horizon via a checkpoint buffer and an optimizer rollout strategy, combined with an architecture that encodes gradient scale information in its latent states. Results show improved early novel view synthesis quality while remaining stable over long horizons, with zero-shot generalization to unseen reconstruction settings. To support our findings, we introduce the first unified framework for training and evaluating both learned and conventional optimizers across sparse and dense view settings. Code and models will be released publicly. Our project page is available at https://naamapearl.github.io/learn2splat .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18866v1">FLUIDSPLAT: Reconstructing Physical Fields from Sparse Sensors via Gaussian Primitives</a></div>
    <div class="paper-meta">
      📅 2026-05-15
      | 💬 23 pages, 4 figures,preprint
    </div>
    <details class="paper-abstract">
      Reconstructing continuous flow fields from sparse surface-mounted sensors is central to aerodynamic design, flow control, and digital-twin instrumentation. Existing neural methods for this task typically encode sensor readings into implicit latent codes with little spatial interpretability and limited formal guidance on how representational capacity should scale with observation count. Inspired by 3D Gaussian Splatting, we introduce FLUIDSPLAT, a sensor-conditioned model that predicts K anisotropic Gaussian primitives forming a partition-of-unity scaffold, a spatially explicit and interpretable intermediate representation of the flow. For an idealized Gaussian primitive estimator, we prove an $O(K^{-s/d})$ approximation rate for fields with Sobolev smoothness $s$; incorporating $N$ noisy observations yields a squared-risk decomposition with bias $O(K^{-2s/d})$ and variance $O(σ^{2}K/N)$.Balancing the two yields $K^{*}\!\sim\!(N/σ^{2})^{d/(2s+d)}$: primitive count cannot grow freely under sparse sensing, revealing a variance bottleneck that motivates complementing the scaffold with a state-conditioned residual decoder. On a standard cylinder-flow benchmark, FLUIDSPLAT achieves the best mean error across all surface-sensor layouts; on AirfRANS with 8 surface-pressure sensors, it reduces error by 11-23% over the strongest baseline across three standard splits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15010v2">3D Skew-Normal Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading representation for real-time novel view synthesis and has been widely adopted in various downstream applications. The core strength of 3DGS lies in its efficient kernel-based scene representation, where Gaussian primitives provide favorable mathematical and computational properties. However, under a finite primitive budget, the symmetric shape of each primitive directly affects representation compactness, especially near asymmetric structures such as object boundaries and one-sided surfaces. Recent works have explored more complex kernel distributions; however, they either remain within the elliptical family or rely on hard truncation, which limits continuous shape control and introduces distributional discontinuities. In this paper, we propose Skew-Normal Splatting (SNS), which adopts the Azzalini Skew-Normal distribution as the fundamental primitive. By introducing a learnable and bounded skewness parameter, SNS can continuously interpolate between symmetric Gaussians and Half-Gaussian-like shapes, enabling flexible modeling of both sharp boundaries and interior regions. Moreover, SNS preserves analytical tractability under affine transformations and marginalization. This property allows seamless integration into existing Gaussian Splatting rasterization pipelines. Furthermore, to address the strong coupling between scale, rotation, and skewness parameters, we introduce a decoupled parameterization and a block-wise optimization strategy to enhance training stability and accuracy. Extensive experiments on standard novel-view synthesis benchmarks show that SNS consistently improves reconstruction quality over Gaussian and recent non-Gaussian kernels, with clearer benefits on sharp boundaries and thin or one-sided structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13073v2">HarmoGS: Robust 3D Gaussian Splatting in the Wild via Conflict-Aware Gradient Harmonization</a></div>
    <div class="paper-meta">
      📅 2026-05-15
    </div>
    <details class="paper-abstract">
      In-the-wild 3D Gaussian Splatting remains challenging due to transient distractors and illumination-induced cross-view appearance inconsistencies. Existing methods mainly rely on image-level masking to suppress unreliable supervision, but masking alone cannot fully eliminate residual occlusions or resolve illumination-induced inconsistencies, both of which can introduce conflicting cross-view gradients. These unresolved conflicts may destabilize Gaussian optimization and lead to visible reconstruction artifacts. We propose a conflict-aware 3DGS framework that addresses this problem from both image-space supervision and gradient-level optimization. Semantic Consistency-Guided Masking learns pixel-wise consistency scores to adaptively refine prior masks and suppress unreliable supervision before gradient formation. A dual-view Conflict-Aware Gradient Harmonization strategy further reconciles view-specific gradients by mutually rotating them into an orthogonal configuration, reducing negative directional interference across views. We also introduce conflict-aware densification and pruning to stabilize Gaussian growth and remove persistently conflicting primitives. Extensive experiments on standard in-the-wild benchmarks demonstrate that our method achieves state-of-the-art rendering quality under complex transient distractors and cross-view inconsistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15398v1">3DEditSafe: Defending 3D Editing Pipelines from Unsafe Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      Recent advances in 3D generative editing, particularly pipelines based on 3D Gaussian Splatting (3DGS), have achieved high-fidelity, multi-view-consistent scene manipulation from text prompts. However, we find that these pipelines also introduce new safety risks when unsafe prompts produce edits that are propagated and optimized across views. In this work, we study unsafe generation in 3D editing pipelines and show that such behavior can lead to coherent, undesirable Not-Safe-For-Work (NSFW) content in the final 3D representation. To address this, we propose 3DEditSafe, a safety-regularized 3D editing framework that constrains unsafe semantic propagation during optimization. 3DEditSafe combines generation-stage safety guidance with rendered-view 3D safety regularization, safe semantic projection, residue suppression, and mask-aware preservation to steer optimization away from unsafe editing directions. We evaluate our approach on EditSplat scenes using an object-compatible unsafe prompt benchmark and show that 2D safety guidance alone is not consistently sufficient to prevent unsafe 3D edits. 3DEditSafe reduces unsafe semantic alignment and view-level attack success rates, while revealing a safety-quality tradeoff in which stronger unsafe suppression can introduce artifacts or reduce unsafe-prompt fidelity. To our knowledge, this work is the first attempt to study and defend against unsafe generation in text-driven 3D editing pipelines, highlighting the need for safety mechanisms that operate directly on optimized 3D representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14880v1">Denoising-GS: Gaussian Splatting with Spatial-aware Denoising</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have achieved remarkable success in high-fidelity Novel View Synthesis (NVS), yet the optimization process inevitably introduces noisy Gaussian primitives due to the sparse and incomplete initialization from Structure-from-Motion (SfM) point clouds. Most existing methods focus solely on adjusting the positions of primitives during optimization, while neglecting the underlying spatial structure. To this end, we introduce a new perspective by formulating the optimization of 3DGS as a primitive denoising process and propose Denoising-GS, a spatial-aware denoising framework for Gaussian primitives by taking both the positions and spatial structure into consideration. Specifically, we design an optimizer that preserves the spatial optimization flow of primitives, facilitating coherent and directed denoising rather than random perturbations. Building upon this, the Spatial Gradient-based Denoising strategy jointly considers the spatial supports of primitives to ensure gradient-consistent updates. Furthermore, the Uncertainty-based Denoising module estimates primitive-wise uncertainty to prune redundant or noisy primitives, while the Spatial Coherence Refinement strategy selectively splits primitives in sparse regions to maintain structural completeness. Experiments conducted on three benchmark datasets demonstrate that Denoising-GS consistently enhances NVS fidelity while maintaining representation compactness, achieving state-of-the-art performance across all benchmarks. Source code and models will be made publicly available.
    </details>
</div>
