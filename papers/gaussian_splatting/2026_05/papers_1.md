# gaussian splatting - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01493v1">Splatshot: 3D Face Avatar Generation from a Single Unconstrained Photo</a></div>
    <div class="paper-meta">
      📅 2026-05-31
      | 💬 28 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Reconstructing a photorealistic 3D face avatar from a single unconstrained photograph is challenging: feed-forward 3D Gaussian Splatting (3DGS) models degrade on out-of-distribution inputs, while pretrained diffusion models produce high-fidelity images but lack multi-view consistency. We observe that these paradigms are fundamentally complementary: explicit 3D representations guarantee geometric consistency, whereas 2D diffusion priors ensure photorealism. Building on this, we propose SplatShot, a training-free framework that couples these representations directly within the denoising process. Given a base 3DGS face model and a single reference image, we jointly denoise all target views using a per-step 3D feedback loop. At each timestep, we predict clean images from the noisy latents, refit the 3DGS to these multi-view predictions, and back-propagate the photometric discrepancy between the 3DGS re-renderings and 2D predictions into the noise estimate. This steers the sampling trajectory toward strictly 3D-coherent, identity-faithful outputs. Experiments on diverse in-the-wild images demonstrate that SplatShot produces 3D avatars with superior identity preservation, photorealism, and multi-view consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01458v1">LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World</a></div>
    <div class="paper-meta">
      📅 2026-05-31
      | 💬 https://legsvla.github.io/
    </div>
    <details class="paper-abstract">
      Training vision-language-action (VLA) policies for humanoid loco-manipulation is constrained by the high cost and complexity of collecting human teleoperation demonstrations. VLA policies fine-tuned in simulators have, until now, failed to transfer effectively in humanoid loco-manipulation tasks. We present LEGS (Loco-manipulation via Embodied Gaussian Splatting), a hybrid simulator that composites a mesh foreground (robot, objects, props) over a photorealistic 3D Gaussian Splatting (3DGS) background reconstructed from a handheld scene capture. LEGS uses a procedural motion-primitive generator to synthesize labeled demonstrations at scale without human teleoperation, and a deterministic two-stage color calibration to align the rendered 3DGS image to the robot's deployment camera. On a Unitree G1 humanoid robot, across three pick-and-place tasks of increasing whole-body difficulty and three VLA backbones (psi_0, pi_0.5, GR00T N1.6), a policy trained purely on LEGS data matches or exceeds one trained on human teleoperation demos on every experiment. It also outperforms a mesh-only simulation baseline that ablates the effect of the 3DGS background, showing that photorealistic rendering is a key enabler for synthetic data transfer. Humanoid motion is recorded independently of scene appearance in LEGS, allowing the same auto-generated demonstrations to be re-rendered under new backgrounds and object meshes--covering a new scene at more than 15x lower cost than teleoperation--to augment training data for robustness to scene variations. Under combined object-and-scene appearance shift, the policy trained on re-rendered LEGS-AUG data maintains task success while the baseline trained on teleoperation data fails entirely. Our project page is located at https://legsvla.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01315v1">DeblurNVS: Geometric Latent Diffusion for Novel View Synthesis from Sparse Motion-Blurred Images</a></div>
    <div class="paper-meta">
      📅 2026-05-31
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) is a fundamental problem in computer vision and graphics. Recent advances in neural radiance fields (NeRF), 3D Gaussian Splatting (3DGS), and generative view synthesis have substantially improved its quality. Yet most methods still rely on clean observations, where image structures and cross-view geometric cues are well preserved. Motion blur breaks this assumption by corrupting local details and weakening multi-view correspondences. Such blur commonly arises from camera shake, scene motion, or finite exposure in practical capture. Blur-aware NVS methods address this degradation by modeling image formation, but their reliance on costly per-scene optimization limits efficient and generalizable sparse-view synthesis. To address this, we propose DeblurNVS, a novel framework for synthesizing high-fidelity novel views directly from sparse motion-blurred images, without requiring per-scene optimization. DeblurNVS restores the intermediate geometric representations needed for multi-view reasoning, enabling blurred inputs to recover reliable structure and correspondence cues. The restored representations are then combined with target camera information to synthesize the target-view representation and reconstruct a sharp RGB novel view. To enable the large-scale training, we construct a motion-blurred NVS dataset from DL3DV-10K using interpolation-based finite-exposure blur synthesis. Extensive experiments demonstrate that DeblurNVS outperforms existing baselines on synthetic motion-blur benchmarks and generalizes to real motion-blurred scenes, producing perceptually sharper and structurally more stable novel views while avoiding costly per-scene optimization. Project page: https://github.com/PKU-YuanGroup/DeblurNVS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.15491v5">GSDeformer: Direct, Real-time and Extensible Cage-based Deformation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-31
      | 💬 Project Page: https://jhuangbu.github.io/gsdeformer, Video: https://www.youtube.com/watch?v=-ecrj48-MqM
    </div>
    <details class="paper-abstract">
      We present GSDeformer, a method that enables cage-based deformation on 3D Gaussian Splatting (3DGS). Our approach bridges cage-based deformation and 3DGS by using a proxy point-cloud representation. This point cloud is generated from 3D Gaussians, and deformations applied to the point cloud are translated into transformations on the 3D Gaussians. To handle potential bending caused by deformation, we incorporate a splitting process to approximate it. Our method does not modify or extend the core architecture of 3D Gaussian Splatting, making it compatible with any trained vanilla 3DGS or its variants. Additionally, we automate cage construction for 3DGS and its variants using a render-and-reconstruct approach. Experiments demonstrate that GSDeformer delivers superior deformation results compared to existing methods, is robust under extreme deformations, requires no retraining for editing, runs in real-time, and can be extended to other 3DGS variants. Project Page: https://jhuangbu.github.io/gsdeformer/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06989v3">MipSLAM: Alias-Free Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-05-31
      | 💬 Accepted to ICRA 2026
    </div>
    <details class="paper-abstract">
      This paper introduces MipSLAM, a frequency-aware 3D Gaussian Splatting (3DGS) SLAM framework capable of high-fidelity anti-aliased novel view synthesis and robust pose estimation under varying camera configurations. Existing 3DGS-based SLAM systems often suffer from aliasing artifacts and trajectory drift due to inadequate filtering and purely spatial optimization. To overcome these limitations, we propose an Elliptical Adaptive Anti-aliasing (EAA) algorithm that approximates Gaussian contributions via geometry-aware numerical integration, avoiding costly analytic computation. Furthermore, we present a Spectral-Aware Pose Graph Optimization (SA-PGO) module that reformulates trajectory estimation in the frequency domain, effectively suppressing high-frequency noise and drift through graph Laplacian analysis. Extensive evaluations on Replica and TUM datasets demonstrate that MipSLAM achieves state-of-the-art rendering quality and localization accuracy across multiple resolutions. Code is available at https://github.com/yzli1998/MipSLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07806v2">Multi-view Pyramid Transformer: Look Coarser to See Broader</a></div>
    <div class="paper-meta">
      📅 2026-05-31
      | 💬 Project page: see https://gynjn.github.io/MVP/
    </div>
    <details class="paper-abstract">
      We propose Multi-view Pyramid Transformer (MVP), a scalable multi-view transformer architecture that directly reconstructs large 3D scenes from tens to hundreds of images in a single forward pass. Drawing on the idea of ``looking broader to see the whole, looking finer to see the details," MVP is built on two core design principles: 1) a local-to-global inter-view hierarchy that gradually broadens the model's perspective from local views to groups and ultimately the full scene, and 2) a fine-to-coarse intra-view hierarchy that starts from detailed spatial representations and progressively aggregates them into compact, information-dense tokens. This dual hierarchy achieves both computational efficiency and representational richness, enabling fast reconstruction of large and complex scenes. We validate MVP on diverse datasets and show that, when coupled with 3D Gaussian Splatting as the underlying 3D representation, it achieves state-of-the-art generalizable reconstruction quality while maintaining high efficiency and scalability across a wide range of view configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.23277v3">iLRM: An Iterative Large 3D Reconstruction Model</a></div>
    <div class="paper-meta">
      📅 2026-05-31
      | 💬 Project page: https://gynjn.github.io/iLRM/
    </div>
    <details class="paper-abstract">
      Feed-forward 3D modeling has emerged as a promising approach for rapid and high-quality 3D reconstruction. In particular, directly generating explicit 3D representations, such as 3D Gaussian splatting, has attracted significant attention due to its fast and high-quality rendering. However, many state-of-the-art methods, primarily based on transformer architectures, suffer from severe scalability issues because they rely on full attention across image tokens from multiple input views, resulting in prohibitive computational costs as the number of views or image resolution increases. Toward a scalable and efficient feed-forward 3D reconstruction, we introduce an iterative Large 3D Reconstruction Model (iLRM) that generates 3D Gaussian representations through an iterative refinement mechanism, guided by three core principles: (1) decoupling the scene representation from input images to enable compact 3D representations; (2) decomposing global multi-view interactions into a two-stage attention scheme to reduce computational costs; and (3) injecting high-resolution information at every layer to achieve high-fidelity reconstruction. Experimental results on widely used datasets, such as RE10K and DL3DV, demonstrate that iLRM outperforms existing methods in both reconstruction quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19065v3">A Geometric Algebra-Informed 3DGS Framework for Wireless Channel Prediction</a></div>
    <div class="paper-meta">
      📅 2026-05-31
      | 💬 Accepted by CVPR 2026
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Geometric Algebra-Informed 3D Gaussian Splatting (GAI-GS), a framework for wireless modeling that couples 3D Gaussian splatting with a geometric algebra-based attention mechanism to explicitly model ray-object interactions in complex propagation environments. GAI-GS encodes joint spatial-electromagnetic (EM) relations into token representations, enabling scene-level aggregation within a unified, end-to-end neural architecture. This design grounds wireless ray propagation in electromagnetic principles, allowing token interactions to model key effects such as multipath, attenuation, and reflection/diffraction. Through extensive evaluations on multiple real-world indoor datasets, GAI-GS consistently surpasses current baselines across various wireless tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00817v1">Directed Distance Fields for Constant-Time Ray Queries on Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-30
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) renders new views of a scene in real time. Like every rasterizer, it answers only primary rays, the rays from the camera through the image. It cannot trace the secondary rays that shadows, ambient occlusion, and global illumination need. We turn a trained 3DGS scene into a ray oracle by distilling a Directed Distance Function (DDF). The DDF is a small neural field. It takes a ray, given by an origin and a direction, and returns the distance to the first surface and whether the ray hits anything. Each query is one forward pass. The field is 52~MB, and its size does not depend on the number of Gaussians, so its cost and memory stay flat as the scene grows. We make three points. First, we study what supervision a DDF needs. Depth rendered from the Gaussians is too blurry to teach thin parts, while clean distance supervision recovers them. Second, we measure speed. The DDF is 26 to 72 times faster than sphere tracing an equivalent signed distance field, and unlike a bounding volume hierarchy built over the Gaussians, even on dedicated RT-core hardware, its query time and memory do not grow with the scene. Third, we show a pipeline that needs no mesh: images give a 3DGS scene, a neural surface gives clean distances, and the DDF learns from them. We use the DDF as a secondary-ray oracle for global illumination. It reproduces reference ray-traced shadows at 30.3~dB and ambient occlusion at 21.3~dB across 142 objects, and on real captured scenes. Our codes are available at https://github.com/smlab-niser/ddf-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09877v2">Genie 4D: Semantic-Prior-Guided 4D Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-30
    </div>
    <details class="paper-abstract">
      At the intersection of computer vision and robotic perception, 4D reconstruction of dynamic scenes connects low-level geometric sensing with high-level semantic understanding. We present Genie 4D, a framework that turns hand-held phone capture into a semantically grounded, action-controllable 4D world model. Genie 4D couples a real-time visual-inertial Gaussian splatting front-end for metric geometry with a feed-forward 4D backbone regularized by frozen DINOv3 features acting as structural priors. The semantic priors suppress identity drift during dynamic tracking, while a short conditional diffusion refiner recovers high-frequency surface detail that regression backbones smooth away. Finally, a lightweight latent-action head exposes the reconstructed 4D state to a Genie-style world model trained with a JEPA-style next-embedding objective, so that the scene can be rolled forward under user actions. On the Point Odyssey and TUM-Dynamics benchmarks, Genie 4D retains the linear time complexity O(T) of feed-forward baselines while improving 3D tracking accuracy (APD) and reconstruction completeness, and it runs interactively on a single consumer GPU (RTX 5090) from iPhone, Mac, Windows, and Linux capture clients. Genie 4D offers a practical, semantic-prior-guided path toward physically grounded world models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00452v1">Beyond Static Gaussians: An Empirical Investigation of Architectural Paradigms for Dynamic 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-30
      | 💬 Accepted in Journal of Computational Vision and Imaging Systems (JCVIS)
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction via 3D Gaussian Splatting (3DGS) has emerged as a compelling approach for representing evolving environments, yet understanding trade-offs between methodologies remains crucial. This paper presents a comprehensive analysis of dynamic 3DGS methods, categorizing them into two paradigms: structure-guided methods employing auxiliary representations (deformation fields, canonical spaces, grids) to model temporal changes, and gaussian-centric methods encoding dynamics directly into primitives via continuous functions or 4D representations. We evaluate representative methods from both paradigms on the D-NeRF benchmark. Our findings reveal that structure-guided methods achieve superior reconstruction fidelity and compact model sizes, while gaussian-centric approaches demonstrate significantly higher rendering speeds enabling real-time performance, though with greater quality variability and potentially substantial storage overhead. This analysis highlights a fundamental trade-off between reconstruction quality/compactness versus rendering speed, providing insights to guide future research and application development in dynamic scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00450v1">Optimizing 3D Gaussian Splatting via Point Cloud Upsampling</a></div>
    <div class="paper-meta">
      📅 2026-05-30
      | 💬 Accepted in Journal of Computational Vision and Imaging Systems (JCVIS)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a technique for creating and rendering 3D scenes, however its performance depends heavily on the quality of initial seed points. To improve 3DGS initialization, this study presents and evaluates several point cloud upsampling approaches: linear interpolation, triangular interpolation, spline-based surface reconstruction, moving least squares surface fitting, and Voronoi-based point generation. Additionally, this research introduces a depth-guided point lifting method that leverages depth maps to maintain geometric consistency with Structure-from-Motion (SfM) reconstructions. Through extensive experiments on the Mip-NeRF360 and Replica datasets, the proposed methods demonstrate improvements in reconstruction quality across diverse scene types. Results indicate that different upsampling strategies excel in different scenarios: surface reconstruction methods perform better with organic, detailed scenes, while simpler interpolation approaches are more suited for scenes dominated by piecewise-smooth geometries. In comparison, the depth-guided approach shows promise for adding geometry-aware points across the entire scene, importantly in texture-less regions. These findings, which provide preliminary practical guidelines for selecting appropriate upsampling methods based on scene characteristics and computational constraints, advances the understanding of how point cloud initialization affects 3DGS quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00447v1">GeoSAM-3D: Geodesic Prompt Propagation for Open-Vocabulary 3D Scene Segmentation from Monocular Video</a></div>
    <div class="paper-meta">
      📅 2026-05-30
    </div>
    <details class="paper-abstract">
      Open-vocabulary 3D scene segmentation usually assumes RGB-D video, calibrated multi-view imagery, or a reconstructed mesh. GeoSAM-3D studies a lighter setting: a user uploads a short monocular video, clicks or names an object in one frame, and receives a propagated 3D mask over a Gaussian scene. The implementation combines frozen image and video foundation models with a monocular 3D Gaussian Splatting reconstruction and a differentiable graph-geodesic propagation kernel over Gaussian centroids. The central design choice is to propagate prompts by heat-kernel distance on the reconstructed scene graph, rather than by Euclidean nearest neighbors in 3D. This preserves continuity around curved surfaces and reduces leakage across nearby but disconnected objects. This paper describes the repository state, the mathematical kernel implemented in geosam3d.propagate, the feature head trained from Segment Anything masks, and the validation already present in the codebase. The evaluation protocol separates implementation validation, graph propagation quality, leakage control, and interactive latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00444v1">Real-Time Physics Simulation with Dynamic Mesh-Gaussian Reconstructions</a></div>
    <div class="paper-meta">
      📅 2026-05-30
    </div>
    <details class="paper-abstract">
      Integrating dynamic 3D reconstructions into physics simulation requires fixed mesh topology for efficient collision detection, but state-of-the-art methods like DG-Mesh produce varying topology optimized for geometric quality. We investigate whether topology conversion can enable physics integration while preserving reconstruction fidelity. We propose a dual-representation framework combining fixed-topology meshes for physics with Gaussian splatting for rendering, achieving 4.65$\times$ speedup over varying-topology baselines through runtime vertex buffer updates. We evaluate two conversion strategies, temporal correspondence tracking and template-based projection, against native fixed-topology methods (MaGS) on the DG-Mesh dataset. Our evaluation reveals that both conversion approaches incur 65-80% geometric degradation, producing results inferior to MaGS despite DG-Mesh's superior initial quality. This demonstrates that high-quality reconstruction and physics-compatible topology represent fundamentally distinct objectives that cannot be reconciled through post-processing. Our findings inform future development of physics-aware reconstruction methods and our framework enables real-time simulation with any fixed-topology approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00352v1">HiGS: A Hierarchical Rendering Architecture for Real-Time 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-29
      | 💬 Project Page: https://research.nvidia.com/labs/sil/projects/higs/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become the standard for real-time novel view synthesis on commodity GPUs. Its pipeline ties spatial partitioning and rasterization to one tile size, yet the two pull in opposite directions: partitioning, which bins and depth-sorts gaussians, grows cheaper with larger tiles, while rasterization gets cheaper with smaller ones. Prior acceleration work reduces the cost of individual stages but keeps both locked to that single scale, where a few dense tiles dominate frame time. We present Hierarchically Tiled Gaussian Splatting (HiGS), which gives each its own scale: partitioning runs over coarse macro-tiles, while rasterization runs over the fine render tiles within them. Rasterization work is then issued in proportion to the gaussians in each macro-tile rather than per tile, so dense regions spread across many parallel units instead of serializing through one. Across tested scenes, HiGS renders up to 15.8x faster than the original 3DGS and outperforms every other rasterizer we evaluate, while preserving exact front-to-back alpha compositing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31419v1">Triangle Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-05-29
      | 💬 26 pages, 11 figures
    </div>
    <details class="paper-abstract">
      We present a dense RGB-D SLAM system using differentiable triangles as the 3D map representation. While 3D Gaussian Splatting has emerged as the leading method for novel-view synthesis, triangles remain the standard primitive for traditional rendering hardware, game engines, and downstream tasks requiring explicit geometry such as simulation, collision, and editing. Recent offline methods have demonstrated that an unstructured 'triangle soup' can be optimised into a photorealistic mesh via Delaunay triangulation across a set of posed images. Building upon this insight, we present the first dense SLAM system to employ Triangle Splatting to perform both tracking and mapping through online differentiable rendering of a triangle soup. The map can be converted into a connected mesh on-the-fly via restricted Delaunay triangulation, enabling new online capabilities such as mesh deformation and collision checking. On Replica and TUM-RGBD, our system outperforms baselines on 3D geometry, matches the camera-tracking accuracy, and enables online mesh-based scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31376v1">LiftNav: Path Planning via Semantic Lifting in TSDF-Guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-29
    </div>
    <details class="paper-abstract">
      Autonomous robots in unknown indoor environments require both reliable collision avoidance and object-level understanding. Classical representations such as TSDF support safe planning but lack semantics, while photorealistic methods like Gaussian Splatting (GS) provide rich appearance yet suffer from soft geometry, limiting precise obstacle avoidance. We present LiftNav, a hybrid navigation framework built on GSFusion's TSDF+GS dual map, augmented with a real-time pipeline of YOLO-based detection, TSDF-based 3D lifting, and B-spline trajectory optimization. This design enables flexible semantic navigation without dense 3D embeddings. We further introduce a hinge-loss-based collision penalty that improves trajectory smoothness and safety. We evaluate our approach in a simulation using the Replica dataset. Compared against a state-of-the-art radiance field baseline we show a 100% feasibility rate and shorter trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09364v2">VAD-GS: Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-29
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has demonstrated impressive performance in synthesizing high-fidelity novel views. Nonetheless, its effectiveness critically depends on the quality of the initialized point cloud. Specifically, achieving uniform and complete point coverage over the underlying scene structure requires overlapping observation frustums, an assumption that is often violated in unbounded, dynamic urban environments. Training Gaussian models with partially initialized point clouds often leads to distortions and artifacts, as camera rays may fail to intersect valid surfaces, resulting in incorrect gradient propagation to Gaussian primitives associated with occluded or invisible geometry. Additionally, existing densification strategies simply clone and split Gaussian primitives from existing ones, incapable of reconstructing geometry from missing structures. To address these limitations, we propose VAD-GS, a 3DGS framework tailored for geometry recovery in challenging urban scenes. Our method identifies unreliable geometry structures via voxel-based visibility reasoning, selects informative supporting views through diversity-aware view selection, and recovers missing structures via multi-view stereo reconstruction. This design enables the generation of new Gaussian primitives guided by reliable geometric priors, even in regions lacking initial points. Extensive experiments on the Waymo and nuScenes datasets demonstrate that VAD-GS outperforms state-of-the-art 3DGS approaches and significantly improves the quality of reconstructed geometry for both static and dynamic objects. Our project webpage is at mias.group/VAD-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30987v1">Benchmarking Single-Step Inpainting Methods for Multi-Object 3D Gaussian Splatting Scenes</a></div>
    <div class="paper-meta">
      📅 2026-05-29
      | 💬 Accepted as an extended abstract to the CVEU Workshop at CVPR 2026
    </div>
    <details class="paper-abstract">
      The tasks of object removal and inpainting 3D Gaussian Splatting (3DGS) scenes face challenges such as 3D consistency across camera views. In comparing 2D inpainters and their suitability for the 3D domain, we find that reconstruction-based inpainters outperform generative diffusion models in 3D consistency. Integrating these 2D inpainters into different single-step methods for creating and finetuning 3DGS scenes, our results indicate that initializing the scene from scratch produces higher quality results than finetuning the existing scene. Using a state-of-the-art generative 2D inpainter, we create a straightforward baseline to underline the importance of object removal before inpainting in the 3D setting. Since 360° datasets rarely include real-world ground truths, and challenging occlusion scenarios are equally sparse, we introduce a novel multi-object scene with recorded ground truth data and many views with object occlusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.03337v2">FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles</a></div>
    <div class="paper-meta">
      📅 2026-05-29
      | 💬 Project page: https://yklcs.com/ftgspp
    </div>
    <details class="paper-abstract">
      The recent surge in 4D Gaussian Splatting (4DGS) has achieved impressive dynamic scene reconstruction. While these methods demonstrate remarkable performance, the specific drivers behind such gains remain less explored, making a systematic understanding of the underlying principles challenging. In this paper, we perform a comprehensive analysis of these hidden factors to provide a clearer perspective on the 4DGS framework. We first establish a controlled baseline, FreeTimeGS_ours, by formalizing and reproducing the heuristics of the state-of-the-art FreeTimeGS. Using this framework, we dissect 4DGS along its fundamental axes and uncover key secrets, including the emergent temporal partitioning driven by Gaussian durations and the discrepancy between photometric fidelity and spatiotemporal consistency. Based on these insights, we propose FreeTimeGS++, a principled method that employs gated marginalization and neural velocity fields to achieve superior stability and robust dynamic representations. Our approach yields reproducible results with reduced run-to-run variance. We will release our implementation to provide a reliable foundation for future 4DGS research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09632v4">X-GS: An Extensible Framework for Perceiving and Thinking via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-29
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods operate in isolation, focusing on specific domains. In this paper, we introduce X-GS, an extensible framework consisting of two major components. The X-GS-Perceiver unifies a broad range of 3DGS techniques to enable real-time online SLAM with semantic distillation. The X-GS-Thinker accommodates multimodal models, enabling them to seamlessly interface with the Perceiver to complete downstream tasks. In our implementation of X-GS, the Perceiver leverages the latest vision foundation models to improve online SLAM performance and employs three key mechanisms to accelerate semantic distillation. The Thinker can be built upon both contrastive and generative vision-language models and utilizes the Perceiver's semantic Gaussian splats to unlock capabilities such as 3D visual grounding and scene captioning. Experimental results on diverse benchmarks demonstrate the efficiency and newly unlocked multimodal capabilities of the X-GS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30863v1">DSD-GS: Dynamic-Static Decomposition of Gaussian Splatting for Efficient and High-Fidelity Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-29
      | 💬 23 pages, 9 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction and novel view synthesis are fundamental to next-generation visual intelligence applications such as virtual reality, robotics, and digital twins. However, high-fidelity reconstruction of complex, time-varying scenes from arbitrary viewpoints remains a significant challenge. Existing dynamic 3DGS methods suffer from computational inefficiency, since they model all Gaussians as dynamic components. While recent decomposition-based approaches address this issue, they still struggle with degraded reconstruction quality and prolonged training time. To mitigate these limitations, we propose a novel dynamic reconstruction framework built upon an efficient static-dynamic decomposition strategy using a Feed-Forward Gaussian Splatting encoder and an optical flow model. By eliminating redundant computations on static regions, our method achieves state-of-the-art performance, outperforming existing baselines across rendering quality, training and rendering speed, and storage efficiency. Notably, on the Neural 3D dataset, our framework requires only 10 minutes for training and achieves a rendering speed of over 700 FPS on a single NVIDIA RTX 5090 GPU at resolution of 1352x1014. Furthermore, our decomposition strategy eliminates the need for COLMAP preprocessing and enables deterministic initialization, thereby enhancing both efficiency and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18898v2">DeblurSplat: SfM-free 3D Gaussian Splatting with Event Camera for Robust Deblurring</a></div>
    <div class="paper-meta">
      📅 2026-05-29
      | 💬 Accepted by TMM 2026
    </div>
    <details class="paper-abstract">
      In this paper, we propose the first Structure-from-Motion (SfM)-free deblurring 3D Gaussian Splatting method via event camera, dubbed DeblurSplat. We address the motion-deblurring problem in two ways. First, we leverage the pretrained capability of the dense stereo module (DUSt3R) to directly obtain accurate initial point clouds from blurred images. Without calculating camera poses as an intermediate result, we avoid the cumulative errors transfer from inaccurate camera poses to the initial point clouds' positions. Second, we introduce the event stream into the deblur pipeline for its high sensitivity to dynamic change. By decoding the latent sharp images from the event stream and blurred images, we can provide a fine-grained supervision signal for scene reconstruction optimization. Extensive experiments across a range of scenes demonstrate that DeblurSplat not only excels in generating high-fidelity novel views but also achieves significant rendering efficiency compared to the SOTAs in deblur 3D-GS.
    </details>
</div>
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
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.00137v1">Advances in Neural 3D Mesh Texturing: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 Eurographics STAR (Computer Graphics Forum), 2026. Project Page: https://sairajk.github.io/neural-mesh-texturing/
    </div>
    <details class="paper-abstract">
      Texturing 3D meshes plays a vital role in determining the visual realism of digital objects and scenes. Although recent generative 3D approaches based on Neural Radiance Fields and Gaussian Splatting can produce textured assets directly, polygonal meshes remain the core representation across modeling, animation, visual effects, and gaming pipelines. Neural 3D mesh texturing therefore continues to be an essential and active area of research. In this survey, we present a comprehensive review of recent advances in neural 3D mesh texturing, covering methods for texture synthesis, transfer, and completion. We first summarize key foundations in mesh geometry, texture mapping, differentiable rendering, and neural generative models, and then organize the literature into a unified taxonomy spanning early GAN-based methods to modern diffusion-based pipelines. We further analyze common architectures and supervision strategies, review datasets and evaluation protocols, and discuss emerging applications, practical/commercial systems, and open challenges. Together, these insights provide a structured perspective on the current landscape and help guide future developments in learning-based 3D mesh texturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26262v3">Semantic Foam: Unifying Spatial and Semantic Scene Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-05-28
      | 💬 15 pages, 10 figures, Accepted to CVPR 2026 (Highlight) , Project page: http://semanticfoam.github.io/
    </div>
    <details class="paper-abstract">
      Modern scene reconstruction methods, such as 3D Gaussian Splatting, deliver photo-realistic novel view synthesis at real-time speeds, yet their adoption in interactive graphics applications has been limited. A major bottleneck is the difficulty of interacting with these representations compared to traditional, human-authored 3D assets. While previous research has attempted to impose semantic decomposition on these models, significant challenges remain regarding segmentation quality and consistency. To address this, we introduce Semantic Foam, extending the recently proposed Radiant Foam representations to semantic decomposition tasks. Our approach integrates the natural spatial volumetric decomposition of Radiant Foam's Voronoi mesh with an explicit semantic feature field parameterized at the cell level. This explicit structure enables direct spatial regularization, which prevents artifacts caused by occlusion or inconsistent supervision across views - common pitfalls for other point-based representations. Experimental results show that our method achieves comparable or superior object-level segmentation performance compared to state-of-the-art methods like Gaussian Grouping and SAGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30396v1">Smaller and Faster 3DGS via Post-Training Dictionary Learning</a></div>
    <div class="paper-meta">
      📅 2026-05-28
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a promising neural scene representation for real-time rendering, but trained models often suffer from large memory footprints, limiting deployment on less powerful devices. Existing compression techniques often lead to architectures with several additional trainable parameters. While achieving outstanding compression ratios, they introduce noticeable drops in image quality. In this work, we introduce the first dictionary-learning-based compression framework for 3DGS. The proposed post-training compression pipeline can be deployed in virtually any 3DGS model without the need for re-training or modifications to existing 3DGS models. Our compression framework is straightforward to implement, yet provides significant compression capabilities, preserves image quality, and improves real-time rendering performance. Across 13 benchmark scenes, our approach achieves an average compression ratio of 3.95x, 3.10x, and 4.55x when applied to 3DGS, 3DGS-MCMC, and PixelGS, respectively. This yields consistent rendering speedups of 23.3%, 24.3%, and 25.3%, while maintaining image quality.
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
