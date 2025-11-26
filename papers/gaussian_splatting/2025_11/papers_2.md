# gaussian splatting - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12370v1">Changes in Real Time: Online Scene Change Detection with Multi-View Fusion</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Online Scene Change Detection (SCD) is an extremely challenging problem that requires an agent to detect relevant changes on the fly while observing the scene from unconstrained viewpoints. Existing online SCD methods are significantly less accurate than offline approaches. We present the first online SCD approach that is pose-agnostic, label-free, and ensures multi-view consistency, while operating at over 10 FPS and achieving new state-of-the-art performance, surpassing even the best offline approaches. Our method introduces a new self-supervised fusion loss to infer scene changes from multiple cues and observations, PnP-based fast pose estimation against the reference scene, and a fast change-guided update strategy for the 3D Gaussian Splatting scene representation. Extensive experiments on complex real-world datasets demonstrate that our approach outperforms both online and offline baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12304v1">LiDAR-GS++:Improving LiDAR Gaussian Reconstruction via Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted by AAAI-26
    </div>
    <details class="paper-abstract">
      Recent GS-based rendering has made significant progress for LiDAR, surpassing Neural Radiance Fields (NeRF) in both quality and speed. However, these methods exhibit artifacts in extrapolated novel view synthesis due to the incomplete reconstruction from single traversal scans. To address this limitation, we present LiDAR-GS++, a LiDAR Gaussian Splatting reconstruction method enhanced by diffusion priors for real-time and high-fidelity re-simulation on public urban roads. Specifically, we introduce a controllable LiDAR generation model conditioned on coarsely extrapolated rendering to produce extra geometry-consistent scans and employ an effective distillation mechanism for expansive reconstruction. By extending reconstruction to under-fitted regions, our approach ensures global geometric consistency for extrapolative novel views while preserving detailed scene surfaces captured by sensors. Experiments on multiple public datasets demonstrate that LiDAR-GS++ achieves state-of-the-art performance for both interpolated and extrapolated viewpoints, surpassing existing GS and NeRF-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.08305v4">ELECTRA: A Cartesian Network for 3D Charge Density Prediction with Floating Orbitals</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 10 pages, 4 figures, 5 tables
    </div>
    <details class="paper-abstract">
      We present the Electronic Tensor Reconstruction Algorithm (ELECTRA) - an equivariant model for predicting electronic charge densities using floating orbitals. Floating orbitals are a long-standing concept in the quantum chemistry community that promises more compact and accurate representations by placing orbitals freely in space, as opposed to centering all orbitals at the position of atoms. Finding the ideal placement of these orbitals requires extensive domain knowledge, though, which thus far has prevented widespread adoption. We solve this in a data-driven manner by training a Cartesian tensor network to predict the orbital positions along with orbital coefficients. This is made possible through a symmetry-breaking mechanism that is used to learn position displacements with lower symmetry than the input molecule while preserving the rotation equivariance of the charge density itself. Inspired by recent successes of Gaussian Splatting in representing densities in space, we are using Gaussian orbitals and predicting their weights and covariance matrices. Our method achieves a state-of-the-art balance between computational efficiency and predictive accuracy on established benchmarks. Furthermore, ELECTRA is able to lower the compute time required to arrive at converged DFT solutions - initializing calculations using our predicted densities yields an average 50.72 % reduction in self-consistent field (SCF) iterations on unseen molecules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12040v1">SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 AAAI2026-Oral. Project Page: https://xinyuanhu66.github.io/SRSplat/
    </div>
    <details class="paper-abstract">
      Feed-forward 3D reconstruction from sparse, low-resolution (LR) images is a crucial capability for real-world applications, such as autonomous driving and embodied AI. However, existing methods often fail to recover fine texture details. This limitation stems from the inherent lack of high-frequency information in LR inputs. To address this, we propose \textbf{SRSplat}, a feed-forward framework that reconstructs high-resolution 3D scenes from only a few LR views. Our main insight is to compensate for the deficiency of texture information by jointly leveraging external high-quality reference images and internal texture cues. We first construct a scene-specific reference gallery, generated for each scene using Multimodal Large Language Models (MLLMs) and diffusion models. To integrate this external information, we introduce the \textit{Reference-Guided Feature Enhancement (RGFE)} module, which aligns and fuses features from the LR input images and their reference twin image. Subsequently, we train a decoder to predict the Gaussian primitives using the multi-view fused feature obtained from \textit{RGFE}. To further refine predicted Gaussian primitives, we introduce \textit{Texture-Aware Density Control (TADC)}, which adaptively adjusts Gaussian density based on the internal texture richness of the LR inputs. Extensive experiments demonstrate that our SRSplat outperforms existing methods on various datasets, including RealEstate10K, ACID, and DTU, and exhibits strong cross-dataset and cross-resolution generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16410v2">REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Bridging the gap between complex human instructions and precise 3D object grounding remains a significant challenge in vision and robotics. Existing 3D segmentation methods often struggle to interpret ambiguous, reasoning-based instructions, while 2D vision-language models that excel at such reasoning lack intrinsic 3D spatial understanding. In this paper, we introduce REALM, an innovative MLLM-agent framework that enables open-world reasoning-based segmentation without requiring extensive 3D-specific post-training. We perform segmentation directly on 3D Gaussian Splatting representations, capitalizing on their ability to render photorealistic novel views that are highly suitable for MLLM comprehension. As directly feeding one or more rendered views to the MLLM can lead to high sensitivity to viewpoint selection, we propose a novel Global-to-Local Spatial Grounding strategy. Specifically, multiple global views are first fed into the MLLM agent in parallel for coarse-level localization, aggregating responses to robustly identify the target object. Then, several close-up novel views of the object are synthesized to perform fine-grained local segmentation, yielding accurate and consistent 3D masks. Extensive experiments show that REALM achieves remarkable performance in interpreting both explicit and implicit instructions across LERF, 3D-OVS, and our newly introduced REALM3D benchmarks. Furthermore, our agent framework seamlessly supports a range of 3D interaction tasks, including object removal, replacement, and style transfer, demonstrating its practical utility and versatility. Project page: https://ChangyueShi.github.io/REALM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11231v1">3D Gaussian and Diffusion-Based Gaze Redirection</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      High-fidelity gaze redirection is critical for generating augmented data to improve the generalization of gaze estimators. 3D Gaussian Splatting (3DGS) models like GazeGaussian represent the state-of-the-art but can struggle with rendering subtle, continuous gaze shifts. In this paper, we propose DiT-Gaze, a framework that enhances 3D gaze redirection models using a novel combination of Diffusion Transformer (DiT), weak supervision across gaze angles, and an orthogonality constraint loss. DiT allows higher-fidelity image synthesis, while our weak supervision strategy using synthetically generated intermediate gaze angles provides a smooth manifold of gaze directions during training. The orthogonality constraint loss mathematically enforces the disentanglement of internal representations for gaze, head pose, and expression. Comprehensive experiments show that DiT-Gaze sets a new state-of-the-art in both perceptual quality and redirection accuracy, reducing the state-of-the-art gaze error by 4.1% to 6.353 degrees, providing a superior method for creating synthetic training data. Our code and models will be made available for the research community to benchmark against.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11213v1">RealisticDreamer: Guidance Score Distillation for Few-shot Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently gained great attention in the 3D scene representation for its high-quality real-time rendering capabilities. However, when the input comprises sparse training views, 3DGS is prone to overfitting, primarily due to the lack of intermediate-view supervision. Inspired by the recent success of Video Diffusion Models (VDM), we propose a framework called Guidance Score Distillation (GSD) to extract the rich multi-view consistency priors from pretrained VDMs. Building on the insights from Score Distillation Sampling (SDS), GSD supervises rendered images from multiple neighboring views, guiding the Gaussian splatting representation towards the generative direction of VDM. However, the generative direction often involves object motion and random camera trajectories, making it challenging for direct supervision in the optimization process. To address this problem, we introduce an unified guidance form to correct the noise prediction result of VDM. Specifically, we incorporate both a depth warp guidance based on real depth maps and a guidance based on semantic image features, ensuring that the score update direction from VDM aligns with the correct camera pose and accurate geometry. Experimental results show that our method outperforms existing approaches across multiple datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11175v1">Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      Multi-view video reconstruction plays a vital role in computer vision, enabling applications in film production, virtual reality, and motion analysis. While recent advances such as 4D Gaussian Splatting (4DGS) have demonstrated impressive capabilities in dynamic scene reconstruction, they typically rely on the assumption that input video streams are temporally synchronized. However, in real-world scenarios, this assumption often fails due to factors like camera trigger delays or independent recording setups, leading to temporal misalignment across views and reduced reconstruction quality. To address this challenge, a novel temporal alignment strategy is proposed for high-quality 4DGS reconstruction from unsynchronized multi-view videos. Our method features a coarse-to-fine alignment module that estimates and compensates for each camera's time shift. The method first determines a coarse, frame-level offset and then refines it to achieve sub-frame accuracy. This strategy can be integrated as a readily integrable module into existing 4DGS frameworks, enhancing their robustness when handling asynchronous data. Experiments show that our approach effectively processes temporally misaligned videos and significantly enhances baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03180v2">Duplex-GS: Proxy-Guided Weighted Blending for Real-Time Order-Independent Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 submitted to TCSVT
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated remarkable rendering fidelity and efficiency. However, these methods still rely on computationally expensive sequential alpha-blending operations, resulting in significant overhead, particularly on resource-constrained platforms. In this paper, we propose Duplex-GS, a dual-hierarchy framework that integrates proxy Gaussian representations with order-independent rendering techniques to achieve photorealistic results while sustaining real-time performance. To mitigate the overhead caused by view-adaptive radix sort, we introduce cell proxies for local Gaussians management and propose cell search rasterization for further acceleration. By seamlessly combining our framework with Order-Independent Transparency (OIT), we develop a physically inspired weighted sum rendering technique that simultaneously eliminates "popping" and "transparency" artifacts, yielding substantial improvements in both accuracy and efficiency. Extensive experiments on a variety of real-world datasets demonstrate the robustness of our method across diverse scenarios, including multi-scale training views and large-scale environments. Our results validate the advantages of the OIT rendering paradigm in Gaussian Splatting, achieving high-quality rendering with an impressive 1.5 to 4 speedup over existing OIT based Gaussian Splatting approaches and 52.2% to 86.9% reduction of the radix sort overhead without quality degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11048v1">PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted at AAAI 2026. Supplementary material included after references. 27 pages, 21 figures, 11 tables
    </div>
    <details class="paper-abstract">
      4D flow magnetic resonance imaging (MRI) is a reliable, non-invasive approach for estimating blood flow velocities, vital for cardiovascular diagnostics. Unlike conventional MRI focused on anatomical structures, 4D flow MRI requires high spatiotemporal resolution for early detection of critical conditions such as stenosis or aneurysms. However, achieving such resolution typically results in prolonged scan times, creating a trade-off between acquisition speed and prediction accuracy. Recent studies have leveraged physics-informed neural networks (PINNs) for super-resolution of MRI data, but their practical applicability is limited as the prohibitively slow training process must be performed for each patient. To overcome this limitation, we propose PINGS-X, a novel framework modeling high-resolution flow velocities using axes-aligned spatiotemporal Gaussian representations. Inspired by the effectiveness of 3D Gaussian splatting (3DGS) in novel view synthesis, PINGS-X extends this concept through several non-trivial novel innovations: (i) normalized Gaussian splatting with a formal convergence guarantee, (ii) axes-aligned Gaussians that simplify training for high-dimensional data while preserving accuracy and the convergence guarantee, and (iii) a Gaussian merging procedure to prevent degenerate solutions and boost computational efficiency. Experimental results on computational fluid dynamics (CFD) and real 4D flow MRI datasets demonstrate that PINGS-X substantially reduces training time while achieving superior super-resolution accuracy. Our code and datasets are available at https://github.com/SpatialAILab/PINGS-X.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16533v2">Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a high-fidelity and efficient paradigm for online free-viewpoint video (FVV) reconstruction, offering viewers rapid responsiveness and immersive experiences. However, existing online methods face challenge in prohibitive storage requirements primarily due to point-wise modeling that fails to exploit the motion properties. To address this limitation, we propose a novel Compact Gaussian Streaming (ComGS) framework, leveraging the locality and consistency of motion in dynamic scene, that models object-consistent Gaussian point motion through keypoint-driven motion representation. By transmitting only the keypoint attributes, this framework provides a more storage-efficient solution. Specifically, we first identify a sparse set of motion-sensitive keypoints localized within motion regions using a viewspace gradient difference strategy. Equipped with these keypoints, we propose an adaptive motion-driven mechanism that predicts a spatial influence field for propagating keypoint motion to neighboring Gaussian points with similar motion. Moreover, ComGS adopts an error-aware correction strategy for key frame reconstruction that selectively refines erroneous regions and mitigates error accumulation without unnecessary overhead. Overall, ComGS achieves a remarkable storage reduction of over 159 X compared to 3DGStream and 14 X compared to the SOTA method QUEEN, while maintaining competitive visual fidelity and rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.00225v2">Understanding while Exploring: Semantics-driven Active Mapping</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Effective robotic autonomy in unknown environments demands proactive exploration and precise understanding of both geometry and semantics. In this paper, we propose ActiveSGM, an active semantic mapping framework designed to predict the informativeness of potential observations before execution. Built upon a 3D Gaussian Splatting (3DGS) mapping backbone, our approach employs semantic and geometric uncertainty quantification, coupled with a sparse semantic representation, to guide exploration. By enabling robots to strategically select the most beneficial viewpoints, ActiveSGM efficiently enhances mapping completeness, accuracy, and robustness to noisy semantic data, ultimately supporting more adaptive scene exploration. Our experiments on the Replica and Matterport3D datasets highlight the effectiveness of ActiveSGM in active semantic mapping tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09827v1">AHA! Animating Human Avatars in Diverse Scenes with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      We present a novel framework for animating humans in 3D scenes using 3D Gaussian Splatting (3DGS), a neural scene representation that has recently achieved state-of-the-art photorealistic results for novel-view synthesis but remains under-explored for human-scene animation and interaction. Unlike existing animation pipelines that use meshes or point clouds as the underlying 3D representation, our approach introduces the use of 3DGS as the 3D representation to the problem of animating humans in scenes. By representing humans and scenes as Gaussians, our approach allows for geometry-consistent free-viewpoint rendering of humans interacting with 3D scenes. Our key insight is that the rendering can be decoupled from the motion synthesis and each sub-problem can be addressed independently, without the need for paired human-scene data. Central to our method is a Gaussian-aligned motion module that synthesizes motion without explicit scene geometry, using opacity-based cues and projected Gaussian structures to guide human placement and pose alignment. To ensure natural interactions, we further propose a human-scene Gaussian refinement optimization that enforces realistic contact and navigation. We evaluate our approach on scenes from Scannet++ and the SuperSplat library, and on avatars reconstructed from sparse and dense multi-view human capture. Finally, we demonstrate that our framework allows for novel applications such as geometry-consistent free-viewpoint rendering of edited monocular RGB videos with new animated humans, showcasing the unique advantage of 3DGS for monocular video-based human animation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10316v1">Depth-Consistent 3D Gaussian Splatting via Physical Defocus Modeling and Multi-View Geometric Supervision</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Three-dimensional reconstruction in scenes with extreme depth variations remains challenging due to inconsistent supervisory signals between near-field and far-field regions. Existing methods fail to simultaneously address inaccurate depth estimation in distant areas and structural degradation in close-range regions. This paper proposes a novel computational framework that integrates depth-of-field supervision and multi-view consistency supervision to advance 3D Gaussian Splatting. Our approach comprises two core components: (1) Depth-of-field Supervision employs a scale-recovered monocular depth estimator (e.g., Metric3D) to generate depth priors, leverages defocus convolution to synthesize physically accurate defocused images, and enforces geometric consistency through a novel depth-of-field loss, thereby enhancing depth fidelity in both far-field and near-field regions; (2) Multi-View Consistency Supervision employing LoFTR-based semi-dense feature matching to minimize cross-view geometric errors and enforce depth consistency via least squares optimization of reliable matched points. By unifying defocus physics with multi-view geometric constraints, our method achieves superior depth fidelity, demonstrating a 0.8 dB PSNR improvement over the state-of-the-art method on the Waymo Open Dataset. This framework bridges physical imaging principles and learning-based depth regularization, offering a scalable solution for complex depth stratification in urban environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12174v2">UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      In this paper, we propose UniGS, a unified map representation and differentiable framework for high-fidelity multimodal 3D reconstruction based on 3D Gaussian Splatting. Our framework integrates a CUDA-accelerated rasterization pipeline capable of rendering photo-realistic RGB images, geometrically accurate depth maps, consistent surface normals, and semantic logits simultaneously. We redesign the rasterization to render depth via differentiable ray-ellipsoid intersection rather than using Gaussian centers, enabling effective optimization of rotation and scale attribute through analytic depth gradients. Furthermore, we derive the analytic gradient formulation for surface normal rendering, ensuring geometric consistency among reconstructed 3D scenes. To improve computational and storage efficiency, we introduce a learnable attribute that enables differentiable pruning of Gaussians with minimal contribution during training. Quantitative and qualitative experiments demonstrate state-of-the-art reconstruction accuracy across all modalities, validating the efficacy of our geometry-aware paradigm. Source code and multimodal viewer will be available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09944v1">TSPE-GS: Probabilistic Depth Extraction for Semi-Transparent Surface Reconstruction via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 AAAI26 Poster
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting offers a strong speed-quality trade-off but struggles to reconstruct semi-transparent surfaces because most methods assume a single depth per pixel, which fails when multiple surfaces are visible. We propose TSPE-GS (Transparent Surface Probabilistic Extraction for Gaussian Splatting), which uniformly samples transmittance to model a pixel-wise multi-modal distribution of opacity and depth, replacing the prior single-peak assumption and resolving cross-surface depth ambiguity. By progressively fusing truncated signed distance functions, TSPE-GS reconstructs external and internal surfaces separately within a unified framework. The method generalizes to other Gaussian-based reconstruction pipelines without extra training overhead. Extensive experiments on public and self-collected semi-transparent and opaque datasets show TSPE-GS significantly improves semi-transparent geometry reconstruction while maintaining performance on opaque scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07743v1">UltraGS: Gaussian Splatting for Ultrasound Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Ultrasound imaging is a cornerstone of non-invasive clinical diagnostics, yet its limited field of view complicates novel view synthesis. We propose \textbf{UltraGS}, a Gaussian Splatting framework optimized for ultrasound imaging. First, we introduce a depth-aware Gaussian splatting strategy, where each Gaussian is assigned a learnable field of view, enabling accurate depth prediction and precise structural representation. Second, we design SH-DARS, a lightweight rendering function combining low-order spherical harmonics with ultrasound-specific wave physics, including depth attenuation, reflection, and scattering, to model tissue intensity accurately. Third, we contribute the Clinical Ultrasound Examination Dataset, a benchmark capturing diverse anatomical scans under real-world clinical protocols. Extensive experiments on three datasets demonstrate UltraGS's superiority, achieving state-of-the-art results in PSNR (up to 29.55), SSIM (up to 0.89), and MSE (as low as 0.002) while enabling real-time synthesis at 64.69 fps. The code and dataset are open-sourced at: https://github.com/Bean-Young/UltraGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.06161v2">Feature-EndoGaussian: Feature Distilled Gaussian Splatting in Surgical Deformable Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 17 pages, 5 figures; Accepted to ML4H 2025
    </div>
    <details class="paper-abstract">
      Minimally invasive surgery (MIS) requires high-fidelity, real-time visual feedback of dynamic and low-texture surgical scenes. To address these requirements, we introduce FeatureEndo-4DGS (FE-4DGS), the first real time pipeline leveraging feature-distilled 4D Gaussian Splatting for simultaneous reconstruction and semantic segmentation of deformable surgical environments. Unlike prior feature-distilled methods restricted to static scenes, and existing 4D approaches that lack semantic integration, FE-4DGS seamlessly leverages pre-trained 2D semantic embeddings to produce a unified 4D representation-where semantics also deform with tissue motion. This unified approach enables the generation of real-time RGB and semantic outputs through a single, parallelized rasterization process. Despite the additional complexity from feature distillation, FE-4DGS sustains real-time rendering (61 FPS) with a compact footprint, achieves state-of-the-art rendering fidelity on EndoNeRF (39.1 PSNR) and SCARED (27.3 PSNR), and delivers competitive EndoVis18 segmentation, matching or exceeding strong 2D baselines for binary segmentation tasks (0.93 DSC) and remaining competitive for multi-label segmentation (0.77 DSC).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09695v1">A Shared-Autonomy Construction Robotic System for Overhead Works</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 4pages, 8 figures, ICRA construction workshop
    </div>
    <details class="paper-abstract">
      We present the ongoing development of a robotic system for overhead work such as ceiling drilling. The hardware platform comprises a mobile base with a two-stage lift, on which a bimanual torso is mounted with a custom-designed drilling end effector and RGB-D cameras. To support teleoperation in dynamic environments with limited visibility, we use Gaussian splatting for online 3D reconstruction and introduce motion parameters to model moving objects. For safe operation around dynamic obstacles, we developed a neural configuration-space barrier approach for planning and control. Initial feasibility studies demonstrate the capability of the hardware in drilling, bolting, and anchoring, and the software in safe teleoperation in a dynamic environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.08305v3">ELECTRA: A Cartesian Network for 3D Charge Density Prediction with Floating Orbitals</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 10 pages, 4 figures, 5 tables
    </div>
    <details class="paper-abstract">
      We present the Electronic Tensor Reconstruction Algorithm (ELECTRA) - an equivariant model for predicting electronic charge densities using floating orbitals. Floating orbitals are a long-standing concept in the quantum chemistry community that promises more compact and accurate representations by placing orbitals freely in space, as opposed to centering all orbitals at the position of atoms. Finding the ideal placement of these orbitals requires extensive domain knowledge, though, which thus far has prevented widespread adoption. We solve this in a data-driven manner by training a Cartesian tensor network to predict the orbital positions along with orbital coefficients. This is made possible through a symmetry-breaking mechanism that is used to learn position displacements with lower symmetry than the input molecule while preserving the rotation equivariance of the charge density itself. Inspired by recent successes of Gaussian Splatting in representing densities in space, we are using Gaussian orbitals and predicting their weights and covariance matrices. Our method achieves a state-of-the-art balance between computational efficiency and predictive accuracy on established benchmarks. Furthermore, ELECTRA is able to lower the compute time required to arrive at converged DFT solutions - initializing calculations using our predicted densities yields an average 50.72 \% reduction in self-consistent field (SCF) iterations on unseen molecules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09397v1">OUGS: Active View Selection via Object-aware Uncertainty Estimation in 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 11 pages (10 main + 1 appendix), 7 figures, 3 tables. Preprint, under review for Eurographics 2026
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have achieved state-of-the-art results for novel view synthesis. However, efficiently capturing high-fidelity reconstructions of specific objects within complex scenes remains a significant challenge. A key limitation of existing active reconstruction methods is their reliance on scene-level uncertainty metrics, which are often biased by irrelevant background clutter and lead to inefficient view selection for object-centric tasks. We present OUGS, a novel framework that addresses this challenge with a more principled, physically-grounded uncertainty formulation for 3DGS. Our core innovation is to derive uncertainty directly from the explicit physical parameters of the 3D Gaussian primitives (e.g., position, scale, rotation). By propagating the covariance of these parameters through the rendering Jacobian, we establish a highly interpretable uncertainty model. This foundation allows us to then seamlessly integrate semantic segmentation masks to produce a targeted, object-aware uncertainty score that effectively disentangles the object from its environment. This allows for a more effective active view selection strategy that prioritizes views critical to improving object fidelity. Experimental evaluations on public datasets demonstrate that our approach significantly improves the efficiency of the 3DGS reconstruction process and achieves higher quality for targeted objects compared to existing state-of-the-art methods, while also serving as a robust uncertainty estimator for the global scene.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.03536v2">HumanDreamer-X: Photorealistic Single-image Human Avatars Reconstruction via Gaussian Restoration</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Project Page: https://humandreamer-x.github.io/
    </div>
    <details class="paper-abstract">
      Single-image human reconstruction is vital for digital human modeling applications but remains an extremely challenging task. Current approaches rely on generative models to synthesize multi-view images for subsequent 3D reconstruction and animation. However, directly generating multiple views from a single human image suffers from geometric inconsistencies, resulting in issues like fragmented or blurred limbs in the reconstructed models. To tackle these limitations, we introduce \textbf{HumanDreamer-X}, a novel framework that integrates multi-view human generation and reconstruction into a unified pipeline, which significantly enhances the geometric consistency and visual fidelity of the reconstructed 3D models. In this framework, 3D Gaussian Splatting serves as an explicit 3D representation to provide initial geometry and appearance priority. Building upon this foundation, \textbf{HumanFixer} is trained to restore 3DGS renderings, which guarantee photorealistic results. Furthermore, we delve into the inherent challenges associated with attention mechanisms in multi-view human generation, and propose an attention modulation strategy that effectively enhances geometric details identity consistency across multi-view. Experimental results demonstrate that our approach markedly improves generation and reconstruction PSNR quality metrics by 16.45% and 12.65%, respectively, achieving a PSNR of up to 25.62 dB, while also showing generalization capabilities on in-the-wild data and applicability to various human reconstruction backbone models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13713v4">SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 9 pages, 8 figures, submitted to The International Journal of Robotics Research (IJRR)
    </div>
    <details class="paper-abstract">
      Models and methods originally developed for Novel View Synthesis and Scene Rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as sequential operations and, in many settings, multi-modality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. Additionally, the data are often collected using sensors which are handheld or mounted on drones or mobile robots, which complicates the accurate reproduction of sensor motions. To bridge these gaps, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM, Novel View Rendering and Gaussian Splatting. Recorded with a robot manipulator, it uniquely includes 40 sequences with time-synchronized RGB-D images, IMU readings, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of recent integrations of SLAM paradigms within robotic applications. The dataset features five setups with consumer and industrial objects under four controlled lighting conditions, each with separate training and test trajectories. All sequences are static with different levels of object rearrangements and occlusions. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.13639v4">4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Our code and results can be publicly accessed at: https://github.com/robotics-upo/gaussian-rio-cpp
    </div>
    <details class="paper-abstract">
      4D millimeter-wave (mmWave) radars are sensors that provide robustness against adverse weather conditions (rain, snow, fog, etc.), and as such they are increasingly used for odometry and SLAM (Simultaneous Location and Mapping). However, the noisy and sparse nature of the returned scan data proves to be a challenging obstacle for existing registration algorithms, especially those originally intended for more accurate sensors such as LiDAR. Following the success of 3D Gaussian Splatting for vision, in this paper we propose a summarized representation for radar scenes based on global simultaneous optimization of 3D Gaussians as opposed to voxel-based approaches, and leveraging its inherent Probability Density Function (PDF) for registration. Moreover, we propose tackling the problem of radar noise entirely within the scan matching process by optimizing multiple registration hypotheses for better protection against local optima of the PDF. Finally, following existing practice we implement an Extended Kalman Filter-based Radar-Inertial Odometry pipeline in order to evaluate the effectiveness of our system. Experiments using publicly available 4D radar datasets show that our Gaussian approach is comparable to existing registration algorithms, outperforming them in several sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07743v1">UltraGS: Gaussian Splatting for Ultrasound Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Ultrasound imaging is a cornerstone of non-invasive clinical diagnostics, yet its limited field of view complicates novel view synthesis. We propose \textbf{UltraGS}, a Gaussian Splatting framework optimized for ultrasound imaging. First, we introduce a depth-aware Gaussian splatting strategy, where each Gaussian is assigned a learnable field of view, enabling accurate depth prediction and precise structural representation. Second, we design SH-DARS, a lightweight rendering function combining low-order spherical harmonics with ultrasound-specific wave physics, including depth attenuation, reflection, and scattering, to model tissue intensity accurately. Third, we contribute the Clinical Ultrasound Examination Dataset, a benchmark capturing diverse anatomical scans under real-world clinical protocols. Extensive experiments on three datasets demonstrate UltraGS's superiority, achieving state-of-the-art results in PSNR (up to 29.55), SSIM (up to 0.89), and MSE (as low as 0.002) while enabling real-time synthesis at 64.69 fps. The code and dataset are open-sourced at: https://github.com/Bean-Young/UltraGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08294v1">SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 WACV 2026
    </div>
    <details class="paper-abstract">
      Accurate 3D human pose estimation is fundamental for applications such as augmented reality and human-robot interaction. State-of-the-art multi-view methods learn to fuse predictions across views by training on large annotated datasets, leading to poor generalization when the test scenario differs. To overcome these limitations, we propose SkelSplat, a novel framework for multi-view 3D human pose estimation based on differentiable Gaussian rendering. Human pose is modeled as a skeleton of 3D Gaussians, one per joint, optimized via differentiable rendering to enable seamless fusion of arbitrary camera views without 3D ground-truth supervision. Since Gaussian Splatting was originally designed for dense scene reconstruction, we propose a novel one-hot encoding scheme that enables independent optimization of human joints. SkelSplat outperforms approaches that do not rely on 3D ground truth in Human3.6M and CMU, while reducing the cross-dataset error up to 47.8% compared to learning-based methods. Experiments on Human3.6M-Occ and Occlusion-Person demonstrate robustness to occlusions, without scenario-specific fine-tuning. Our project page is available here: https://skelsplat.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02660v2">PMGS: Reconstruction of Projectile Motion Across Large Spatiotemporal Spans via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Modeling complex rigid motion across large spatiotemporal spans remains an unresolved challenge in dynamic reconstruction. Existing paradigms are mainly confined to short-term, small-scale deformation and offer limited consideration for physical consistency. This study proposes PMGS, focusing on reconstructing Projectile Motion via 3D Gaussian Splatting. The workflow comprises two stages: 1) Target Modeling: achieving object-centralized reconstruction through dynamic scene decomposition and an improved point density control; 2) Motion Recovery: restoring full motion sequences by learning per-frame SE(3) poses. We introduce an acceleration consistency constraint to bridge Newtonian mechanics and pose estimation, and design a dynamic simulated annealing strategy that adaptively schedules learning rates based on motion states. Futhermore, we devise a Kalman fusion scheme to optimize error accumulation from multi-source observations to mitigate disturbances. Experiments show PMGS's superior performance in reconstructing high-speed nonlinear rigid motion compared to mainstream dynamic methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08032v1">Perceptual Quality Assessment of 3D Gaussian Splatting: A Subjective Dataset and Prediction Metric</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      With the rapid advancement of 3D visualization, 3D Gaussian Splatting (3DGS) has emerged as a leading technique for real-time, high-fidelity rendering. While prior research has emphasized algorithmic performance and visual fidelity, the perceptual quality of 3DGS-rendered content, especially under varying reconstruction conditions, remains largely underexplored. In practice, factors such as viewpoint sparsity, limited training iterations, point downsampling, noise, and color distortions can significantly degrade visual quality, yet their perceptual impact has not been systematically studied. To bridge this gap, we present 3DGS-QA, the first subjective quality assessment dataset for 3DGS. It comprises 225 degraded reconstructions across 15 object types, enabling a controlled investigation of common distortion factors. Based on this dataset, we introduce a no-reference quality prediction model that directly operates on native 3D Gaussian primitives, without requiring rendered images or ground-truth references. Our model extracts spatial and photometric cues from the Gaussian representation to estimate perceived quality in a structure-aware manner. We further benchmark existing quality assessment methods, spanning both traditional and learning-based approaches. Experimental results show that our method consistently achieves superior performance, highlighting its robustness and effectiveness for 3DGS content evaluation. The dataset and code are made publicly available at https://github.com/diaoyn/3DGSQA to facilitate future research in 3DGS quality assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06299v2">Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted by AAAI-26
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS), an explicit scene representation technique, has shown significant promise for dynamic novel-view synthesis from monocular video input. However, purely data-driven 3DGS often struggles to capture the diverse physics-driven motion patterns in dynamic scenes. To fill this gap, we propose Physics-Informed Deformable Gaussian Splatting (PIDG), which treats each Gaussian particle as a Lagrangian material point with time-varying constitutive parameters and is supervised by 2D optical flow via motion projection. Specifically, we adopt static-dynamic decoupled 4D decomposed hash encoding to reconstruct geometry and motion efficiently. Subsequently, we impose the Cauchy momentum residual as a physics constraint, enabling independent prediction of each particle's velocity and constitutive stress via a time-evolving material field. Finally, we further supervise data fitting by matching Lagrangian particle flow to camera-compensated optical flow, which accelerates convergence and improves generalization. Experiments on a custom physics-driven dataset as well as on standard synthetic and real-world datasets demonstrate significant gains in physical consistency and monocular dynamic reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2505.21890v2">Diffusion Denoised Hyperspectral Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted to 3DV 2026
    </div>
    <details class="paper-abstract">
      Hyperspectral imaging (HSI) has been widely used in agricultural applications for non-destructive estimation of plant nutrient composition and precise determination of nutritional elements of samples. Recently, 3D reconstruction methods have been used to create implicit neural representations of HSI scenes, which can help localize the target object's nutrient composition spatially and spectrally. Neural Radiance Field (NeRF) is a cutting-edge implicit representation that can be used to render hyperspectral channel compositions of each spatial location from any viewing direction. However, it faces limitations in training time and rendering speed. In this paper, we propose Diffusion-Denoised Hyperspectral Gaussian Splatting (DD-HGS), which enhances the state-of-the-art 3D Gaussian Splatting (3DGS) method with wavelength-aware spherical harmonics, a Kullback-Leibler divergence-based spectral loss, and a diffusion-based denoiser to enable 3D explicit reconstruction of hyperspectral scenes across the full spectral range. We present extensive evaluations on diverse real-world hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of DD-HGS. The results demonstrate that DD-HGS achieves new state-of-the-art performance among previously published methods. Project page: https://dragonpg2000.github.io/DDHGS-website/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04665v2">Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 The first two authors contributed equally. Website: https://real2sim-eval.github.io/
    </div>
    <details class="paper-abstract">
      Robotic manipulation policies are advancing rapidly, but their direct evaluation in the real world remains costly, time-consuming, and difficult to reproduce, particularly for tasks involving deformable objects. Simulation provides a scalable and systematic alternative, yet existing simulators often fail to capture the coupled visual and physical complexity of soft-body interactions. We present a real-to-sim policy evaluation framework that constructs soft-body digital twins from real-world videos and renders robots, objects, and environments with photorealistic fidelity using 3D Gaussian Splatting. We validate our approach on representative deformable manipulation tasks, including plush toy packing, rope routing, and T-block pushing, demonstrating that simulated rollouts correlate strongly with real-world execution performance and reveal key behavioral patterns of learned policies. Our results suggest that combining physics-informed reconstruction with high-quality rendering enables reproducible, scalable, and accurate evaluation of robotic manipulation policies. Website: https://real2sim-eval.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07321v1">YoNoSplat: You Only Need One Model for Feedforward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Fast and flexible 3D scene reconstruction from unstructured image collections remains a significant challenge. We present YoNoSplat, a feedforward model that reconstructs high-quality 3D Gaussian Splatting representations from an arbitrary number of images. Our model is highly versatile, operating effectively with both posed and unposed, calibrated and uncalibrated inputs. YoNoSplat predicts local Gaussians and camera poses for each view, which are aggregated into a global representation using either predicted or provided poses. To overcome the inherent difficulty of jointly learning 3D Gaussians and camera parameters, we introduce a novel mixing training strategy. This approach mitigates the entanglement between the two tasks by initially using ground-truth poses to aggregate local Gaussians and gradually transitioning to a mix of predicted and ground-truth poses, which prevents both training instability and exposure bias. We further resolve the scale ambiguity problem by a novel pairwise camera-distance normalization scheme and by embedding camera intrinsics into the network. Moreover, YoNoSplat also predicts intrinsic parameters, making it feasible for uncalibrated inputs. YoNoSplat demonstrates exceptional efficiency, reconstructing a scene from 100 views (at 280x518 resolution) in just 2.69 seconds on an NVIDIA GH200 GPU. It achieves state-of-the-art performance on standard benchmarks in both pose-free and pose-dependent settings. Our project page is at https://botaoye.github.io/yonosplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14270v3">GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Scene reconstruction has emerged as a central challenge in computer vision, with approaches such as Neural Radiance Fields (NeRF) and Gaussian Splatting achieving remarkable progress. While Gaussian Splatting demonstrates strong performance on large-scale datasets, it often struggles to capture fine details or maintain realism in regions with sparse coverage, largely due to the inherent limitations of sparse 3D training data. In this work, we propose GauSSmart, a hybrid method that effectively bridges 2D foundational models and 3D Gaussian Splatting reconstruction. Our approach integrates established 2D computer vision techniques, including convex filtering and semantic feature supervision from foundational models such as DINO, to enhance Gaussian-based scene reconstruction. By leveraging 2D segmentation priors and high-dimensional feature embeddings, our method guides the densification and refinement of Gaussian splats, improving coverage in underrepresented areas and preserving intricate structural details. We validate our approach across three datasets, where GauSSmart consistently outperforms existing Gaussian Splatting in the majority of evaluated scenes. Our results demonstrate the significant potential of hybrid 2D-3D approaches, highlighting how the thoughtful combination of 2D foundational models with 3D reconstruction pipelines can overcome the limitations inherent in either approach alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07241v1">4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted by AAAI 2026.The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      Remarkable advances in recent 2D image and 3D shape generation have induced a significant focus on dynamic 4D content generation. However, previous 4D generation methods commonly struggle to maintain spatial-temporal consistency and adapt poorly to rapid temporal variations, due to the lack of effective spatial-temporal modeling. To address these problems, we propose a novel 4D generation network called 4DSTR, which modulates generative 4D Gaussian Splatting with spatial-temporal rectification. Specifically, temporal correlation across generated 4D sequences is designed to rectify deformable scales and rotations and guarantee temporal consistency. Furthermore, an adaptive spatial densification and pruning strategy is proposed to address significant temporal variations by dynamically adding or deleting Gaussian points with the awareness of their pre-frame movements. Extensive experiments demonstrate that our 4DSTR achieves state-of-the-art performance in video-to-4D generation, excelling in reconstruction quality, spatial-temporal consistency, and adaptation to rapid temporal movements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07122v1">Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      Dynamic Gaussian Splatting approaches have achieved remarkable performance for 4D scene reconstruction. However, these approaches rely on dense-frame video sequences for photorealistic reconstruction. In real-world scenarios, due to equipment constraints, sometimes only sparse frames are accessible. In this paper, we propose Sparse4DGS, the first method for sparse-frame dynamic scene reconstruction. We observe that dynamic reconstruction methods fail in both canonical and deformed spaces under sparse-frame settings, especially in areas with high texture richness. Sparse4DGS tackles this challenge by focusing on texture-rich areas. For the deformation network, we propose Texture-Aware Deformation Regularization, which introduces a texture-based depth alignment loss to regulate Gaussian deformation. For the canonical Gaussian field, we introduce Texture-Aware Canonical Optimization, which incorporates texture-based noise into the gradient descent process of canonical Gaussians. Extensive experiments show that when taking sparse frames as inputs, our method outperforms existing dynamic or few-shot techniques on NeRF-Synthetic, HyperNeRF, NeRF-DS, and our iPhone-4D datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06953v1">GFix: Perceptually Enhanced Gaussian Splatting Video Compression</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enhances 3D scene reconstruction through explicit representation and fast rendering, demonstrating potential benefits for various low-level vision tasks, including video compression. However, existing 3DGS-based video codecs generally exhibit more noticeable visual artifacts and relatively low compression ratios. In this paper, we specifically target the perceptual enhancement of 3DGS-based video compression, based on the assumption that artifacts from 3DGS rendering and quantization resemble noisy latents sampled during diffusion training. Building on this premise, we propose a content-adaptive framework, GFix, comprising a streamlined, single-step diffusion model that serves as an off-the-shelf neural enhancer. Moreover, to increase compression efficiency, We propose a modulated LoRA scheme that freezes the low-rank decompositions and modulates the intermediate hidden states, thereby achieving efficient adaptation of the diffusion backbone with highly compressible updates. Experimental results show that GFix delivers strong perceptual quality enhancement, outperforming GSVC with up to 72.1% BD-rate savings in LPIPS and 21.4% in FID.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06830v1">MUGSQA: Novel Multi-Uncertainty-Based Gaussian Splatting Quality Assessment Method, Dataset, and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has recently emerged as a promising technique for 3D object reconstruction, delivering high-quality rendering results with significantly improved reconstruction speed. As variants continue to appear, assessing the perceptual quality of 3D objects reconstructed with different GS-based methods remains an open challenge. To address this issue, we first propose a unified multi-distance subjective quality assessment method that closely mimics human viewing behavior for objects reconstructed with GS-based methods in actual applications, thereby better collecting perceptual experiences. Based on it, we also construct a novel GS quality assessment dataset named MUGSQA, which is constructed considering multiple uncertainties of the input data. These uncertainties include the quantity and resolution of input views, the view distance, and the accuracy of the initial point cloud. Moreover, we construct two benchmarks: one to evaluate the robustness of various GS-based reconstruction methods under multiple uncertainties, and the other to evaluate the performance of existing quality assessment metrics. Our dataset and benchmark code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06810v1">ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reconstruction with Fewer Primitives</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves state-of-the-art image quality and real-time performance in novel view synthesis but often suffers from a suboptimal spatial distribution of primitives. This issue stems from cloning-based densification, which propagates Gaussians along existing geometry, limiting exploration and requiring many primitives to adequately cover the scene. We present ConeGS, an image-space-informed densification framework that is independent of existing scene geometry state. ConeGS first creates a fast Instant Neural Graphics Primitives (iNGP) reconstruction as a geometric proxy to estimate per-pixel depth. During the subsequent 3DGS optimization, it identifies high-error pixels and inserts new Gaussians along the corresponding viewing cones at the predicted depth values, initializing their size according to the cone diameter. A pre-activation opacity penalty rapidly removes redundant Gaussians, while a primitive budgeting strategy controls the total number of primitives, either by a fixed budget or by adapting to scene complexity, ensuring high reconstruction quality. Experiments show that ConeGS consistently enhances reconstruction quality and rendering performance across Gaussian budgets, with especially strong gains under tight primitive constraints where efficient placement is crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06765v1">Robust and High-Fidelity 3D Gaussian Splatting: Fusing Pose Priors and Geometry Constraints for Texture-Deficient Outdoor Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 7 pages, 3 figures. Accepted by IROS 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a key rendering pipeline for digital asset creation due to its balance between efficiency and visual quality. To address the issues of unstable pose estimation and scene representation distortion caused by geometric texture inconsistency in large outdoor scenes with weak or repetitive textures, we approach the problem from two aspects: pose estimation and scene representation. For pose estimation, we leverage LiDAR-IMU Odometry to provide prior poses for cameras in large-scale environments. These prior pose constraints are incorporated into COLMAP's triangulation process, with pose optimization performed via bundle adjustment. Ensuring consistency between pixel data association and prior poses helps maintain both robustness and accuracy. For scene representation, we introduce normal vector constraints and effective rank regularization to enforce consistency in the direction and shape of Gaussian primitives. These constraints are jointly optimized with the existing photometric loss to enhance the map quality. We evaluate our approach using both public and self-collected datasets. In terms of pose optimization, our method requires only one-third of the time while maintaining accuracy and robustness across both datasets. In terms of scene representation, the results show that our method significantly outperforms conventional 3DGS pipelines. Notably, on self-collected datasets characterized by weak or repetitive textures, our approach demonstrates enhanced visualization capabilities and achieves superior overall performance. Codes and data will be publicly available at https://github.com/justinyeah/normal_shape.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06734v1">Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted by AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Rain degrades the visual quality of multi-view images, which are essential for 3D scene reconstruction, resulting in inaccurate and incomplete reconstruction results. Existing datasets often overlook two critical characteristics of real rainy 3D scenes: the viewpoint-dependent variation in the appearance of rain streaks caused by their projection onto 2D images, and the reduction in ambient brightness resulting from cloud coverage during rainfall. To improve data realism, we construct a new dataset named OmniRain3D that incorporates perspective heterogeneity and brightness dynamicity, enabling more faithful simulation of rain degradation in 3D scenes. Based on this dataset, we propose an end-to-end reconstruction framework named REVR-GSNet (Rain Elimination and Visibility Recovery for 3D Gaussian Splatting). Specifically, REVR-GSNet integrates recursive brightness enhancement, Gaussian primitive optimization, and GS-guided rain elimination into a unified architecture through joint alternating optimization, achieving high-fidelity reconstruction of clean 3D scenes from rain-degraded inputs. Extensive experiments show the effectiveness of our dataset and method. Our dataset and method provide a foundation for future research on multi-view image deraining and rainy 3D scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06632v1">DIAL-GS: Dynamic Instance Aware Reconstruction for Label-free Street Scenes with 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Urban scene reconstruction is critical for autonomous driving, enabling structured 3D representations for data synthesis and closed-loop testing. Supervised approaches rely on costly human annotations and lack scalability, while current self-supervised methods often confuse static and dynamic elements and fail to distinguish individual dynamic objects, limiting fine-grained editing. We propose DIAL-GS, a novel dynamic instance-aware reconstruction method for label-free street scenes with 4D Gaussian Splatting. We first accurately identify dynamic instances by exploiting appearance-position inconsistency between warped rendering and actual observation. Guided by instance-level dynamic perception, we employ instance-aware 4D Gaussians as the unified volumetric representation, realizing dynamic-adaptive and instance-aware reconstruction. Furthermore, we introduce a reciprocal mechanism through which identity and dynamics reinforce each other, enhancing both integrity and consistency. Experiments on urban driving scenarios show that DIAL-GS surpasses existing self-supervised baselines in reconstruction quality and instance-level editing, offering a concise yet powerful solution for urban scene modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06457v1">Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-09
      | 💬 WACV 2026, project page: https://dfki-av.github.io/inpaint360gs/
    </div>
    <details class="paper-abstract">
      Despite recent advances in single-object front-facing inpainting using NeRF and 3D Gaussian Splatting (3DGS), inpainting in complex 360° scenes remains largely underexplored. This is primarily due to three key challenges: (i) identifying target objects in the 3D field of 360° environments, (ii) dealing with severe occlusions in multi-object scenes, which makes it hard to define regions to inpaint, and (iii) maintaining consistent and high-quality appearance across views effectively. To tackle these challenges, we propose Inpaint360GS, a flexible 360° editing framework based on 3DGS that supports multi-object removal and high-fidelity inpainting in 3D space. By distilling 2D segmentation into 3D and leveraging virtual camera views for contextual guidance, our method enables accurate object-level editing and consistent scene completion. We further introduce a new dataset tailored for 360° inpainting, addressing the lack of ground truth object-free scenes. Experiments demonstrate that Inpaint360GS outperforms existing baselines and achieves state-of-the-art performance. Project page: https://dfki-av.github.io/inpaint360gs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.12742v2">Effective Gaussian Management for High-fidelity Object Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-09
    </div>
    <details class="paper-abstract">
      This paper presents an effective Gaussian management framework for high-fidelity scene reconstruction of appearance and geometry. Departing from recent Gaussian Splatting (GS) methods that rely on indiscriminate attribute assignment, our approach introduces a novel densification strategy called \emph{GauSep} that selectively activates Gaussian color or normal attributes. Together with a tailored rendering pipeline, termed \emph{Separate Rendering}, this strategy alleviates gradient conflicts arising from dual supervision and yields improved reconstruction quality. In addition, we develop \emph{GauRep}, an adaptive and integrated Gaussian representation that reduces redundancy both at the individual and global levels, effectively balancing model capacity and number of parameters. To provide reliable geometric supervision essential for effective management, we also introduce \emph{CoRe}, a novel surface reconstruction module that distills normal fields from the SDF branch to the Gaussian branch through a confidence mechanism. Notably, our management framework is model-agnostic and can be seamlessly incorporated into other architectures, simultaneously improving performance and reducing model size. Extensive experiments demonstrate that our approach achieves superior performance in reconstructing both appearance and geometry compared with state-of-the-art methods, while using significantly fewer parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.13055v3">MGSO: Monocular Real-time Photometric SLAM with Efficient 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-08
      | 💬 This is the pre-print version of a work that has been published in ICRA 2025 with doi: 10.1109/ICRA55743.2025.11127380. This version may no longer be accessible without notice. Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses. Please cite the official version
    </div>
    <details class="paper-abstract">
      Real-time SLAM with dense 3D mapping is computationally challenging, especially on resource-limited devices. The recent development of 3D Gaussian Splatting (3DGS) offers a promising approach for real-time dense 3D reconstruction. However, existing 3DGS-based SLAM systems struggle to balance hardware simplicity, speed, and map quality. Most systems excel in one or two of the aforementioned aspects but rarely achieve all. A key issue is the difficulty of initializing 3D Gaussians while concurrently conducting SLAM. To address these challenges, we present Monocular GSO (MGSO), a novel real-time SLAM system that integrates photometric SLAM with 3DGS. Photometric SLAM provides dense structured point clouds for 3DGS initialization, accelerating optimization and producing more efficient maps with fewer Gaussians. As a result, experiments show that our system generates reconstructions with a balance of quality, memory efficiency, and speed that outperforms the state-of-the-art. Furthermore, our system achieves all results using RGB inputs. We evaluate the Replica, TUM-RGBD, and EuRoC datasets against current live dense reconstruction systems. Not only do we surpass contemporary systems, but experiments also show that we maintain our performance on laptop hardware, making it a practical solution for robotics, A/R, and other real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.19856v3">RaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cues for 3D Object Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-08
    </div>
    <details class="paper-abstract">
      4D millimeter-wave radar is a promising sensing modality for autonomous driving, yet effective 3D object detection from 4D radar and monocular images remains challenging. Existing fusion approaches either rely on instance proposals lacking global context or dense BEV grids constrained by rigid structures, lacking a flexible and adaptive representation for diverse scenes. To address this, we propose RaGS, the first framework that leverages 3D Gaussian Splatting (GS) to fuse 4D radar and monocular cues for 3D object detection. 3D GS models the scene as a continuous field of Gaussians, enabling dynamic resource allocation to foreground objects while maintaining flexibility and efficiency. Moreover, the velocity dimension of 4D radar provides motion cues that help anchor and refine the spatial distribution of Gaussians. Specifically, RaGS adopts a cascaded pipeline to construct and progressively refine the Gaussian field. It begins with Frustum-based Localization Initiation (FLI), which unprojects foreground pixels to initialize coarse Gaussian centers. Then, Iterative Multimodal Aggregation (IMA) explicitly exploits image semantics and implicitly integrates 4D radar velocity geometry to refine the Gaussians within regions of interest. Finally, Multi-level Gaussian Fusion (MGF) renders the Gaussian field into hierarchical BEV features for 3D object detection. By dynamically focusing on sparse and informative regions, RaGS achieves object-centric precision and comprehensive scene perception. Extensive experiments on View-of-Delft, TJ4DRadSet, and OmniHD-Scenes demonstrate its robustness and SOTA performance. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06046v1">StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video</a></div>
    <div class="paper-meta">
      📅 2025-11-08
      | 💬 Accepted by AAAI 2026. Code will be released at https://www.github.com/kkkzh/StreamSTGS
    </div>
    <details class="paper-abstract">
      Streaming free-viewpoint video~(FVV) in real-time still faces significant challenges, particularly in training, rendering, and transmission efficiency. Harnessing superior performance of 3D Gaussian Splatting~(3DGS), recent 3DGS-based FVV methods have achieved notable breakthroughs in both training and rendering. However, the storage requirements of these methods can reach up to $10$MB per frame, making stream FVV in real-time impossible. To address this problem, we propose a novel FVV representation, dubbed StreamSTGS, designed for real-time streaming. StreamSTGS represents a dynamic scene using canonical 3D Gaussians, temporal features, and a deformation field. For high compression efficiency, we encode canonical Gaussian attributes as 2D images and temporal features as a video. This design not only enables real-time streaming, but also inherently supports adaptive bitrate control based on network condition without any extra training. Moreover, we propose a sliding window scheme to aggregate adjacent temporal features to learn local motions, and then introduce a transformer-guided auxiliary training module to learn global motions. On diverse FVV benchmarks, StreamSTGS demonstrates competitive performance on all metrics compared to state-of-the-art methods. Notably, StreamSTGS increases the PSNR by an average of $1$dB while reducing the average frame size to just $170$KB. The code is publicly available on https://github.com/kkkzh/StreamSTGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05109v1">Efficient representation of 3D spatial data for defense-related applications</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Geospatial sensor data is essential for modern defense and security, offering indispensable 3D information for situational awareness. This data, gathered from sources like lidar sensors and optical cameras, allows for the creation of detailed models of operational environments. In this paper, we provide a comparative analysis of traditional representation methods, such as point clouds, voxel grids, and triangle meshes, alongside modern neural and implicit techniques like Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS). Our evaluation reveals a fundamental trade-off: traditional models offer robust geometric accuracy ideal for functional tasks like line-of-sight analysis and physics simulations, while modern methods excel at producing high-fidelity, photorealistic visuals but often lack geometric reliability. Based on these findings, we conclude that a hybrid approach is the most promising path forward. We propose a system architecture that combines a traditional mesh scaffold for geometric integrity with a neural representation like 3DGS for visual detail, managed within a hierarchical scene structure to ensure scalability and performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18090v2">GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted at NeurIPS 2025 (Spotlight). Project page: https://fictionarry.github.io/GeoSVR-project/
    </div>
    <details class="paper-abstract">
      Reconstructing accurate surfaces with radiance fields has achieved remarkable progress in recent years. However, prevailing approaches, primarily based on Gaussian Splatting, are increasingly constrained by representational bottlenecks. In this paper, we introduce GeoSVR, an explicit voxel-based framework that explores and extends the under-investigated potential of sparse voxels for achieving accurate, detailed, and complete surface reconstruction. As strengths, sparse voxels support preserving the coverage completeness and geometric clarity, while corresponding challenges also arise from absent scene constraints and locality in surface refinement. To ensure correct scene convergence, we first propose a Voxel-Uncertainty Depth Constraint that maximizes the effect of monocular depth cues while presenting a voxel-oriented uncertainty to avoid quality degradation, enabling effective and robust scene constraints yet preserving highly accurate geometries. Subsequently, Sparse Voxel Surface Regularization is designed to enhance geometric consistency for tiny voxels and facilitate the voxel-based formation of sharp and accurate surfaces. Extensive experiments demonstrate our superior performance compared to existing methods across diverse challenging scenarios, excelling in geometric accuracy, detail preservation, and reconstruction completeness while maintaining high efficiency. Code is available at https://github.com/Fictionarry/GeoSVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.10473v3">ControlGS: Consistent Structural Compression Control for Deployment-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a highly deployable real-time method for novel view synthesis. In practice, it requires a universal, consistent control mechanism that adjusts the trade-off between rendering quality and model compression without scene-specific tuning, enabling automated deployment across different device performances and communication bandwidths. In this work, we present ControlGS, a control-oriented optimization framework that maps the trade-off between Gaussian count and rendering quality to a continuous, scene-agnostic, and highly responsive control axis. Extensive experiments across a wide range of scene scales and types (from small objects to large outdoor scenes) demonstrate that, by adjusting a globally unified control hyperparameter, ControlGS can flexibly generate models biased toward either structural compactness or high fidelity, regardless of the specific scene scale or complexity, while achieving markedly higher rendering quality with the same or fewer Gaussians compared to potential competing methods. Project page: https://zhang-fengdi.github.io/ControlGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.18533v2">On Scaling Up 3D Gaussian Splatting Training</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 ICLR 2025 Oral; Homepage: https://daohanlu.github.io/scaling-up-3dgs/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is increasingly popular for 3D reconstruction due to its superior visual quality and rendering speed. However, 3DGS training currently occurs on a single GPU, limiting its ability to handle high-resolution and large-scale 3D reconstruction tasks due to memory constraints. We introduce Grendel, a distributed system designed to partition 3DGS parameters and parallelize computation across multiple GPUs. As each Gaussian affects a small, dynamic subset of rendered pixels, Grendel employs sparse all-to-all communication to transfer the necessary Gaussians to pixel partitions and performs dynamic load balancing. Unlike existing 3DGS systems that train using one camera view image at a time, Grendel supports batched training with multiple views. We explore various optimization hyperparameter scaling strategies and find that a simple sqrt(batch size) scaling rule is highly effective. Evaluations using large-scale, high-resolution scenes show that Grendel enhances rendering quality by scaling up 3DGS parameters across multiple GPUs. On the Rubble dataset, we achieve a test PSNR of 27.28 by distributing 40.4 million Gaussians across 16 GPUs, compared to a PSNR of 26.28 using 11.2 million Gaussians on a single GPU. Grendel is an open-source project available at: https://github.com/nyu-systems/Grendel-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04951v1">CLM: Removing the GPU Memory Barrier for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to appear in the 2026 ACM International Conference on Architectural Support for Programming Languages and Operating Systems
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is an increasingly popular novel view synthesis approach due to its fast rendering time, and high-quality output. However, scaling 3DGS to large (or intricate) scenes is challenging due to its large memory requirement, which exceed most GPU's memory capacity. In this paper, we describe CLM, a system that allows 3DGS to render large scenes using a single consumer-grade GPU, e.g., RTX4090. It does so by offloading Gaussians to CPU memory, and loading them into GPU memory only when necessary. To reduce performance and communication overheads, CLM uses a novel offloading strategy that exploits observations about 3DGS's memory access pattern for pipelining, and thus overlap GPU-to-CPU communication, GPU computation and CPU computation. Furthermore, we also exploit observation about the access pattern to reduce communication volume. Our evaluation shows that the resulting implementation can render a large scene that requires 100 million Gaussians on a single RTX4090 and achieve state-of-the-art reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05229v1">4D3R: Motion-Aware Neural Reconstruction and Rendering of Dynamic Scenes from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Novel view synthesis from monocular videos of dynamic scenes with unknown camera poses remains a fundamental challenge in computer vision and graphics. While recent advances in 3D representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have shown promising results for static scenes, they struggle with dynamic content and typically rely on pre-computed camera poses. We present 4D3R, a pose-free dynamic neural rendering framework that decouples static and dynamic components through a two-stage approach. Our method first leverages 3D foundational models for initial pose and geometry estimation, followed by motion-aware refinement. 4D3R introduces two key technical innovations: (1) a motion-aware bundle adjustment (MA-BA) module that combines transformer-based learned priors with SAM2 for robust dynamic object segmentation, enabling more accurate camera pose refinement; and (2) an efficient Motion-Aware Gaussian Splatting (MA-GS) representation that uses control points with a deformation field MLP and linear blend skinning to model dynamic motion, significantly reducing computational cost while maintaining high-quality reconstruction. Extensive experiments on real-world dynamic datasets demonstrate that our approach achieves up to 1.8dB PSNR improvement over state-of-the-art methods, particularly in challenging scenarios with large dynamic objects, while reducing computational requirements by 5x compared to previous dynamic scene representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05152v1">Splatography: Sparse multi-view dynamic Gaussian Splatting for filmmaking challenges</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Deformable Gaussian Splatting (GS) accomplishes photorealistic dynamic 3-D reconstruction from dense multi-view video (MVV) by learning to deform a canonical GS representation. However, in filmmaking, tight budgets can result in sparse camera configurations, which limits state-of-the-art (SotA) methods when capturing complex dynamic features. To address this issue, we introduce an approach that splits the canonical Gaussians and deformation field into foreground and background components using a sparse set of masks for frames at t=0. Each representation is separately trained on different loss functions during canonical pre-training. Then, during dynamic training, different parameters are modeled for each deformation field following common filmmaking practices. The foreground stage contains diverse dynamic features so changes in color, position and rotation are learned. While, the background containing film-crew and equipment, is typically dimmer and less dynamic so only changes in point position are learned. Experiments on 3-D and 2.5-D entertainment datasets show that our method produces SotA qualitative and quantitative results; up to 3 PSNR higher with half the model size on 3-D scenes. Unlike the SotA and without the need for dense mask supervision, our method also produces segmented dynamic reconstructions including transparent and dynamic textures. Code and video comparisons are available online: https://interims-git.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05109v1">Efficient representation of 3D spatial data for defense-related applications</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Geospatial sensor data is essential for modern defense and security, offering indispensable 3D information for situational awareness. This data, gathered from sources like lidar sensors and optical cameras, allows for the creation of detailed models of operational environments. In this paper, we provide a comparative analysis of traditional representation methods, such as point clouds, voxel grids, and triangle meshes, alongside modern neural and implicit techniques like Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS). Our evaluation reveals a fundamental trade-off: traditional models offer robust geometric accuracy ideal for functional tasks like line-of-sight analysis and physics simulations, while modern methods excel at producing high-fidelity, photorealistic visuals but often lack geometric reliability. Based on these findings, we conclude that a hybrid approach is the most promising path forward. We propose a system architecture that combines a traditional mesh scaffold for geometric integrity with a neural representation like 3DGS for visual detail, managed within a hierarchical scene structure to ensure scalability and performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18090v2">GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted at NeurIPS 2025 (Spotlight). Project page: https://fictionarry.github.io/GeoSVR-project/
    </div>
    <details class="paper-abstract">
      Reconstructing accurate surfaces with radiance fields has achieved remarkable progress in recent years. However, prevailing approaches, primarily based on Gaussian Splatting, are increasingly constrained by representational bottlenecks. In this paper, we introduce GeoSVR, an explicit voxel-based framework that explores and extends the under-investigated potential of sparse voxels for achieving accurate, detailed, and complete surface reconstruction. As strengths, sparse voxels support preserving the coverage completeness and geometric clarity, while corresponding challenges also arise from absent scene constraints and locality in surface refinement. To ensure correct scene convergence, we first propose a Voxel-Uncertainty Depth Constraint that maximizes the effect of monocular depth cues while presenting a voxel-oriented uncertainty to avoid quality degradation, enabling effective and robust scene constraints yet preserving highly accurate geometries. Subsequently, Sparse Voxel Surface Regularization is designed to enhance geometric consistency for tiny voxels and facilitate the voxel-based formation of sharp and accurate surfaces. Extensive experiments demonstrate our superior performance compared to existing methods across diverse challenging scenarios, excelling in geometric accuracy, detail preservation, and reconstruction completeness while maintaining high efficiency. Code is available at https://github.com/Fictionarry/GeoSVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10473v3">ControlGS: Consistent Structural Compression Control for Deployment-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a highly deployable real-time method for novel view synthesis. In practice, it requires a universal, consistent control mechanism that adjusts the trade-off between rendering quality and model compression without scene-specific tuning, enabling automated deployment across different device performances and communication bandwidths. In this work, we present ControlGS, a control-oriented optimization framework that maps the trade-off between Gaussian count and rendering quality to a continuous, scene-agnostic, and highly responsive control axis. Extensive experiments across a wide range of scene scales and types (from small objects to large outdoor scenes) demonstrate that, by adjusting a globally unified control hyperparameter, ControlGS can flexibly generate models biased toward either structural compactness or high fidelity, regardless of the specific scene scale or complexity, while achieving markedly higher rendering quality with the same or fewer Gaussians compared to potential competing methods. Project page: https://zhang-fengdi.github.io/ControlGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18533v2">On Scaling Up 3D Gaussian Splatting Training</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 ICLR 2025 Oral; Homepage: https://daohanlu.github.io/scaling-up-3dgs/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is increasingly popular for 3D reconstruction due to its superior visual quality and rendering speed. However, 3DGS training currently occurs on a single GPU, limiting its ability to handle high-resolution and large-scale 3D reconstruction tasks due to memory constraints. We introduce Grendel, a distributed system designed to partition 3DGS parameters and parallelize computation across multiple GPUs. As each Gaussian affects a small, dynamic subset of rendered pixels, Grendel employs sparse all-to-all communication to transfer the necessary Gaussians to pixel partitions and performs dynamic load balancing. Unlike existing 3DGS systems that train using one camera view image at a time, Grendel supports batched training with multiple views. We explore various optimization hyperparameter scaling strategies and find that a simple sqrt(batch size) scaling rule is highly effective. Evaluations using large-scale, high-resolution scenes show that Grendel enhances rendering quality by scaling up 3DGS parameters across multiple GPUs. On the Rubble dataset, we achieve a test PSNR of 27.28 by distributing 40.4 million Gaussians across 16 GPUs, compared to a PSNR of 26.28 using 11.2 million Gaussians on a single GPU. Grendel is an open-source project available at: https://github.com/nyu-systems/Grendel-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04951v1">CLM: Removing the GPU Memory Barrier for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to appear in the 2026 ACM International Conference on Architectural Support for Programming Languages and Operating Systems
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is an increasingly popular novel view synthesis approach due to its fast rendering time, and high-quality output. However, scaling 3DGS to large (or intricate) scenes is challenging due to its large memory requirement, which exceed most GPU's memory capacity. In this paper, we describe CLM, a system that allows 3DGS to render large scenes using a single consumer-grade GPU, e.g., RTX4090. It does so by offloading Gaussians to CPU memory, and loading them into GPU memory only when necessary. To reduce performance and communication overheads, CLM uses a novel offloading strategy that exploits observations about 3DGS's memory access pattern for pipelining, and thus overlap GPU-to-CPU communication, GPU computation and CPU computation. Furthermore, we also exploit observation about the access pattern to reduce communication volume. Our evaluation shows that the resulting implementation can render a large scene that requires 100 million Gaussians on a single RTX4090 and achieve state-of-the-art reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21890v2">Diffusion Denoised Hyperspectral Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to 3DV 2026
    </div>
    <details class="paper-abstract">
      Hyperspectral imaging (HSI) has been widely used in agricultural applications for non-destructive estimation of plant nutrient composition and precise determination of nutritional elements of samples. Recently, 3D reconstruction methods have been used to create implicit neural representations of HSI scenes, which can help localize the target object's nutrient composition spatially and spectrally. Neural Radiance Field (NeRF) is a cutting-edge implicit representation that can be used to render hyperspectral channel compositions of each spatial location from any viewing direction. However, it faces limitations in training time and rendering speed. In this paper, we propose Diffusion-Denoised Hyperspectral Gaussian Splatting (DD-HGS), which enhances the state-of-the-art 3D Gaussian Splatting (3DGS) method with wavelength-aware spherical harmonics, a Kullback-Leibler divergence-based spectral loss, and a diffusion-based denoiser to enable 3D explicit reconstruction of hyperspectral scenes across the full spectral range. We present extensive evaluations on diverse real-world hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of DD-HGS. The results demonstrate that DD-HGS achieves new state-of-the-art performance among previously published methods. Project page: https://dragonpg2000.github.io/DDHGS-website/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21890v2">Diffusion Denoised Hyperspectral Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to 3DV 2026
    </div>
    <details class="paper-abstract">
      Hyperspectral imaging (HSI) has been widely used in agricultural applications for non-destructive estimation of plant nutrient composition and precise determination of nutritional elements of samples. Recently, 3D reconstruction methods have been used to create implicit neural representations of HSI scenes, which can help localize the target object's nutrient composition spatially and spectrally. Neural Radiance Field (NeRF) is a cutting-edge implicit representation that can be used to render hyperspectral channel compositions of each spatial location from any viewing direction. However, it faces limitations in training time and rendering speed. In this paper, we propose Diffusion-Denoised Hyperspectral Gaussian Splatting (DD-HGS), which enhances the state-of-the-art 3D Gaussian Splatting (3DGS) method with wavelength-aware spherical harmonics, a Kullback-Leibler divergence-based spectral loss, and a diffusion-based denoiser to enable 3D explicit reconstruction of hyperspectral scenes across the full spectral range. We present extensive evaluations on diverse real-world hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of DD-HGS. The results demonstrate that DD-HGS achieves new state-of-the-art performance among previously published methods. Project page: https://dragonpg2000.github.io/DDHGS-website/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04090v2">Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to ICCV 2025. Project website: https://consistent3dsr.github.io/
    </div>
    <details class="paper-abstract">
      We propose 3D Super Resolution (3DSR), a novel 3D Gaussian-splatting-based super-resolution framework that leverages off-the-shelf diffusion-based 2D super-resolution models. 3DSR encourages 3D consistency across views via the use of an explicit 3D Gaussian-splatting-based scene representation. This makes the proposed 3DSR different from prior work, such as image upsampling or the use of video super-resolution, which either don't consider 3D consistency or aim to incorporate 3D consistency implicitly. Notably, our method enhances visual quality without additional fine-tuning, ensuring spatial coherence within the reconstructed scene. We evaluate 3DSR on MipNeRF360 and LLFF data, demonstrating that it produces high-resolution results that are visually compelling, while maintaining structural consistency in 3D reconstructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05229v1">4D3R: Motion-Aware Neural Reconstruction and Rendering of Dynamic Scenes from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Novel view synthesis from monocular videos of dynamic scenes with unknown camera poses remains a fundamental challenge in computer vision and graphics. While recent advances in 3D representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have shown promising results for static scenes, they struggle with dynamic content and typically rely on pre-computed camera poses. We present 4D3R, a pose-free dynamic neural rendering framework that decouples static and dynamic components through a two-stage approach. Our method first leverages 3D foundational models for initial pose and geometry estimation, followed by motion-aware refinement. 4D3R introduces two key technical innovations: (1) a motion-aware bundle adjustment (MA-BA) module that combines transformer-based learned priors with SAM2 for robust dynamic object segmentation, enabling more accurate camera pose refinement; and (2) an efficient Motion-Aware Gaussian Splatting (MA-GS) representation that uses control points with a deformation field MLP and linear blend skinning to model dynamic motion, significantly reducing computational cost while maintaining high-quality reconstruction. Extensive experiments on real-world dynamic datasets demonstrate that our approach achieves up to 1.8dB PSNR improvement over state-of-the-art methods, particularly in challenging scenarios with large dynamic objects, while reducing computational requirements by 5x compared to previous dynamic scene representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05152v1">Splatography: Sparse multi-view dynamic Gaussian Splatting for filmmaking challenges</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Deformable Gaussian Splatting (GS) accomplishes photorealistic dynamic 3-D reconstruction from dense multi-view video (MVV) by learning to deform a canonical GS representation. However, in filmmaking, tight budgets can result in sparse camera configurations, which limits state-of-the-art (SotA) methods when capturing complex dynamic features. To address this issue, we introduce an approach that splits the canonical Gaussians and deformation field into foreground and background components using a sparse set of masks for frames at t=0. Each representation is separately trained on different loss functions during canonical pre-training. Then, during dynamic training, different parameters are modeled for each deformation field following common filmmaking practices. The foreground stage contains diverse dynamic features so changes in color, position and rotation are learned. While, the background containing film-crew and equipment, is typically dimmer and less dynamic so only changes in point position are learned. Experiments on 3-D and 2.5-D entertainment datasets show that our method produces SotA qualitative and quantitative results; up to 3 PSNR higher with half the model size on 3-D scenes. Unlike the SotA and without the need for dense mask supervision, our method also produces segmented dynamic reconstructions including transparent and dynamic textures. Code and video comparisons are available online: https://interims-git.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01110v3">A LoD of Gaussians: Unified Training and Rendering for Ultra-Large Scale Reconstruction with External Memory</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a high-performance technique for novel view synthesis, enabling real-time rendering and high-quality reconstruction of small scenes. However, scaling to larger environments has so far relied on partitioning the scene into chunks -- a strategy that introduces artifacts at chunk boundaries, complicates training across varying scales, and is poorly suited to unstructured scenarios such as city-scale flyovers combined with street-level views. Moreover, rendering remains fundamentally limited by GPU memory, as all visible chunks must reside in VRAM simultaneously. We introduce A LoD of Gaussians, a framework for training and rendering ultra-large-scale Gaussian scenes on a single consumer-grade GPU -- without partitioning. Our method stores the full scene out-of-core (e.g., in CPU memory) and trains a Level-of-Detail (LoD) representation directly, dynamically streaming only the relevant Gaussians. A hybrid data structure combining Gaussian hierarchies with Sequential Point Trees enables efficient, view-dependent LoD selection, while a lightweight caching and view scheduling system exploits temporal coherence to support real-time streaming and rendering. Together, these innovations enable seamless multi-scale reconstruction and interactive visualization of complex scenes -- from broad aerial views to fine-grained ground-level details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04283v1">FastGS: Training 3D Gaussian Splatting in 100 Seconds</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Project page: https://fastgs.github.io/
    </div>
    <details class="paper-abstract">
      The dominant 3D Gaussian splatting (3DGS) acceleration methods fail to properly regulate the number of Gaussians during training, causing redundant computational time overhead. In this paper, we propose FastGS, a novel, simple, and general acceleration framework that fully considers the importance of each Gaussian based on multi-view consistency, efficiently solving the trade-off between training time and rendering quality. We innovatively design a densification and pruning strategy based on multi-view consistency, dispensing with the budgeting mechanism. Extensive experiments on Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets demonstrate that our method significantly outperforms the state-of-the-art methods in training speed, achieving a 3.32$\times$ training acceleration and comparable rendering quality compared with DashGaussian on the Mip-NeRF 360 dataset and a 15.45$\times$ acceleration compared with vanilla 3DGS on the Deep Blending dataset. We demonstrate that FastGS exhibits strong generality, delivering 2-7$\times$ training acceleration across various tasks, including dynamic scene reconstruction, surface reconstruction, sparse-view reconstruction, large-scale reconstruction, and simultaneous localization and mapping. The project page is available at https://fastgs.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03992v1">CaRF: Enhancing Multi-View Consistency in Referring 3D Gaussian Splatting Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Referring 3D Gaussian Splatting Segmentation (R3DGS) aims to interpret free-form language expressions and localize the corresponding 3D regions in Gaussian fields. While recent advances have introduced cross-modal alignment between language and 3D geometry, existing pipelines still struggle with cross-view consistency due to their reliance on 2D rendered pseudo supervision and view specific feature learning. In this work, we present Camera Aware Referring Field (CaRF), a fully differentiable framework that operates directly in the 3D Gaussian space and achieves multi view consistency. Specifically, CaRF introduces Gaussian Field Camera Encoding (GFCE), which incorporates camera geometry into Gaussian text interactions to explicitly model view dependent variations and enhance geometric reasoning. Building on this, In Training Paired View Supervision (ITPVS) is proposed to align per Gaussian logits across calibrated views during training, effectively mitigating single view overfitting and exposing inter view discrepancies for optimization. Extensive experiments on three representative benchmarks demonstrate that CaRF achieves average improvements of 16.8%, 4.3%, and 2.0% in mIoU over state of the art methods on the Ref LERF, LERF OVS, and 3D OVS datasets, respectively. Moreover, this work promotes more reliable and view consistent 3D scene understanding, with potential benefits for embodied AI, AR/VR interaction, and autonomous perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15680v2">Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Project page: https://kywind.github.io/pgnd
    </div>
    <details class="paper-abstract">
      Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information. We address these challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation. Our particle-grid model captures global shape and motion information while predicting dense particle movements, enabling the modeling of objects with varied shapes and materials. Particles represent object shapes, while the spatial grid discretizes the 3D space to ensure spatial continuity and enhance learning efficiency. Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos. Through experiments, we demonstrate that our model learns the dynamics of diverse objects -- such as ropes, cloths, stuffed animals, and paper bags -- from sparse-view RGB-D recordings of robot-object interactions, while also generalizing at the category level to unseen instances. Our approach outperforms state-of-the-art learning-based and physics-based simulators, particularly in scenarios with limited camera views. Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks. The project page is available at https://kywind.github.io/pgnd .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04665v1">Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Website: https://real2sim-eval.github.io/
    </div>
    <details class="paper-abstract">
      Robotic manipulation policies are advancing rapidly, but their direct evaluation in the real world remains costly, time-consuming, and difficult to reproduce, particularly for tasks involving deformable objects. Simulation provides a scalable and systematic alternative, yet existing simulators often fail to capture the coupled visual and physical complexity of soft-body interactions. We present a real-to-sim policy evaluation framework that constructs soft-body digital twins from real-world videos and renders robots, objects, and environments with photorealistic fidelity using 3D Gaussian Splatting. We validate our approach on representative deformable manipulation tasks, including plush toy packing, rope routing, and T-block pushing, demonstrating that simulated rollouts correlate strongly with real-world execution performance and reveal key behavioral patterns of learned policies. Our results suggest that combining physics-informed reconstruction with high-quality rendering enables reproducible, scalable, and accurate evaluation of robotic manipulation policies. Website: https://real2sim-eval.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16924v2">Optimized Minimal 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Project page: https://maincold2.github.io/omg/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for real-time, high-performance rendering, enabling a wide range of applications. However, representing 3D scenes with numerous explicit Gaussian primitives imposes significant storage and memory overhead. Recent studies have shown that high-quality rendering can be achieved with a substantially reduced number of Gaussians when represented with high-precision attributes. Nevertheless, existing 3DGS compression methods still rely on a relatively large number of Gaussians, focusing primarily on attribute compression. This is because a smaller set of Gaussians becomes increasingly sensitive to lossy attribute compression, leading to severe quality degradation. Since the number of Gaussians is directly tied to computational costs, it is essential to reduce the number of Gaussians effectively rather than only optimizing storage. In this paper, we propose Optimized Minimal Gaussians representation (OMG), which significantly reduces storage while using a minimal number of primitives. First, we determine the distinct Gaussian from the near ones, minimizing redundancy without sacrificing quality. Second, we propose a compact and precise attribute representation that efficiently captures both continuity and irregularity among primitives. Additionally, we propose a sub-vector quantization technique for improved irregularity representation, maintaining fast training with a negligible codebook size. Extensive experiments demonstrate that OMG reduces storage requirements by nearly 50% compared to the previous state-of-the-art and enables 600+ FPS rendering while maintaining high rendering quality. Our source code is available at https://maincold2.github.io/omg/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13713v3">SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 9 pages, 8 figures, submitted to The International Journal of Robotics Research (IJRR)
    </div>
    <details class="paper-abstract">
      Models and methods originally developed for Novel View Synthesis and Scene Rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as sequential operations and, in many settings, multi-modality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. Additionally, the data are often collected using sensors which are handheld or mounted on drones or mobile robots, which complicates the accurate reproduction of sensor motions. To bridge these gaps, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM, Novel View Rendering and Gaussian Splatting. Recorded with a robot manipulator, it uniquely includes 40 sequences with time-synchronized RGB-D images, IMU readings, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of recent integrations of SLAM paradigms within robotic applications. The dataset features five setups with consumer and industrial objects under four controlled lighting conditions, each with separate training and test trajectories. All sequences are static with different levels of object rearrangements and occlusions. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01110v3">A LoD of Gaussians: Unified Training and Rendering for Ultra-Large Scale Reconstruction with External Memory</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a high-performance technique for novel view synthesis, enabling real-time rendering and high-quality reconstruction of small scenes. However, scaling to larger environments has so far relied on partitioning the scene into chunks -- a strategy that introduces artifacts at chunk boundaries, complicates training across varying scales, and is poorly suited to unstructured scenarios such as city-scale flyovers combined with street-level views. Moreover, rendering remains fundamentally limited by GPU memory, as all visible chunks must reside in VRAM simultaneously. We introduce A LoD of Gaussians, a framework for training and rendering ultra-large-scale Gaussian scenes on a single consumer-grade GPU -- without partitioning. Our method stores the full scene out-of-core (e.g., in CPU memory) and trains a Level-of-Detail (LoD) representation directly, dynamically streaming only the relevant Gaussians. A hybrid data structure combining Gaussian hierarchies with Sequential Point Trees enables efficient, view-dependent LoD selection, while a lightweight caching and view scheduling system exploits temporal coherence to support real-time streaming and rendering. Together, these innovations enable seamless multi-scale reconstruction and interactive visualization of complex scenes -- from broad aerial views to fine-grained ground-level details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04283v1">FastGS: Training 3D Gaussian Splatting in 100 Seconds</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Project page: https://fastgs.github.io/
    </div>
    <details class="paper-abstract">
      The dominant 3D Gaussian splatting (3DGS) acceleration methods fail to properly regulate the number of Gaussians during training, causing redundant computational time overhead. In this paper, we propose FastGS, a novel, simple, and general acceleration framework that fully considers the importance of each Gaussian based on multi-view consistency, efficiently solving the trade-off between training time and rendering quality. We innovatively design a densification and pruning strategy based on multi-view consistency, dispensing with the budgeting mechanism. Extensive experiments on Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets demonstrate that our method significantly outperforms the state-of-the-art methods in training speed, achieving a 3.32$\times$ training acceleration and comparable rendering quality compared with DashGaussian on the Mip-NeRF 360 dataset and a 15.45$\times$ acceleration compared with vanilla 3DGS on the Deep Blending dataset. We demonstrate that FastGS exhibits strong generality, delivering 2-7$\times$ training acceleration across various tasks, including dynamic scene reconstruction, surface reconstruction, sparse-view reconstruction, large-scale reconstruction, and simultaneous localization and mapping. The project page is available at https://fastgs.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03992v1">CaRF: Enhancing Multi-View Consistency in Referring 3D Gaussian Splatting Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Referring 3D Gaussian Splatting Segmentation (R3DGS) aims to interpret free-form language expressions and localize the corresponding 3D regions in Gaussian fields. While recent advances have introduced cross-modal alignment between language and 3D geometry, existing pipelines still struggle with cross-view consistency due to their reliance on 2D rendered pseudo supervision and view specific feature learning. In this work, we present Camera Aware Referring Field (CaRF), a fully differentiable framework that operates directly in the 3D Gaussian space and achieves multi view consistency. Specifically, CaRF introduces Gaussian Field Camera Encoding (GFCE), which incorporates camera geometry into Gaussian text interactions to explicitly model view dependent variations and enhance geometric reasoning. Building on this, In Training Paired View Supervision (ITPVS) is proposed to align per Gaussian logits across calibrated views during training, effectively mitigating single view overfitting and exposing inter view discrepancies for optimization. Extensive experiments on three representative benchmarks demonstrate that CaRF achieves average improvements of 16.8%, 4.3%, and 2.0% in mIoU over state of the art methods on the Ref LERF, LERF OVS, and 3D OVS datasets, respectively. Moreover, this work promotes more reliable and view consistent 3D scene understanding, with potential benefits for embodied AI, AR/VR interaction, and autonomous perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01826v3">GSRF: Complex-Valued 3D Gaussian Splatting for Efficient Radio-Frequency Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Synthesizing radio-frequency (RF) data given the transmitter and receiver positions, e.g., received signal strength indicator (RSSI), is critical for wireless networking and sensing applications, such as indoor localization. However, it remains challenging due to complex propagation interactions, including reflection, diffraction, and scattering. State-of-the-art neural radiance field (NeRF)-based methods achieve high-fidelity RF data synthesis but are limited by long training times and high inference latency. We introduce GSRF, a framework that extends 3D Gaussian Splatting (3DGS) from the optical domain to the RF domain, enabling efficient RF data synthesis. GSRF realizes this adaptation through three key innovations: First, it introduces complex-valued 3D Gaussians with a hybrid Fourier-Legendre basis to model directional and phase-dependent radiance. Second, it employs orthographic splatting for efficient ray-Gaussian intersection identification. Third, it incorporates a complex-valued ray tracing algorithm, executed on RF-customized CUDA kernels and grounded in wavefront propagation principles, to synthesize RF data in real time. Evaluated across various RF technologies, GSRF preserves high-fidelity RF data synthesis while achieving significant improvements in training efficiency, shorter training time, and reduced inference latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04797v1">3D Gaussian Point Encoders</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 10 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      In this work, we introduce the 3D Gaussian Point Encoder, an explicit per-point embedding built on mixtures of learned 3D Gaussians. This explicit geometric representation for 3D recognition tasks is a departure from widely used implicit representations such as PointNet. However, it is difficult to learn 3D Gaussian encoders in end-to-end fashion with standard optimizers. We develop optimization techniques based on natural gradients and distillation from PointNets to find a Gaussian Basis that can reconstruct PointNet activations. The resulting 3D Gaussian Point Encoders are faster and more parameter efficient than traditional PointNets. As in the 3D reconstruction literature where there has been considerable interest in the move from implicit (e.g., NeRF) to explicit (e.g., Gaussian Splatting) representations, we can take advantage of computational geometry heuristics to accelerate 3D Gaussian Point Encoders further. We extend filtering techniques from 3D Gaussian Splatting to construct encoders that run 2.7 times faster as a comparable accuracy PointNet while using 46% less memory and 88% fewer FLOPs. Furthermore, we demonstrate the effectiveness of 3D Gaussian Point Encoders as a component in Mamba3D, running 1.27 times faster and achieving a reduction in memory and FLOPs by 42% and 54% respectively. 3D Gaussian Point Encoders are lightweight enough to achieve high framerates on CPU-only devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.01826v3">GSRF: Complex-Valued 3D Gaussian Splatting for Efficient Radio-Frequency Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Synthesizing radio-frequency (RF) data given the transmitter and receiver positions, e.g., received signal strength indicator (RSSI), is critical for wireless networking and sensing applications, such as indoor localization. However, it remains challenging due to complex propagation interactions, including reflection, diffraction, and scattering. State-of-the-art neural radiance field (NeRF)-based methods achieve high-fidelity RF data synthesis but are limited by long training times and high inference latency. We introduce GSRF, a framework that extends 3D Gaussian Splatting (3DGS) from the optical domain to the RF domain, enabling efficient RF data synthesis. GSRF realizes this adaptation through three key innovations: First, it introduces complex-valued 3D Gaussians with a hybrid Fourier-Legendre basis to model directional and phase-dependent radiance. Second, it employs orthographic splatting for efficient ray-Gaussian intersection identification. Third, it incorporates a complex-valued ray tracing algorithm, executed on RF-customized CUDA kernels and grounded in wavefront propagation principles, to synthesize RF data in real time. Evaluated across various RF technologies, GSRF preserves high-fidelity RF data synthesis while achieving significant improvements in training efficiency, shorter training time, and reduced inference latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04797v1">3D Gaussian Point Encoders</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 10 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      In this work, we introduce the 3D Gaussian Point Encoder, an explicit per-point embedding built on mixtures of learned 3D Gaussians. This explicit geometric representation for 3D recognition tasks is a departure from widely used implicit representations such as PointNet. However, it is difficult to learn 3D Gaussian encoders in end-to-end fashion with standard optimizers. We develop optimization techniques based on natural gradients and distillation from PointNets to find a Gaussian Basis that can reconstruct PointNet activations. The resulting 3D Gaussian Point Encoders are faster and more parameter efficient than traditional PointNets. As in the 3D reconstruction literature where there has been considerable interest in the move from implicit (e.g., NeRF) to explicit (e.g., Gaussian Splatting) representations, we can take advantage of computational geometry heuristics to accelerate 3D Gaussian Point Encoders further. We extend filtering techniques from 3D Gaussian Splatting to construct encoders that run 2.7 times faster as a comparable accuracy PointNet while using 46% less memory and 88% fewer FLOPs. Furthermore, we demonstrate the effectiveness of 3D Gaussian Point Encoders as a component in Mamba3D, running 1.27 times faster and achieving a reduction in memory and FLOPs by 42% and 54% respectively. 3D Gaussian Point Encoders are lightweight enough to achieve high framerates on CPU-only devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15680v2">Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Project page: https://kywind.github.io/pgnd
    </div>
    <details class="paper-abstract">
      Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information. We address these challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation. Our particle-grid model captures global shape and motion information while predicting dense particle movements, enabling the modeling of objects with varied shapes and materials. Particles represent object shapes, while the spatial grid discretizes the 3D space to ensure spatial continuity and enhance learning efficiency. Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos. Through experiments, we demonstrate that our model learns the dynamics of diverse objects -- such as ropes, cloths, stuffed animals, and paper bags -- from sparse-view RGB-D recordings of robot-object interactions, while also generalizing at the category level to unseen instances. Our approach outperforms state-of-the-art learning-based and physics-based simulators, particularly in scenarios with limited camera views. Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks. The project page is available at https://kywind.github.io/pgnd .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16924v2">Optimized Minimal 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Project page: https://maincold2.github.io/omg/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for real-time, high-performance rendering, enabling a wide range of applications. However, representing 3D scenes with numerous explicit Gaussian primitives imposes significant storage and memory overhead. Recent studies have shown that high-quality rendering can be achieved with a substantially reduced number of Gaussians when represented with high-precision attributes. Nevertheless, existing 3DGS compression methods still rely on a relatively large number of Gaussians, focusing primarily on attribute compression. This is because a smaller set of Gaussians becomes increasingly sensitive to lossy attribute compression, leading to severe quality degradation. Since the number of Gaussians is directly tied to computational costs, it is essential to reduce the number of Gaussians effectively rather than only optimizing storage. In this paper, we propose Optimized Minimal Gaussians representation (OMG), which significantly reduces storage while using a minimal number of primitives. First, we determine the distinct Gaussian from the near ones, minimizing redundancy without sacrificing quality. Second, we propose a compact and precise attribute representation that efficiently captures both continuity and irregularity among primitives. Additionally, we propose a sub-vector quantization technique for improved irregularity representation, maintaining fast training with a negligible codebook size. Extensive experiments demonstrate that OMG reduces storage requirements by nearly 50% compared to the previous state-of-the-art and enables 600+ FPS rendering while maintaining high rendering quality. Our source code is available at https://maincold2.github.io/omg/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23734v3">ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 NeurIPS 2025, Project Page: https://lhmd.top/zpressor, Code: https://github.com/ziplab/ZPressor
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) models have recently emerged as a promising solution for novel view synthesis, enabling one-pass inference without the need for per-scene 3DGS optimization. However, their scalability is fundamentally constrained by the limited capacity of their models, leading to degraded performance or excessive memory consumption as the number of input views increases. In this work, we analyze feed-forward 3DGS frameworks through the lens of the Information Bottleneck principle and introduce ZPressor, a lightweight architecture-agnostic module that enables efficient compression of multi-view inputs into a compact latent state $Z$ that retains essential scene information while discarding redundancy. Concretely, ZPressor enables existing feed-forward 3DGS models to scale to over 100 input views at 480P resolution on an 80GB GPU, by partitioning the views into anchor and support sets and using cross attention to compress the information from the support views into anchor views, forming the compressed latent state $Z$. We show that integrating ZPressor into several state-of-the-art feed-forward 3DGS models consistently improves performance under moderate input views and enhances robustness under dense view settings on two large-scale benchmarks DL3DV-10K and RealEstate10K. The video results, code and trained models are available on our project page: https://lhmd.top/zpressor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04789v3">Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Learning effective multi-modal 3D representations of objects is essential for numerous applications, such as augmented reality and robotics. Existing methods often rely on task-specific embeddings that are tailored either for semantic understanding or geometric reconstruction. As a result, these embeddings typically cannot be decoded into explicit geometry and simultaneously reused across tasks. In this paper, we propose Object-X, a versatile multi-modal object representation framework capable of encoding rich object embeddings (e.g. images, point cloud, text) and decoding them back into detailed geometric and visual reconstructions. Object-X operates by geometrically grounding the captured modalities in a 3D voxel grid and learning an unstructured embedding fusing the information from the voxels with the object attributes. The learned embedding enables 3D Gaussian Splatting-based object reconstruction, while also supporting a range of downstream tasks, including scene alignment, single-image 3D object reconstruction, and localization. Evaluations on two challenging real-world datasets demonstrate that Object-X produces high-fidelity novel-view synthesis comparable to standard 3D Gaussian Splatting, while significantly improving geometric accuracy. Moreover, Object-X achieves competitive performance with specialized methods in scene alignment and localization. Critically, our object-centric descriptors require 3-4 orders of magnitude less storage compared to traditional image- or point cloud-based approaches, establishing Object-X as a scalable and highly practical solution for multi-modal 3D scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03099v1">DentalSplat: Dental Occlusion Novel View Synthesis from Sparse Intra-Oral Photographs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      In orthodontic treatment, particularly within telemedicine contexts, observing patients' dental occlusion from multiple viewpoints facilitates timely clinical decision-making. Recent advances in 3D Gaussian Splatting (3DGS) have shown strong potential in 3D reconstruction and novel view synthesis. However, conventional 3DGS pipelines typically rely on densely captured multi-view inputs and precisely initialized camera poses, limiting their practicality. Orthodontic cases, in contrast, often comprise only three sparse images, specifically, the anterior view and bilateral buccal views, rendering the reconstruction task especially challenging. The extreme sparsity of input views severely degrades reconstruction quality, while the absence of camera pose information further complicates the process. To overcome these limitations, we propose DentalSplat, an effective framework for 3D reconstruction from sparse orthodontic imagery. Our method leverages a prior-guided dense stereo reconstruction model to initialize the point cloud, followed by a scale-adaptive pruning strategy to improve the training efficiency and reconstruction quality of 3DGS. In scenarios with extremely sparse viewpoints, we further incorporate optical flow as a geometric constraint, coupled with gradient regularization, to enhance rendering fidelity. We validate our approach on a large-scale dataset comprising 950 clinical cases and an additional video-based test set of 195 cases designed to simulate real-world remote orthodontic imaging conditions. Experimental results demonstrate that our method effectively handles sparse input scenarios and achieves superior novel view synthesis quality for dental occlusion visualization, outperforming state-of-the-art techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23734v3">ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 NeurIPS 2025, Project Page: https://lhmd.top/zpressor, Code: https://github.com/ziplab/ZPressor
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) models have recently emerged as a promising solution for novel view synthesis, enabling one-pass inference without the need for per-scene 3DGS optimization. However, their scalability is fundamentally constrained by the limited capacity of their models, leading to degraded performance or excessive memory consumption as the number of input views increases. In this work, we analyze feed-forward 3DGS frameworks through the lens of the Information Bottleneck principle and introduce ZPressor, a lightweight architecture-agnostic module that enables efficient compression of multi-view inputs into a compact latent state $Z$ that retains essential scene information while discarding redundancy. Concretely, ZPressor enables existing feed-forward 3DGS models to scale to over 100 input views at 480P resolution on an 80GB GPU, by partitioning the views into anchor and support sets and using cross attention to compress the information from the support views into anchor views, forming the compressed latent state $Z$. We show that integrating ZPressor into several state-of-the-art feed-forward 3DGS models consistently improves performance under moderate input views and enhances robustness under dense view settings on two large-scale benchmarks DL3DV-10K and RealEstate10K. The video results, code and trained models are available on our project page: https://lhmd.top/zpressor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04789v3">Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Learning effective multi-modal 3D representations of objects is essential for numerous applications, such as augmented reality and robotics. Existing methods often rely on task-specific embeddings that are tailored either for semantic understanding or geometric reconstruction. As a result, these embeddings typically cannot be decoded into explicit geometry and simultaneously reused across tasks. In this paper, we propose Object-X, a versatile multi-modal object representation framework capable of encoding rich object embeddings (e.g. images, point cloud, text) and decoding them back into detailed geometric and visual reconstructions. Object-X operates by geometrically grounding the captured modalities in a 3D voxel grid and learning an unstructured embedding fusing the information from the voxels with the object attributes. The learned embedding enables 3D Gaussian Splatting-based object reconstruction, while also supporting a range of downstream tasks, including scene alignment, single-image 3D object reconstruction, and localization. Evaluations on two challenging real-world datasets demonstrate that Object-X produces high-fidelity novel-view synthesis comparable to standard 3D Gaussian Splatting, while significantly improving geometric accuracy. Moreover, Object-X achieves competitive performance with specialized methods in scene alignment and localization. Critically, our object-centric descriptors require 3-4 orders of magnitude less storage compared to traditional image- or point cloud-based approaches, establishing Object-X as a scalable and highly practical solution for multi-modal 3D scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03099v1">DentalSplat: Dental Occlusion Novel View Synthesis from Sparse Intra-Oral Photographs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      In orthodontic treatment, particularly within telemedicine contexts, observing patients' dental occlusion from multiple viewpoints facilitates timely clinical decision-making. Recent advances in 3D Gaussian Splatting (3DGS) have shown strong potential in 3D reconstruction and novel view synthesis. However, conventional 3DGS pipelines typically rely on densely captured multi-view inputs and precisely initialized camera poses, limiting their practicality. Orthodontic cases, in contrast, often comprise only three sparse images, specifically, the anterior view and bilateral buccal views, rendering the reconstruction task especially challenging. The extreme sparsity of input views severely degrades reconstruction quality, while the absence of camera pose information further complicates the process. To overcome these limitations, we propose DentalSplat, an effective framework for 3D reconstruction from sparse orthodontic imagery. Our method leverages a prior-guided dense stereo reconstruction model to initialize the point cloud, followed by a scale-adaptive pruning strategy to improve the training efficiency and reconstruction quality of 3DGS. In scenarios with extremely sparse viewpoints, we further incorporate optical flow as a geometric constraint, coupled with gradient regularization, to enhance rendering fidelity. We validate our approach on a large-scale dataset comprising 950 clinical cases and an additional video-based test set of 195 cases designed to simulate real-world remote orthodontic imaging conditions. Experimental results demonstrate that our method effectively handles sparse input scenarios and achieves superior novel view synthesis quality for dental occlusion visualization, outperforming state-of-the-art techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02777v1">PercHead: Perceptual Head Model for Single-Image 3D Head Reconstruction & Editing</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Project Page: https://antoniooroz.github.io/PercHead/ Video: https://www.youtube.com/watch?v=4hFybgTk4kE
    </div>
    <details class="paper-abstract">
      We present PercHead, a method for single-image 3D head reconstruction and semantic 3D editing - two tasks that are inherently challenging due to severe view occlusions, weak perceptual supervision, and the ambiguity of editing in 3D space. We develop a unified base model for reconstructing view-consistent 3D heads from a single input image. The model employs a dual-branch encoder followed by a ViT-based decoder that lifts 2D features into 3D space through iterative cross-attention. Rendering is performed using Gaussian Splatting. At the heart of our approach is a novel perceptual supervision strategy based on DINOv2 and SAM2.1, which provides rich, generalized signals for both geometric and appearance fidelity. Our model achieves state-of-the-art performance in novel-view synthesis and, furthermore, exhibits exceptional robustness to extreme viewing angles compared to established baselines. Furthermore, this base model can be seamlessly extended for semantic 3D editing by swapping the encoder and finetuning the network. In this variant, we disentangle geometry and style through two distinct input modalities: a segmentation map to control geometry and either a text prompt or a reference image to specify appearance. We highlight the intuitive and powerful 3D editing capabilities of our model through a lightweight, interactive GUI, where users can effortlessly sculpt geometry by drawing segmentation maps and stylize appearance via natural language or image prompts. Project Page: https://antoniooroz.github.io/PercHead Video: https://www.youtube.com/watch?v=4hFybgTk4kE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14501v4">Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 A project page associated with this survey is available at https://fnzhan.com/projects/Feed-Forward-3D
    </div>
    <details class="paper-abstract">
      3D reconstruction and view synthesis are foundational problems in computer vision, graphics, and immersive technologies such as augmented reality (AR), virtual reality (VR), and digital twins. Traditional methods rely on computationally intensive iterative optimization in a complex chain, limiting their applicability in real-world scenarios. Recent advances in feed-forward approaches, driven by deep learning, have revolutionized this field by enabling fast and generalizable 3D reconstruction and view synthesis. This survey offers a comprehensive review of feed-forward techniques for 3D reconstruction and view synthesis, with a taxonomy according to the underlying representation architectures including point cloud, 3D Gaussian Splatting (3DGS), Neural Radiance Fields (NeRF), etc. We examine key tasks such as pose-free reconstruction, dynamic 3D reconstruction, and 3D-aware image and video synthesis, highlighting their applications in digital humans, SLAM, robotics, and beyond. In addition, we review commonly used datasets with detailed statistics, along with evaluation protocols for various downstream tasks. We conclude by discussing open research challenges and promising directions for future work, emphasizing the potential of feed-forward approaches to advance the state of the art in 3D vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02207v1">Object-Centric 3D Gaussian Splatting for Strawberry Plant Reconstruction and Phenotyping</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 11 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Strawberries are among the most economically significant fruits in the United States, generating over $2 billion in annual farm-gate sales and accounting for approximately 13% of the total fruit production value. Plant phenotyping plays a vital role in selecting superior cultivars by characterizing plant traits such as morphology, canopy structure, and growth dynamics. However, traditional plant phenotyping methods are time-consuming, labor-intensive, and often destructive. Recently, neural rendering techniques, notably Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have emerged as powerful frameworks for high-fidelity 3D reconstruction. By capturing a sequence of multi-view images or videos around a target plant, these methods enable non-destructive reconstruction of complex plant architectures. Despite their promise, most current applications of 3DGS in agricultural domains reconstruct the entire scene, including background elements, which introduces noise, increases computational costs, and complicates downstream trait analysis. To address this limitation, we propose a novel object-centric 3D reconstruction framework incorporating a preprocessing pipeline that leverages the Segment Anything Model v2 (SAM-2) and alpha channel background masking to achieve clean strawberry plant reconstructions. This approach produces more accurate geometric representations while substantially reducing computational time. With a background-free reconstruction, our algorithm can automatically estimate important plant traits, such as plant height and canopy width, using DBSCAN clustering and Principal Component Analysis (PCA). Experimental results show that our method outperforms conventional pipelines in both accuracy and efficiency, offering a scalable and non-destructive solution for strawberry plant phenotyping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01619v2">3DBonsai: Structure-Aware Bonsai Modeling Using Conditioned 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Recent advancements in text-to-3D generation have shown remarkable results by leveraging 3D priors in combination with 2D diffusion. However, previous methods utilize 3D priors that lack detailed and complex structural information, limiting them to generating simple objects and presenting challenges for creating intricate structures such as bonsai. In this paper, we propose 3DBonsai, a novel text-to-3D framework for generating 3D bonsai with complex structures. Technically, we first design a trainable 3D space colonization algorithm to produce bonsai structures, which are then enhanced through random sampling and point cloud augmentation to serve as the 3D Gaussian priors. We introduce two bonsai generation pipelines with distinct structural levels: fine structure conditioned generation, which initializes 3D Gaussians using a 3D structure prior to produce detailed and complex bonsai, and coarse structure conditioned generation, which employs a multi-view structure consistency module to align 2D and 3D structures. Moreover, we have compiled a unified 2D and 3D Chinese-style bonsai dataset. Our experimental results demonstrate that 3DBonsai significantly outperforms existing methods, providing a new benchmark for structure-aware 3D bonsai generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.11252v2">Anti-Aliased 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 NeurIPS 2025. Code will be available at https://github.com/maeyounes/AA-2DGS
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting (2DGS) has recently emerged as a promising method for novel view synthesis and surface reconstruction, offering better view-consistency and geometric accuracy than volumetric 3DGS. However, 2DGS suffers from severe aliasing artifacts when rendering at different sampling rates than those used during training, limiting its practical applications in scenarios requiring camera zoom or varying fields of view. We identify that these artifacts stem from two key limitations: the lack of frequency constraints in the representation and an ineffective screen-space clamping approach. To address these issues, we present AA-2DGS, an anti-aliased formulation of 2D Gaussian Splatting that maintains its geometric benefits while significantly enhancing rendering quality across different scales. Our method introduces a world-space flat smoothing kernel that constrains the frequency content of 2D Gaussian primitives based on the maximal sampling frequency from training views, effectively eliminating high-frequency artifacts when zooming in. Additionally, we derive a novel object-space Mip filter by leveraging an affine approximation of the ray-splat intersection mapping, which allows us to efficiently apply proper anti-aliasing directly in the local space of each splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11878v2">GS-Verse: Mesh-based Gaussian Splatting for Physics-aware Interaction in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      As the demand for immersive 3D content grows, the need for intuitive and efficient interaction methods becomes paramount. Current techniques for physically manipulating 3D content within Virtual Reality (VR) often face significant limitations, including reliance on engineering-intensive processes and simplified geometric representations, such as tetrahedral cages, which can compromise visual fidelity and physical accuracy. In this paper, we introduce GS-Verse (Gaussian Splatting for Virtual Environment Rendering and Scene Editing), a novel method designed to overcome these challenges by directly integrating an object's mesh with a Gaussian Splatting (GS) representation. Our approach enables more precise surface approximation, leading to highly realistic deformations and interactions. By leveraging existing 3D mesh assets, GS-Verse facilitates seamless content reuse and simplifies the development workflow. Moreover, our system is designed to be physics-engine-agnostic, granting developers robust deployment flexibility. This versatile architecture delivers a highly realistic, adaptable, and intuitive approach to interactive 3D manipulation. We rigorously validate our method against the current state-of-the-art technique that couples VR with GS in a comparative user study involving 18 participants. Specifically, we demonstrate that our approach is statistically significantly better for physics-aware stretching manipulation and is also more consistent in other physics-based manipulations like twisting and shaking. Further evaluation across various interactions and scenes confirms that our method consistently delivers high and reliable performance, showing its potential as a plausible alternative to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02777v1">PercHead: Perceptual Head Model for Single-Image 3D Head Reconstruction & Editing</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Project Page: https://antoniooroz.github.io/PercHead/ Video: https://www.youtube.com/watch?v=4hFybgTk4kE
    </div>
    <details class="paper-abstract">
      We present PercHead, a method for single-image 3D head reconstruction and semantic 3D editing - two tasks that are inherently challenging due to severe view occlusions, weak perceptual supervision, and the ambiguity of editing in 3D space. We develop a unified base model for reconstructing view-consistent 3D heads from a single input image. The model employs a dual-branch encoder followed by a ViT-based decoder that lifts 2D features into 3D space through iterative cross-attention. Rendering is performed using Gaussian Splatting. At the heart of our approach is a novel perceptual supervision strategy based on DINOv2 and SAM2.1, which provides rich, generalized signals for both geometric and appearance fidelity. Our model achieves state-of-the-art performance in novel-view synthesis and, furthermore, exhibits exceptional robustness to extreme viewing angles compared to established baselines. Furthermore, this base model can be seamlessly extended for semantic 3D editing by swapping the encoder and finetuning the network. In this variant, we disentangle geometry and style through two distinct input modalities: a segmentation map to control geometry and either a text prompt or a reference image to specify appearance. We highlight the intuitive and powerful 3D editing capabilities of our model through a lightweight, interactive GUI, where users can effortlessly sculpt geometry by drawing segmentation maps and stylize appearance via natural language or image prompts. Project Page: https://antoniooroz.github.io/PercHead Video: https://www.youtube.com/watch?v=4hFybgTk4kE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14501v4">Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 A project page associated with this survey is available at https://fnzhan.com/projects/Feed-Forward-3D
    </div>
    <details class="paper-abstract">
      3D reconstruction and view synthesis are foundational problems in computer vision, graphics, and immersive technologies such as augmented reality (AR), virtual reality (VR), and digital twins. Traditional methods rely on computationally intensive iterative optimization in a complex chain, limiting their applicability in real-world scenarios. Recent advances in feed-forward approaches, driven by deep learning, have revolutionized this field by enabling fast and generalizable 3D reconstruction and view synthesis. This survey offers a comprehensive review of feed-forward techniques for 3D reconstruction and view synthesis, with a taxonomy according to the underlying representation architectures including point cloud, 3D Gaussian Splatting (3DGS), Neural Radiance Fields (NeRF), etc. We examine key tasks such as pose-free reconstruction, dynamic 3D reconstruction, and 3D-aware image and video synthesis, highlighting their applications in digital humans, SLAM, robotics, and beyond. In addition, we review commonly used datasets with detailed statistics, along with evaluation protocols for various downstream tasks. We conclude by discussing open research challenges and promising directions for future work, emphasizing the potential of feed-forward approaches to advance the state of the art in 3D vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02207v1">Object-Centric 3D Gaussian Splatting for Strawberry Plant Reconstruction and Phenotyping</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 11 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Strawberries are among the most economically significant fruits in the United States, generating over $2 billion in annual farm-gate sales and accounting for approximately 13% of the total fruit production value. Plant phenotyping plays a vital role in selecting superior cultivars by characterizing plant traits such as morphology, canopy structure, and growth dynamics. However, traditional plant phenotyping methods are time-consuming, labor-intensive, and often destructive. Recently, neural rendering techniques, notably Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have emerged as powerful frameworks for high-fidelity 3D reconstruction. By capturing a sequence of multi-view images or videos around a target plant, these methods enable non-destructive reconstruction of complex plant architectures. Despite their promise, most current applications of 3DGS in agricultural domains reconstruct the entire scene, including background elements, which introduces noise, increases computational costs, and complicates downstream trait analysis. To address this limitation, we propose a novel object-centric 3D reconstruction framework incorporating a preprocessing pipeline that leverages the Segment Anything Model v2 (SAM-2) and alpha channel background masking to achieve clean strawberry plant reconstructions. This approach produces more accurate geometric representations while substantially reducing computational time. With a background-free reconstruction, our algorithm can automatically estimate important plant traits, such as plant height and canopy width, using DBSCAN clustering and Principal Component Analysis (PCA). Experimental results show that our method outperforms conventional pipelines in both accuracy and efficiency, offering a scalable and non-destructive solution for strawberry plant phenotyping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.01619v2">3DBonsai: Structure-Aware Bonsai Modeling Using Conditioned 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Recent advancements in text-to-3D generation have shown remarkable results by leveraging 3D priors in combination with 2D diffusion. However, previous methods utilize 3D priors that lack detailed and complex structural information, limiting them to generating simple objects and presenting challenges for creating intricate structures such as bonsai. In this paper, we propose 3DBonsai, a novel text-to-3D framework for generating 3D bonsai with complex structures. Technically, we first design a trainable 3D space colonization algorithm to produce bonsai structures, which are then enhanced through random sampling and point cloud augmentation to serve as the 3D Gaussian priors. We introduce two bonsai generation pipelines with distinct structural levels: fine structure conditioned generation, which initializes 3D Gaussians using a 3D structure prior to produce detailed and complex bonsai, and coarse structure conditioned generation, which employs a multi-view structure consistency module to align 2D and 3D structures. Moreover, we have compiled a unified 2D and 3D Chinese-style bonsai dataset. Our experimental results demonstrate that 3DBonsai significantly outperforms existing methods, providing a new benchmark for structure-aware 3D bonsai generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11878v2">GS-Verse: Mesh-based Gaussian Splatting for Physics-aware Interaction in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      As the demand for immersive 3D content grows, the need for intuitive and efficient interaction methods becomes paramount. Current techniques for physically manipulating 3D content within Virtual Reality (VR) often face significant limitations, including reliance on engineering-intensive processes and simplified geometric representations, such as tetrahedral cages, which can compromise visual fidelity and physical accuracy. In this paper, we introduce GS-Verse (Gaussian Splatting for Virtual Environment Rendering and Scene Editing), a novel method designed to overcome these challenges by directly integrating an object's mesh with a Gaussian Splatting (GS) representation. Our approach enables more precise surface approximation, leading to highly realistic deformations and interactions. By leveraging existing 3D mesh assets, GS-Verse facilitates seamless content reuse and simplifies the development workflow. Moreover, our system is designed to be physics-engine-agnostic, granting developers robust deployment flexibility. This versatile architecture delivers a highly realistic, adaptable, and intuitive approach to interactive 3D manipulation. We rigorously validate our method against the current state-of-the-art technique that couples VR with GS in a comparative user study involving 18 participants. Specifically, we demonstrate that our approach is statistically significantly better for physics-aware stretching manipulation and is also more consistent in other physics-based manipulations like twisting and shaking. Further evaluation across various interactions and scenes confirms that our method consistently delivers high and reliable performance, showing its potential as a plausible alternative to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00159v3">SonarSplat: Novel View Synthesis of Imaging Sonar via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      In this paper, we present SonarSplat, a novel Gaussian splatting framework for imaging sonar that demonstrates realistic novel view synthesis and models acoustic streaking phenomena. Our method represents the scene as a set of 3D Gaussians with acoustic reflectance and saturation properties. We develop a novel method to efficiently rasterize Gaussians to produce a range/azimuth image that is faithful to the acoustic image formation model of imaging sonar. In particular, we develop a novel approach to model azimuth streaking in a Gaussian splatting framework. We evaluate SonarSplat using real-world datasets of sonar images collected from an underwater robotic platform in a controlled test tank and in a real-world river environment. Compared to the state-of-the-art, SonarSplat offers improved image synthesis capabilities (+3.2 dB PSNR) and more accurate 3D reconstruction (77% lower Chamfer Distance). We also demonstrate that SonarSplat can be leveraged for azimuth streak removal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14270v2">GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Scene reconstruction has emerged as a central challenge in computer vision, with approaches such as Neural Radiance Fields (NeRF) and Gaussian Splatting achieving remarkable progress. While Gaussian Splatting demonstrates strong performance on large-scale datasets, it often struggles to capture fine details or maintain realism in regions with sparse coverage, largely due to the inherent limitations of sparse 3D training data. In this work, we propose GauSSmart, a hybrid method that effectively bridges 2D foundational models and 3D Gaussian Splatting reconstruction. Our approach integrates established 2D computer vision techniques, including convex filtering and semantic feature supervision from foundational models such as DINO, to enhance Gaussian-based scene reconstruction. By leveraging 2D segmentation priors and high-dimensional feature embeddings, our method guides the densification and refinement of Gaussian splats, improving coverage in underrepresented areas and preserving intricate structural details. We validate our approach across three datasets, where GauSSmart consistently outperforms existing Gaussian Splatting in the majority of evaluated scenes. Our results demonstrate the significant potential of hybrid 2D-3D approaches, highlighting how the thoughtful combination of 2D foundational models with 3D reconstruction pipelines can overcome the limitations inherent in either approach alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.00159v3">SonarSplat: Novel View Synthesis of Imaging Sonar via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      In this paper, we present SonarSplat, a novel Gaussian splatting framework for imaging sonar that demonstrates realistic novel view synthesis and models acoustic streaking phenomena. Our method represents the scene as a set of 3D Gaussians with acoustic reflectance and saturation properties. We develop a novel method to efficiently rasterize Gaussians to produce a range/azimuth image that is faithful to the acoustic image formation model of imaging sonar. In particular, we develop a novel approach to model azimuth streaking in a Gaussian splatting framework. We evaluate SonarSplat using real-world datasets of sonar images collected from an underwater robotic platform in a controlled test tank and in a real-world river environment. Compared to the state-of-the-art, SonarSplat offers improved image synthesis capabilities (+3.2 dB PSNR) and more accurate 3D reconstruction (77% lower Chamfer Distance). We also demonstrate that SonarSplat can be leveraged for azimuth streak removal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19588v2">Gaussian Splashing: Direct Volumetric Rendering Underwater</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      In underwater images, most useful features are occluded by water. The extent of the occlusion depends on imaging geometry and can vary even across a sequence of burst images. As a result, 3D reconstruction methods robust on in-air scenes, like Neural Radiance Field methods (NeRFs) or 3D Gaussian Splatting (3DGS), fail on underwater scenes. While a recent underwater adaptation of NeRFs achieved state-of-the-art results, it is impractically slow: reconstruction takes hours and its rendering rate, in frames per second (FPS), is less than 1. Here, we present a new method that takes only a few minutes for reconstruction and renders novel underwater scenes at 140 FPS. Named Gaussian Splashing, our method unifies the strengths and speed of 3DGS with an image formation model for capturing scattering, introducing innovations in the rendering and depth estimation procedures and in the 3DGS loss function. Despite the complexities of underwater adaptation, our method produces images at unparalleled speeds with superior details. Moreover, it reveals distant scene details with far greater clarity than other methods, dramatically improving reconstructed and rendered images. We demonstrate results on existing datasets and a new dataset we have collected. Additional visual results are available at: https://bgu-cs-vil.github.io/gaussiansplashingUW.github.io/ .
    </details>
</div>
