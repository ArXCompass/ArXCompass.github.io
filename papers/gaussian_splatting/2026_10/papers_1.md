# gaussian splatting - 2026_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.22052v1">Compression of Uniformly Distributed Unit Quaternions</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Unit quaternions are widely used in computer graphics making their efficient compression critical for reducing memory consumption and bandwidth requirements. In this paper, we present a novel method for encoding uniformly distributed unit quaternions. Our thorough analysis of several quaternion parameterizations compares their theoretical error properties, quantization behavior, computational cost, and memory efficiency and provides both an analytical and empirical characterization. We show that an infinity-norm parameterization with our optimized warp function consistently offers the smallest maximum error and the best storage efficiency, at a decompression cost equal to or lower than that of parameterization methods found in practice. We demonstrate the practical benefits of the proposed approach in several graphics applications, including tangent-space normal mapping, skeletal animation with blend weights, and Gaussian splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21668v1">Edit4D: Editable 4D Scene Generation with Textual Control</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 21 pages, 16 figures. Project page: https://chanwo0kim.github.io/Edit4D/
    </div>
    <details class="paper-abstract">
      We introduce Edit4D, an editable text-to-4D scene generation framework that constructs 4D scenes from separate background and dynamic character representations. Recent advances in generative models have enabled text-driven dynamic scene synthesis, but editable 4D scene generation remains challenging due to the need to preserve spatial geometry and temporal dynamics. Existing approaches typically generate holistic scene videos, where the background and dynamic characters are coupled, making object-level editing and flexible camera control difficult. Edit4D addresses these limitations through a stage-specific generation pipeline. Given a user instruction, Edit4D first decomposes it into stage-specific prompts for background, dynamic character, and scene-character composition. Using these prompts, we synthesize multi-view scene images with text-to-image and camera-conditioned generative priors and reconstruct a background 3D Gaussian splatting representation. We then generate a character image, animate it with an image-conditioned video prior, reconstruct a 4D mesh, and bind Gaussian primitives to obtain a character representation. Finally, both components are composed through text-guided segmentation and contact-aware optimization. A global scene refinement is then performed to improve harmonization and rendering consistency. By integrating stage-specific components, Edit4D enables controllable camera motion, flexible object-level editing, and the practical construction of editable 4D scenes. Project page: https://chanwo0kim.github.io/Edit4D/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21634v1">Anchor3DGS: Feed-Forward 3D Gaussian Splatting with Compact Anchor-Based Representation</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Recent feed-forward 3D Gaussian Splatting (3DGS) methods have enabled effective multi-view 3D reconstruction by predicting Gaussian primitives in a single forward pass. However, existing approaches predominantly adopt a pixel-aligned paradigm that predicts one Gaussian per input pixel, coupling the Gaussian count to the input resolution and number of views. This also leads to redundant primitives in textureless regions and insufficient coverage in geometrically complex areas. We present Anchor3DGS, a pose-free feed-forward 3DGS framework that replaces pixel-aligned prediction with a compact anchor-based representation. Guided by information entropy, our method places a set of 3D anchors in the scene, concentrating them in regions of higher visual complexity while maintaining broad spatial coverage. Each anchor performs occlusion-aware aggregation of multi-view image features, exchanges information with other anchors, and is decoded into a set of local Gaussians. A lightweight feature enhancement module further refines the coarse RGB output from Gaussian splatting and improves rendering fidelity. Extensive experiments on diverse benchmark datasets demonstrate that our Anchor3DGS achieves state-of-the-art novel view synthesis quality while only using an order of magnitude fewer Gaussian primitives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21619v1">Training-Free Target-Aware Refinement of 3D Gaussian Language Map</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 8 pages, 6 figures. Project page: https://lab-of-ai-and-robotics.github.io/target-aware-langmap/
    </div>
    <details class="paper-abstract">
      Distilling vision-language features into 3D Gaussian Splatting yields a language map that can be queried with arbitrary text. In deployment, however, a small set of user-specified targets must be reliably located, which current distilled maps often miss, while open-vocabulary queryability for other words must remain intact. We propose a training-free, target-aware refinement of an already-distilled language map. Guided by a text-promptable segmenter, the visual embedding inside each detected target region is blended with the target text embedding, a convex combination whose coefficient sets the amount of injected prior, while non-target regions are left untouched. Injecting too much overwrites appearance cues and general queryability, so we seek, for each target, the smallest coefficient that still lets it win, and we obtain every coefficient in closed form, leaving no hyperparameters to tune. We further identify a failure mode of injection in language-goal localization. Text embeddings share a common component, and injecting them makes the standard cosine read-out query-agnostic, which a centering step in the read-out removes. Across three benchmarks, the refinement consistently improves open-vocabulary segmentation, roughly doubling the base mIoU on Replica, and localizes language goals more reliably than the base map. It also transfers as a plug-in to off-the-shelf field-based and object-level language maps and supports a real-world demonstration in which a legged robot navigates to natural-language goals. Project page and code: https://lab-of-ai-and-robotics.github.io/target-aware-langmap/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15737v3">RUSplatting: Robust 3D Gaussian Splatting for Sparse-View Underwater Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 Accepted by BMVC 2025
    </div>
    <details class="paper-abstract">
      Reconstructing high-fidelity underwater scenes remains a challenging task due to light absorption, scattering, and limited visibility inherent in aquatic environments. This paper presents an enhanced Gaussian Splatting-based framework that improves both the visual quality and geometric accuracy of deep underwater rendering. We propose decoupled learning for RGB channels, guided by the physics of underwater attenuation, to enable more accurate colour restoration. To address sparse-view limitations and improve view consistency, we introduce a frame interpolation strategy with a novel adaptive weighting scheme. Additionally, we introduce a new loss function aimed at reducing noise while preserving edges, which is essential for deep-sea content. We also release a newly collected dataset, Submerged3D, captured specifically in deep-sea environments. Experimental results demonstrate that our framework consistently outperforms state-of-the-art methods with PSNR gains up to 1.90dB, delivering superior perceptual quality and robustness, and offering promising directions for marine robotics and underwater visual analytics. The code of RUSplatting is available at https://github.com/theflash987/RUSplatting and the dataset Submerged3D can be downloaded at https://zenodo.org/records/15482420.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19541v2">GS-PRM: Gaussian Visual Roadmap for Image-Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2026-10-15
    </div>
    <details class="paper-abstract">
      Image-goal navigation requires a global planner that connects visually specified endpoints while providing observations that a local policy can follow. Three-dimensional Gaussian Splatting (3DGS) provides map-registered RGB, opacity, and depth renderings, but does not explicitly encode navigable topology. We introduce GS-PRM, a Gaussian visual roadmap whose milestones, local connections, and visual references are constructed directly through 3DGS rendering without deriving an occupancy grid or signed distance field. We define render-based collision-free (RC-free) configurations and paths. Panoramic depth supplies conservative clearance for milestone validation and adaptive coverage, while edge-aligned orthographic rendering tests the robot's swept volume. We establish conditional soundness and sufficient conditions for deterministic query completeness and probabilistic construction completeness of GS-PRM. To validate the performance of GS-PRM, we conduct experiments in both simulated and real-world environments, demonstrating long-horizon navigation using goal-conditioned policies guided by GS-PRM. Compared with a state-of-the-art 3DGS-based navigation system, our GS-PRM-based navigation system achieves up to a twice higher average success rate and reduces travel time by up to 57.2% when both systems complete the navigation task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.21121v1">ResStreamGS: Residual-Sensitive Dual-Stream Gaussian Splatting for Mobile Volumetric Video Streaming</a></div>
    <div class="paper-meta">
      📅 2026-10-15
      | 💬 11 pages,9 figures
    </div>
    <details class="paper-abstract">
      Real-time mobile volumetric video enables immersive free-viewpoint experiences, but delivering high-quality 3D content under constrained bandwidth and computation remains challenging. Recent 3D Gaussian Splatting (3DGS) systems, such as V$^3$, enable efficient mobile streaming by encoding time-varying Gaussian attributes with commodity video codecs and hardware decoding. However, existing pipelines treat all Gaussians as frame-varying signals, which is poorly suited to full-scene content where large portions remain temporally stable. Our empirical study shows that frame-wise adaptation and video coding of all Gaussians cause quality degradation in complex full-scene content, particularly in temporally stable regions. To address this issue, we propose ResStreamGS, a residual-sensitive dual-stream framework that processes Gaussians according to their temporal behavior. ResStreamGS identifies residual-sensitive primitives through lightweight probe adaptation and reuses stable Gaussians within each Group of Pictures (GOP). Stable Gaussians are transmitted once per GOP, while only residual-sensitive Gaussians remain in the frame-wise hardware-decodable video stream. Experiments on complex full-scene volumetric videos show that ResStreamGS improves post-codec PSNR by 7.51--8.35~dB over a controlled V$^3$-style all-Gaussians baseline, reduces amortized payload by up to 56.5%, and achieves 30.883~FPS on a Qualcomm Snapdragon 778G platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20590v1">Feed-Forward Multi-View Feature Coding for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) methods enable efficient reconstruction of 3D scenes from sparse multi-view images in a single forward pass, without scene-specific optimization. However, existing feed-forward scene-coding pipelines first generate a complete Gaussian representation and then encode its geometry and appearance attributes separately, often involving non-differentiable operations. The resulting 'generate-then-compress' pipeline requires complex attribute-specific codec designs and limits rate-distortion performance by restricting the exploitation of cross-attribute redundancy. We propose multi-view feature coding for Gaussian splatting (MFGS), an end-to-end feed-forward framework for scene coding. Instead of compressing an explicit Gaussian representation, MFGS extracts multi-view features from the input images and encodes them into a compact bitstream. Through end-to-end rate-distortion training, MFGS learns compact multi-view features that jointly represent geometry and appearance, enabling the receiver to predict Gaussian attributes from the decoded features. We further introduce importance-guided Gaussian selection to accommodate different rendering budgets, allowing the receiver to adjust the number of Gaussians used for rendering from the same bitstream. Across RE10K, ACID, and DL3DV, MFGS achieves 99.76-99.86% bitrate savings over the generate-then-compress pipeline, as measured by BD-rate, while reducing total GMACs by 52-55% and providing an approximately 12 times end-to-end speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11142v2">Naka-GS: Naka-Guided Chroma Correction and Point-Prior Compaction for Low-Light 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Low-light inputs obscure scene content, distort color, and corrupt geometric priors, substantially degrading 3D Gaussian Splatting. Although image enhancement can recover visibility, it often introduces chromatic deviations and fine-scale artifacts that propagate through multi-view reconstruction. Moreover, dense point priors can further amplify these errors and increase optimization cost. To address the coupled challenges, we present Naka-GS, a low-light 3DGS framework that jointly improves observations and Gaussian initialization while retaining the standard 3DGS renderer and training objective. Our Naka-Guided Chroma Correction treats a fixed bionic response as a photometric anchor and learns bounded, frequency-decoupled luminance and chromatic corrections, preserving fine structures while suppressing residual color and boundary artifacts. The corrected views are reconstructed into dense scene priors, which a lightweight, non-learned Point-Prior Compaction Module refines through voxel pooling and distance-adaptive progressive pruning. On eight scene-disjoint RealX3D scenes, Naka-GS achieves 20.735~dB PSNR, 0.7408 SSIM, and 0.3340 LPIPS. PPM reduces initialization from 1.00M to 59.52K points and shortens average 3DGS optimization time by 37.7%.The code is available at https://github.com/RunyuZhu/Naka-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.20397v1">Where Should the Gaussians Go? Reusable Label-Aware Allocation for Budget-Constrained Gaussian Splatting of Cinematic CT</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Anatomical labels help 3D Gaussian splatting preserve vessels in cinematic renderings of CT, but the representation is still found by optimising each patient's scene, where densification decides where Gaussians go, while real-time displays cap their number. We address one part of this process: deciding where a fixed budget of B Gaussians should be placed. We propose a reusable, labelaware allocation front end that maps a patient's CT, labels and transfer function to a density over 2mm cells and draws exactly B positions from it without per-patient optimisation; the other Gaussian attributes are still fitted per patient. The front end is learned once from per-patient optimised scenes and implemented as a population-trained neural prior and an interpretable seven-parameter visibility-aware closed form (VACF). With identical attribute fitting on 20 held-out patients at 10K-100K Gaussians, both raise vessel-masked PSNR over uniform allocation by 4.5-7.1dB and over pruning a dense voxel-aligned model by 1.6-3.3dB. The neural prior does not differ significantly from per-patient optimised allocation at any budget, and VACF is within 0.6dB of it on vessels; on 19 external patients the neural prior again matches it on vessels and VACF is within 0.7dB. As an initialisation for position optimisation, VACF allocation beats per-patient densification on vessels at 10K Gaussians and, at 25K, reaches a higher vessel PSNR after 118s of optimisation than densification after 248s. Under altered rendering presets, VACF transfers to new colours and lighting but not to a label-free transfer function, whereas the neural prior shows the opposite pattern.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13643v3">DexBioLab: Benchmarking Dexterous Manipulation with Dynamic Experimental Phenomena in Realistic Laboratories</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Automating biological laboratories requires robots to perform dexterous tool interactions under realistic visual conditions while handling dynamic experimental phenomena, presenting significant challenges for robotic manipulation. To enable systematic evaluation under these conditions, we introduce DexBioLab, a simulation benchmark and framework for dexterous manipulation in realistic biological laboratory environments. DexBioLab comprises 14 tasks across 4 robot embodiments, spanning functional tool use, precision manipulation, language-conditioned visual grounding, and instrument operation. We introduce controllable dynamic experimental phenomena, including flames, vapor, and bubbles, to capture the evolving visual conditions of laboratory procedures. To further enhance environmental realism, we develop a hybrid rendering pipeline that integrates interactive simulated foreground objects with Gaussian-splatting backgrounds reconstructed from 6 real laboratory scenes. For efficient large-scale data generation and evaluation, we further build a parallel simulation framework on top of mjlab, proposing Parallel Finite-State Machine (PFSM) and integrating batched rendering with decoupled trajectory replay for demonstration generation and model-independent closed-loop policy evaluation. Our evaluation reveals substantial variation in policy performance across tasks, exposing distinct strengths and weaknesses of current manipulation policies. Real-world experiments further demonstrate a reduced sim-to-real gap, supporting the validity of AutoBio+ for realistic manipulation evaluation. Supplementary materials are available at https://cumulusalpha.github.io/DexBioLabWebPage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.24049v2">Physics-Integrated Operator Learning via Gaussian Splatting Representations</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Neural operators provide efficient surrogates for spatiotemporal PDE systems, but purely data-driven formulations often accumulate substantial errors during long-horizon autoregressive prediction and may fail to exploit available governing-equation structure. Existing approaches incorporate physics primarily through residual-based training objectives or PDE-specific architectural constraints, which can introduce optimization difficulties or limit architectural generality. In this work, we introduce a representation-level approach to physics integration in which a feed-forward Gaussian splatting (FFGS) representation serves as a continuous interface between discretized solution fields and governing operators. The FFGS representation reconstructs the state as a continuous Gaussian field with closed-form spatial derivatives, allowing available physical PDE operators to be integrated directly within the learned evolution map without introducing a physics-residual loss. We evaluate the framework across two- and three-dimensional PDE systems, including advection, diffusion, nonlinear self-advection, and reaction dynamics. Over long-horizon autoregressive rollouts, the proposed framework reduces relative $\ell_2$ error by $1.5\times$--$2.2\times$ compared with the strongest purely data-driven baseline across the benchmark suite, while consistently improving spectral fidelity. The framework also remains effective when the governing equations are partially known, demonstrating robustness to incomplete physics. These results demonstrate that continuous field representations can provide a practical interface for incorporating known physical structure into generic neural-operator surrogates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19847v1">VisibilitySplat: Ray-Traced Occlusion for Relighting with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 10 pages, 5 figures, 3 tables. Published at WSCG 2026
    </div>
    <details class="paper-abstract">
      We address inverse rendering with 3D Gaussian Splatting (3DGS) by introducing explicit visibility modeling for relighting as well as material and light decomposition. Our approach augments the standard rasterized 3DGS pipeline with a sparse visibility oracle that estimates transmittance-weighted occlusion for Gaussian primitives from sampled lighting directions. This oracle is implemented using a point-based ray tracer accelerated by a bounding volume hierarchy (BVH) and queried sparsely within the inverse-rendering objectives, while the original 3DGS loop is retained for training efficiency, interactive visualization, and tooling compatibility. Shading follows a physically based rendering (PBR) formulation with a principled microfacet bidirectional reflectance distribution function (BRDF) and low-resolution high-dynamic-range (HDR) environment maps. We evaluate our method on the TensoIR synthetic dataset against baselines based on neural radiance fields (NeRFs), tensor representations, and 3DGS. The results show competitive performance and improvements over several earlier baselines for novel-view synthesis (NVS), albedo reconstruction, and relighting. Ablations indicate that explicit visibility modeling is an important contributor to relighting and material-illumination disentanglement, while pure PBR supervision can favor appearance fitting in NVS. Overall, the findings suggest that sparse, point-based visibility queries effectively complement rasterized splatting for inverse rendering while preserving the practical advantages of the 3DGS ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19742v1">HyperLoop: Hyperspectral-Assisted Loop Closure Detection in 3D-SLAM for Planetary Exploration</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 Paper accepted at International Astronautical Congress 2026, IAC-26,A3,IP,163,x107111
    </div>
    <details class="paper-abstract">
      Autonomous navigation of planetary rovers is challenged by visually repetitive, featureless terrain. SLAM systems based on 3D Gaussian Splatting (3DGS) offer dense mapping but rely on RGB appearance descriptors that degrade under perceptual aliasing - a common failure mode in extraterrestrial environments where physically distinct locations appear visually identical. This paper presents HyperLoop, the first framework integrating hyperspectral imaging (HSI) into the loop closure module of a 3DGS-based SLAM system. Hyperspectral descriptors encode material-level spectral signatures complementing RGB features, providing robustness to illumination changes and visual repetitiveness. Two HSI descriptor families are introduced: Bag-of-Spectral-Words (BoSW), encoding spectral material distributions via TF-IDF weighted histograms, and PCA-K VLAD (PCAK-VLAD), capturing first-order residual statistics of PCA-compressed spectral signatures. Both are evaluated with SIFT-VLAD and NetVLAD RGB descriptors under Strict-AND and Weighted fusion strategies on a custom dataset from the Moon-Mars Outdoor Test Site (MMOTS) at DLR, comprising synchronized RGB-D and hyperspectral sequences across four planetary analog traverses (65.5 m total). A key finding is that the sensor's Red-NIR range (600-860 nm) provides limited discriminability for rock terrain: the best HSI fusion (SIFT-VLAD + BoSW, Strict-AND) achieves 5.04 cm ATE RMSE, within 1.8% of SIFT-VLAD alone (4.95 cm). The gain over NetVLAD (8.81 cm) is attributable to SIFT-VLAD, not HSI fusion. This motivates future adoption of SWIR sensors capturing diagnostic mineral absorption features absent from the 600-860 nm range for planetary SLAM missions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19538v1">RDCompGS: Rate-Distortion Optimization for Compression of 3D Gaussian Splatting with Image Codecs</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 This manuscript has been submitted to IEEE Transactions on Multimedia (TMM)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables high-quality, real-time novel view synthesis but requires significant storage and transmission resources. Mapping Gaussian attributes to 2D representations enables efficient compression using mature image codecs. However, Gaussian representations that achieve comparable rendering quality are generally treated similarly during optimization, despite substantial differences in coding rate, leading to limited rate-distortion (R-D) performance after compression. To address this limitation, we propose RDCompGS, a 3DGS compression method that enables overall R-D optimization of Gaussian representations with respect to subsequent coding. A differentiable estimate of the coding rate is incorporated into the R-D optimization with rendering distortion. The rate constraint adjustment model further regulates the R-D trade-off across different compression conditions. Extensive experiments on three benchmark datasets demonstrate that RDCompGS achieves average Bjøntegaard Delta Rate reductions of 37.84% and 49.31% with H.265/HEVC and JPEG XL, respectively. The estimated rate also shows a strong correlation with the actual bit-rate under different compression conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19502v1">Holi-Spatial 360: Generative Reconstruction from Sparse Panoramas for Spatial Rendering and Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-10-14
    </div>
    <details class="paper-abstract">
      Spatial intelligence requires scalable 3D environments that are both photorealistic and semantically grounded, yet existing indoor datasets are dominated by room-scale captures and costly LiDAR acquisition. Online real-estate platforms offer abundant panoramas of diverse multi-room homes, but sparse viewpoints and texture-poor regions make reconstruction severely under-constrained. We introduce Holi-Spatial 360, a generative reconstruction and data curation framework for converting these panoramas into annotated multi-room 3D environments. Because reconstruction quality bounds subsequent generation, our Correspondence-Guided Geometric Optimization converts feed-forward reconstruction priors into reliable cross-view constraints and builds a coherent 3D Gaussian Splatting representation as 3D spatial memory. Adapted single-image and video priors then restore appearance and complete continuous trajectories while preserving observed content. The reconstructed scenes support arbitrary RGB-D rendering and an automated curation pipeline that combines end-to-end 3D grounding with VLM-based verification. Holi360 contains over 1K multi-room 3DGS scenes, more than 100K instance annotations, and 2M spatial question-answer pairs. We also introduce the Holi360 benchmark to evaluate the spatial understanding of VLMs in real-world, multi-room indoor environments. Experiments demonstrate superior reconstruction and curation quality under sparse panoramic inputs, while fine-tuning VLMs on Holi360 substantially improves spatial reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.01458v2">LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World</a></div>
    <div class="paper-meta">
      📅 2026-10-14
      | 💬 https://hojjunekim.github.io/legs/
    </div>
    <details class="paper-abstract">
      Training vision-language-action (VLA) policies for humanoid loco-manipulation is constrained by the high cost and complexity of collecting human teleoperation demonstrations. VLA policies fine-tuned in simulators have so far transferred poorly to humanoid loco-manipulation tasks. We present LEGS (Loco-manipulation via Embodied Gaussian Splatting), a hybrid simulator that composites a mesh foreground (robot, objects, props) over a photorealistic 3D Gaussian Splatting (3DGS) background reconstructed from a handheld scene capture. LEGS uses a procedural motion-primitive generator to synthesize labeled demonstrations at scale without human teleoperation, and a two-stage color calibration that aligns renders to the deployment camera. On a Unitree G1 humanoid robot, across three pick-and-place tasks of increasing whole-body difficulty and three VLA backbones ($ψ_0$, $π_{0.5}$, GR00T N1.6), a policy trained purely on LEGS data matches or exceeds one trained on human teleoperation demos on every experiment. It also outperforms a mesh-only baseline trained on the same motions without the calibrated 3DGS frontend. Because motion is recorded independently of appearance, the same demonstrations can be re-rendered for a target scene and objects at over 15x lower cost than teleoperation. When both the objects and the scene change, a policy trained on LEGS-AUG data re-rendered for that target maintains task success, while the teleop baseline fails entirely. Code and videos are available at https://hojjunekim.github.io/legs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.19120v1">DoReMe: Dual optimization of Reconstruction and Motion estimation for self-navigated free-breathing abdominal MRI using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 25 pages, 2 tables and 7 figures
    </div>
    <details class="paper-abstract">
      Purpose: Free-breathing abdominal MRI is limited by respiratory motion, which introduces image artifacts and k-space inconsistencies. Existing reconstruction approaches often depend on pre-estimated motion fields or external respiratory surrogate signals. We propose a coupled reconstruction-motion estimation algorithm for self-navigated free-breathing acquisition, in which motion is represented by compact Gaussian primitives and jointly refined with the image estimates during iterative reconstruction. Methods: An iterative reconstruction that alternates between data consistency updates and motion modelling was proposed. Respiratory motion was represented using Gaussian primitives. A Gaussian splatting was integrated into the reconstruction algorithm. The method was evaluated on self-navigated free-breathing Cartesian gradient-echo (GRE) acquisition in four healthy volunteers, including retrospective scan shortening experiments, and was compared with regularized compressed sensing (CS) reconstruction. Results: Respiratory motion states were reconstructed in approximately 1.5 min. Compared with CS, the proposed method improved visual image quality, particularly in coronal liver views, and better preserved anatomical structures in retrospectively shortened acquisitions. These findings provide proof of concept for the proposed method. Conclusion: The proposed coupled reconstruction-motion estimation algorithm improved visual image quality compared with CS alone. Gaussian primitives provided a compact and sufficiently flexible representation of nonrigid respiratory motion with practical runtime, supporting further scanner integration and validation in larger cohorts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03877v2">MLP Splatting: Object-Centric Neural Fields</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 Presented at ECCV 2026
    </div>
    <details class="paper-abstract">
      3D representations are fundamental for scene rendering, understanding, and interaction. Approaches like Neural Radiation Fields and 3D Gaussian Splatting achieve impressive photorealistic novel-view synthesis, but lack the ability to easily decompose scene elements into object level, requiring additional segmentation or grouping. We present MLP-Splatting, a method that enables scene decomposition via a few expressive light-field primitives while providing photorealistic novel-view synthesis. MLP-Splatting models each primitive as an independent compact MLP with localized spatial support that predicts radiance and opacity. In contrast to low-level Gaussian primitives or a single global radiance field, our neural primitives provide greater expressive capacity while remaining spatially localized. Rendering is performed through efficient sparse volumetric compositing over ray-primitive interactions. Our primitives are supervised using RGB supervision alone, which yields primitives that represent local scene regions often corresponding to objects or object parts, enabling interactive object-level editing without segmentation masks by selecting a handful of primitives. Our method, augmented with optional semantic feature distillation, enables open-vocabulary scene interaction and open-set instant segmentation. Compared to state-of-the-art semantic 3DGS methods, we achieve substantially lower memory usage (1/7$\times$) and faster rendering (5$\times$). Project Page: https://shinjeongkim.com/mlp-splatting
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01581v4">Satellite-Free Training for Drone-View Geo-Localization</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 The authors previously withdrew this paper to allow for substantial revisions to the original manuscript, including extensive new experiments, additional analyses, and significant improvements to both the methodology and presentation. The thoroughly revised version of the paper has now been released, and we warmly welcome readers to explore the updated work
    </div>
    <details class="paper-abstract">
      Drone-view geo-localization (DVGL) typically assumes access to satellite imagery during task-specific development. We study a late-arriving-gallery setting, where model development relies only on grouped source UAV observations and the target satellite gallery remains unavailable until the model is frozen. We propose \ours, Orbit Completion Learning, which uses source-side multi-view geometry as privileged training information while preserving image-only deployment. OCL reconstructs each source scene using structure-from-motion and 3D Gaussian Splatting, and renders canonical complementary views to complete its observation orbit. An exponential-moving-average teacher aggregates the completed orbit into a scene-level target, while a variable-cardinality student learns to recover this target from partial observations through distillation and contrastive learning. At inference, the frozen student independently encodes UAV queries and satellite gallery images using the same single-image path, without reconstruction or target-gallery adaptation. Experiments on University-1652, SUES-200, and IRVL328, together with nine weather-style appearance shifts, evaluate generalization across altitude, modality, and appearance variations. OCL achieves 57.85\% R@1 and 68.44\% mAP on University-1652, and obtains the best R@1 among the evaluated target-satellite-free methods in eight of nine weather conditions. Controlled ablations further show that completed teacher context provides gains beyond student-side synthetic-view augmentation. The code and data are publicly available at https://github.com/liutao23/OCL-DVGL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.18168v1">Back to the Future: Reverse-Consequence Modeling for Long-Horizon Particle Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Accurate prediction of object trajectories is fundamental to mechanical understanding and robotic control. However, autoregressive models inherently suffer from cumulative errors, leading to severe drift in long-horizon rollouts. Observing that in many physical systems with dissipative or convergence-to-rest behavior, the final states are often more predictable than transient dynamics (e.g., a sliding object resting under friction or a pendulum stabilizing at equilibrium), we propose ReStep, a cascaded approach that reverses the conventional forward-prediction paradigm. ReStep first employs a stochastic model to directly predict the final state, then leverages Anchored Diffusion to generate goal-conditioned intermediate trajectories. By conditioning on both ''boundary states'', it effectively constrains dynamics and reduces uncertainty, producing physically plausible and temporally coherent transitions. We further integrate ReStep dynamics as the physical engine within a 3D Gaussian Splatting pipeline, enabling a unified physical-visual framework for intuitive inference from images. Extensive experiments on diverse rigid and non-rigid collision datasets show that ReStep achieves superior long-term accuracy, substantially mitigating error accumulation via reverse-consequence modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.17936v1">PUGS: Physics-Guided Uncertainty Modeling for Underwater Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Dynamic 3D Gaussian Splatting (3DGS) provides an efficient framework for dynamic scene reconstruction. Most existing approaches model scene dynamics with deterministic deformation fields, assuming each Gaussian follows a distinct geometric trajectory over time. This assumption breaks down underwater, where light scattering, refraction, marine snow and low visibility create substantial physical ambiguity in captured images. We present PUGS, a physics-guided uncertainty model for underwater dynamic 3DGS. PUGS learns inherent uncertainty induced by underwater image formation and leverages it as a physical signal for dynamic scene modeling. Rather than estimating a single deterministic displacement, it parameterizes each Gaussian's motion as a probability distribution with expected deformation and physical uncertainty. We adopt stochastic deformation optimization to sample Gaussian trajectories via local reparameterization, exploring multiple physically plausible solutions during training. We further propose uncertainty stochastic annealing and a dual-path self-consistency distillation objective, converting stochastic exploration to deterministic reconstruction while preserving temporal coherence. Extensive experiments on underwater dynamic datasets and physics-aware benchmarks demonstrate superior rendering quality, temporal consistency and reconstruction robustness. Our method achieves 1.16 dB PSNR gain over the strongest baseline on UVEB and 3.99 dB on DRUVA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.17605v1">PLEDGE: Progressive Lossy Error-Bounded Compression of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-13
    </div>
    <details class="paper-abstract">
      Existing 3D Gaussian Splatting (3DGS) codecs report reconstruction quality after compression, and give users little control over the distortion of individual primitives. We present PLEDGE, a training-free, progressive, error-bounded codec that translates four user-declared tolerances into per-primitive quantization bounds: screen-space position, footprint deformation due to rotation and scale, view-dependent color, and opacity. The bounds apply throughout a declared camera domain, including previously unseen viewpoints. Position and opacity are bounded directly; footprint deformation is bounded under the rasterizer's affine projection model, and color is bounded over all viewing directions. PLEDGE organizes quantized residuals into a progressive bitstream with explicit error bounds for every valid prefix. Experiments across thirteen scenes verify all declared per-mechanism bounds and demonstrate competitive rate-distortion performance. The four tolerances incur substantially different rate costs, so users can spend rate where their application needs it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04203v3">Sparse-GS2Mesh: 3D Gaussian Splatting Guided by Novel Stereo Views and 2DGS for Sparse View Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 Accepted to NIPS 2026 PhysWorldAI workshops. Project Page: https://yhnoh8623.github.io/projects/s-gs2mesh/
    </div>
    <details class="paper-abstract">
      Surface reconstruction under sparse-view settings remains challenging due to limited geometric cues. Volume rendering methods based on signed distance functions often produce over-smoothed surfaces, while 3D Gaussian Splatting (3DGS), though time-efficient, suffers from incomplete geometry due to the lack of reliable depth supervision and the limitation of being optimized only from given input views. In this paper, we present Sparse-GS2Mesh, a stereo-aware framework for surface reconstruction from sparse views. While 3DGS and stereo matching have been leveraged for surface reconstruction under dense view settings, we extend them to operate effectively under sparse view conditions by first initializing 3DGS using epipolar depth priors to mitigate the 3DGS overfitting problem, followed by our three key components: (I) adaptive baseline selection, (II) fine-tuning with a stereo matching network, and (III) 2D/3D co-regularized fine-tuning. Given a warmed-up 3DGS initialized with epipolar depth, the adaptive baseline selection automatically determines a baseline to synthesize for each sparse view. We then fine-tune 3DGS by backpropagating depth-refining gradients from the stereo matching network, effectively specializing the 3DGS for stereo matching. The 2D/3D co-regularization further helps obtain stable reconstruction, addressing weak geometric cues in close stereo views. Sparse-GS2Mesh achieves a 15\% improvement over state-of-the-art methods in little-overlap settings and comparable results in large-overlap settings. Project page: https://yhnoh8623.github.io/projects/s-gs2mesh/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.17521v1">YUBI-Real2Sim: 3DGS Scene Verification for Reliable Policy Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      Evaluating VLA policies on real robots is costly, time-consuming, and difficult to reproduce, making real-to-sim simulation essential for scalable evaluation. While 3D Gaussian Splatting (3DGS) enables rapid real-to-sim scene generation, the reconstruction quality required for reliable policy evaluation remains unquantified. In particular, policies relying on wrist-mounted fisheye cameras (e.g., YUBI handheld gripper data) are heavily exposed to visual artifacts (holes, floaters) in wide-angle views, directly degrading task performance. To address this, we present YUBI-Real2Sim, a unified framework for 3DGS-based scene reconstruction and real-to-sim policy evaluation under dual fisheye wrist views. We develop an integrated pipeline combining 200-degree fisheye camera capture, 3DGS training, 3D foundation models for object modeling, and MuJoCo-based physical simulation, reproducing the observed real-world policy ordering in eight of nine scene-task comparisons. Using this framework, we construct a benchmark across two bimanual robot platforms, three scenes, three tasks, and dual pi0.5-based policies. This systematically measures real-to-sim rank agreement and evaluates policy sensitivity against counterfactual 3DGS artifacts (ceiling holes, floaters, warping) and capture disturbances. We further analyze whether novel-view quality scores can indicate policy-ranking reliability under a scene-held-out evaluation protocol. The results show task-dependent performance and limitations of quality-based reliability prediction, highlighting the need to assess reconstruction quality when using Real2Sim for policy evaluation. These results do not yet establish a generally reliable scene-acceptance rule. Project Page: https://yubi-real2sim.airoa.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.17513v1">Gaussian Splatting for Real-Time Scene Reconstruction: Methods, Systems, and Trade-offs</a></div>
    <div class="paper-meta">
      📅 2026-10-13
      | 💬 Accepted version to Els Pattern Recognition
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) allows for high-quality real-time novel-view synthesis. Since its inception in 2023, its explicit and highly configurable scene representation has been extended toward sparse-view generation, dynamic reconstruction, efficiency and compression, streaming, and simultaneous localization and mapping (SLAM). Although Gaussian Splatting can provide real-time rendering, rendering an existing representation is distinct from constructing, updating, or maintaining the scene itself. In this survey, we synthesize current literature, categorize them and provide a connection of current approaches and tools towards an end-to-end pipeline. Explicitly, we categorize GS scene reconstruction into five overlapping families: single-shot generation, dynamic representations, efficiency and compression, streaming and immersive systems, and Gaussian Splatting SLAM. We distinguish render-only throughput from task-specific scene generation, update, tracking, mapping, and optimization costs, and combine controlled sparse-view benchmarking with native evaluations to identify cross-family quality, latency, memory, and update-cost trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15431v2">SplatBus: A Gaussian Splatting Viewer Framework via GPU Interprocess Communication</a></div>
    <div class="paper-meta">
      📅 2026-10-12
      | 💬 4 pages, 2 figures, 4 tables. Accepted to SIGGRAPH Asia 2026 Technical Communications. v2: camera-ready version
    </div>
    <details class="paper-abstract">
      Radiance field-based rendering methods have attracted significant interest from the computer vision and computer graphics communities. They enable high-fidelity reconstruction with complex real-world lighting effects, but at the cost of high rendering time. 3D Gaussian Splatting solves this issue with a rasterization-based approach for real-time rendering, enabling applications such as autonomous driving, robotics, virtual reality, and extended reality. However, due to the lack of format and rendering standardization, the ecosystem is fragmented into specific viewers. This makes it difficult to integrate these methods into traditional mesh-based rendering pipelines, which is a common use case for interactive applications. To address this limitation, we propose a framework that uses Nvidia's interprocess communication (IPC) APIs to provide an implementation-agnostic rendering server, enabling the results to be viewed in external clients such as Unity, Blender, and OpenGL viewers. The code is available at https://github.com/RockyXu66/splatbus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23675v2">GS-DOT: Gaussian splatting-based image reconstruction for diffuse optical tomography</a></div>
    <div class="paper-meta">
      📅 2026-10-12
    </div>
    <details class="paper-abstract">
      This work presents GS-DOT, a novel image reconstruction framework based on Gaussian Splatting (GS) for diffuse optical tomography (DOT). Inspired by GS for rendering applications, absorption coefficients are represented as a sparse sum of anisotropic Gaussian primitives optimized to fit measured time-resolved point-spread functions through analytic gradients and Adam optimization. This is the first adaptation of GS algorithms in the photon diffusion regime, where the ray transport function is replaced by the diffusion functions to enable accurate modeling of light transport in highly scattering media. Validation on synthetic tissue models demonstrate high accuracy in localization and quantification of reconstructed absorption maps for both clean and noisy signals. GS-DOT has demonstrated high robustness to noise and showed a huge reduction in memory demand.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.16292v1">3D Gaussian Splatting for Fine-grained Urban Scene Reconstruction via Semantic-guided Density Control and Loss Design</a></div>
    <div class="paper-meta">
      📅 2026-10-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has been becoming a promising 3D reconstruction technique because of its advantage of high-fidelity rendering effect with high efficiency. However, it still suffers from coarse density control and semantically blind loss functions when reconstructing the fine-grained structures in complex urban scenes. Thus, this study proposes SG-GS, a semantic-guided 3DGS method for fine-grained structure reconstruction. It takes the semantic segmentation deduced by the Segment Anything Model (SAM) as a constraint and systematically integrates it into both Gaussian density control and loss function design. On the one hand, a semantic-constrained adaptive density control mechanism is constructed, which dynamically allocates gradient thresholds to different semantic regions of the scene and introduces an additional boundary-enhanced densification strategy, so that new Gaussians are directed to structurally critical regions. On the other hand, a multi-feature guided loss function is designed, and fine-grained supervision signals are gradually introduced through a phased progressive training strategy, alleviating the artifacts caused by the lack of semantic constraints in the basic loss. Experiments on three datasets show that the proposed method consistently improves reconstruction accuracy. Compared with the original 3DGS, the PSNR increases by approximately 2.7% to 5.2%, while artifacts on linear structures and boundary blurring are effectively mitigated. The method can serve the fine-grained 3D reconstruction of urban scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.16177v1">DIGS: Distribution-Informed Gaussian Splatting for Training-Free Open-Vocabulary 3D Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-10-12
    </div>
    <details class="paper-abstract">
      Open-vocabulary 3D Gaussian segmentation typically binds semantic features to the per-Gaussian state via per-scene gradient training (1-4 h) or the more recent closed-form distillation. Both commit each primitive to a single feature; therefore, boundary Gaussians spanning multiple 2D instances are forced into a hard decision that degrades retrieval accuracy and yields unstable mask boundaries at render time. We present DIGS, a training-free framework with two stages, class-agnostic mapping and multi-modal encoding, that decouples instance discovery from linguistic naming. At the Gaussian level, our distribution-informed representation maintains an explicit top-$M$ label distribution per primitive, updated by a tempered Dirichlet rule; preservation of multi-peaked label distributions at boundary primitives allows a late-argmax renderer to recover clean instance boundaries. A bidirectional state machine aggregates these distributions into a persistent class-agnostic instance ontology. The instance-level encoding stage then fuses the CLIP visual embedding of each instance's isolated 3D footprint with the CLIP-text embedding of an LLM-generated attribute description, disambiguating visually similar instances. On a single A100, DIGS completes mapping and instance encoding on LERF-OVS in 1.0-3.2 min per scene, depending on whether LLM fusion is enabled, and serves native 3D queries in 5.4 ms. It reaches 71.95% mean mIoU on LERF-OVS, outperforming both the training-free SFS and training-based LaGa baselines. DIGS also achieves the highest mean mIoU on LERF-Mask and 3D-OVS and surpasses Gaga and ObjectGS on Replica under their respective evaluation protocols. Code is available at https://github.com/ZephyrZos/DIGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.14963v1">AD-GS: Anatomical Density-guided 3D Gaussian Splatting for Sparse-View Dental CBCT Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-10-10
    </div>
    <details class="paper-abstract">
      Sparse-view Cone-Beam Computed Tomography (CBCT) reconstruction is essential for reducing radiation dose and scan effort, but remains underconstrained due to limited projection information. Recent 3D Gaussian Splatting (3DGS)-based CBCT reconstruction methods rely on gradient-driven densification, which becomes unstable under sparse acquisition and often leads to over-smoothed structures. We propose AD-GS, an Anatomical Density-guided 3D Gaussian Splatting framework that formulates Gaussian densification as an attenuation-aware Markov Chain Monte Carlo (MCMC) sampling process. By interpreting Gaussians as samples from an underlying anatomical attenuation distribution, AD-GS adaptively concentrates representation capacity in high-attenuation regions, enabling sharper structural boundaries under sparse projections. To further constrain reconstruction, we incorporate a semantic regularization based on zero-shot features from Segment Anything in Medical Images, enforcing multi-view consistency through semantic priors. Experiments on the ToothFairy3 dental CBCT dataset demonstrate that AD-GS achieves superior structural fidelity and boundary preservation compared with existing 3DGS-based approaches under sparse-view settings within the dental and maxillofacial setting. Code is available at https://github.com/yulinjiaxue/AD-GS-BMVC2026.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.21258v2">Spectral GS-SLAM: Observability-Aware, Degeneracy-Robust Tracking for Real-Time 3D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-10-10
      | 💬 This work has been accepted to IROS 2026. Project site: https://edward62740.github.io/spectralgsslam, Code: https://github.com/edward62740/Spectral-GS-SLAM
    </div>
    <details class="paper-abstract">
      Recent 3DGS-SLAM systems enable real-time operation by leveraging conventional feature matching or ICP-based tracking, thereby avoiding the heavy dense photometric optimization used in earlier approaches. However, feature matching remains prone to failure in textureless environments, while ICP-based tracking struggles in structureless or geometrically degenerate scenes due to ill-conditioned optimization. To address this issue, we propose Spectral GS-SLAM, an efficient yet robust tracking framework that integrates ICP with complementary feature-based constraints. Our method mitigates numerical instability by adaptively compensating under-constrained directions in degenerate scenarios, without interfering with the shared Gaussian representation used for mapping. We further introduce a Gaussian-aware planarity weighting mechanism that exploits the intrinsic covariance structure of 3D Gaussians to characterize scene geometry and guide information fusion. Extensive evaluations on challenging TUM RGB-D sequences demonstrate that Spectral GS-SLAM achieves real-time performance (40.14 FPS) while maintaining consistent tracking in both structureless and featureless environments. The proposed method preserves trajectory integrity in degenerate scenes while maintaining competitive performance in non-adverse conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.14685v1">Demiurge: Forging Navigable 4D Worlds via Autoregressive Video Rerendering</a></div>
    <div class="paper-meta">
      📅 2026-10-10
    </div>
    <details class="paper-abstract">
      Recent advances in video diffusion enable novel-view video synthesis from a single input video, supporting world modeling, embodied intelligence, and robotic simulation. However, existing methods process camera trajectories independently, lacking the cross-trajectory consistency needed for coherent, navigable 4D worlds. We introduce Demiurge, an autoregressive video rerendering pipeline built on a dynamic 4D point-cloud memory cache. It promotes cross-trajectory consistency through four components: (1) an appearance video diffusion model that synthesizes photorealistic novel-view videos conditioned on the 4D memory; (2) a memory-consistent video depth completion model that aligns known-region depths with cached geometry and completes missing regions, mitigating scale ambiguity and discontinuities during memory updates; (3) memory-aware dual-stream warping that provides high-fidelity visual guidance and occlusion cues for novel-view synthesis; and (4) geometry-aware video refinement that enhances perceptual quality. Extensive experiments demonstrate state-of-the-art performance and three key capabilities: coherent spatiotemporal lattice construction over dense multi-view trajectories; large-parallax video rerendering under challenging camera motions, including 180-degree rotations with no input-view overlap; and geometry-faithful outputs that can be directly optimized into high-quality 3D Gaussian Splatting representations with minimal artifacts. Project page: https://hanxinzhu-lab.github.io/demiurge_project_page/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.14663v1">TETRIS-4DGS: Efficient Post-Training Compression of 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-10
      | 💬 ICME 2026
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) has emerged as a powerful dynamic scene representation, supporting real-time high quality rendering but suffers from prohibitive storage overhead. Existing methods typically attempt to solve this by treating compression as a per-scene optimization problem, limiting generalizability and applicability where low-latency compression is required. In this work, we leverage existing image and video codecs and propose a generalized post-training compression framework for 4DGS to address these issues. Specifically, we apply importance-based adaptive quantization to Gaussians according to their importance levels, effectively refining bitrate allocation while remaining fully compatible with existing codecs. To further improve compression efficiency, we utilize the statistical properties of 4DGS and propose motion- and importance-aware packing to reorganize attributes into codec-friendly 2D maps. Our approach can achieve over 40x compression rate in seconds while maintaining high quality, facilitating the practical deployment of 4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.05289v2">Mobile-4DGS: Unified Static-Dynamic Real-time Mobile Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-10
      | 💬 Code has been released: https://xiaobiaodu.github.io/mobile-4dgs-project/
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have achieved remarkable performance in novel view synthesis, yet deploying both static and dynamic Gaussian representations on resource-constrained mobile devices remains challenging due to heavy storage, redundant primitives, and costly per-frame computation. We present Mobile-4DGS, a unified lightweight framework for high-fidelity real-time static and dynamic Gaussian rendering on mobile platforms. For compact appearance modeling, we introduce a Monte Carlo Specular Energy Aggregator that compresses high-order radiance residuals into the first-order Spherical Harmonics (SH), together with an Attribute-Conditioned SH Enhancement module whose predicted offsets are pre-baked before inference. We further propose a Multi-View Alpha-Based Densification and Pruning strategy to suppress redundant primitives while maintaining multi-view consistency. For dynamic scenes, we develop a compact explicit 4D representation by constructing second-order Gaussian motion, learnable temporal support, and a binary static-dynamic partition, enabling continuous-time modeling without runtime deformation networks. Based on this partition, a Depth-Order Certificate selectively reuses previously committed depth orders to reduce re-projection, sorting, merging, and index-buffer updates during playback. Extensive experiments on static and dynamic scenes demonstrate that Mobile-4DGS substantially reduces storage and rendering overhead while maintaining competitive visual quality, enabling real-time 3D and 4D Gaussian Splatting on mobile devices. \textcolor{magenta}{\href{https://xiaobiaodu.github.io/mobile-4dgs-project/}{Code has been released: https://xiaobiaodu.github.io/mobile-4dgs-project/}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13945v1">Splat2Act: From Captured Scenes to Personalized VLA Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-10-09
      | 💬 Project page: https://splat2act.github.io
    </div>
    <details class="paper-abstract">
      A user who brings a generalist manipulation policy into their home or office wants it to pick up their objects from their shelves, counters, and tables. Yet even a strong generalist like $π_{0.5}$ achieves 0% pickup success in our captured deployment scenes without adaptation, and collecting real demonstrations in every user's environment is prohibitively expensive. We present Splat2Act, an automated real-to-sim-to-real pipeline that personalizes a pretrained vision-language-action (VLA) policy to a specific scene and object set using only a handheld phone scan and images of the user's objects. Unlike prior real-to-sim work that reconstructs individual tabletops, Splat2Act operates on building-scale indoor environments: it builds a photorealistic digital twin with 3D Gaussian Splatting and collision geometry, automatically discovers tabletop and non-tabletop receptacles, constructs feasible pickup tasks, and generates successful demonstrations with a privileged simulation oracle, with only scene and object capture required from a human. Applied to 101 scanned campus environments with a median navigable area of 104.87 m$^2$ and 7 receptacles, the pipeline yields 852 receptacles and over 68,000 pickup demonstrations. We fine-tune $π_{0.5}$ in two stages: Stage 1 adapts to hundreds of public object assets, drawn either from the deployment scene alone or from all 101 scenes; Stage 2 personalizes to the user's own objects on the deployment scene's receptacles. Our strongest personalized policy reaches 62.5% aggregate pickup success in the target scene. Deployed on a real Stretch 3, the same policy achieves 63.9% success on a 12-task subset, closely tracking its 66.7% simulated success on those tasks. These results suggest that automated real-to-sim-to-real training can personalize VLA manipulation policies using only simulation-oracle data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13860v1">Make It Flow: Animating Water in 3D Gaussian Splatting Scenes</a></div>
    <div class="paper-meta">
      📅 2026-10-09
      | 💬 Overview video at https://youtu.be/qf04D73kgrc
    </div>
    <details class="paper-abstract">
      Explicit 3D scene representations such as 3D Gaussian Splatting (3DGS) have recently enabled high-quality, real-time novel view synthesis. However, these methods assume static scenes, limiting realism in the presence of dynamic phenomena ubiquitous in real-world environments. While coherent motion (e.g., humans or vehicles) can be captured and reconstructed, high-frequency effects such as flowing water remain intractable under casual capture conditions, where dense multi-view, high-speed acquisition is not feasible. We present a method to animate water in static 3DGS scenes, enabling visually plausible motion without requiring dynamic capture or physical simulation. Our approach estimates 2D optical flow from reference images or model renderings and lifts it into a consistent 3D Eulerian motion field. We propose a cinemagraph-inspired algorithm that applies this motion field to deform Gaussian primitives, producing temporally seamless looping animations. The resulting representation retains real-time rendering performance, significantly enhances perceptual realism, and preserves free-viewpoint navigation within the scene. We evaluate our method on a newly captured dataset of in-the-wild waterfall scenes and demonstrate clear improvements over prior work in dynamic Gaussian splatting and 3D cinemagraph generation in visual quality and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13781v1">SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models</a></div>
    <div class="paper-meta">
      📅 2026-10-09
      | 💬 Project page: https://opendrivelab.com/SimVTA/
    </div>
    <details class="paper-abstract">
      Driving world action models (WAMs) learn from paired trajectories and future observations. Post-training with reinforcement learning allows the policy to explore alternative trajectories and improve planning using driving rewards. However, driving logs contain future videos only for the recorded trajectories, leaving these alternatives without corresponding visual supervision. Thus, we introduce SimVTA, a simulation-grounded post-training framework that combines trajectory optimization with video supervision. First, we sample candidate trajectories for each observation and use a real-world simulator (e.g., World Engine) to render future videos for high-reward candidates in real-world scenes reconstructed with 3D Gaussian Splatting (3DGS). This provides trajectory-matched supervision independent of the model's own predictions. Under a limited rendering budget, we prioritize video supervision for higher-error futures among these rendered candidates. Finally, we optimize trajectory planning with Group Relative Policy Optimization (GRPO) and use a flow-matching loss on the matched videos to update shared representations. At inference time, the model predicts trajectories directly, without simulator calls. On NAVSIM v1, applying SimVTA to a WAM initialization improves PDMS from 86.2 to 91.3 (+5.1). It achieves 88.2 EPDMS on NAVSIM v2 and an average HD-Score of 38.8 in closed-loop evaluation on HUGSIM. Ablations show that updating shared representations with video supervision improves planning, while prioritizing higher-error futures among high-reward trajectories further improves performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13757v1">Gaussian Splatting with Reflections Guided by What is Seen</a></div>
    <div class="paper-meta">
      📅 2026-10-09
      | 💬 Accepted to IEEE ICIP 2026
    </div>
    <details class="paper-abstract">
      While novel view synthesis methods like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have shown impressive performance on common scenes, they often struggle with capturing view-dependent specular effects, particularly near-field reflections. Existing methods utilize environment maps, which only represent lights from infinity, or environment Gaussians, which rely solely on indirect supervision from reflections, are insufficient for accurately modeling such effects. Here, we propose a framework based on 2D Gaussian Splatting (2DGS) that leverages directly observable environment information in input images. Our approach also integrates a hybrid reflection appearance representation and view-dependent specular modeling. Experiments on public benchmarks and a newly curated near-field reflection dataset demonstrate that our method achieves superior rendering quality and geometric accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13541v1">Efficient Confidence-Aware Densification for 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-10-09
    </div>
    <details class="paper-abstract">
      3DGS-MCMC has replaced the gradient-driven density control of 3D Gaussian Splatting with opacity-based sampling and relocation, and is now the de facto standard for prior-free densification under a fixed primitive budget. Because opacity stays high where the scene is already well reconstructed, this sampling sends most new primitives to the foreground and leaves the background under-reconstructed. At low budgets, prior-based methods such as ConeGS and EDGS outperform it, at the cost of a pre-trained radiance field or dense image matches and a separate initialization stage. We argue for a comeback of error-driven densification within the fixed-budget MCMC framework. Our confidence-aware densification keeps the fixed budget and gradual growth of 3DGS-MCMC, but places primitives by the image error instead of the opacity. New primitives are drawn mostly where screen-space gradients are large. We favor primitives with a low learned confidence, which marks regions the reconstruction still explains poorly, and relocate the primitives that contribute least to the rendered images to these places. Averaged over 24 scenes, our method outperforms 3DGS-MCMC in PSNR, SSIM and LPIPS at every budget from 100k to 2M primitives. At 100k, without a prior or an initialization stage, it recovers 0.84dB of the 1.05dB PSNR by which ConeGS leads 3DGS-MCMC, and from 500k on, it has the highest PSNR and SSIM of all fixed-budget methods we compare with, including ConeGS and EDGS. While inherently prior-free, our densification can also exploit priors: from the ConeGS or EDGS initialization, it matches ConeGS in PSNR and SSIM at 100k and is best in all three metrics at 500k.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13503v1">SplatQ: Interaction-Aware Pruning of 3D Gaussian Splatting via Quadratic Selection</a></div>
    <div class="paper-meta">
      📅 2026-10-09
      | 💬 Under review at ICLR 2027 (main paper 9 pages, 5 figures, 2 tables; appendix included)
    </div>
    <details class="paper-abstract">
      Existing 3D Gaussian Splatting (3DGS) pruning methods primarily rely on global Top-K selection based on independent importance scores, lacking regional budget control and ignoring joint deletion effects. Consequently, they cannot ensure balanced reconstruction quality across a scene. We propose SplatQ, a combinatorial pruning algorithm that coordinates regional budget allocation and joint Gaussian selection under a fixed budget. SplatQ partitions Gaussians by spatial proximity and appearance similarity into size-limited blocks, decomposing global selection into subproblems of a specific size. It then smooths importance-based block retention quotas to prevent excessive budget concentration. Finally, it constructs a quadratic approximation of joint deletion effects using the responses of representative pixel colors to changes in Gaussian opacity. Individual responses, pairwise interactions, and quota penalties are incorporated into a quadratic unconstrained binary optimization (QUBO) objective, solved using simulated annealing (SA) or a coherent Ising machine (CIM) to jointly select Gaussians. Experiments on 5 benchmarks demonstrate that SplatQ outperforms state-of-the-art pruning methods in aggregate reconstruction quality, spatial quality balance, and rendering speed. Ablations confirm that budget smoothing improves local detail reconstruction, while interaction modeling improves Gaussian selection quality under a fixed budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13498v1">A Hybrid and Hierarchical Ray Tracing Method for Efficient 3D Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2026-10-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a powerful representation for real-time neural reconstruction and rendering. While existing 3DGS systems rely on rasterization for efficiency, recent work explores ray tracing to enable secondary lighting, highly distorted cameras, and depth-of-field effects. However, compared with rasterization, ray-traced 3DGS introduces three major challenges: (1) the highly non-uniform spatial distribution of massive numbers of Gaussian primitives creates severe load imbalance; (2) neighboring pixels often generate rays with highly similar traversal paths, leading to substantial redundant traversal overhead; and (3) supporting both Gaussian and triangle primitives yields heterogeneous intersection workloads that are not well matched to existing GPU architectures. We present H2-Ray, a hybrid and hierarchical ray tracing architecture for efficient 3D Gaussian rendering. H2-Ray introduces three techniques. Load-Balanced Spatial BVH (LB-SBVH) partitions the scene into uniform grids and uses density-aware representations to convert an imbalanced global workload into balanced local traversal units. Ray-Cluster Path Reuse (RCPR) lets a proxy ray share traversal state with neighboring rays to amortize redundant traversal. The Unified Intersection Engine (UIE) supports mixed Gaussian and triangle intersections through a mode-multiplexed datapath that shares 40% of its floating-point units across both primitive types. Across 8 scenes, H2-Ray achieves a geometric-mean speedup of 3.77x and a maximum speedup of 9.26x over RaySplats on the NVIDIA RTX A5000, combining a cycle-accurate model of the modified RT Core with measured CUDA timings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.24114v2">COSY: Compositional 3DGS Synthesis for Disentangled Human Head Editing</a></div>
    <div class="paper-meta">
      📅 2026-10-09
    </div>
    <details class="paper-abstract">
      Recent 3D Gaussian Splatting (3DGS) GANs for human heads synthesize and render photorealistic 3D models in real-time and offer a vast variety in identity and appearance. However, controlling specific semantic attributes such as hair color or glasses remains challenging, as edits in the entangled latent space often induce unintended changes in identity or appearance. Although there are several methods that aim to disentangle the latent space post training by estimating directions that only modify certain features, these methods cannot guarantee complete disentanglement and often require pre-trained classifiers. In our approach, we propose a new generator architecture that synthesizes components, such as hair, skin, glasses, and torso, completely independently. This allows for changing the latent vector for one region while keeping the remaining parts fixed. Further, we achieve this separation using only sparse information such as the hair or skin color, eliminating the requirement of segmentation masks or geometric priors, often seen in prior work. To ensure matching shape and lighting conditions during editing, we allow minimal shared information via context tokens between the independent generators. These tokens even allow us to control the shape and light, without any prior annotation. Compared to existing works on GAN-based generation and editing, our method shows better disentanglement, more precise editing control, and competitive visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13309v1">NovaSplat: Model-Free 6D Object Pose Estimation via 3D Gaussian Splatting Onboarding and Refinement</a></div>
    <div class="paper-meta">
      📅 2026-10-09
    </div>
    <details class="paper-abstract">
      6D pose estimation of novel objects without CAD models remains an open challenge: model-free methods must build the object representation from scratch during onboarding and have long trailed their model-based counterparts. We present NovaSplat, a model-free pipeline that closes this gap. From a short smartphone video captured at onboarding, we train a 3D Gaussian Splatting (3DGS) model that serves both as a template source for detection and coarse pose estimation and as a differentiable backbone for pose refinement. Our refiner optimizes the object-to-camera pose with combined photometric and geometric losses under a continuous rotation parametrization free of gradient singularities. On a household-object benchmark, NovaSplat surpasses all model-based methods. On a challenging industrial benchmark with textureless and specular surfaces, it outperforms all model-free competitors while substantially narrowing the gap to model-based ones. We further introduce NovaSplat-Lite, an inference variant that is $3.1\times$ faster at negligible accuracy cost. Code will be made available at \href{https://github.com/POSE-Lab/novasplat}{https://github.com/POSE-Lab/novasplat}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13177v1">CompactSplat: Spatially Adaptive Gaussian Distribution for Feedforward Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-10-09
      | 💬 NeuRIPS2026, Generalizable reconstruction, 3DGS, 2DGS
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes from sparse views without per-scene optimization remains highly challenging, especially for recovering accurate geometry and fine textures. While recent feedforward approaches leverage generalizable 3D Gaussian Splatting (3DGS) for scene generation, they typically assign one or multiple Gaussians to each pixel. Such uniform allocation not only produces highly redundant representations by wasting primitives in homogeneous areas, but also fails to exploit the inherent geometric priors of planar surfaces, often resulting in sub-optimal structural modeling. To address these limitations, we present CompactSplat, a novel feedforward framework that enables a content-aware allocation of 3D Gaussians. By integrating texture-guided spatial partitioning with hierarchical sampling, CompactSplat adaptively distributes primitives - concentrating dense Gaussians in complex regions while tiling flat areas with expanded, scale-modulated primitives. Moreover, our framework inherently supports dynamic sparsity control at inference time, enabling seamless trade-offs between rendering fidelity and memory footprint without requiring network retraining. Extensive experiments on RealEstate10K, DL3DV, and ScanNet demonstrate that CompactSplat consistently outperforms prior methods on both standard metrics and high-resolution rendering consistency, achieving high-fidelity reconstructions with significantly fewer primitives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.13143v1">Measurable Semantic 3D Gaussian Splatting with Joint Region and Depth Supervision for Interactive Building Inspection</a></div>
    <div class="paper-meta">
      📅 2026-10-09
    </div>
    <details class="paper-abstract">
      Image-based 3D reconstruction provides a scalable representation for building inspection and maintenance. However, photorealistic reconstruction alone does not fully provide the geometric measurability, semantic structure, and interactive information access required for engineering applications. This study proposes a measurable semantic 3D Gaussian Splatting (3DGS) framework with region and depth supervision for interactive building inspection from multi-view imagery. Mask-based region supervision suppresses irrelevant backgrounds and focuses optimization on target regions. Point-wise depth supervision guides Gaussian centers toward physical surfaces using reference geometry derived from multi-view stereo (MVS) point clouds, providing geometric anchors for measuring physical dimensions and areas. A semantic Gaussian field associates learnable class features with individual Gaussian primitives, enabling semantic rendering and component-level inspection. These capabilities are operationalized through AEC-SplatTwin, an interactive building inspection platform supporting real-time visualization, semantic analysis, quantitative measurement, and multimodal large language model (MLLM)-assisted interaction. On the synthetic UAV-inspection QuakeCity and smartphone-captured HKUPB datasets, the framework reduces Gaussian-center depth MAE by 58.1% and 86.8%, while its semantic field achieves mIoU values of 0.8529 and 0.8642. On HKUPB, the mean relative errors for height, width, and area measurements are 0.33%, 1.07%, and 0.60%. These results show that the proposed framework extends 3DGS beyond visual reconstruction into a measurable, semantically structured, and interactive engineering representation for building inspection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12461v1">OuroWorld: Bringing Any 3D World Alive as Diverse, Endlessly Looping 3D Cinemagraphs</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Project page: https://ouroworld.userwei.com
    </div>
    <details class="paper-abstract">
      Recent 3D world models generate photorealistic, explorable scenes that remain frozen in time. OuroWorld is a mask-free framework that turns any static 3D Gaussian Splatting scene into a 3D cinemagraph: a dynamic scene with vivid, diverse motion looping seamlessly from any viewpoint. A vision-language model infers plausible dynamics and guides a video model to synthesize a reference video, which we lift and complete into multi-view videos. To learn from this imperfect supervision, we propose Inconsistency-Robust Periodic 4DGS: a Fourier-series deformation field guarantees looping by construction, while a Grounded Drift Field anchored at the reference view absorbs cross-view inconsistency. Unlike prior Eulerian methods limited to fluid-like motion, we capture general deformation, object motion, and illumination change. We introduce a ground-truth-free evaluation covering vividness, naturalness, loop seam coherence, and scene quality. On 39 reconstructed and generated scenes, OuroWorld outperforms all baselines and wins 70.8%-99.0% of user-study comparisons. Project page: https://ouroworld.userwei.com
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12127v1">LVS: Local View Synthesis from Relative Camera Pose by Reusing Previous Views</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Interactive scene exploration requires frequent view updates, although small camera motions preserve much of the visible content. Conventional 3D Gaussian Splatting nevertheless renders each target view, leaving this image overlap unexploited. Reusing rendered images offers an alternative. Geometric warping alone cannot recover newly exposed content and remains sensitive to depth errors. We propose a per-scene framework that replaces repeated scene rendering for nearby views with relative-pose-guided RGB-D image reuse. Geometric warping uses depth and relative pose to transport source content, while a lightweight multiscale network predicts RGB residuals to correct artifacts and infer missing appearance. Cached source features further reduce repeated computation. On GS-render, residual refinement improves PSNR by 0.72~dB over pure warping; evaluations on captured and rendered scenes demonstrate low query latency. This separation of scene rendering from local view updates supports responsive scene exploration, with potential applications in augmented and virtual reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10396v2">Gaussian Density Splatting Network</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 This is the preprint version of the paper and supplemental material to appear in NeurIPS, 2026. Please cite the final published version
    </div>
    <details class="paper-abstract">
      This paper proposes a novel crowd counting approach, the Gaussian Density Splatting Network (GDSNet). Unlike methods that rely on conventional, grid-based density maps and are sensitive to spatial resolution, GDSNet represents a crowd as a superposition of continuous 2D Gaussian primitives. Our approach is built upon two key contributions. First, a control-point-based fitting mechanism is introduced to structure the prediction of Gaussian parameters. A set of control points is adaptively allocated to define local regions, from which features are pooled to regress each primitive's parameters. Second, we adapt a differentiable Gaussian Splatting framework to the counting task by parameterizing each primitive with geometric parameters and a scalar density mass. This formulation allows the network to be trained end-to-end via spatial matching of differentiably rendered density maps, naturally providing both local density supervision and global count optimization. Extensive evaluations on four standard benchmarks show GDSNet consistently outperforms the state of the art. The code is available at https://github.com/infinite0522/GDSNet-Gaussian-Density-Splatting-Network.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11752v1">2DGS-Planner: Rasterization-based Path Planning in 2D Gaussian Splatting Map</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Gaussian splatting provides an explicit and efficiently rasterizable scene representation for robot navigation. However, individual Gaussian primitives may not reliably represent obstacles as they are jointly optimized through alpha-composited rendering from a finite set of reconstruction views. We propose 2DGS-Planner, a path planner for ground robots that reads planning-relevant geometry from a 2D Gaussian splatting (2DGS) map through rasterization, rather than treating individual Gaussian primitives as obstacles. During offline roadmap construction, multi-view attribution converts rendered normal dispersion into structural scores for non-ground disks supported by the reconstruction views. These scores guide adaptive node sampling on the ground. Path-aligned orthographic queries validate candidate edges, while cylindrical queries estimate local clearance fields that are cached on the edges. During online planning, graph search initializes a route, and path refinement reuses the cached fields while accounting for the robot's dimensions and ground constraints. Experiments demonstrate improved roadmap connectivity, more accurate clearance estimation, and higher planning success compared with the tested baselines. These results support rasterization as an effective geometric query interface for planning directly on Gaussian maps. Code and data are available at https://2dgs-planner.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11572v1">PAM-ToD: Plug-and-Play Appearance Modeling for Cross-Time-of-Day 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 21 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Adapting a pre-trained 3D Gaussian Splatting (3DGS) road scene to a new time of day requires learning appearance changes from a few anchor images while preserving consistent, real-time rendering. We propose PAM-ToD, a lightweight plug-in that learns color corrections while keeping the pre-trained 3DGS parameters fixed. PAM-ToD scales each Gaussian's existing color to model illumination changes and uses an additive term for additional brightness, such as when street lamps turn on at night. Under a simplified image formation model, unchanged surface albedo can be eliminated from the relation between source and target appearances, allowing us to learn these corrections without separately estimating albedo and illumination. The model corrects colors across the scene while allowing the corrections to vary by location and by Gaussian. To guide learning from a few anchor images, it discourages abrupt spatial changes in these corrections. We also introduce CARLA-ToD, a benchmark with matching geometry, camera poses, and moving-object trajectories across three times of day. A few target-time anchor images are used to train each plug-in, while separate views are used for evaluation. Across the static and dynamic settings, PAM-ToD achieves higher PSNR and lower LPIPS than the baselines, even when the anchor images come from a single synchronized capture across multiple cameras.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11547v1">OX-NeRF: 3D X-ray Tomography Reconstruction from Sparse Views Using Implicit Neural Representation</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      NeRF and Gaussian splatting methods have been successfully applied on X-ray scenes where the views are too sparse for 3D reconstruction via classical methods. Ultra-sparse scenes with 10 or fewer views such as those with high-rate or low-dose acquisition still, however, present a significant challenge. To address this problem we present a new framework, Optimised X-ray Neural Radiance Fields (OX-NeRF), that combines cross-scene feature learning with scene-specific optimisation to reconstruct sets of related scenes. OX-NeRF employs a convolutional neural network (CNN) to identify cross-scene features while maintaining scene-specific multi-resolution hash grids of spatial features. The paired representations are fused and passed to a multilayer perceptron (MLP); the CNN, hash grids and MLP are then jointly optimised end-to-end. Benchmarking on parallel-beam and cone-beam X-ray datasets shows OX-NeRF provides significantly higher reconstruction accuracy on ultra-sparse scenes compared to existing radiance field methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03538v2">Constrained Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      While Dynamic Gaussian Splatting enables high-fidelity 4D reconstruction, its deployment is severely hindered by a fundamental dilemma: unconstrained densification leads to excessive memory consumption incompatible with edge devices, whereas heuristic pruning fails to achieve optimal rendering quality under preset Gaussian budgets. In this work, we propose Constrained Dynamic Gaussian Splatting (CDGS), a novel framework that formulates dynamic scene reconstruction as a budget-constrained optimization problem to enforce a strict, user-defined Gaussian budget during training. Our key insight is to introduce a differentiable budget controller as the core optimization driver. Guided by a multi-modal unified importance score, this controller fuses geometric, motion, and perceptual cues for precise capacity regulation. To maximize the utility of this fixed budget, we further introduce an adaptive static-dynamic allocation strategy that separates the Gaussian representation into static and dynamic branches and distributes the shared global capacity between them according to motion complexity. Furthermore, we implement a three-phase training strategy to seamlessly integrate these constraints, ensuring precise adherence to the target count. After training, a dual-mode hybrid compression scheme further reduces storage overhead. CDGS therefore not only strictly adheres to the specified Gaussian-count budget (error<2%) but also achieves favorable rate-distortion performance. Extensive experiments demonstrate that CDGS delivers optimal rendering quality under varying capacity limits and favorable rate-distortion performance, achieving over 3x model compression compared with the state-of-the-art method Ex4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11364v1">FlyMark: Training-Free Invisible Watermarking of 3D Gaussian Splatting via a Fruit Fly Connectome</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      A trained 3D Gaussian Splatting (3DGS) scene ships as a portable parameter array that can be copied, pruned, requantized, or repackaged outside its training pipeline, so ownership evidence is most useful when it lives in the released parameters and remains checkable long after the embedding tooling is gone. Existing 3DGS watermarks typically tie embedding or extraction to scene optimization, a learned decoder, or rendered views, so the evidence survives only as long as a second trained artifact does. FlyMark instead writes a keyed message into the parameters a 3DGS file already stores. Its carrier directions are derived from the photoreceptors of a published connectome, a citable versioned artifact that fixes the geometry exhaustively and leaves nothing to tune per scene. A virtual observer reads cone-wise apparent luminance along a scene-normalized orbit from stored centers, colors, and opacities; a keyed dithered quantization-index-modulation code replicates each message bit across these observations; and one sparse bounded least-squares solve realizes the targets through achromatic shifts of existing degree-zero colors under a hard per-channel linear-RGB bound. All geometry and higher-order appearance parameters are preserved bit-identically, and extraction needs only cone queries, rounding, and majority voting. Under a model-domain threat model on synthetic and real scenes, FlyMark attains high clean bit accuracy and visual fidelity while cleanly separating matched from wrong keys.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11039v1">Rendering-Free Lookahead for Question-Guided Active Vision</a></div>
    <div class="paper-meta">
      📅 2026-10-08
      | 💬 Project page: https://k0uya.github.io/rfl-proj/
    </div>
    <details class="paper-abstract">
      Active robot vision requires controlling the camera to reveal task-relevant information that is hidden from the current viewpoint. For example, determining what is inside a box may require raising the camera and looking down into it. For viewpoint-dependent question answering, the challenge is to select camera motions that expose the visual evidence needed to answer the question. Although vision-language models (VLMs) can interpret observed images, selecting such motions requires anticipating the usefulness of unseen views. We quantify this usefulness as answerability, a VLM's estimate that a view suffices to answer the question, and present Rendering-Free Lookahead (RFL), a viewpoint-selection policy that ranks candidate camera motions by predicted future answerability. RFL transfers visual lookahead from deployment to offline training. At training, a privileged teacher renders candidate future views in 3D Gaussian Splatting (3DGS) scenes and uses a frozen VLM to compute one- and two-step answerability targets. Through two-stage distillation, a student learns to predict these action values from the question, recent visual observations, and a candidate camera motion. At deployment, RFL uses these predicted values to select camera motions without rendering future views. On 377 E3VS-Bench test episodes in unseen environments, RFL improves the mean judge score by 43\% over a direct-action baseline using the same VLM. These results support learning camera-control policies from privileged visual lookahead for viewpoint-dependent question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06912v3">Fast LapSum: Exact Differentiable Top-$k$ at Million Scale</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Selecting the top-$k$ elements is a fundamental operation for inducing sparsity in large-scale models and optimization problems, enabling robust expert activation, token routing or attention pruning. However, hard top-$k$ is non-differentiable, while existing differentiable alternatives become increasingly expensive as the number of coordinates grows. We introduce Fast LapSum, a scalable solver for the LapSum soft top-$k$ formulation that preserves an exact selection mass of $k$, while supporting end-to-end differentiation. In Fast LapSum, we reduce sorting cost using probabilistic bracketing, which restricts sorting to a narrow band of scores around the threshold using a binomial order-statistic from kernel-noised samples. A certification pass upgrades the probabilistic localization to a verified one at the cost of one additional linear pass, with a full-sort fallback that covers the worst-case scenario. Our certified GPU implementation processes $10^6$, $10^7$, and $10^8$ scores in median times of $0.92$, $1.39$, and $7.24$ ms, respectively, making exact-budget soft top-$k$ practical within million-scale optimization loops. We demonstrate this capability in two applications: megapixel sparse adversarial examples with a small fraction of initial image pixels, where Fast LapSum achieves an order-of-magnitude speedup over state-of-the-art methods, and 3D Gaussian splatting. In the latter, we use Fast LapSum to reduce the number of Gaussians produced by Adaptive Density Control to a substantially smaller number while retaining nearly the same rendering quality and massively reducing computation. These results demonstrate that exact-budget differentiable top-$k$ can be incorporated into practical million-scale optimization pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.14090v1">When pose graphs are not enough: a failure analysis of pose-graph-based camera estimation for Gaussian splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Exterior Gaussian-splat twins for real-estate listings are captured to a fixed recipe: a consumer drone orbits a single property in several overlapping passes at stepped altitudes, with occasional vertical ascents, so the imagery is dense, sequential, GPS- and time-tagged, and revisits each facade from many angles. This structure is tantalizing: it suggests camera poses could be recovered by a dense pose-graph optimization (PGO) whose only unknowns are the poses, far fewer than the joint pose-and-point problem bundle adjustment (BA) solves. PGO is mature in robotics, but almost always as post-processing: it redistributes the drift of an already-decent trajectory once loop closures tie it together. Splatting captures push that regime to its limit, every frame closing loops with dozens of others, raising a clean question: does pose-only optimization reach BA accuracy? We built a system specialized to this capture structure, not a general-purpose SfM tool, to test it. It beat COLMAP on speed but never on quality; this paper is the candid post-mortem. The shortfall arises from several interacting causes, hand-set weights, greedy merging and a noisy GPS ruler, together with two limits inherent to the formulation: SE(3) edges carry no scale degree of freedom, so independently scaled sub-maps drift when merged, and a two-view edge stops short of the coupling one 3D point imposes on every frame that sees it. A viable design must add back Sim(3) constraints, absolute GPS priors, retained structure, and uncertainty-aware weighting
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04203v2">Sparse-GS2Mesh: 3D Gaussian Splatting Guided by Novel Stereo Views and 2DGS for Sparse View Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Accepted to NIPS 2026 PhysWorldAI workshops. Project Page: https://yhnoh8623.github.io/projects/s-gs2mesh/
    </div>
    <details class="paper-abstract">
      Surface reconstruction under sparse-view settings remains challenging due to limited geometric cues. Volume rendering methods based on signed distance functions often produce over-smoothed surfaces, while 3D Gaussian Splatting (3DGS), though time-efficient, suffers from incomplete geometry due to the lack of reliable depth supervision and the limitation of being optimized only from given input views. In this paper, we present Sparse-GS2Mesh, a stereo-aware framework for surface reconstruction from sparse views. While 3DGS and stereo matching have been leveraged for surface reconstruction under dense view settings, we extend them to operate effectively under sparse view conditions by first initializing 3DGS using epipolar depth priors to mitigate the 3DGS overfitting problem, followed by our three key components: (I) adaptive baseline selection, (II) fine-tuning with a stereo matching network, and (III) 2D/3D co-regularized fine-tuning. Given a warmed-up 3DGS initialized with epipolar depth, the adaptive baseline selection automatically determines a baseline to synthesize for each sparse view. We then fine-tune 3DGS by backpropagating depth-refining gradients from the stereo matching network, effectively specializing the 3DGS for stereo matching. The 2D/3D co-regularization further helps obtain stable reconstruction, addressing weak geometric cues in close stereo views. Sparse-GS2Mesh achieves a 15\% improvement over state-of-the-art methods in little-overlap settings and comparable results in large-overlap settings. Codes will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11011v1">PCAsplat: Gaussian Splatting with Local PCA Regularization</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Gaussian splatting has emerged as a flexible representation for 3D reconstruction from posed images. However, existing methods are optimized primarily using rasterization-based losses, which supervise a splat only when it contributes to sampled camera rays. Gaussians that are occluded or contribute little to the sampled view therefore receive weak or no geometric gradients and may drift away from the underlying surface, producing undesired floaters. We introduce PCAsplat, a geometry-aware regularization framework for Gaussian splatting based on differentiable local principal component analysis (PCA). Our PCA regularizer acts directly on neighborhoods of Gaussian centers and can therefore update Gaussians that do not contribute to the current training view. We regularize the PCA eigenvalues to encourage Gaussians to move to the underlying surface with isotropic tangent-plane coverage. We also align each Gaussian normal with the PCA-estimated neighborhood normal to enforce consistent orientation. Experiments on DTU, Tanks and Temples, and NeRF Synthetic show that the splats produced by PCAsplat better approximate samples of the reference surface while substantially reducing undesired floaters. These surface-aligned splats enable downstream geometry-processing tasks, including point cloud segmentation, and direct Poisson reconstruction. Additionally, PCAsplat remains competitive under conventional novel view synthesis and mesh extraction tasks. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22342v2">4D-GSW: Kinematic-Aware Spatio-Temporal Consistent Watermarking for 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 9 pages main paper, 7 figures, 18 pages in total
    </div>
    <details class="paper-abstract">
      While 4D Gaussian Splatting (4DGS) has revolutionized high-fidelity dynamic reconstruction, safeguarding the intellectual property of these assets remains an open challenge. Conventional steganographic techniques often neglect the underlying kinematic manifolds, triggering non-physical artifacts such as severe temporal flickering and "FVD collapse". To address this, we propose \textbf{4D-GSW}, a kinematic-aware watermarking framework designed to embed robust copyright information while preserving high spatio-temporal consistency. Unlike prior 4D steganography that primarily focuses on opacity-guided invisibility, our approach explicitly addresses the physical coherence of motion trajectories. We introduce a \textbf{Spatio-Temporal Curvature (STC)} metric to identify "Dynamic Instants," adaptively gating watermark gradient injection to shield critical motion manifolds from non-physical perturbations. To ensure global coherence across complex deformations, we formulate a joint \textbf{HMM-MRF energy minimization} model that synchronizes watermark phases within both temporal trajectories and spatial neighborhoods. Furthermore, an \textbf{anisotropic gradient routing} mechanism ensures that watermark embedding remains strictly decoupled from photometric reconstruction fidelity. Extensive experiments have demonstrated the superior performance of our method in robustly hiding watermarks while resisting various attacks and maintaining high rendering quality and spatiotemporal consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20714v4">The Role of Initialization in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Accepted to ACCV 2026. Sources available at https://github.com/deivse/ivd_splat
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become the method of choice for photo-realistic novel view synthesis (NVS), due to its efficiency and compelling visual quality. 3DGS represents the scene as a set of 3D Gaussians, parameterized by their position, spatial extent, and view-dependent color. Starting from an initial point cloud, 3DGS refines the Gaussians' parameters to reconstruct a set of training images as accurately as possible. Typically, a sparse Structure-from-Motion point cloud is used as initialization. Thus, in order to obtain a full scene representation, 3DGS methods rely on a densification stage. In this paper, we systematically study how initialization affects 3DGS NVS performance and geometric quality, using several densification strategies. We show that dense initialization does not lead to consistent visual improvements when paired with strong densification. Despite that, it helps in generalization to off-trajectory views and significantly improves geometric accuracy of the scenes. Our code is available at https://github.com/deivse/ivd_splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09853v1">DeltaSplat: Iterative Gaussian Refinement for Pose-Free Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 11 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Pose-free feed-forward 3D Gaussian Splatting (3DGS) reconstructs a scene from sparse, unposed images in a single network pass, removing the need for camera calibration and per-scene optimization. However, camera estimation errors propagate into the predicted Gaussians and compound the geometric and photometric inaccuracies of single-pass prediction. To correct these errors, we introduce DeltaSplat, a lightweight Gaussian refinement module for pose-free feed-forward 3DGS. It iteratively renders the current Gaussians at the input context views and predicts per-Gaussian updates from the resulting residuals. A 2D residual alone, however, underdetermines the 3D correction. DeltaSplat therefore conditions each update on per-pixel Plücker rays and rendered depth as a soft geometric prior. A dual-branch convolutional mixer efficiently encodes these inputs, and per-attribute heads decode the fused features into position, opacity, and color updates. The module adds only ~2.2% parameters to the backbone and remains fully feed-forward at inference. On DL3DV, DeltaSplat reaches 26.64 dB PSNR in the pose-free setting, improving its state-of-the-art backbone by 1.75 dB and surpassing even baselines supplied with ground-truth cameras; consistent gains hold across 6-24 views and all camera regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04606v2">Sparse-View 4D Gaussian Splatting via Spatiotemporal Priors and Generative Assistance</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 4 pages, 5 figures, Accepted to SIGGRAPH Asia 2026 Workshops (SA Workshops '26)
    </div>
    <details class="paper-abstract">
      We present a 4D Gaussian Splatting framework for the Sparse-View Track of the SIGGRAPH Asia 2026 Volumetric Video Challenge, which requires dynamic scene reconstruction from only six cameras with wide baselines. To achieve robust dynamic reconstruction under such sparse views, our framework integrates three components. (1) Region-adaptive spatial priors: We use foreground masks to guide Gaussian initialization and mask voting to control densification separately for the dynamic foreground and static background. Background geometry is regularized using monocular depth aligned to metric scale. (2) Motion-consistent temporal priors: We provide supervision at intermediate times through frame interpolation and constrain projected Gaussian motion with estimated optical flow. (3) Generative assistance: We place virtual cameras in the widest angular gaps and restore their rendered images using a diffusion-based model conditioned on camera poses. The restored images are iteratively incorporated into training as pseudo-supervision. On the validation set, our framework improves full-frame PSNR from 25.60 dB for the baseline to 29.75 dB. On the official test benchmark, it achieves 30.04 dB full-frame PSNR and 27.88 dB foreground PSNR, ranking first overall in the Sparse-View Track.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19156v2">Hand-4DGS: Feed-Forward 3D Gaussian Splatting for 4D Hand Reconstruction from Egocentric Videos</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Dynamic 3D hand reconstruction from egocentric videos is essential for next-generation computing platforms such as AR/VR and AI glasses. Despite its importance, most prior works focus either on multi-view 3D hand reconstruction or on 4D human body reconstruction. Egocentric 4D hand reconstruction remains difficult due to rapid hand and camera motion, hand-object and inter-hand interactions, and inherent ambiguity from single-view observations. To address these challenges, we introduce Hand-4DGS, a feed-forward framework for dynamic 4D hand reconstruction from egocentric videos. Our approach incorporates a mesh-guided representation for structural priors and temporal convolutions to model dynamic motion. We evaluate our framework on H2O and ARCTIC, two egocentric hand-object interaction datasets, and show improvements over baselines. Our model can efficiently adapt to unseen videos from datasets that are not included in training. In addition, image supervision through differentiable Gaussian rasterization provides an additional signal for appearance optimization and pose adjustment during training and test-time optimization, without ground-truth 3D hand pose annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09343v1">TileSkipper: Region-Adaptive Tile Pruning for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Tiled 3D Gaussian Splatting rasterizers often use one scene-wide contribution cutoff for tile enumeration, although content differs in its sensitivity to support truncation. TileSkipper selects a static per-Gaussian cutoff policy for a frozen checkpoint. Calibration renders measure candidate pair savings and an isolated-removal distortion proxy that accounts for front transmittance and background color. The method allocates cutoffs across 64 Gaussian groups and accepts policies only after complete renders on disjoint selection views. The exported policy uses one byte per Gaussian, with no parameter updates, additional kernel, or per-frame policy inference. Across 13 scenes from Mip-NeRF 360, Tanks & Temples, and Deep Blending, a fixed-policy AccuTile sweep gives dataset-macro speedups of $1.088\times$ at standard resolution and $1.238\times$ at 3840 pixels wide, with $-0.007/-0.023$ dB mean PSNR change. Six integrations with existing opacity-aware bounds yield $1.009\times$--$1.121\times$ compiler-only speedups. For four ports from $3σ$ rasterizers, we separately attribute the prior exact-bound transition and our incremental gain. Matched-quality ablations show modest gains over scene-global calibration and parity with per-Gaussian control; the standalone comparison with AdaGScale is regime-dependent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09116v1">SPLATIFY: Reproduce, Discover, Innovate! From Papers and Ideas to Trainable 3DGS Code</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      The rapid growth of 3D Gaussian Splatting (3DGS) research demands significant effort to reimplement papers before building on them. We introduce SPLATIFY, a multi-agent framework that converts 3DGS papers into trainable gsplat-based implementations, where generic paper-to-code methods and frontier models fail. SPLATIFY achieves this through five innovations: (1) A context-free grammar for gsplat over a modular method template with extension points for losses, densification, rendering, and optimization, constraining synthesis so generated code satisfies gsplat's architectural invariants by construction. (2) Architectural elements for faithful reproduction: fork-aware citation recovery retrieving component-level code at function-level granularity, Graph-of-Thought synthesis in topological dependency order, RAG-guided in-context example selection from over 20 verified implementations, and visual feedback combining PSNR-guided regeneration, Gaussian-level structural checks, and VLM-driven patching. (3) Knowledge-driven compositional improvement that autonomously finds weaknesses and composes complementary regularizers, losses, and densification strategies to improve upon original results. (4) Interdisciplinary method discovery where agents retrieve physical priors from outside the 3DGS literature and compose them with rendering knowledge to produce methods for previously unaddressed scene types. (5) SPLATIFY-Bench, an evaluation framework across 30 diverse 3DGS papers. On papers without public code, SPLATIFY matches expert implementations while reducing development time from weeks to minutes, and through compositional discovery further improves PSNR by up to 2.4 dB. We additionally demonstrate novel methods for volumetric nebula rendering and other scientific domains, synthesized entirely by SPLATIFY.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08756v1">Post-Training Semantic Lifting for 3D Gaussian Splatting: Separating Detector, Lifting and Representation Error</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 18 pages, 11 figures, 9 tables. Code: https://github.com/ivanver02/semantic-lifting-3dgs
    </div>
    <details class="paper-abstract">
      The same Gaussian of a 3D Gaussian Splatting model is seen from many views, and these views do not always agree on the class it belongs to. The Gaussian may be occluded in some of them, and the confidence of the detector is not the same from one view to another. The ground truth, on the other hand, is given as an annotated mesh, because two training runs do not produce the same Gaussians. In this work, we propose a post-training lifting method that works with one target class at a time and combines the information coming from all the views. Target and non-target evidence are accumulated simultaneously, weighted by the visibility of each Gaussian in each view. After that, the Gaussians are filtered with two thresholds: a main threshold $β$ selects the high-confidence seeds, and a lower one $γβ$ adds the connected components around them. For the evaluation, the labels are transferred from the Gaussians to the mesh vertices that are both visible and annotated. With this design, we can separate three sources of error: the 2D detector, the lifting and the transfer between representations. The thresholds and the transfer operator are chosen on seven Replica validation scenes, and the method is evaluated on ten held-out ScanNet++ scenes with the same values for every scene and class. The mean mIoU on the validation scenes was 0.93 with masks from the dataset annotations and 0.65 with YOLO masks, and on the ScanNet++ test scenes it was 0.80 and 0.54. Compared with thresholding the evidence per view, as a previous version of the method did, the fraction improves the test mIoU by 0.24 and makes it possible to use a single threshold for all the classes and scenes of both datasets. Finally, the error analysis shows that most of the remaining error comes from the detector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04007v2">VolS-GS: Relightable Gaussian Splatting with Volumetric Subsurface Scattering</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      We present VolS-GS, a relightable Gaussian splatting framework that reconstructs objects from one-light-at-a-time (OLAT) captures and renders them under novel lighting and viewpoints. Relightable Gaussian Splatting methods typically model appearance independently at each primitive, which makes non-local effects difficult to represent. This limitation is particularly apparent for subsurface scattering, where light entering the object at one location can emerge at another. Rather than modeling this effect solely with a neural network or a local kernel at each primitive, we use the spatial support of the Gaussian scene as the domain of a differentiable finite-volume transport solver, so that light can propagate through the object's interior. A small network predicts scattering and absorption coefficients for each Gaussian, and the solve redistributes incident light through the resulting field. The coefficients are fit to images rather than measured, so the solve supplies a transport-shaped path for aggregating per-primitive appearance, not a measurement of the material. To keep the learned shadow and specular terms from taking over the other components, our shadow term is predicted from visibility together with the transmittances and the scattering the solve produces, and a regularizer suppresses specular highlights in regions the shadow term predicts to be unlit. Experiments on three OLAT benchmarks show that VolS-GS consistently improves relighting quality on held-out lights and views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.05576v2">SteadySplats: Resampling of Low-Variance Gaussians for High-Fidelity Stochastic Rendering</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Stochastic order-independent transparency enables efficient and elegant rendering of primitive-based radiance fields like 3D Gaussian Splatting models, but remains impractical due to the inherent visible noise in the output. We propose a principled approach to minimize high-frequency noise, addressing its sources at the representation and image synthesis level. During stochastic rendering, our history-based spatial resampling scheme drastically accelerates image convergence, while temporal importance resampling ensures coherence under camera movement. During training, a color regularizer implicitly reduces the variance along view rays in the 3DGS models. With these properties, our optimized, Vulkan-based renderer effectively mitigates output noise at low and high sample counts, achieving a substantial 13~dB PSNR increase in quality over previous stochastic methods at 1 sample per pixel and quickly converging to sorted 3DGS with an average L1 error of less than $10^{-4}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39960v2">Reconstructing the Dynamic World: A Representation-Centric View of 4D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      4D scene reconstruction aims to recover the evolving geometry, appearance, and motion of dynamic environments from visual observations. Despite substantial progress in neural scene representations, reconstructing dynamic scenes remains challenging due to non-rigid motion, occlusions, temporal inconsistencies, and the trade-offs between reconstruction fidelity and computational efficiency. Recent advances in Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have introduced diverse approaches to representing and reconstructing dynamic scenes, yet their relationships, underlying design choices, and evaluation protocols remain fragmented. In this paper, we present a unified perspective on 4D scene reconstruction, organizing existing methods around their scene representations, temporal modeling strategies, reconstruction pipelines, and optimization objectives. Through this framework, we examine how different design choices affect geometric fidelity, appearance consistency, motion representation, and computational efficiency. We further consolidate commonly used datasets and evaluation metrics, identify limitations in current experimental practices, and discuss open challenges in reconstructing complex, dynamic real-world environments. By connecting methodological developments with their underlying assumptions and evaluation evidence, this work provides a structured foundation for understanding existing approaches and identifying future research directions. An evolving collection of relevant papers and resources is available at https://github.com/ZiyangYan/Awesome-4D-Scene-Reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.07958v1">DensiTok: Making Feed-Forward 3D Gaussian Splatting See More Views Than It Is Given</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) reconstructs a scene in a single forward pass, replacing per-scene optimization with a network trained across many scenes. Its quality, however, degrades sharply as the number of input images drops. The bottleneck is upstream of the reconstruction heads: from a few unposed views, the internal representation they read carries no evidence for unobserved regions, leaving holes, floaters, and blur. The common remedy supplies that evidence as pixels, synthesizing extra views with an image or video generator and re-encoding them, which is costly and not 3D-consistent by construction. We instead densify the evidence itself. We present DensiTok, a plug-in module for pretrained feed-forward 3DGS models that densifies their internal geometry tokens directly, making a frozen backbone behave as though it had observed many more views than it was given. DensiTok compresses those tokens into a compact latent space, completes the latents of the unobserved viewpoints in a single flow-matching step conditioned on camera geometry, and decodes them back into tokens that the original reconstruction heads. The same module design can be integrated into different pretrained predictors while keeping each backbone and its reconstruction heads frozen. Completion in a low-dimensional latent space requires no image synthesis or additional encoder passes. Across three pretrained backbones and two benchmarks, DensiTok consistently improves sparse-view reconstruction and recovers much of the gap to dense-view reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09045v2">Scene-Agnostic Object-Centric Representation Learning for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Published at the Third Workshop for Learning 3D with Multi-View Supervision (3DMV), CVPR 2026
    </div>
    <details class="paper-abstract">
      Recent works on 3D scene understanding leverage 2D masks from visual foundation models (VFMs) to supervise radiance fields, enabling instance-level 3D segmentation. However, the supervision signals from foundation models are not fundamentally object-centric and often require additional mask pre/post-processing or specialized training and loss design to resolve mask identity conflicts across views. The learned identity of the 3D scene is scene-dependent, limiting generalizability across scenes. Therefore, we propose a dataset-level, object-centric supervision scheme to learn object representations in 3D Gaussian Splatting (3DGS). Building on a pre-trained slot attention-based Global Object Centric Learning (GOCL) module, we learn a scene-agnostic object codebook that provides consistent, identity-anchored representations across views and scenes. By coupling the codebook with the module's unsupervised object masks, we can directly supervise the identity features of 3D Gaussians without additional mask pre-/post-processing or explicit multi-view alignment. The learned scene-agnostic codebook enables object supervision and identification without per-scene fine-tuning or retraining. Our method thus introduces unsupervised object-centric learning (OCL) into 3DGS, yielding more structured representations and better generalization for downstream tasks such as robotic interaction, scene understanding, and cross-scene generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08784v2">GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted to ICRA 2026. 8 pages. v2: published version, with typo, citation and acronym fixes
    </div>
    <details class="paper-abstract">
      Robust and accurate perception of dynamic objects and map elements is crucial for autonomous vehicles performing safe navigation in complex traffic scenarios. While vision-only methods have become the de facto standard due to their technical advances, they can benefit from effective and cost-efficient fusion with radar measurements. In this work, we advance fusion methods by repurposing Gaussian Splatting as an efficient universal view transformer that bridges the view disparity gap, mapping both image pixels and radar points into a common Bird's-Eye View (BEV) representation. Our main contribution is GaussianCaR, an end-to-end network for BEV segmentation that, unlike prior BEV fusion methods, leverages Gaussian Splatting to map raw sensor information into latent features for efficient camera-radar fusion. Our architecture combines multi-scale fusion with a transformer decoder to efficiently extract BEV features. Experimental results demonstrate that our approach achieves performance on par with, or even surpassing, the state of the art on BEV segmentation tasks (57.3%, 82.9%, and 50.1% IoU for vehicles, roads, and lane dividers) on the nuScenes dataset, while maintaining a 3.2x faster inference runtime. Code and project page are available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.07795v1">Efficient Gaussian Splatting Sequence Compression with Standard Video Codecs</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted by MM Asia 2026
    </div>
    <details class="paper-abstract">
      This paper presents a novel effective Gaussian Splatting (GS) sequence Compression method that utilizes the Video codec (GSCV). Existing video-based GS sequence compression relies on the Parallel Linear Assignment Sorting (PLAS) and tracked primitive information to convert GS into smooth 2D videos. However, tracked information is not available for most practical applications, and without it, using the vanilla PLAS can generate images exhibiting weak inter-frame correlation, due to its stochastic nature. GSCV incorporates a simple yet efficient Inter-PLAS method to produce close images between the I- and P-frames of GS, enhancing the inter-frame performance of video codec greatly. GSCV also realizes a new pipeline based on the state-of-the-art video codecs with high bit-depth GS images, achieving higher compressibility while simultaneously providing a higher quality upper bound. Experimental results show that the proposed GSCV exhibits obviously improved performance over MPEG video and point cloud-based anchors in GS sequence compression. The code is available at https://github.com/Qi-Yangsjtu/GSCV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.07569v1">OpenSplatGraph: From Dense Semantic Maps to Structured Scene Graphs for Open-Vocabulary Robot Perception</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted to ACCV 2026
    </div>
    <details class="paper-abstract">
      Dense 3D mapping with semantic understanding is essential for robotic perception in complex environments. Recent 3D Gaussian Splatting-based mapping approaches enable high-fidelity geometry and efficient open-vocabulary perception, but typically represent semantics as unstructured feature fields that limit object-centric reasoning. In contrast, 3D scene graphs explicitly model objects and their relationships for structured reasoning, but are commonly constructed from sparse geometric representations that do not fully exploit dense semantic maps. In this work, we present OpenSplatGraph, a unified framework that constructs persistent 3D scene graphs directly from an online Gaussian-based open-vocabulary semantic map. The proposed framework augments the dense semantic map with a reliability-aware semantic field that maintains lightweight observation statistics for confidence-aware, query-conditioned object extraction. Extracted object instances are associated with persistent graph nodes, allowing object attributes and relationships to be incrementally updated across observations and queries. By tightly coupling dense semantic mapping with persistent object-centric representations, our framework supports both language-guided object grounding and structured relational reasoning while preserving the geometric fidelity of Gaussian-based mapping. Comprehensive evaluations on standard 3D scene understanding benchmarks and real-world robotic experiments demonstrate that OpenSplatGraph achieves competitive performance for online open-vocabulary perception and downstream robotic tasks. Project page: https://csiro-robotics.github.io/OpenSplatGraph.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.07472v1">SURGE: Sonar-fUsed Reconstruction and localization via image-gated Graph Estimation</a></div>
    <div class="paper-meta">
      📅 2026-10-05
    </div>
    <details class="paper-abstract">
      Remotely operated vehicles (ROVs) are widely used to explore and inspect underwater environments such as caves, shipwrecks, and submerged infrastructure. These missions require accurate 3D understanding of the surrounding environment, which depends on both reliable vehicle localization and metric scene reconstruction. However, external positioning is often unavailable underwater, requiring small ROVs to rely primarily on onboard perception. Optic vision provides rich visual and geometric information but suffers from scale ambi- guity and trajectory drift, whereas 2D imaging sonar provides metric range but incomplete 3D geometry. Existing underwater reconstruction approaches typically address these limitations separately or assume known sensor poses, leaving localization and reconstruction disconnected. We present SURGE, a camera sonar framework that jointly estimates the ROV trajectory and target location by integrating visual and acoustic observations within a factor graph, then uses the recovered metric poses for sonar Gaussian splatting. Experiments on real underwater RGB sonar observations show that SURGE substantially improves localization consistency over conventional vision based pose estimation and produces a more compact, natively metric reconstruction than RGB Gaussian splatting baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.06688v1">GS-Pool: Object-Level Change Detection in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-05
    </div>
    <details class="paper-abstract">
      Factories, museums and surveyors photograph the same space months apart and need to know which objects changed. When each visit is reconstructed with 3D Gaussian Splatting (3DGS), a direct comparison of the two reconstructions does not answer this. Training is stochastic, so two reconstructions of an unchanged space never coincide, and the second visit is often a quick re-scan with far fewer photographs. We propose GS-Pool, which takes two independently reconstructed Gaussian fields of the same space and returns the changed objects in each, together with their masks. SAM2 masks of each visit's photographs are lifted onto the Gaussians that render them and merged into an object pool, so every decision is taken once per object in 3D. We introduce a photographic carrier, the 3DGS training loss of each input reconstruction against the other visit's photographs, backpropagated to the Gaussians that rendered each pixel. We combine it with GS-Diff's geometry and colour terms and our distilled DINOv3 features. This evidence is compared with that of the objects present in both visits, which sets a change threshold for each scene. On PASLCD, GS-Pool reaches mIoU/F1 scores of 0.751/0.846 against 0.644/0.758 for GS-Diff, the strongest prior method, a gain of 17%/12%. Its mIoU is also 36%, 40% and 57% above that of O-SCD, PlenoCI and MV-3DCD, and it reaches 0.855 mIoU on CL-Splats, 33% above MV-3DCD. Each changed object is returned as a set of Gaussians with the evidence behind its decision, which an inspector can review in 3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.07110v1">MoonGS: High-quality Representation of the Lunar Surface via Gaussian Splatting Using Robust Depth Features from Image Pairs</a></div>
    <div class="paper-meta">
      📅 2026-10-05
    </div>
    <details class="paper-abstract">
      High-quality 3D reconstruction of lunar terrain from sparse rover images is indispensable for autonomous lunar exploration, but remains challenging because viewpoint overlap is insufficient, surface textures are weak, and data volume is limited. We propose MoonGS, the first feed-forward 3D Gaussian Splatting framework tailored to lunar scenes. Given only two input images, MoonGS predicts pixel-aligned Gaussian primitives in a single forward pass and renders photorealistic novel views without any per-scene optimization. MoonGS (i) adopts an adaptable backbone design that seamlessly integrates advanced vision foundation models to extract robust depth features; (ii) integrates semantic priors in two manners: merging semantic cues with visual features to refine Gaussian parameter estimation, and adopting a semantic ranking loss that regularizes background depth; and (iii) employs an entropy-guided heuristic resampling strategy to augment sparse observations by selecting the most informative distant viewpoints with negligible overhead. Experiments on the LuSNAR benchmark and our synthetic weak-texture MoonBlender dataset show that MoonGS surpasses state-of-the-art feed-forward NeRF/3DGS baselines by +4.9 dB PSNR, +0.29 SSIM, and 40\% lower LPIPS while maintaining sub-second inference. Furthermore, we validate the broad applicability of our framework by demonstrating that it effectively leverages state-of-the-art backbones, including VGGT, to significantly boost performance. Qualitative evaluations on Chang'e mission imagery also show the best visual quality among compared methods, indicating robustness on real lunar data. The source code and dataset are publicly available at https://github.com/InRobots/MoonBlender.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.06472v1">MaRO-GS: Mask-Robust Object-Centric Gaussian Splatting from Inconsistent Multi-view Masks</a></div>
    <div class="paper-meta">
      📅 2026-10-05
      | 💬 Accepted to ACCV 2026. Project page: https://eunjikim02.github.io/marogs/
    </div>
    <details class="paper-abstract">
      We address the challenge of accurate 3D object reconstruction from multi-view images in Gaussian Splatting. Existing object-level 3DGS methods reconstruct the entire scene rather than directly optimizing the target object, even when only the target object is needed, which incurs substantial computational overhead. They also rely on 2D segmentation masks to associate Gaussians with objects, but these masks are often inconsistent across views. Such inconsistencies corrupt Gaussian optimization and produce incorrectly supervised Gaussians that degrade object reconstruction fidelity. To overcome these limitations, we propose MaRO-GS, a 3DGS framework that directly optimizes target-object Gaussians from object-masked multi-view images and remains robust to inconsistent supervision. For reliable supervision, mask-reliability view filtering excludes unreliable views. Object-supported Gaussian density control suppresses Gaussians irrelevant to the target object and prevents background densification, while Silhouette-aligned Object Loss maintains object-focused optimization. Extensive experiments across diverse datasets demonstrate that MaRO-GS improves PSNR, segmentation accuracy, and computational efficiency, with the largest PSNR gain of 2.05 dB on the small-object LERF-Mask dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.02322v2">SCION: Scene Composition with Instanced Neural Primitives</a></div>
    <div class="paper-meta">
      📅 2026-10-05
      | 💬 Accepted to NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Real-world scenes are compositional: bricks, blades of grass, pebbles, and tree leaves recur across human-built and natural environments. Existing neural scene representations model these elements independently. Most 3D Gaussian Splatting and follow-up abstraction and compression methods treat each element as unique, fitting millions of independent Gaussians per scene. Prior methods like Splat and Replace fit template objects, but they require mostly manual selection of repeated elements. As a result, these representations store redundant parameters and provide weak manipulation handles for downstream tasks. We introduce SCION, a hierarchical compositional scene representation that replaces independent Gaussians with a compact vocabulary of reusable primitives and lightweight world-space instances that place transformed copies throughout the scene. We fit this representation to multi-view captures via a joint optimization over discrete and continuous scene parameters, combining two-level densification over splats and instances with an adversarial loss that preserves detail across shared primitives. The recovered structure yields a compact, controllable representation while maintaining high quality even at 1.2 MB. SCION achieves rate-distortion favorable to existing Gaussian compression methods, and it enables instance-level scene editing and animation without retraining. Our results show that neural scene representations need not memorize scenes as independent primitives; they can discover reusable parts. Project webpage: https://light.princeton.edu/SCION
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.06171v1">Controllable and Photorealistic Pedestrian Risky Motion Generation for End-to-End Driving Safety Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-05
      | 💬 9 pages, 7 figures, Website at https://controlped.netlify.app
    </div>
    <details class="paper-abstract">
      Evaluating end-to-end autonomous driving under rare, safety-critical vehicle-pedestrian interactions requires photorealistic, sensor-level scenarios. However, trajectory-based scenario generators cannot synthesize raw visual observations, whereas video-based approaches lack controllability. To bridge this gap, we present ControlPed, a novel framework that combines trajectory-level conflict synthesis with 3D Gaussian Splatting (3DGS) to generate photorealistic, motion-controllable safety-critical scenarios. Built upon HazardPed, a dataset derived from 10,352 traffic videos comprising 422 conflict trajectories, HD maps, and 857 annotated 3D human motions, ControlPed first generates conflict trajectories, lifts them into 3D human motion sequences via text-conditioned motion diffusion, and finally renders multi-view sensor observations using animatable 3DGS avatars. Safety evaluation in 88 rendered photorealistic scenarios reveals that seven leading end-to-end driving models suffer a severe performance drop, with their mean HDScore plunging from 88.8 to 47.4, exposing major failure modes under dangerous pedestrian behaviors. The dataset and testing benchmarks will be released to facilitate safety assessment of vehicle-pedestrian interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22197v2">Multi4D: High-Fidelity Dynamic Gaussian Splatting via Multi-Level Competitive Allocation</a></div>
    <div class="paper-meta">
      📅 2026-10-05
      | 💬 Accepted by ECCV 2026, project page:https://batfacewayne.github.io/Multi4D.io/
    </div>
    <details class="paper-abstract">
      Dynamic 3D Gaussian splatting faces a fundamental tension between motion consistency and visual fidelity. Deformation-based approaches preserve temporal correspondence but suffer from motion over-factorization, oversmoothing high-frequency dynamics. In contrast, 4D-primitive methods capture fine visual details yet incur temporal overparameterization, breaking object identity and leading to severe storage overhead. To resolve this, we introduce Multi4D, a framework for high-fidelity dynamic Gaussian Splatting based on multi-level competitive allocation. Instead of a monolithic representation, we distribute modeling capacity across three structured levels: static structure, persistent dynamic geometry, and transient appearance primitives. Through shared rasterization and residual-driven optimization, these levels dynamically compete to explain photometric error, enabling adaptive specialization without pre-assigned decomposition. This allocation preserves long-term motion consistency while capturing fine dynamic detail, achieving state-of-the-art rendering quality and real-time performance with significantly fewer dynamic primitives. Furthermore, because our representation explicitly tracks compact persistent Gaussians over time, semantic features can be embedded afterward, enabling Multi4D to achieve state-of-the-art 4D segmentation accuracy with an order-of-magnitude speedup. Project page: https://batfacewayne.github.io/Multi4D.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.06035v1">Casual Flash Lighting for Gaussian Splat Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2026-10-05
    </div>
    <details class="paper-abstract">
      Recovering geometry, materials, and lighting from photographs is highly ambiguous when only static illumination is available. Active-lighting setups reduce the ambiguity but require dark rooms or specialized hardware. Instead, we synergize both static and flash lighting from casual indoor capture, with the flash on or off, each from independent viewpoints. The flash residual constrains albedo and the BRDF, while static lighting captures grazing-angle specular highlights that flash misses. With a 2DGS reconstruction framing, our key contribution is a GS-anchored diffuse field: a hash-encoded MLP is queried at the rasterized 2DGS depth. As it depends only on world position, it is view consistent in 3D and allows the flash residual to drive material decomposition instead of being absorbed by alpha-blending drift across views. At the same time, we render static lighting with deferred shading such that it can also supervise material decomposition. On five synthetic and three real indoor scenes, our method outperforms six recent baselines on diffuse color, albedo and roughness material parameters, and in relighting where PSNR improves by 4.17 dB over the next-best baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04770v2">Cam2Sim: Neural Scenario Reconstruction for Closed-Loop Autonomous Driving Simulation</a></div>
    <div class="paper-meta">
      📅 2026-10-05
      | 💬 In proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE '26) - Tools and Datasets track
    </div>
    <details class="paper-abstract">
      Simulation-based testing enables safe and repeatable evaluation of autonomous driving systems, but its effectiveness is limited by the gap between synthetic simulator outputs and real-world camera observations. To address this problem, we present Cam2Sim, a tool that transforms real-world driving recordings into playable CARLA simulation scenarios. Starting from camera images and poses, Cam2Sim reconstructs road geometry, ego trajectories, parked vehicles, and simulation assets, and augments the reconstructed environment with Gaussian Splatting to render camera observations that resemble the original recording. The framework supports ROS-based data extraction, parked-vehicle detection, OpenStreetMap-based map generation, CARLA scenario construction, Gaussian Splatting training, trajectory replay, and closed-loop execution with a system under test. We validate Cam2Sim on a real-world urban-driving scenario with a camera-based end-to-end driving model, comparing reconstruction quality, image-generation quality, and closed-loop behavior against both a simulation-only baseline and the real-world target. Results show that Gaussian-Splatting-based rendering reduces the visual gap with respect to standard simulator rendering and improves behavioral similarity to the real-world reference runs. The artifact is publicly available at https: //github.com/ast-fortiss-tum/cam2sim, and a screencast showing the tool is available at https://youtu.be/KmZ74l1__lI
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.06017v3">FujinSplat: Seeing Through Smoke with RAW-Domain Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-04
      | 💬 20 pages, 11 figures, including supplementary material. Updated author affiliations and funding acknowledgments; scientific content unchanged. Code: https://github.com/I2WM/FujinSplat
    </div>
    <details class="paper-abstract">
      The appearance of a smoky scene is shaped by two processes that a camera records together: the participating medium alters scene radiance in a view-dependent way, and the image signal processor (ISP) then remaps the result through a nonlinear tone and color transformation. Recovering a clean 3D scene requires separating both. Per-view sRGB dehazing acts only after the ISP has entangled them; standard 3D reconstruction ignores the medium and absorbs it into scene geometry and radiance. FujinSplat addresses the problem in the RAW domain, where the two processes remain separable. A per-scene Base ISP is fitted from the scene's hazy RAW captures to its own camera renderings and then frozen, providing a fixed photometric anchor that performs no dehazing. Analyzing expert corrections reveals a compact, low-dimensional correction space identifiable from RAW alone. FujinSplat therefore fits per-view action answers at the training poses and trains a single scene-agnostic controller to regress them from RAW; the corrected views supervise one static 3D Gaussian representation, jointly with a bounded per-view residual that reconciles cross-view photometric inconsistencies. On the RealX3D real-world smoke benchmark FujinSplat clearly outperforms the strongest comparable baseline, ahead of both physics-based reconstruction and restoration-then-3DGS pipelines. Code is available at https://github.com/I2WM/FujinSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.05289v1">Mobile-4DGS: Unified Static-Dynamic Real-time Mobile Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-04
      | 💬 Code has been released: https://xiaobiaodu.github.io/mobile-4dgs-project/
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have achieved remarkable performance in novel view synthesis, yet deploying both static and dynamic Gaussian representations on resource-constrained mobile devices remains challenging due to heavy storage, redundant primitives, and costly per-frame computation. We present Mobile-4DGS, a unified lightweight framework for high-fidelity real-time static and dynamic Gaussian rendering on mobile platforms. For compact appearance modeling, we introduce a Monte Carlo Specular Energy Aggregator that compresses high-order radiance residuals into the first-order Spherical Harmonics (SH), together with an Attribute-Conditioned SH Enhancement module whose predicted offsets are pre-baked before inference. We further propose a Multi-View Alpha-Based Densification and Pruning strategy to suppress redundant primitives while maintaining multi-view consistency. For dynamic scenes, we develop a compact explicit 4D representation by constructing second-order Gaussian motion, learnable temporal support, and a binary static-dynamic partition, enabling continuous-time modeling without runtime deformation networks. Based on this partition, a Depth-Order Certificate selectively reuses previously committed depth orders to reduce re-projection, sorting, merging, and index-buffer updates during playback. Extensive experiments on static and dynamic scenes demonstrate that Mobile-4DGS substantially reduces storage and rendering overhead while maintaining competitive visual quality, enabling real-time 3D and 4D Gaussian Splatting on mobile devices. \textcolor{magenta}{\href{https://xiaobiaodu.github.io/mobile-4dgs-project/}{Code has been released: https://xiaobiaodu.github.io/mobile-4dgs-project/}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23087v2">EndoWave: 4D Gaussian Splatting with Rational Wavelet for Endoscopic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-10-04
    </div>
    <details class="paper-abstract">
      In robot-assisted minimally invasive surgery, accurate 3D reconstruction from endoscopic video is vital for downstream tasks and improved outcomes. However, endoscopic scenarios present unique challenges, including photometric inconsistencies, non-rigid tissue motion, and view-dependent highlights. Most 3DGS-based methods that rely solely on appearance constraints for optimizing 3DGS are often insufficient in this context, as these dynamic visual artifacts can mislead the optimization process and lead to inaccurate reconstructions. To address these limitations, we present EndoWave, a unified spatio-temporal Gaussian Splatting framework by incorporating an optical flow-based geometric constraint and a multi-resolution rational wavelet supervision. First, we adopt a unified spatio-temporal Gaussian representation that directly optimizes primitives in a 4D domain. Second, we propose a geometric constraint derived from optical flow to enhance temporal coherence and effectively constrain the 3D structure of the scene. Third, we propose a multi-resolution rational orthogonal wavelet as a constraint, which can effectively separate the details of the endoscope and enhance the rendering performance. Extensive evaluations on two real surgical datasets, EndoNeRF and StereoMIS, demonstrate that our method EndoWave achieves state-of-the-art reconstruction quality and visual accuracy compared to the baseline method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04651v1">GS-Codec: A Gaussian-Splatting Bottleneck for Neural Audio Coding</a></div>
    <div class="paper-meta">
      📅 2026-10-03
      | 💬 Accepted to Neurips 2026
    </div>
    <details class="paper-abstract">
      Neural audio codecs compress waveforms into compact discrete tokens that underpin speech language models, real-time communication, and large-scale audio storage. Almost every dominant design, including residual vector quantization, finite scalar quantization, and single-codebook variants, follows the VQ-VAE template by partitioning the encoder latent through a learned codebook or a fixed scalar grid. We ask whether this partition is necessary. We introduce GS-Codec, a neural speech codec whose bottleneck is a parametric signal decomposition rather than a quantizer. We adapt Gaussian splatting from 3D scene reconstruction to one-dimensional latents. An inner optimization loop fits each encoder segment as a weighted sum of 1D Gaussian primitives. The decoder then reconstructs the waveform from the rendered sum. To avoid the cost of this iterative inner loop at inference time, we additionally train a lightweight GS Predictor Net that regresses the primitive parameters in a single forward pass. The encoder and decoder are trained end-to-end through the inner loop, with no quantizer anywhere in the training pipeline: the bottleneck is the decomposition itself, and scalar quantization is applied only post-training to the fitted parameters. Rather than relying on discrete codebook stages for bitrate control, our representation exposes a fine-grained rate-quality tradeoff: a single trained checkpoint supports post-training bitrate control by varying the number of primitives and the per-parameter bit depth, with no retraining required. GS-Codec matches or exceeds well-established open-source codecs such as EnCodec and DAC on speaker similarity (SIM), intelligibility (STOI), and perceptual quality (UTMOS) at comparable bitrates, while achieving comparable semantic performance (WER). Code and audio samples are available at https://ronaluf.github.io/gs-codec/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04351v1">LoCoSplat: Real-Time Feed-Forward 3D Gaussian Splatting with Minimal 3D Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-10-03
      | 💬 23 pages. Under review
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) increasingly aggregates multi-view evidence with heavy learned 3D networks. We propose LoCoSplat (Local-Context Splatting), motivated by the observation that a Gaussian is a local primitive: once depth is predicted, what the 3D stage must add (scale, rotation, opacity) depends on the point cloud around each anchor, and a fixed local average of that neighbourhood is enough to supply it, no heavy network required. LoCoSplat realises exactly this average: it splats a 16-d linear projection of the point features into a fine and a coarse grid and reads both back at each anchor with a 0.14M-parameter pointwise MLP; with no learned 3D network and no dynamic sparse computation, its whole encoder runs as one fp16 CUDA graph. On RealEstate10K, LoCoSplat outperforms every prior feed-forward method on PSNR, SSIM, and LPIPS at 6, 12, and 24 views, with a margin that widens as views densify (+3.3 PSNR over VolSplat, the prior voxel-aligned state of the art, at 24 views) and grows further under zero-shot transfer to ACID and fine-tuning on ScanNet. It reconstructs a 6-view scene in 33 ms on one NVIDIA RTX PRO 6000 GPU, the fastest of seven feed-forward methods and $4.2\times$ faster than the previous state of the art, trains $2.7\times$ faster ($5.9\times$ at 24 views), and uses $6.7\times$ less inference memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04336v1">A differentiable Lagrangian-coupled 3D Gaussian Splatting-SPH model for forward simulation and inverse analysis in solid mechanics</a></div>
    <div class="paper-meta">
      📅 2026-10-03
    </div>
    <details class="paper-abstract">
      Recent advances in generative world models have increased interest in digital models that reproduce both the appearance of real objects and their response to physical interaction. Three-dimensional reconstruction techniques, including 3D Gaussian Splatting, capture detailed surface geometry and appearance from images and videos. However, extending these representations beyond plausible animation to mechanically interpretable models for constitutive behavior, boundary conditions, and inverse parameter identification remains less explored. In this work, a differentiable Lagrangian-coupled 3DGS-smoothed particle hydrodynamics (SPH) model is proposed for forward simulation and inverse analysis of deformable solids. The observed object is first reconstructed from multi-view calibrated visual dataset as a 3DGS rendering model. An envelope-based procedure then generates an independent SPH support for the solid-mechanics model, avoiding the direct use of rendering primitives as mechanical particles. A reference-configuration Lagrangian transfer maps SPH deformation to Gaussian positions and covariances, thereby coupling the physical model and the image observation model while preserving a differentiable computational path. The SPH formulation supports linear elastic, hyperelastic, and Kelvin--Voigt viscoelastic responses, together with fixed, free, and Robin-type boundary conditions. Numerical studies validate the SPH response against finite-element results, assess accuracy and efficiency against a conventional model using Gaussian centers as surface SPH particles, and demonstrate forward simulations on beam, bridge, and liver-shaped examples. Inverse analyses further estimate constitutive and boundary parameters from rendered deformation observations, including noisy cases, demonstrating the feasibility of the proposed model for mechanics-based parameter identification from image data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04199v1">CellSplat4D: PSF-Aware 4D Gaussian Splatting for Sparse Robotic Live-Cell Imaging</a></div>
    <div class="paper-meta">
      📅 2026-10-03
    </div>
    <details class="paper-abstract">
      A robotic microscope watching living cells cannot afford to look as often as it would like. Every volume it acquires costs photons the specimen does not get back, and time owed to other wells. What such a platform exists to produce is a record of individual cells through time: which cell is which from one volume to the next, and which cell divided into which two. Sampling sparsely breaks that record exactly where it matters, and the fault lies in the acquisition schedule rather than in the analysis software. We fill the gaps by reconstructing them, fitting a 4D Gaussian model to whatever volumes the hardware could afford. The model is a cloud of light-emitting blobs, each carrying a position, a shape and a lifetime. Being continuous in time, it renders any missing volume on demand, decoupling how often the robot analyses from how often it can afford to look. The microscope's point-spread function is measured from the data rather than inherited from acquisition metadata or left to the optimizer, because metadata inflates it and the optimizer cannot recover it at all: a wider blur around a smaller blob fits the images equally well. Each blob's lifetime is stored in frames rather than as a fraction of the recording, so that it denotes a fixed duration on any sequence. Unmeasured timesteps are supervised at coarse scale by a 3D U-Net that predicts the intermediate volume directly and estimates no motion field, since a dividing cell becomes two and no motion describes that. On two Cell Tracking Challenge sequences, a C. elegans embryo and a Chinese Hamster Ovarian (CHO), with fidelity scored per cell nucleus, our reconstruction holds the highest nucleus fidelity at every distance from an acquired frame, has the flattest decay across the gap, and best recovers focal planes it was never shown with graceful degradation across the gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04076v1">VCURF: Virtual Camera-based Uncertainty of Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2026-10-02
    </div>
    <details class="paper-abstract">
      Radiance fields, implemented with either implicit (NeRF) or explicit (Gaussian Splatting) representations, are advancing the state of the art in novel view synthesis at a rapid pace. Even though the rendered views they generate are often compelling, they are not free of errors. In this paper, we propose a new approach for pixel-wise uncertainty quantification based on measuring the inconsistencies among renderings by the radiance field model in virtual cameras sampled near the target viewpoint. We named our approach VCURF for Virtual Camera-based Uncertainty of Radiance Fields. VCURF treats the radiance field model as a black box, only assuming that it is capable of rendering color and depth on demand. This property makes our approach applicable to both NeRF and GS models without any modification. Our experiments on a combination of datasets, radiance field models and baselines demonstrate VCURF's effectiveness in pixel-wise uncertainty estimation. We conclude the paper with findings that question the way view selection is tackled by the majority of the current literature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04013v1">Return-to-Home Feasible Micro-Aerial Vehicle Exploration for 3D Gaussian Splatting Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-10-02
      | 💬 IEEE/RSJ International Conference on INTELLIGENT ROBOTS & SYSTEMS 2026
    </div>
    <details class="paper-abstract">
      Micro aerial vehicles (MAVs) enable rapid indoor 3D reconstruction for inspection and time-critical situational awareness, but must operate under strict flight-time budgets and safety constraints that require explicit return-to-home (RTH) feasibility with a conservative margin. We present a topology-first active reconstruction framework that decouples high-fidelity rendering from navigation. A dense 3D Gaussian Splatting (3DGS) map is optimized as the reconstruction target, while a lightweight Truncated Signed Distance Field (TSDF)/occupancy scaffold supports conservative collision checking and online construction of a sparse 3D Voronoi skeleton roadmap. Since directly extracting roadmaps from TSDF geometry can be unstable under noisy and incomplete online fusion, we validate nodes and edges using carved free-space consistency and dense visibility checks, which suppress behind-wall phantom structure and stabilize planning. Viewpoints are selected on the roadmap using a flight-time-budget-aware objective and a lightweight receding-horizon lookahead to improve non-myopic exploration behavior. We evaluate on photorealistic indoor simulation benchmarks (ReplicaCAD, Gibson, and HM3D), reporting reconstruction coverage/error, RTH success, and computational cost. Results demonstrate improved or competitive performance relative to recent GS-based baselines under identical flight-time budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.03599v1">ManifoldSplat: Language-Guided Semantic Shape Editing of 3D Gaussian Head Avatars</a></div>
    <div class="paper-meta">
      📅 2026-10-02
      | 💬 GCPR 2026
    </div>
    <details class="paper-abstract">
      High-fidelity 3D head avatars have reached near-photorealistic quality. While recent methods enable text-driven manipulation, they struggle to provide fine-grained localized control, often entangling features or lacking geometric consistency. Modifying geometry through natural language currently requires slow per-prompt optimization or compromises identity and rigging. We present ManifoldSplat, the first end-toend framework for language-guided semantic shape editing of animatable 3D Gaussian Splatting avatars reconstructed from monocular videos. By performing edits within the structured FLAME manifold rather than directly optimizing an unstructured Gaussian cloud, we strictly preserve identity and animation. We introduce DeltaRegion, a per-region disentangled Conditional Variational Autoencoder (CVAE) delivering feedforward shape deltas, alongside a refining stage to recover view-consistent details. ManifoldSplat reconstructs and edits an avatar in ~90 seconds on a consumer GPU, rendering at ~800 FPS. Extensive evaluations demonstrate our approach sets a new state-of-the-art in localized prompt alignment, geometric coherence, and identity preservation. Project page and code: https://a-canela.github.io/manifoldsplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.29963v2">ADATEX4D: adaptive texture capacity allocation for 4D gaussian splatting</a></div>
    <div class="paper-meta">
      📅 2026-10-02
    </div>
    <details class="paper-abstract">
      Textured Gaussians improve local appearance capacity, but assigning the same texture resolution to every primitive wastes storage on low-detail or weakly visible regions. We introduce AdaTex4D, an adaptive texture-capacity module for deformation-based 4D Gaussian Splatting. Each Gaussian carries packed RGBA triplanes whose two axes grow independently according to visibility normalized screen-space gradients and deformed local scales. Experiments on N3DV and PanopticSports show that AdaTex4D reduces texture storage by more than half while preserving reconstruction quality. Under fixed memory budgets, adaptive allocation also improves quality over uniform texture assignment and reduces overall model and peak memory. These results show that dynamic, anisotropic texture allocation provides a more efficient way to distribute local appearance capacity in 4D Gaussian representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00731v2">Measuring Asset and Scene Reconstruction Effects in Real-to-Sim Robot Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-02
      | 💬 v2: added link to the code and data repository. Code: https://github.com/Kaedim/yam-sim-harness-oss
    </div>
    <details class="paper-abstract">
      Simulated evaluation is increasingly used alongside real-world evaluation of robot policies because it is cheaper and easier to repeat; however, its value depends on how closely its outcomes track the real robot's. We test whether our reconstruction pipeline, combining metrically scaled object geometry, authored physical parameters and scene reconstruction, reduces disagreement between simulated and real robot scores relative to a default open-source recipe. We constructed two simulated versions of one bimanual robot cell: an authored reconstruction, using object geometry at estimated metric scale, projected textures, authored physics and our own scene splat; and a baseline, referred to as the default reconstruction, using the open-source recipe of a generative single-image mesh, engine-default physics and a Gaussian-splat scene. Both reconstructions use the same object photographs and scene video. Two policies ran five tasks each, giving ten task-policy pairs, which we call cells; each cell was run twenty times in each reconstruction with all other settings held fixed. Both reconstructions were scored against the same real trials, graded by a third-party evaluator. Pearson correlation between the ten simulated and real cell means is r = 0.90 for the authored reconstruction and 0.51 for the default. Mean score error is 6.97 percentage points for the authored reconstruction and 17.54 for the default, a reduction of 10.56 percentage points. These results show that improving the quality of the environment reconstruction through higher visual fidelity, authored physics and metric scale makes the simulation more faithful to the real world and narrows the sim-to-real gap. We release the harness, the per-trial scores, every reported run's configuration, and the assets and scenes of both reconstructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.03162v1">Budgeted-GS: Real-Time Large-Scale Gaussian Splatting via Factoring LOD</a></div>
    <div class="paper-meta">
      📅 2026-10-02
      | 💬 28 pages, 21 figures. Preprint of the EG 2027 submission (paper1075)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting achieves excellent visual quality with real-time rendering, but at the scale of entire cities it does not fit: a trained model carries millions of primitives and gigabytes of memory, and real-time rendering at high quality on a consumer GPU remains out of reach. We introduce Budgeted-GS, a post-hoc method that turns any trained 3DGS model into a factoring tree, a multi-resolution hierarchy of moment-matched aggregates. After a construction pass of a few seconds, a single quality parameter selects, for each view, the level of detail that fits the memory of the target device, so the same city-scale model serves GPUs with widely different memory capacities. When a new scene is to be trained, the same theory applies: instead of growing a full-sized model and compressing it afterwards, budget-centered training first measures how many primitives the scene needs and then trains the model directly at that size, avoiding the wasted effort of optimizing primitives that are later discarded. Both methods are grounded in a measurable capacity floor, a budget-error law derived from optimal transport in phase space; selection rules certified by recent covering theorems decide which primitives are redundant. The floor answers how many primitives a scene actually needs and how many can safely be given up. We validate the floor on 13 public scenes under a preregistered protocol, and exercise both methods from object scenes to an official city capture, rendering it at native 1920x1080, full SH, in real time on one consumer GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29997v2">FRUC: Feedforward Dynamic Scene Reconstruction from Uncalibrated Collaborative Driving Views</a></div>
    <div class="paper-meta">
      📅 2026-10-02
      | 💬 Accepted by NeurIPS 2026
    </div>
    <details class="paper-abstract">
      We present FRUC, a feedforward 3D Gaussian Splatting framework for dynamic scene reconstruction from uncalibrated collaborative driving views. Existing multi-agent reconstruction frameworks are often hindered by rigid prerequisites, demanding precise spatial calibration and slow per-scene optimization. In this paper, we rethink this task by conceptualizing a distributed multi-vehicle network as a spatio-temporally unstructured ego-centric multi-camera system, where the core challenge lies in enhancing ego-centric occluded geometry through collaboration without degrading the ego's accurately observed visible geometry, while preserving reconstruction efficiency. For efficient reconstruction, FRUC is built upon a visual grounded geometric Transformer backbone to enable one-shot, calibration-free inference from a flexible number of multi-vehicle views. To achieve non-destructive geometric supplementation under uncalibrated cross-agent misalignment, FRUC first introduces an ego-centric causal occlusion field that explicitly derives occlusion evolution as latent priors by modeling agent-wise spatio-temporal correlations. Guided by these occlusion priors, it further formulates cross-agent integration as a deterministic residual denoising process via zero-initialized injection, turning challenging cross-agent fusion into bounded residual learning for robust collaborative blind-spot completion. Through extensive evaluations on real-world V2X-Real and UrbanIng-V2X datasets, FRUC is shown to be a new state-of-the-art for the scene reconstruction of dynamic collaborative driving environments, significantly outperforming existing methods in both rendering quality and efficiency. Code is available at https://github.com/yihangtao/FRUC.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.02851v1">ByteSplat: Efficient Distributed 3D Gaussian Splatting Training via Intra- and Inter-GPU communication reduction</a></div>
    <div class="paper-meta">
      📅 2026-10-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables photorealistic scene reconstruction, but training large-scale scenes requires substantial memory and computation. Distributing training across multiple GPUs increases available memory capacity, yet its performance is strictly constrained by data movement. We identify two dominant bottlenecks: repeated off-chip memory accesses during forward and backward rasterization, and inter-GPU communication of partial Gaussian gradients. To address these bottlenecks, we present ByteSplat, a distributed 3DGS training framework that jointly reduces intra- and inter-GPU data movement. First, ByteSplat fuses forward rasterization and backward rasterization into a single GPU kernel, retaining intermediate results on-chip to eliminate redundant off-chip transfers. Second, to alleviate the increased on-chip storage by fused rasterization, ByteSplat introduces hardware-aware pruning that considers both rendering quality and per-tile shared-memory constraints, increasing the overall performance of fused execution. Lastly, ByteSplat exploits gradient sparsity to eliminate zero partial gradients from inter-GPU communication. GPU-efficient encoder and decoder kernels compact the remaining records and directly aggregate the received gradients on their owner GPUs, reducing both communication volume and local aggregation overhead. We evaluate ByteSplat across six datasets. Compared with the baseline, ByteSplat reduces intra-GPU off-chip traffic and backward inter-GPU communication volume by 63.4% and 65.8%, respectively. On eight GPUs, ByteSplat achieves up to 6.1$\times$ training speedup while preserving reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.30320v2">MonoPhysics: Estimating Geometry, Appearance, and Physical Parameters from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2026-10-01
      | 💬 NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Existing inverse physics methods recover physical parameters from multi-view videos, where geometric constraints across views resolve scale and 3D structure. In monocular settings, however, such constraints are absent, leading to severe scale ambiguity, inaccurate geometry, and weak coupling between appearance optimization and physical simulation. To address these challenges, we propose MonoPhysics, a framework for monocular inverse physics estimation of deformable objects that jointly optimizes geometry, appearance, and physical parameters using a differentiable simulator and 3D Gaussian Splatting. Our key contribution is removing the multi-view capture requirement of existing methods, a necessary step toward handling in-the-wild video. MonoPhysics introduces three visual-physical bridges: scene re-parameterization, physics-aware geometry refinement, and a differentiable position map. We evaluate on Vid2Sim, real-world captures, and a new dataset of elastic and plasticine objects that we introduce. MonoPhysics outperforms monocular baselines in future prediction and recovers Young's modulus on Vid2Sim with accuracy comparable to a multi-view baseline. Code and data are available at https://daniel03c1.github.io/MonoPhysics/.
    </details>
</div>
