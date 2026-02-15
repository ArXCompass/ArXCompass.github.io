# gaussian splatting - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13204v2">EDGS: Eliminating Densification for Efficient Convergence of 3DGS</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting reconstructs scenes by starting from a sparse Structure-from-Motion initialization and refining under-reconstructed regions. This process is slow, as it requires multiple densification steps where Gaussians are repeatedly split and adjusted, following a lengthy optimization path. Moreover, this incremental approach often yields suboptimal renderings in high-frequency regions. We propose a fundamentally different approach: eliminate densification with a one-step approximation of scene geometry using triangulated pixels from dense image correspondences. This dense initialization allows us to estimate the rough geometry of the scene while preserving rich details from input RGB images, providing each Gaussian with well-informed color, scale, and position. As a result, we dramatically shorten the optimization path and remove the need for densification. Unlike methods that rely on sparse keypoints, our dense initialization ensures uniform detail across the scene, even in high-frequency regions where other methods struggle. Moreover, since all splats are initialized in parallel at the start of optimization, we remove the need to wait for densification to adjust new Gaussians. EDGS reaches LPIPS and SSIM performance of standard 3DGS significantly faster than existing efficiency-focused approaches. When trained further, it exceeds the reconstruction quality of state-of-the-art models aimed at maximizing fidelity. Our method is fully compatible with other acceleration techniques, making it a versatile and efficient solution that can be integrated with existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12159v1">3DGSNav: Enhancing Vision-Language Model Reasoning for Object Navigation via Active 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Object navigation is a core capability of embodied intelligence, enabling an agent to locate target objects in unknown environments. Recent advances in vision-language models (VLMs) have facilitated zero-shot object navigation (ZSON). However, existing methods often rely on scene abstractions that convert environments into semantic maps or textual representations, causing high-level decision making to be constrained by the accuracy of low-level perception. In this work, we present 3DGSNav, a novel ZSON framework that embeds 3D Gaussian Splatting (3DGS) as persistent memory for VLMs to enhance spatial reasoning. Through active perception, 3DGSNav incrementally constructs a 3DGS representation of the environment, enabling trajectory-guided free-viewpoint rendering of frontier-aware first-person views. Moreover, we design structured visual prompts and integrate them with Chain-of-Thought (CoT) prompting to further improve VLM reasoning. During navigation, a real-time object detector filters potential targets, while VLM-driven active viewpoint switching performs target re-verification, ensuring efficient and reliable recognition. Extensive evaluations across multiple benchmarks and real-world experiments on a quadruped robot demonstrate that our method achieves robust and competitive performance against state-of-the-art approaches.The Project Page:https://aczheng-cai.github.io/3dgsnav.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26455v2">Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      We present Stylos, a single-forward 3D Gaussian framework for 3D style transfer that operates on unposed content, from a single image to a multi-view collection, conditioned on a separate reference style image. Stylos synthesizes a stylized 3D Gaussian scene without per-scene optimization or precomputed poses, achieving geometry-aware, view-consistent stylization that generalizes to unseen categories, scenes, and styles. At its core, Stylos adopts a Transformer backbone with two pathways: geometry predictions retain self-attention to preserve geometric fidelity, while style is injected via global cross-attention to enforce visual consistency across views. With the addition of a voxel-based 3D style loss that aligns aggregated scene features to style statistics, Stylos enforces view-consistent stylization while preserving geometry. Experiments across multiple datasets demonstrate that Stylos delivers high-quality zero-shot stylization, highlighting the effectiveness of global style-content coupling, the proposed 3D style loss, and the scalability of our framework from single view to large-scale multi-view settings. Our codes are available at https://github.com/HanzhouLiu/Stylos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2210.00379v8">NeRF: Neural Radiance Field in 3D Vision: A Comprehensive Review (Updated Post-Gaussian Splatting)</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Updated Post-Gaussian Splatting
    </div>
    <details class="paper-abstract">
      In March 2020, Neural Radiance Field (NeRF) revolutionized Computer Vision, allowing for implicit, neural network-based scene representation and novel view synthesis. NeRF models have found diverse applications in robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, and more. In August 2023, Gaussian Splatting, a direct competitor to the NeRF-based framework, was proposed, gaining tremendous momentum and overtaking NeRF-based research in terms of interest as the dominant framework for novel view synthesis. We present a comprehensive survey of NeRF papers from the past five years (2020-2025). These include papers from the pre-Gaussian Splatting era, where NeRF dominated the field for novel view synthesis and 3D implicit and hybrid representation neural field learning. We also include works from the post-Gaussian Splatting era where NeRF and implicit/hybrid neural fields found more niche applications. Our survey is organized into architecture and application-based taxonomies in the pre-Gaussian Splatting era, as well as a categorization of active research areas for NeRF, neural field, and implicit/hybrid neural representation methods. We provide an introduction to the theory of NeRF and its training via differentiable volume rendering. We also present a benchmark comparison of the performance and speed of classical NeRF, implicit and hybrid neural representation, and neural field models, and an overview of key datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11714v1">GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 8 pages, 6 figures, RA-L accepted
    </div>
    <details class="paper-abstract">
      We propose GSO-SLAM, a real-time monocular dense SLAM system that leverages Gaussian scene representation. Unlike existing methods that couple tracking and mapping with a unified scene, incurring computational costs, or loosely integrate them with well-structured tracking frameworks, introducing redundancies, our method bidirectionally couples Visual Odometry (VO) and Gaussian Splatting (GS). Specifically, our approach formulates joint optimization within an Expectation-Maximization (EM) framework, enabling the simultaneous refinement of VO-derived semi-dense depth estimates and the GS representation without additional computational overhead. Moreover, we present Gaussian Splat Initialization, which utilizes image information, keyframe poses, and pixel associations from VO to produce close approximations to the final Gaussian scene, thereby eliminating the need for heuristic methods. Through extensive experiments, we validate the effectiveness of our method, showing that it not only operates in real time but also achieves state-of-the-art geometric/photometric fidelity of the reconstructed scene and tracking accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11705v1">TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Accepted to AAAI 2026. Project page: https://vcc.tech/research/2026/TG-Field
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized 3D scene representation with superior efficiency and quality. While recent adaptations for computed tomography (CT) show promise, they struggle with severe artifacts under highly sparse-view projections and dynamic motions. To address these challenges, we propose Tomographic Geometry Field (TG-Field), a geometry-aware Gaussian deformation framework tailored for both static and dynamic CT reconstruction. A multi-resolution hash encoder is employed to capture local spatial priors, regularizing primitive parameters under ultra-sparse settings. We further extend the framework to dynamic reconstruction by introducing time-conditioned representations and a spatiotemporal attention block to adaptively aggregate features, thereby resolving spatiotemporal ambiguities and enforcing temporal coherence. In addition, a motion-flow network models fine-grained respiratory motion to track local anatomical deformations. Extensive experiments on synthetic and real-world datasets demonstrate that TG-Field consistently outperforms existing methods, achieving state-of-the-art reconstruction accuracy under highly sparse-view conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11638v1">Variation-aware Flexible 3D Gaussian Editing</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Indirect editing methods for 3D Gaussian Splatting (3DGS) have recently witnessed significant advancements. These approaches operate by first applying edits in the rendered 2D space and subsequently projecting the modifications back into 3D. However, this paradigm inevitably introduces cross-view inconsistencies and constrains both the flexibility and efficiency of the editing process. To address these challenges, we present VF-Editor, which enables native editing of Gaussian primitives by predicting attribute variations in a feedforward manner. To accurately and efficiently estimate these variations, we design a novel variation predictor distilled from 2D editing knowledge. The predictor encodes the input to generate a variation field and employs two learnable, parallel decoding functions to iteratively infer attribute changes for each 3D Gaussian. Thanks to its unified design, VF-Editor can seamlessly distill editing knowledge from diverse 2D editors and strategies into a single predictor, allowing for flexible and effective knowledge transfer into the 3D domain. Extensive experiments on both public and private datasets reveal the inherent limitations of indirect editing pipelines and validate the effectiveness and flexibility of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00148v2">Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 43 pages, ICLR 2026
    </div>
    <details class="paper-abstract">
      Predicting physical dynamics from raw visual data remains a major challenge in AI. While recent video generation models have achieved impressive visual quality, they still cannot consistently generate physically plausible videos due to a lack of modeling of physical laws. Recent approaches combining 3D Gaussian splatting and physics engines can produce physically plausible videos, but are hindered by high computational costs in both reconstruction and simulation, and often lack robustness in complex real-world scenarios. To address these issues, we introduce Neural Gaussian Force Field (NGFF), an end-to-end neural framework that integrates 3D Gaussian perception with physics-based dynamic modeling to generate interactive, physically realistic 4D videos from multi-view RGB inputs, achieving two orders of magnitude faster than prior Gaussian simulators. To support training, we also present GSCollision, a 4D Gaussian dataset featuring diverse materials, multi-object interactions, and complex scenes, totaling over 640k rendered physical videos (~4 TB). Evaluations on synthetic and real 3D scenarios show NGFF's strong generalization and robustness in physical reasoning, advancing video prediction towards physics-grounded world models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11577v1">LeafFit: Plant Assets Creation from 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Our source code is publicly available at https://github.com/netbeifeng/leaf_fit
    </div>
    <details class="paper-abstract">
      We propose LeafFit, a pipeline that converts 3D Gaussian Splatting (3DGS) of individual plants into editable, instanced mesh assets. While 3DGS faithfully captures complex foliage, its high memory footprint and lack of mesh topology make it incompatible with traditional game production workflows. We address this by leveraging the repetition of leaf shapes; our method segments leaves from the unstructured 3DGS, with optional user interaction included as a fallback. A representative leaf group is selected and converted into a thin, sharp mesh to serve as a template; this template is then fitted to all other leaves via differentiable Moving Least Squares (MLS) deformation. At runtime, the deformation is evaluated efficiently on-the-fly using a vertex shader to minimize storage requirements. Experiments demonstrate that LeafFit achieves higher segmentation quality and deformation accuracy than recent baselines while significantly reducing data size and enabling parameter-level editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11575v1">ReaDy-Go: Real-to-Sim Dynamic 3D Gaussian Splatting Simulation for Environment-Specific Visual Navigation with Moving Obstacles</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 Project page: https://syeon-yoo.github.io/ready-go-site/
    </div>
    <details class="paper-abstract">
      Visual navigation models often struggle in real-world dynamic environments due to limited robustness to the sim-to-real gap and the difficulty of training policies tailored to target deployment environments (e.g., households, restaurants, and factories). Although real-to-sim navigation simulation using 3D Gaussian Splatting (GS) can mitigate this gap, prior works have assumed only static scenes or unrealistic dynamic obstacles, despite the importance of safe navigation in dynamic environments. To address these issues, we propose ReaDy-Go, a novel real-to-sim simulation pipeline that synthesizes photorealistic dynamic scenarios for target environments. ReaDy-Go generates photorealistic navigation datasets for dynamic environments by combining a reconstructed static GS scene with dynamic human GS obstacles, and trains policies robust to both the sim-to-real gap and moving obstacles. The pipeline consists of three components: (1) a dynamic GS simulator that integrates scene GS with a human animation module, enabling the insertion of animatable human GS avatars and the synthesis of plausible human motions from 2D trajectories, (2) navigation dataset generation for dynamic environments that leverages the simulator, a robot expert planner designed for dynamic GS representations, and a human planner, and (3) policy learning using the generated datasets. ReaDy-Go outperforms baselines across target environments in both simulation and real-world experiments, demonstrating improved navigation performance even after sim-to-real transfer and in the presence of moving obstacles. Moreover, zero-shot sim-to-real deployment in an unseen environment indicates its generalization potential. Project page: https://syeon-yoo.github.io/ready-go-site/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.06109v2">LighthouseGS: Indoor Structure-aware 3D Gaussian Splatting for Panorama-Style Mobile Captures</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 WACV 2026
    </div>
    <details class="paper-abstract">
      We introduce LighthouseGS, a practical novel view synthesis framework based on 3D Gaussian Splatting that utilizes simple panorama-style captures from a single mobile device. While convenient, this rotation-dominant motion and narrow baseline make accurate camera pose and 3D point estimation challenging, especially in textureless indoor scenes. To address these challenges, LighthouseGS leverages rough geometric priors, such as mobile device camera poses and monocular depth estimation, and utilizes indoor planar structures. Specifically, we propose a new initialization method called plane scaffold assembly to generate consistent 3D points on these structures, followed by a stable pruning strategy to enhance geometry and optimization stability. Additionally, we present geometric and photometric corrections to resolve inconsistencies from motion drift and auto-exposure in mobile devices. Tested on real and synthetic indoor scenes, LighthouseGS delivers photorealistic rendering, outperforming state-of-the-art methods and enabling applications like panoramic view synthesis and object placement. Project page: https://vision3d-lab.github.io/lighthousegs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.00437v2">ADGaussian: Generalizable Gaussian Splatting for Autonomous Driving via Multi-modal Joint Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 The paper is accepted by ICRA 2026 and the project page can be found at https://maggiesong7.github.io/research/ADGaussian/
    </div>
    <details class="paper-abstract">
      We present a novel approach, termed ADGaussian, for generalizable street scene reconstruction. The proposed method enables high-quality rendering from merely single-view input. Unlike prior Gaussian Splatting methods that primarily focus on geometry refinement, we emphasize the importance of joint optimization of image and depth features for accurate Gaussian prediction. To this end, we first incorporate sparse LiDAR depth as an additional input modality, formulating the Gaussian prediction process as a joint learning framework of visual information and geometric clue. Furthermore, we propose a Multi-modal Feature Matching strategy coupled with a Multi-scale Gaussian Decoding model to enhance the joint refinement of multi-modal features, thereby enabling efficient multi-modal Gaussian learning. Extensive experiments on Waymo and KITTI demonstrate that our ADGaussian achieves state-of-the-art performance and exhibits superior zero-shot generalization capabilities in novel-view shifting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07493v2">Thermal odometry and dense mapping using learned odometry and Gaussian splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 11 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Thermal infrared sensors, with wavelengths longer than smoke particles, can capture imagery independent of darkness, dust, and smoke. This robustness has made them increasingly valuable for motion estimation and environmental perception in robotics, particularly in adverse conditions. Existing thermal odometry and mapping approaches, however, are predominantly geometric and often fail across diverse datasets while lacking the ability to produce dense maps. Motivated by the efficiency and high-quality reconstruction ability of recent Gaussian Splatting (GS) techniques, we propose TOM-GS, a thermal odometry and mapping method that integrates learning-based odometry with GS-based dense mapping. TOM-GS is among the first GS-based SLAM systems tailored for thermal cameras, featuring dedicated thermal image enhancement and monocular depth integration. Extensive experiments on motion estimation and novel-view rendering demonstrate that TOM-GS outperforms existing learning-based methods, confirming the benefits of learning-based pipelines for robust thermal odometry and dense reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10278v1">ERGO: Excess-Risk-Guided Optimization for High-Fidelity Monocular 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Generating 3D content from a single image remains a fundamentally challenging and ill-posed problem due to the inherent absence of geometric and textural information in occluded regions. While state-of-the-art generative models can synthesize auxiliary views to provide additional supervision, these views inevitably contain geometric inconsistencies and textural misalignments that propagate and amplify artifacts during 3D reconstruction. To effectively harness these imperfect supervisory signals, we propose an adaptive optimization framework guided by excess risk decomposition, termed ERGO. Specifically, ERGO decomposes the optimization losses in 3D Gaussian splatting into two components, i.e., excess risk that quantifies the suboptimality gap between current and optimal parameters, and Bayes error that models the irreducible noise inherent in synthesized views. This decomposition enables ERGO to dynamically estimate the view-specific excess risk and adaptively adjust loss weights during optimization. Furthermore, we introduce geometry-aware and texture-aware objectives that complement the excess-risk-derived weighting mechanism, establishing a synergistic global-local optimization paradigm. Consequently, ERGO demonstrates robustness against supervision noise while consistently enhancing both geometric fidelity and textural quality of the reconstructed 3D content. Extensive experiments on the Google Scanned Objects dataset and the OmniObject3D dataset demonstrate the superiority of ERGO over existing state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.05515v2">Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Project page: https://vala3d.github.io
    </div>
    <details class="paper-abstract">
      Recently, distilling open-vocabulary language features from 2D images into 3D Gaussians has attracted significant attention. Although existing methods achieve impressive language-based interactions of 3D scenes, we observe two fundamental issues: background Gaussians contributing negligibly to a rendered pixel get the same feature as the dominant foreground ones, and multi-view inconsistencies due to view-specific noise in language embeddings. We introduce Visibility-Aware Language Aggregation (VALA), a lightweight yet effective method that computes marginal contributions for each ray and applies a visibility-aware gate to retain only visible Gaussians. Moreover, we propose a streaming weighted geometric median in cosine space to merge noisy multi-view features. Our method yields a robust, view-consistent language feature embedding in a fast and memory-efficient manner. VALA improves open-vocabulary localization and segmentation across reference datasets, consistently surpassing existing works. More results are available at https://vala3d.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10239v1">XSPLAIN: XAI-enabling Splat-based Prototype Learning for Attribute-aware INterpretability</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has rapidly become a standard for high-fidelity 3D reconstruction, yet its adoption in multiple critical domains is hindered by the lack of interpretability of the generation models as well as classification of the Splats. While explainability methods exist for other 3D representations, like point clouds, they typically rely on ambiguous saliency maps that fail to capture the volumetric coherence of Gaussian primitives. We introduce XSPLAIN, the first ante-hoc, prototype-based interpretability framework designed specifically for 3DGS classification. Our approach leverages a voxel-aggregated PointNet backbone and a novel, invertible orthogonal transformation that disentangles feature channels for interpretability while strictly preserving the original decision boundaries. Explanations are grounded in representative training examples, enabling intuitive ``this looks like that'' reasoning without any degradation in classification performance. A rigorous user study (N=51) demonstrates a decisive preference for our approach: participants selected XSPLAIN explanations 48.4\% of the time as the best, significantly outperforming baselines $(p<0.001)$, showing that XSPLAIN provides transparency and user trust. The source code for this work is available at: https://github.com/Solvro/ml-splat-xai
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09999v1">Faster-GS: Analyzing and Improving Gaussian Splatting Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Project page: https://fhahlbohm.github.io/faster-gaussian-splatting
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have focused on accelerating optimization while preserving reconstruction quality. However, many proposed methods entangle implementation-level improvements with fundamental algorithmic modifications or trade performance for fidelity, leading to a fragmented research landscape that complicates fair comparison. In this work, we consolidate and evaluate the most effective and broadly applicable strategies from prior 3DGS research and augment them with several novel optimizations. We further investigate underexplored aspects of the framework, including numerical stability, Gaussian truncation, and gradient approximation. The resulting system, Faster-GS, provides a rigorously optimized algorithm that we evaluate across a comprehensive suite of benchmarks. Our experiments demonstrate that Faster-GS achieves up to 5$\times$ faster training while maintaining visual quality, establishing a new cost-effective and resource efficient baseline for 3DGS optimization. Furthermore, we demonstrate that optimizations can be applied to 4D Gaussian reconstruction, leading to efficient non-rigid scene optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10173v1">ArtisanGS: Interactive Tools for Gaussian Splat Selection with AI and Human in the Loop</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 12 pages, includes supplementary material
    </div>
    <details class="paper-abstract">
      Representation in the family of 3D Gaussian Splats (3DGS) are growing into a viable alternative to traditional graphics for an expanding number of application, including recent techniques that facilitate physics simulation and animation. However, extracting usable objects from in-the-wild captures remains challenging and controllable editing techniques for this representation are limited. Unlike the bulk of emerging techniques, focused on automatic solutions or high-level editing, we introduce an interactive suite of tools centered around versatile Gaussian Splat selection and segmentation. We propose a fast AI-driven method to propagate user-guided 2D selection masks to 3DGS selections. This technique allows for user intervention in the case of errors and is further coupled with flexible manual selection and segmentation tools. These allow a user to achieve virtually any binary segmentation of an unstructured 3DGS scene. We evaluate our toolset against the state-of-the-art for Gaussian Splat selection and demonstrate their utility for downstream applications by developing a user-guided local editing approach, leveraging a custom Video Diffusion Model. With flexible selection tools, users have direct control over the areas that the AI can modify. Our selection and editing tools can be used for any in-the-wild capture without additional optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08958v2">Grow with the Flow: 4D Reconstruction of Growing Plants with Gaussian Flow Fields</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Project page: https://weihanluo.ca/growflow/
    </div>
    <details class="paper-abstract">
      Modeling the time-varying 3D appearance of plants during their growth poses unique challenges: unlike many dynamic scenes, plants generate new geometry over time as they expand, branch, and differentiate. Recent motion modeling techniques are ill-suited to this problem setting. For example, deformation fields cannot introduce new geometry, and 4D Gaussian splatting constrains motion to a linear trajectory in space and time and cannot track the same set of Gaussians over time. Here, we introduce a 3D Gaussian flow field representation that models plant growth as a time-varying derivative over Gaussian parameters -- position, scale, orientation, color, and opacity -- enabling nonlinear and continuous-time growth dynamics. To initialize a sufficient set of Gaussian primitives, we reconstruct the mature plant and learn a process of reverse growth, effectively simulating the plant's developmental history in reverse. Our approach achieves superior image quality and geometric accuracy compared to prior methods on multi-view timelapse datasets of plant growth, providing a new approach for appearance modeling of growing 3D structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09816v1">CompSplat: Compression-aware 3D Gaussian Splatting for Real-world Video</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      High-quality novel view synthesis (NVS) from real-world videos is crucial for applications such as cultural heritage preservation, digital twins, and immersive media. However, real-world videos typically contain long sequences with irregular camera trajectories and unknown poses, leading to pose drift, feature misalignment, and geometric distortion during reconstruction. Moreover, lossy compression amplifies these issues by introducing inconsistencies that gradually degrade geometry and rendering quality. While recent studies have addressed either long-sequence NVS or unposed reconstruction, compression-aware approaches still focus on specific artifacts or limited scenarios, leaving diverse compression patterns in long videos insufficiently explored. In this paper, we propose CompSplat, a compression-aware training framework that explicitly models frame-wise compression characteristics to mitigate inter-frame inconsistency and accumulated geometric errors. CompSplat incorporates compression-aware frame weighting and an adaptive pruning strategy to enhance robustness and geometric consistency, particularly under heavy compression. Extensive experiments on challenging benchmarks, including Tanks and Temples, Free, and Hike, demonstrate that CompSplat achieves state-of-the-art rendering quality and pose accuracy, significantly surpassing most recent state-of-the-art NVS approaches under severe compression conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08625v2">OpenMonoGS-SLAM: Monocular Gaussian Splatting SLAM with Open-set Semantics</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Work in progress. Project page: https://jisang1528.github.io/OpenMonoGS-SLAM/
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) is a foundational component in robotics, AR/VR, and autonomous systems. With the rising focus on spatial AI in recent years, combining SLAM with semantic understanding has become increasingly important for enabling intelligent perception and interaction. Recent efforts have explored this integration, but they often rely on depth sensors or closed-set semantic models, limiting their scalability and adaptability in open-world environments. In this work, we present OpenMonoGS-SLAM, the first monocular SLAM framework that unifies 3D Gaussian Splatting (3DGS) with open-set semantic understanding. To achieve our goal, we leverage recent advances in Visual Foundation Models (VFMs), including MASt3R for visual geometry and SAM and CLIP for open-vocabulary semantics. These models provide robust generalization across diverse tasks, enabling accurate monocular camera tracking and mapping, as well as a rich understanding of semantics in open-world environments. Our method operates without any depth input or 3D semantic ground truth, relying solely on self-supervised learning objectives. Furthermore, we propose a memory mechanism specifically designed to manage high-dimensional semantic features, which effectively constructs Gaussian semantic feature maps, leading to strong overall performance. Experimental results demonstrate that our approach achieves performance comparable to or surpassing existing baselines in both closed-set and open-set segmentation tasks, all without relying on supplementary sensors such as depth maps or semantic annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09736v1">Toward Fine-Grained Facial Control in 3D Talking Head Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Audio-driven talking head generation is a core component of digital avatars, and 3D Gaussian Splatting has shown strong performance in real-time rendering of high-fidelity talking heads. However, achieving precise control over fine-grained facial movements remains a significant challenge, particularly due to lip-synchronization inaccuracies and facial jitter, both of which can contribute to the uncanny valley effect. To address these challenges, we propose Fine-Grained 3D Gaussian Splatting (FG-3DGS), a novel framework that enables temporally consistent and high-fidelity talking head generation. Our method introduces a frequency-aware disentanglement strategy to explicitly model facial regions based on their motion characteristics. Low-frequency regions, such as the cheeks, nose, and forehead, are jointly modeled using a standard MLP, while high-frequency regions, including the eyes and mouth, are captured separately using a dedicated network guided by facial area masks. The predicted motion dynamics, represented as Gaussian deltas, are applied to the static Gaussians to generate the final head frames, which are rendered via a rasterizer using frame-specific camera parameters. Additionally, a high-frequency-refined post-rendering alignment mechanism, learned from large-scale audio-video pairs by a pretrained model, is incorporated to enhance per-frame generation and achieve more accurate lip synchronization. Extensive experiments on widely used datasets for talking head generation demonstrate that our method outperforms recent state-of-the-art approaches in producing high-fidelity, lip-synced talking head videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09415v1">Stability and Concentration in Nonlinear Inverse Problems with Block-Structured Parameters: Lipschitz Geometry, Identifiability, and an Application to Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      We develop an operator-theoretic framework for stability and statistical concentration in nonlinear inverse problems with block-structured parameters. Under a unified set of assumptions combining blockwise Lipschitz geometry, local identifiability, and sub-Gaussian noise, we establish deterministic stability inequalities, global Lipschitz bounds for least-squares misfit functionals, and nonasymptotic concentration estimates. These results yield high-probability parameter error bounds that are intrinsic to the forward operator and independent of any specific reconstruction algorithm. As a concrete instantiation, we verify that the Gaussian Splatting rendering operator satisfies the proposed assumptions and derive explicit constants governing its Lipschitz continuity and resolution-dependent observability. This leads to a fundamental stability--resolution tradeoff, showing that estimation error is inherently constrained by the ratio between image resolution and model complexity. Overall, the analysis characterizes operator-level limits for a broad class of high-dimensional nonlinear inverse problems arising in modern imaging and differentiable rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09162v2">GTAvatar: Bridging Gaussian Splatting and Texture Mapping for Relightable and Editable Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2026-02-09
      | 💬 Eurographics 2026 Project page: https://kelianb.github.io/GTAvatar/
    </div>
    <details class="paper-abstract">
      Recent advancements in Gaussian Splatting have enabled increasingly accurate reconstruction of photorealistic head avatars, opening the door to numerous applications in visual effects, videoconferencing, and virtual reality. This, however, comes with the lack of intuitive editability offered by traditional triangle mesh-based methods. In contrast, we propose a method that combines the accuracy and fidelity of 2D Gaussian Splatting with the intuitiveness of UV texture mapping. By embedding each canonical Gaussian primitive's local frame into a patch in the UV space of a template mesh in a computationally efficient manner, we reconstruct continuous editable material head textures from a single monocular video on a conventional UV domain. Furthermore, we leverage an efficient physically based reflectance model to enable relighting and editing of these intrinsic material maps. Through extensive comparisons with state-of-the-art methods, we demonstrate the accuracy of our reconstructions, the quality of our relighting results, and the ability to provide intuitive controls for modifying an avatar's appearance and geometry via texture mapping without additional optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08909v1">Analysis of Converged 3D Gaussian Splatting Solutions: Density Effects and Prediction Limit</a></div>
    <div class="paper-meta">
      📅 2026-02-09
    </div>
    <details class="paper-abstract">
      We investigate what structure emerges in 3D Gaussian Splatting (3DGS) solutions from standard multi-view optimization. We term these Rendering-Optimal References (RORs) and analyze their statistical properties, revealing stable patterns: mixture-structured scales and bimodal radiance across diverse scenes. To understand what determines these parameters, we apply learnability probes by training predictors to reconstruct RORs from point clouds without rendering supervision. Our analysis uncovers fundamental density-stratification. Dense regions exhibit geometry-correlated parameters amenable to render-free prediction, while sparse regions show systematic failure across architectures. We formalize this through variance decomposition, demonstrating that visibility heterogeneity creates covariance-dominated coupling between geometric and appearance parameters in sparse regions. This reveals the dual character of RORs: geometric primitives where point clouds suffice, and view synthesis primitives where multi-view constraints are essential. We provide density-aware strategies that improve training robustness and discuss architectural implications for systems that adaptively balance feed-forward prediction and rendering-based refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08784v1">GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion</a></div>
    <div class="paper-meta">
      📅 2026-02-09
      | 💬 8 pages, 6 figures. Accepted to IEEE ICRA 2026
    </div>
    <details class="paper-abstract">
      Robust and accurate perception of dynamic objects and map elements is crucial for autonomous vehicles performing safe navigation in complex traffic scenarios. While vision-only methods have become the de facto standard due to their technical advances, they can benefit from effective and cost-efficient fusion with radar measurements. In this work, we advance fusion methods by repurposing Gaussian Splatting as an efficient universal view transformer that bridges the view disparity gap, mapping both image pixels and radar points into a common Bird's-Eye View (BEV) representation. Our main contribution is GaussianCaR, an end-to-end network for BEV segmentation that, unlike prior BEV fusion methods, leverages Gaussian Splatting to map raw sensor information into latent features for efficient camera-radar fusion. Our architecture combines multi-scale fusion with a transformer decoder to efficiently extract BEV features. Experimental results demonstrate that our approach achieves performance on par with, or even surpassing, the state of the art on BEV segmentation tasks (57.3%, 82.9%, and 50.1% IoU for vehicles, roads, and lane dividers) on the nuScenes dataset, while maintaining a 3.2x faster inference runtime. Code and project page are available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08724v1">Rotated Lights for Consistent and Efficient 2D Gaussians Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2026-02-09
      | 💬 Project Page: https://rotlight-ir.github.io/
    </div>
    <details class="paper-abstract">
      Inverse rendering aims to decompose a scene into its geometry, material properties and light conditions under a certain rendering model. It has wide applications like view synthesis, relighting, and scene editing. In recent years, inverse rendering methods have been inspired by view synthesis approaches like neural radiance fields and Gaussian splatting, which are capable of efficiently decomposing a scene into its geometry and radiance. They then further estimate the material and lighting that lead to the observed scene radiance. However, the latter step is highly ambiguous and prior works suffer from inaccurate color and baked shadows in their albedo estimation albeit their regularization. To this end, we propose RotLight, a simple capturing setup, to address the ambiguity. Compared to a usual capture, RotLight only requires the object to be rotated several times during the process. We show that as few as two rotations is effective in reducing artifacts. To further improve 2DGS-based inverse rendering, we additionally introduce a proxy mesh that not only allows accurate incident light tracing, but also enables a residual constraint and improves global illumination handling. We demonstrate with both synthetic and real world datasets that our method achieves superior albedo estimation while keeping efficient computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08266v1">Informative Object-centric Next Best View for Object-aware 3D Gaussian Splatting in Cluttered Scenes</a></div>
    <div class="paper-meta">
      📅 2026-02-09
      | 💬 9 pages, 8 figures, 4 tables, accepted to ICRA 2026
    </div>
    <details class="paper-abstract">
      In cluttered scenes with inevitable occlusions and incomplete observations, selecting informative viewpoints is essential for building a reliable representation. In this context, 3D Gaussian Splatting (3DGS) offers a distinct advantage, as it can explicitly guide the selection of subsequent viewpoints and then refine the representation with new observations. However, existing approaches rely solely on geometric cues, neglect manipulation-relevant semantics, and tend to prioritize exploitation over exploration. To tackle these limitations, we introduce an instance-aware Next Best View (NBV) policy that prioritizes underexplored regions by leveraging object features. Specifically, our object-aware 3DGS distills instancelevel information into one-hot object vectors, which are used to compute confidence-weighted information gain that guides the identification of regions associated with erroneous and uncertain Gaussians. Furthermore, our method can be easily adapted to an object-centric NBV, which focuses view selection on a target object, thereby improving reconstruction robustness to object placement. Experiments demonstrate that our NBV policy reduces depth error by up to 77.14% on the synthetic dataset and 34.10% on the real-world GraspNet dataset compared to baselines. Moreover, compared to targeting the entire scene, performing NBV on a specific object yields an additional reduction of 25.60% in depth error for that object. We further validate the effectiveness of our approach through real-world robotic manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10152v2">Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer</a></div>
    <div class="paper-meta">
      📅 2026-02-09
      | 💬 ICLR 2026 Project Page https://yecongwan.github.io/Color3D/
    </div>
    <details class="paper-abstract">
      In this work, we present Color3D, a highly adaptable framework for colorizing both static and dynamic 3D scenes from monochromatic inputs, delivering visually diverse and chromatically vibrant reconstructions with flexible user-guided control. In contrast to existing methods that focus solely on static scenarios and enforce multi-view consistency by averaging color variations which inevitably sacrifice both chromatic richness and controllability, our approach is able to preserve color diversity and steerability while ensuring cross-view and cross-time consistency. In particular, the core insight of our method is to colorize only a single key view and then fine-tune a personalized colorizer to propagate its color to novel views and time steps. Through personalization, the colorizer learns a scene-specific deterministic color mapping underlying the reference view, enabling it to consistently project corresponding colors to the content in novel views and video frames via its inherent inductive bias. Once trained, the personalized colorizer can be applied to infer consistent chrominance for all other images, enabling direct reconstruction of colorful 3D scenes with a dedicated Lab color space Gaussian splatting representation. The proposed framework ingeniously recasts complicated 3D colorization as a more tractable single image paradigm, allowing seamless integration of arbitrary image colorization models with enhanced flexibility and controllability. Extensive experiments across diverse static and dynamic 3D colorization benchmarks substantiate that our method can deliver more consistent and chromatically rich renderings with precise user control. Project Page https://yecongwan.github.io/Color3D/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15897v2">ThermoSplat: Cross-Modal 3D Gaussian Splatting with Feature Modulation and Geometry Decoupling</a></div>
    <div class="paper-meta">
      📅 2026-02-09
    </div>
    <details class="paper-abstract">
      Multi-modal scene reconstruction integrating RGB and thermal infrared data is essential for robust environmental perception across diverse lighting and weather conditions. However, extending 3D Gaussian Splatting (3DGS) to multi-spectral scenarios remains challenging. Current approaches often struggle to fully leverage the complementary information of multi-modal data, typically relying on mechanisms that either tend to neglect cross-modal correlations or leverage shared representations that fail to adaptively handle the complex structural correlations and physical discrepancies between spectrums. To address these limitations, we propose ThermoSplat, a novel framework that enables deep spectral-aware reconstruction through active feature modulation and adaptive geometry decoupling. First, we introduce a Spectrum-Aware Adaptive Modulation that dynamically conditions shared latent features on thermal structural priors, effectively guiding visible texture synthesis with reliable cross-modal geometric cues. Second, to accommodate modality-specific geometric inconsistencies, we propose a Modality-Adaptive Geometric Decoupling scheme that learns independent opacity offsets and executes an independent rasterization pass for the thermal branch. Additionally, a hybrid rendering pipeline is employed to integrate explicit Spherical Harmonics with implicit neural decoding, ensuring both semantic consistency and high-frequency detail preservation. Extensive experiments on the RGBT-Scenes dataset demonstrate that ThermoSplat achieves state-of-the-art rendering quality across both visible and thermal spectrums.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03510v2">G2P: Gaussian-to-Point Attribute Alignment for Boundary-Aware 3D Semantic Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-02-08
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Semantic segmentation on point clouds is critical for 3D scene understanding. However, sparse and irregular point distributions provide limited appearance evidence, making geometry-only features insufficient to distinguish objects with similar shapes but distinct appearances (e.g., color, texture, material). We propose Gaussian-to-Point (G2P), which transfers appearance-aware attributes from 3D Gaussian Splatting to point clouds for more discriminative and appearance-consistent segmentation. Our G2P address the misalignment between optimized Gaussians and original point geometry by establishing point-wise correspondences. By leveraging Gaussian opacity attributes, we resolve the geometric ambiguity that limits existing models. Additionally, Gaussian scale attributes enable precise boundary localization in complex 3D scenes. Extensive experiments demonstrate that our approach achieves superior performance on standard benchmarks and shows significant improvements on geometrically challenging classes, all without any 2D or language supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.04597v3">Targetless LiDAR-Camera Calibration with Neural Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-07
      | 💬 IEEE RA-L 2026. Project page: https://zang09.github.io/tlc-calib-site
    </div>
    <details class="paper-abstract">
      Accurate LiDAR-camera calibration is crucial for multi-sensor systems. However, traditional methods often rely on physical targets, which are impractical for real-world deployment. Moreover, even carefully calibrated extrinsics can degrade over time due to sensor drift or external disturbances, necessitating periodic recalibration. To address these challenges, we present a Targetless LiDAR-Camera Calibration (TLC-Calib) that jointly optimizes sensor poses with a neural Gaussian-based scene representation. Reliable LiDAR points are frozen as anchor Gaussians to preserve global structure, while auxiliary Gaussians prevent local overfitting under noisy initialization. Our fully differentiable pipeline with photometric and geometric regularization achieves robust and generalizable calibration, consistently outperforming existing targetless methods on the KITTI-360, Waymo, and Fast-LIVO2 datasets. In addition, it yields more consistent Novel View Synthesis results, reflecting improved extrinsic alignment. The project page is available at: https://www.haebeom.com/tlc-calib-site/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06846v1">DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos</a></div>
    <div class="paper-meta">
      📅 2026-02-06
    </div>
    <details class="paper-abstract">
      Spatial audio is crucial for creating compelling immersive 360-degree video experiences. However, generating realistic spatial audio, such as first-order ambisonics (FOA), from 360-degree videos in complex acoustic scenes remains challenging. Existing methods often overlook the dynamic nature and acoustic complexity of 360-degree scenes, fail to fully account for dynamic sound sources, and neglect complex environmental effects such as occlusion, reflections, and reverberation, which are influenced by scene geometries and materials. We propose DynFOA, a framework based on dynamic acoustic perception and conditional diffusion, for generating high-fidelity FOA from 360-degree videos. DynFOA first performs visual processing via a video encoder, which detects and localizes multiple dynamic sound sources, estimates their depth and semantics, and reconstructs the scene geometry and materials using a 3D Gaussian Splatting. This reconstruction technique accurately models occlusion, reflections, and reverberation based on the geometries and materials of the reconstructed 3D scene and the listener's viewpoint. The audio encoder then captures the spatial motion and temporal 4D sound source trajectories to fine-tune the diffusion-based FOA generator. The fine-tuned FOA generator adjusts spatial cues in real time, ensuring consistent directional fidelity during listener head rotation and complex environmental changes. Extensive evaluations demonstrate that DynFOA consistently outperforms existing methods across metrics such as spatial accuracy, acoustic fidelity, and distribution matching, while also improving the user experience. Therefore, DynFOA provides a robust and scalable approach to rendering realistic dynamic spatial audio for VR and immersive media applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06830v1">GaussianPOP: Principled Simplification Framework for Compact 3D Gaussian Splatting via Error Quantification</a></div>
    <div class="paper-meta">
      📅 2026-02-06
    </div>
    <details class="paper-abstract">
      Existing 3D Gaussian Splatting simplification methods commonly use importance scores, such as blending weights or sensitivity, to identify redundant Gaussians. However, these scores are not driven by visual error metrics, often leading to suboptimal trade-offs between compactness and rendering fidelity. We present GaussianPOP, a principled simplification framework based on analytical Gaussian error quantification. Our key contribution is a novel error criterion, derived directly from the 3DGS rendering equation, that precisely measures each Gaussian's contribution to the rendered image. By introducing a highly efficient algorithm, our framework enables practical error calculation in a single forward pass. The framework is both accurate and flexible, supporting on-training pruning as well as post-training simplification via iterative error re-quantification for improved stability. Experimental results show that our method consistently outperforms existing state-of-the-art pruning methods across both application scenarios, achieving a superior trade-off between model compactness and high rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07101v1">Zero-Shot UAV Navigation in Forests via Relightable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-06
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      UAV navigation in unstructured outdoor environments using passive monocular vision is hindered by the substantial visual domain gap between simulation and reality. While 3D Gaussian Splatting enables photorealistic scene reconstruction from real-world data, existing methods inherently couple static lighting with geometry, severely limiting policy generalization to dynamic real-world illumination. In this paper, we propose a novel end-to-end reinforcement learning framework designed for effective zero-shot transfer to unstructured outdoors. Within a high-fidelity simulation grounded in real-world data, our policy is trained to map raw monocular RGB observations directly to continuous control commands. To overcome photometric limitations, we introduce Relightable 3D Gaussian Splatting, which decomposes scene components to enable explicit, physically grounded editing of environmental lighting within the neural representation. By augmenting training with diverse synthesized lighting conditions ranging from strong directional sunlight to diffuse overcast skies, we compel the policy to learn robust, illumination-invariant visual features. Extensive real-world experiments demonstrate that a lightweight quadrotor achieves robust, collision-free navigation in complex forest environments at speeds up to 10 m/s, exhibiting significant resilience to drastic lighting variations without fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06343v1">Uncertainty-Aware 4D Gaussian Splatting for Monocular Occluded Human Rendering</a></div>
    <div class="paper-meta">
      📅 2026-02-06
    </div>
    <details class="paper-abstract">
      High-fidelity rendering of dynamic humans from monocular videos typically degrades catastrophically under occlusions. Existing solutions incorporate external priors-either hallucinating missing content via generative models, which induces severe temporal flickering, or imposing rigid geometric heuristics that fail to capture diverse appearances. To this end, we reformulate the task as a Maximum A Posteriori estimation problem under heteroscedastic observation noise. In this paper, we propose U-4DGS, a framework integrating a Probabilistic Deformation Network and a Double Rasterization pipeline. This architecture renders pixel-aligned uncertainty maps that act as an adaptive gradient modulator, automatically attenuating artifacts from unreliable observations. Furthermore, to prevent geometric drift in regions lacking reliable visual cues, we enforce Confidence-Aware Regularizations, which leverage the learned uncertainty to selectively propagate spatial-temporal validity. Extensive experiments on ZJU-MoCap and OcMotion demonstrate that U-4DGS achieves SOTA rendering fidelity and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05822v1">NVS-HO: A Benchmark for Novel View Synthesis of Handheld Objects</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      We propose NVS-HO, the first benchmark designed for novel view synthesis of handheld objects in real-world environments using only RGB inputs. Each object is recorded in two complementary RGB sequences: (1) a handheld sequence, where the object is manipulated in front of a static camera, and (2) a board sequence, where the object is fixed on a ChArUco board to provide accurate camera poses via marker detection. The goal of NVS-HO is to learn a NVS model that captures the full appearance of an object from (1), whereas (2) provides the ground-truth images used for evaluation. To establish baselines, we consider both a classical SfM pipeline and a state-of-the-art pre-trained feed-forward neural network (VGGT) as pose estimators, and train NVS models based on NeRF and Gaussian Splatting. Our experiments reveal significant performance gaps in current methods under unconstrained handheld conditions, highlighting the need for more robust approaches. NVS-HO thus offers a challenging real-world benchmark to drive progress in RGB-based novel view synthesis of handheld objects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02989v2">SharpTimeGS: Sharp and Stable Dynamic Gaussian Splatting via Lifespan Modulation</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Novel view synthesis of dynamic scenes is fundamental to achieving photorealistic 4D reconstruction and immersive visual experiences. Recent progress in Gaussian-based representations has significantly improved real-time rendering quality, yet existing methods still struggle to maintain a balance between long-term static and short-term dynamic regions in both representation and optimization. To address this, we present SharpTimeGS, a lifespan-aware 4D Gaussian framework that achieves temporally adaptive modeling of both static and dynamic regions under a unified representation. Specifically, we introduce a learnable lifespan parameter that reformulates temporal visibility from a Gaussian-shaped decay into a flat-top profile, allowing primitives to remain consistently active over their intended duration and avoiding redundant densification. In addition, the learned lifespan modulates each primitives' motion, reducing drift in long-lived static points while retaining unrestricted motion for short-lived dynamic ones. This effectively decouples motion magnitude from temporal duration, improving long-term stability without compromising dynamic fidelity. Moreover, we design a lifespan-velocity-aware densification strategy that mitigates optimization imbalance between static and dynamic regions by allocating more capacity to regions with pronounced motion while keeping static areas compact and stable. Extensive experiments on multiple benchmarks demonstrate that our method achieves state-of-the-art performance while supporting real-time rendering up to 4K resolution at 100 FPS on one RTX 4090.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17059v3">REArtGS++: Generalizable Articulation Reconstruction with Temporal Geometry Constraint via Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Articulated objects are pervasive in daily environments, such as drawers and refrigerators. Towards their part-level surface reconstruction and joint parameter estimation, REArtGS introduces a category-agnostic approach using multi-view RGB images at two different states. However, we observe that REArtGS still struggles with screw-joint or multi-part objects and lacks geometric constraints for unseen states. In this paper, we propose REArtGS++, a novel method towards generalizable articulated object reconstruction with temporal geometry constraint and planar Gaussian splatting. We first model a decoupled screw motion for each joint without type prior, and jointly optimize part-aware Gaussians with joint parameters through part motion blending. To introduce time-continuous geometric constraint for articulated modeling, we encourage Gaussians to be planar and propose a temporally consistent regularization between planar normal and depth through Taylor first-order expansion. Extensive experiments on both synthetic and real-world articulated objects demonstrate our superiority in generalizable part-level surface reconstruction and joint parameter estimation, compared to existing approaches. Project Site: https://sites.google.com/view/reartgs2/home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.12788v2">Efficient Scene Modeling via Structure-Aware and Region-Prioritized 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes with high fidelity and efficiency remains a central pursuit in computer vision and graphics. Recent advances in 3D Gaussian Splatting (3DGS) enable photorealistic rendering with Gaussian primitives, yet the modeling process remains governed predominantly by photometric supervision. This reliance often leads to irregular spatial distribution and indiscriminate primitive adjustments that largely ignore underlying geometric context. In this work, we rethink Gaussian modeling from a geometric standpoint and introduce Mini-Splatting2, an efficient scene modeling framework that couples structure-aware distribution and region-prioritized optimization, driving 3DGS into a geometry-regulated paradigm. The structure-aware distribution enforces spatial regularity through structured reorganization and representation sparsity, ensuring balanced structural coverage for compact organization. The region-prioritized optimization improves training discrimination through geometric saliency and computational selectivity, fostering appropriate structural emergence for fast convergence. These mechanisms alleviate the long-standing tension among representation compactness, convergence acceleration, and rendering fidelity. Extensive experiments demonstrate that Mini-Splatting2 achieves up to 4$\times$ fewer Gaussians and 3$\times$ faster optimization while maintaining state-of-the-art visual quality, paving the way towards structured and efficient 3D Gaussian modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.02103v3">MVGS: Multi-view Regulated Gaussian Splatting for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 Project Page:https://xiaobiaodu.github.io/mvgs-project/
    </div>
    <details class="paper-abstract">
      Recent works in volume rendering, \textit{e.g.} NeRF and 3D Gaussian Splatting (3DGS), significantly advance the rendering quality and efficiency with the help of the learned implicit neural radiance field or 3D Gaussians. Rendering on top of an explicit representation, the vanilla 3DGS and its variants deliver real-time efficiency by optimizing the parametric model with single-view supervision per iteration during training which is adopted from NeRF. Consequently, certain views are overfitted, leading to unsatisfying appearance in novel-view synthesis and imprecise 3D geometries. To solve aforementioned problems, we propose a new 3DGS optimization method embodying four key novel contributions: 1) We transform the conventional single-view training paradigm into a multi-view training strategy. With our proposed multi-view regulation, 3D Gaussian attributes are further optimized without overfitting certain training views. As a general solution, we improve the overall accuracy in a variety of scenarios and different Gaussian variants. 2) Inspired by the benefit introduced by additional views, we further propose a cross-intrinsic guidance scheme, leading to a coarse-to-fine training procedure concerning different resolutions. 3) Built on top of our multi-view regulated training, we further propose a cross-ray densification strategy, densifying more Gaussian kernels in the ray-intersect regions from a selection of views. 4) By further investigating the densification strategy, we found that the effect of densification should be enhanced when certain views are distinct dramatically. As a solution, we propose a novel multi-view augmented densification strategy, where 3D Gaussians are encouraged to get densified to a sufficient number accordingly, resulting in improved reconstruction accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15281v2">StyleMe3D: Stylization with Disentangled Priors by Multiple Encoders on 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 18 pages; Project page: https://styleme3d.github.io/
    </div>
    <details class="paper-abstract">
      Current 3D Gaussian Splatting stylization approaches are limited in their ability to represent diverse artistic styles, frequently defaulting to low-level texture replacement or yielding semantically inconsistent outputs. In this paper, we introduce StyleMe3D, a novel hierarchical framework that achieves comprehensive, high-fidelity stylization by disentangling multi-level style representations while preserving geometric fidelity. The cornerstone of StyleMe3D is Dynamic Style Score Distillation (DSSD), which harnesses latent priors from a style-aware diffusion model to provide high-level semantic guidance, ensuring robust and expressive style transfer. To further refine this distillation process, we propose a multi-modal alignment strategy using the CLIP latent space: a CLIP-based style stream evaluator (Contrastive Style Descriptor) that enforces middle-level stylistic similarity, and a CLIP-based content stream evaluator (3D Gaussian Quality Assessment) that acts as a global regularizer to mitigate typical GS quality degradation. Finally, a VGG-based Simultaneously Optimized Scale module is integrated to refine fine-grained texture details at the low-level. Extensive experiments demonstrate that our method consistently preserves intricate geometric details and achieves coherent stylistic effects across entire scenes, significantly surpassing state-of-the-art baselines in both qualitative and quantitative evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05190v1">PoseGaussian: Pose-Driven Novel View Synthesis for Robust 3D Human Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      We propose PoseGaussian, a pose-guided Gaussian Splatting framework for high-fidelity human novel view synthesis. Human body pose serves a dual purpose in our design: as a structural prior, it is fused with a color encoder to refine depth estimation; as a temporal cue, it is processed by a dedicated pose encoder to enhance temporal consistency across frames. These components are integrated into a fully differentiable, end-to-end trainable pipeline. Unlike prior works that use pose only as a condition or for warping, PoseGaussian embeds pose signals into both geometric and temporal stages to improve robustness and generalization. It is specifically designed to address challenges inherent in dynamic human scenes, such as articulated motion and severe self-occlusion. Notably, our framework achieves real-time rendering at 100 FPS, maintaining the efficiency of standard Gaussian Splatting pipelines. We validate our approach on ZJU-MoCap, THuman2.0, and in-house datasets, demonstrating state-of-the-art performance in perceptual quality and structural accuracy (PSNR 30.86, SSIM 0.979, LPIPS 0.028).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06122v1">From Blurry to Believable: Enhancing Low-quality Talking Heads with 3D Generative Priors</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 Accepted to 3DV 2026. Project Page: https://humansensinglab.github.io/super-head/
    </div>
    <details class="paper-abstract">
      Creating high-fidelity, animatable 3D talking heads is crucial for immersive applications, yet often hindered by the prevalence of low-quality image or video sources, which yield poor 3D reconstructions. In this paper, we introduce SuperHead, a novel framework for enhancing low-resolution, animatable 3D head avatars. The core challenge lies in synthesizing high-quality geometry and textures, while ensuring both 3D and temporal consistency during animation and preserving subject identity. Despite recent progress in image, video and 3D-based super-resolution (SR), existing SR techniques are ill-equipped to handle dynamic 3D inputs. To address this, SuperHead leverages the rich priors from pre-trained 3D generative models via a novel dynamics-aware 3D inversion scheme. This process optimizes the latent representation of the generative model to produce a super-resolved 3D Gaussian Splatting (3DGS) head model, which is subsequently rigged to an underlying parametric head model (e.g., FLAME) for animation. The inversion is jointly supervised using a sparse collection of upscaled 2D face renderings and corresponding depth maps, captured from diverse facial expressions and camera viewpoints, to ensure realism under dynamic facial motions. Experiments demonstrate that SuperHead generates avatars with fine-grained facial details under dynamic motions, significantly outperforming baseline methods in visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05047v1">QuantumGS: Quantum Encoding Framework for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Recent advances in neural rendering, particularly 3D Gaussian Splatting (3DGS), have enabled real-time rendering of complex scenes. However, standard 3DGS relies on spherical harmonics, which often struggle to accurately capture high-frequency view-dependent effects such as sharp reflections and transparency. While hybrid approaches like Viewing Direction Gaussian Splatting (VDGS) mitigate this limitation using classical Multi-Layer Perceptrons (MLPs), they remain limited by the expressivity of classical networks in low-parameter regimes. In this paper, we introduce QuantumGS, a novel hybrid framework that integrates Variational Quantum Circuits (VQC) into the Gaussian Splatting pipeline. We propose a unique encoding strategy that maps the viewing direction directly onto the Bloch sphere, leveraging the natural geometry of qubits to represent 3D directional data. By replacing classical color-modulating networks with quantum circuits generated via a hypernetwork or conditioning mechanism, we achieve higher expressivity and better generalization. Source code is available in the supplementary material. Code is available at https://github.com/gwilczynski95/QuantumGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04549v1">Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) revolutionized novel view rendering. Instead of inferring from dense spatial points, as implicit representations do, 3DGS uses sparse Gaussians. This enables real-time performance but increases space requirements, hindering applications such as immersive communication. 3DGS compression emerged as a field aimed at alleviating this issue. While impressive progress has been made, at low rates, compression introduces artifacts that degrade visual quality significantly. We introduce NiFi, a method for extreme 3DGS compression through restoration via artifact-aware, diffusion-based one-step distillation. We show that our method achieves state-of-the-art perceptual quality at extremely low rates, down to 0.1 MB, and towards 1000x rate improvement over 3DGS at comparable perceptual performance. The code will be open-sourced upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04349v1">VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      3D editing has emerged as a critical research area to provide users with flexible control over 3D assets. While current editing approaches predominantly focus on 3D Gaussian Splatting or multi-view images, the direct editing of 3D meshes remains underexplored. Prior attempts, such as VoxHammer, rely on voxel-based representations that suffer from limited resolution and necessitate labor-intensive 3D mask. To address these limitations, we propose \textbf{VecSet-Edit}, the first pipeline that leverages the high-fidelity VecSet Large Reconstruction Model (LRM) as a backbone for mesh editing. Our approach is grounded on a analysis of the spatial properties in VecSet tokens, revealing that token subsets govern distinct geometric regions. Based on this insight, we introduce Mask-guided Token Seeding and Attention-aligned Token Gating strategies to precisely localize target regions using only 2D image conditions. Also, considering the difference between VecSet diffusion process versus voxel we design a Drift-aware Token Pruning to reject geometric outliers during the denoising process. Finally, our Detail-preserving Texture Baking module ensures that we not only preserve the geometric details of original mesh but also the textural information. More details can be found in our project page: https://github.com/BlueDyee/VecSet-Edit/tree/main
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04251v1">Towards Next-Generation SLAM: A Survey on 3DGS-SLAM Focusing on Performance, Robustness, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Traditional Simultaneous Localization and Mapping (SLAM) systems often face limitations including coarse rendering quality, insufficient recovery of scene details, and poor robustness in dynamic environments. 3D Gaussian Splatting (3DGS), with its efficient explicit representation and high-quality rendering capabilities, offers a new reconstruction paradigm for SLAM. This survey comprehensively reviews key technical approaches for integrating 3DGS with SLAM. We analyze performance optimization of representative methods across four critical dimensions: rendering quality, tracking accuracy, reconstruction speed, and memory consumption, delving into their design principles and breakthroughs. Furthermore, we examine methods for enhancing the robustness of 3DGS-SLAM in complex environments such as motion blur and dynamic environments. Finally, we discuss future challenges and development trends in this area. This survey aims to provide a technical reference for researchers and foster the development of next-generation SLAM systems characterized by high fidelity, efficiency, and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04043v1">AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      The growing demand for rapid and scalable 3D asset creation has driven interest in feed-forward 3D reconstruction methods, with 3D Gaussian Splatting (3DGS) emerging as an effective scene representation. While recent approaches have demonstrated pose-free reconstruction from unposed image collections, integrating stylization or appearance control into such pipelines remains underexplored. Existing attempts largely rely on image-based conditioning, which limits both controllability and flexibility. In this work, we introduce AnyStyle, a feed-forward 3D reconstruction and stylization framework that enables pose-free, zero-shot stylization through multimodal conditioning. Our method supports both textual and visual style inputs, allowing users to control the scene appearance using natural language descriptions or reference images. We propose a modular stylization architecture that requires only minimal architectural modifications and can be integrated into existing feed-forward 3D reconstruction backbones. Experiments demonstrate that AnyStyle improves style controllability over prior feed-forward stylization methods while preserving high-quality geometric reconstruction. A user study further confirms that AnyStyle achieves superior stylization quality compared to an existing state-of-the-art approach. Repository: https://github.com/joaxkal/AnyStyle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03538v1">Constrained Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      While Dynamic Gaussian Splatting enables high-fidelity 4D reconstruction, its deployment is severely hindered by a fundamental dilemma: unconstrained densification leads to excessive memory consumption incompatible with edge devices, whereas heuristic pruning fails to achieve optimal rendering quality under preset Gaussian budgets. In this work, we propose Constrained Dynamic Gaussian Splatting (CDGS), a novel framework that formulates dynamic scene reconstruction as a budget-constrained optimization problem to enforce a strict, user-defined Gaussian budget during training. Our key insight is to introduce a differentiable budget controller as the core optimization driver. Guided by a multi-modal unified importance score, this controller fuses geometric, motion, and perceptual cues for precise capacity regulation. To maximize the utility of this fixed budget, we further decouple the optimization of static and dynamic elements, employing an adaptive allocation mechanism that dynamically distributes capacity based on motion complexity. Furthermore, we implement a three-phase training strategy to seamlessly integrate these constraints, ensuring precise adherence to the target count. Coupled with a dual-mode hybrid compression scheme, CDGS not only strictly adheres to hardware constraints (error < 2%}) but also pushes the Pareto frontier of rate-distortion performance. Extensive experiments demonstrate that CDGS delivers optimal rendering quality under varying capacity limits, achieving over 3x compression compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20348v3">Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 8 pages, 5 figures. Accepted to IEEE Intelligent Vehicles Symposium (IV) 2026. Revised version (v3) presents camera-ready publication
    </div>
    <details class="paper-abstract">
      3D reconstruction for Digital Twins often relies on LiDAR-based methods, which provide accurate geometry but lack the semantics and textures naturally captured by cameras. Traditional LiDAR-camera fusion approaches require complex calibration and still struggle with certain materials like glass, which are visible in images but poorly represented in point clouds. We propose a camera-only pipeline that reconstructs scenes using 3D Gaussian Splatting from multi-view images, extracts semantic material masks via vision models, converts Gaussian representations to mesh surfaces with projected material labels, and assigns physics-based material properties for accurate sensor simulation in modern graphics engines and simulators. This approach combines photorealistic reconstruction with physics-based material assignment, providing sensor simulation fidelity comparable to LiDAR-camera fusion while eliminating hardware complexity and calibration requirements. We validate our camera-only method using an internal dataset from an instrumented test vehicle, leveraging LiDAR as ground truth for reflectivity validation alongside image similarity metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03327v1">Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Novel view synthesis has evolved rapidly, advancing from Neural Radiance Fields to 3D Gaussian Splatting (3DGS), which offers real-time rendering and rapid training without compromising visual fidelity. However, 3DGS relies heavily on accurate camera poses and high-quality point cloud initialization, which are difficult to obtain in sparse-view scenarios. While traditional Structure from Motion (SfM) pipelines often fail in these settings, existing learning-based point estimation alternatives typically require reliable reference views and remain sensitive to pose or depth errors. In this work, we propose a robust method utilizing π^3, a reference-free point cloud estimation network. We integrate dense initialization from π^3 with a regularization scheme designed to mitigate geometric inaccuracies. Specifically, we employ uncertainty-guided depth supervision, normal consistency loss, and depth warping. Experimental results demonstrate that our approach achieves state-of-the-art performance on the Tanks and Temples, LLFF, DTU, and MipNeRF360 datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02000v2">SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 ICLR 2026; Project Page: https://hebing-sjtu.github.io/SurfSplat-website/
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes from sparse images remains a challenging task due to the difficulty of recovering accurate geometry and texture without optimization. Recent approaches leverage generalizable models to generate 3D scenes using 3D Gaussian Splatting (3DGS) primitive. However, they often fail to produce continuous surfaces and instead yield discrete, color-biased point clouds that appear plausible at normal resolution but reveal severe artifacts under close-up views. To address this issue, we present SurfSplat, a feedforward framework based on 2D Gaussian Splatting (2DGS) primitive, which provides stronger anisotropy and higher geometric precision. By incorporating a surface continuity prior and a forced alpha blending strategy, SurfSplat reconstructs coherent geometry together with faithful textures. Furthermore, we introduce High-Resolution Rendering Consistency (HRRC), a new evaluation metric designed to evaluate high-resolution reconstruction quality. Extensive experiments on RealEstate10K, DL3DV, and ScanNet demonstrate that SurfSplat consistently outperforms prior methods on both standard metrics and HRRC, establishing a robust solution for high-fidelity 3D reconstruction from sparse inputs. Project page: https://hebing-sjtu.github.io/SurfSplat-website/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03207v1">WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via WebGPU</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      We present WebSplatter, an end-to-end GPU rendering pipeline for the heterogeneous web ecosystem. Unlike naive ports, WebSplatter introduces a wait-free hierarchical radix sort that circumvents the lack of global atomics in WebGPU, ensuring deterministic execution across diverse hardware. Furthermore, we propose an opacity-aware geometry culling stage that dynamically prunes splats before rasterization, significantly reducing overdraw and peak memory footprint. Evaluation demonstrates that WebSplatter consistently achieves 1.2$\times$ to 4.5$\times$ speedups over state-of-the-art web viewers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02402v1">SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 Project page: https://city-super.github.io/SoMA/
    </div>
    <details class="paper-abstract">
      Simulating deformable objects under rich interactions remains a fundamental challenge for real-to-sim robot manipulation, with dynamics jointly driven by environmental effects and robot actions. Existing simulators rely on predefined physics or data-driven dynamics without robot-conditioned control, limiting accuracy, stability, and generalization. This paper presents SoMA, a 3D Gaussian Splat simulator for soft-body manipulation. SoMA couples deformable dynamics, environmental forces, and robot joint actions in a unified latent neural space for end-to-end real-to-sim simulation. Modeling interactions over learned Gaussian splats enables controllable, stable long-horizon manipulation and generalization beyond observed trajectories without predefined physical models. SoMA improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.17982v3">HI-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-02
    </div>
    <details class="paper-abstract">
      We present HI-SLAM2, a geometry-aware Gaussian SLAM system that achieves fast and accurate monocular scene reconstruction using only RGB input. Existing Neural SLAM or 3DGS-based SLAM methods often trade off between rendering quality and geometry accuracy, our research demonstrates that both can be achieved simultaneously with RGB input alone. The key idea of our approach is to enhance the ability for geometry estimation by combining easy-to-obtain monocular priors with learning-based dense SLAM, and then using 3D Gaussian splatting as our core map representation to efficiently model the scene. Upon loop closure, our method ensures on-the-fly global consistency through efficient pose graph bundle adjustment and instant map updates by explicitly deforming the 3D Gaussian units based on anchored keyframe updates. Furthermore, we introduce a grid-based scale alignment strategy to maintain improved scale consistency in prior depths for finer depth details. Through extensive experiments on Replica, ScanNet, and ScanNet++, we demonstrate significant improvements over existing Neural SLAM methods and even surpass RGB-D-based methods in both reconstruction and rendering quality. The project page and source code will be made available at https://hi-slam2.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03878v1">Intellectual Property Protection for 3D Gaussian Splatting Assets: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 A collection of relevant papers is summarized and will be continuously updated at \url{https://github.com/tmllab/Awesome-3DGS-IP-Protection}
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a mainstream representation for real-time 3D scene synthesis, enabling applications in virtual and augmented reality, robotics, and 3D content creation. Its rising commercial value and explicit parametric structure raise emerging intellectual property (IP) protection concerns, prompting a surge of research on 3DGS IP protection. However, current progress remains fragmented, lacking a unified view of the underlying mechanisms, protection paradigms, and robustness challenges. To address this gap, we present the first systematic survey on 3DGS IP protection and introduce a bottom-up framework that examines (i) underlying Gaussian-based perturbation mechanisms, (ii) passive and active protection paradigms, and (iii) robustness threats under emerging generative AI era, revealing gaps in technical foundations and robustness characterization and indicating opportunities for deeper investigation. Finally, we outline six research directions across robustness, efficiency, and protection paradigms, offering a roadmap toward reliable and trustworthy IP protection for 3DGS assets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02089v1">UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) enables high-quality, real-time rendering for bounded scenes, its extension to large-scale urban environments gives rise to critical challenges in terms of geometric consistency, memory efficiency, and computational scalability. To address these issues, we present UrbanGS, a scalable reconstruction framework that effectively tackles these challenges for city-scale applications. First, we propose a Depth-Consistent D-Normal Regularization module. Unlike existing approaches that rely solely on monocular normal estimators, which can effectively update rotation parameters yet struggle to update position parameters, our method integrates D-Normal constraints with external depth supervision. This allows for comprehensive updates of all geometric parameters. By further incorporating an adaptive confidence weighting mechanism based on gradient consistency and inverse depth deviation, our approach significantly enhances multi-view depth alignment and geometric coherence, which effectively resolves the issue of geometric accuracy in complex large-scale scenes. To improve scalability, we introduce a Spatially Adaptive Gaussian Pruning (SAGP) strategy, which dynamically adjusts Gaussian density based on local geometric complexity and visibility to reduce redundancy. Additionally, a unified partitioning and view assignment scheme is designed to eliminate boundary artifacts and optimize computational load. Extensive experiments on multiple urban datasets demonstrate that UrbanGS achieves superior performance in rendering quality, geometric accuracy, and memory efficiency, providing a systematic solution for high-fidelity large-scale scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.09606v2">Feat2GS: Probing Visual Foundation Models with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 Project Page: https://fanegg.github.io/Feat2GS/
    </div>
    <details class="paper-abstract">
      Given that visual foundation models (VFMs) are trained on extensive datasets but often limited to 2D images, a natural question arises: how well do they understand the 3D world? With the differences in architecture and training protocols (i.e., objectives, proxy tasks), a unified framework to fairly and comprehensively probe their 3D awareness is urgently needed. Existing works on 3D probing suggest single-view 2.5D estimation (e.g., depth and normal) or two-view sparse 2D correspondence (e.g., matching and tracking). Unfortunately, these tasks ignore texture awareness, and require 3D data as ground-truth, which limits the scale and diversity of their evaluation set. To address these issues, we introduce Feat2GS, which readout 3D Gaussians attributes from VFM features extracted from unposed images. This allows us to probe 3D awareness for geometry and texture via novel view synthesis, without requiring 3D data. Additionally, the disentanglement of 3DGS parameters - geometry ($\boldsymbol{x}$, $α$, $Σ$) and texture ($\boldsymbol{c}$) - enables separate analysis of texture and geometry awareness. Under Feat2GS, we conduct extensive experiments to probe the 3D awareness of several VFMs, and investigate the ingredients that lead to a 3D aware VFM. Building on these findings, we develop several variants that achieve state-of-the-art across diverse datasets. This makes Feat2GS useful for probing VFMs, and as a simple-yet-effective baseline for novel-view synthesis. Code and data are available at https://fanegg.github.io/Feat2GS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01844v1">CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Deep learning has demonstrated remarkable capabilities in simulating complex dynamic systems. However, existing methods require known physical properties as supervision or inputs, limiting their applicability under unknown conditions. To explore this challenge, we introduce Cloth Dynamics Grounding (CDG), a novel scenario for unsupervised learning of cloth dynamics from multi-view visual observations. We further propose Cloth Dynamics Splatting (CloDS), an unsupervised dynamic learning framework designed for CDG. CloDS adopts a three-stage pipeline that first performs video-to-geometry grounding and then trains a dynamics model on the grounded meshes. To cope with large non-linear deformations and severe self-occlusions during grounding, we introduce a dual-position opacity modulation that supports bidirectional mapping between 2D observations and 3D geometry via mesh-based Gaussian splatting in video-to-geometry grounding stage. It jointly considers the absolute and relative position of Gaussian components. Comprehensive experimental evaluations demonstrate that CloDS effectively learns cloth dynamics from visual data while maintaining strong generalization capabilities for unseen configurations. Our code is available at https://github.com/whynot-zyl/CloDS. Visualization results are available at https://github.com/whynot-zyl/CloDS_video}.%\footnote{As in this example.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.14552v2">From Slices to Structures: Unsupervised 3D Reconstruction of Female Pelvic Anatomy from Freehand Transvaginal Ultrasound</a></div>
    <div class="paper-meta">
      📅 2026-02-02
    </div>
    <details class="paper-abstract">
      Volumetric ultrasound has the potential to significantly improve diagnostic accuracy and clinical decision-making, yet its widespread adoption remains limited by dependence on specialized hardware and restrictive acquisition protocols. In this work, we present a novel unsupervised framework for reconstructing 3D anatomical structures from freehand 2D transvaginal ultrasound sweeps, without requiring external tracking or learned pose estimators. Our method, TVGS, adapts the principles of Gaussian Splatting to the domain of ultrasound, introducing a slice-aware, differentiable rasterizer tailored to the unique physics and geometry of ultrasound imaging. We model anatomy as a collection of anisotropic 3D Gaussians and optimize their parameters directly from image-level supervision. To ensure robustness against irregular probe motion, we introduce a joint optimization scheme that refines slice poses alongside anatomical structure. The result is a compact, flexible, and memory-efficient volumetric representation that captures anatomical detail with high spatial fidelity. This work demonstrates that accurate 3D reconstruction from 2D ultrasound images can be achieved through purely computational means, offering a scalable alternative to conventional 3D systems and enabling new opportunities for AI-assisted analysis and diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01723v1">FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-02
    </div>
    <details class="paper-abstract">
      Extending 3D Gaussian Splatting (3DGS) to 4D physical simulation remains challenging. Based on the Material Point Method (MPM), existing methods either rely on manual parameter tuning or distill dynamics from video diffusion models, limiting the generalization and optimization efficiency. Recent attempts using LLMs/VLMs suffer from a text/image-to-3D perceptual gap, yielding unstable physics behavior. In addition, they often ignore the surface structure of 3DGS, leading to implausible motion. We propose FastPhysGS, a fast and robust framework for physics-based dynamic 3DGS simulation:(1) Instance-aware Particle Filling (IPF) with Monte Carlo Importance Sampling (MCIS) to efficiently populate interior particles while preserving geometric fidelity; (2) Bidirectional Graph Decoupling Optimization (BGDO), an adaptive strategy that rapidly optimizes material parameters predicted from a VLM. Experiments show FastPhysGS achieves high-fidelity physical simulation in 1 minute using only 7 GB runtime memory, outperforming prior works with broad potential applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01674v1">VRGaussianAvatar: Integrating 3D Gaussian Avatars into VR</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 Accepted as an IEEE TVCG paper at IEEE VR 2026 (journal track)
    </div>
    <details class="paper-abstract">
      We present VRGaussianAvatar, an integrated system that enables real-time full-body 3D Gaussian Splatting (3DGS) avatars in virtual reality using only head-mounted display (HMD) tracking signals. The system adopts a parallel pipeline with a VR Frontend and a GA Backend. The VR Frontend uses inverse kinematics to estimate full-body pose and streams the resulting pose along with stereo camera parameters to the backend. The GA Backend stereoscopically renders a 3DGS avatar reconstructed from a single image. To improve stereo rendering efficiency, we introduce Binocular Batching, which jointly processes left and right eye views in a single batched pass to reduce redundant computation and support high-resolution VR displays. We evaluate VRGaussianAvatar with quantitative performance tests and a within-subject user study against image- and video-based mesh avatar baselines. Results show that VRGaussianAvatar sustains interactive VR performance and yields higher perceived appearance similarity, embodiment, and plausibility. Project page and source code are available at https://vrgaussianavatar.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01513v1">MarkCleaner: High-Fidelity Watermark Removal via Imperceptible Micro-Geometric Perturbation</a></div>
    <div class="paper-meta">
      📅 2026-02-02
    </div>
    <details class="paper-abstract">
      Semantic watermarks exhibit strong robustness against conventional image-space attacks. In this work, we show that such robustness does not survive under micro-geometric perturbations: spatial displacements can remove watermarks by breaking the phase alignment. Motivated by this observation, we introduce MarkCleaner, a watermark removal framework that avoids semantic drift caused by regeneration-based watermark removal. Specifically, MarkCleaner is trained with micro-geometry-perturbed supervision, which encourages the model to separate semantic content from strict spatial alignment and enables robust reconstruction under subtle geometric displacements. The framework adopts a mask-guided encoder that learns explicit spatial representations and a 2D Gaussian Splatting-based decoder that explicitly parameterizes geometric perturbations while preserving semantic content. Extensive experiments demonstrate that MarkCleaner achieves superior performance in both watermark removal effectiveness and visual fidelity, while enabling efficient real-time inference. Our code will be made available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02602v1">Position: 3D Gaussian Splatting Watermarking Should Be Scenario-Driven and Threat-Model Explicit</a></div>
    <div class="paper-meta">
      📅 2026-02-01
    </div>
    <details class="paper-abstract">
      3D content acquisition and creation are expanding rapidly in the new era of machine learning and AI. 3D Gaussian Splatting (3DGS) has become a promising high-fidelity and real-time representation for 3D content. Similar to the initial wave of digital audio-visual content at the turn of the millennium, the demand for intellectual property protection is also increasing, since explicit and editable 3D parameterization makes unauthorized use and dissemination easier. In this position paper, we argue that effective progress in watermarking 3D assets requires articulated security objectives and realistic threat models, incorporating the lessons learned from digital audio-visual asset protection over the past decades. To address this gap in security specification and evaluation, we advocate a scenario-driven formulation, in which adversarial capabilities are formalized through a security model. Based on this formulation, we construct a reference framework that organizes existing methods and clarifies how specific design choices map to corresponding adversarial assumptions. Within this framework, we also examine a legacy spread-spectrum embedding scheme, characterizing its advantages and limitations and highlighting the important trade-offs it entails. Overall, this work aims to foster effective intellectual property protection for 3D assets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03809v1">Split&Splat: Zero-Shot Panoptic Segmentation via Explicit Instance Modeling and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-02-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (GS) enables fast and high-quality scene reconstruction, but it lacks an object-consistent and semantically aware structure. We propose Split&Splat, a framework for panoptic scene reconstruction using 3DGS. Our approach explicitly models object instances. It first propagates instance masks across views using depth, thus producing view-consistent 2D masks. Each object is then reconstructed independently and merged back into the scene while refining its boundaries. Finally, instance-level semantic descriptors are embedded in the reconstructed objects, supporting various applications, including panoptic segmentation, object retrieval, and 3D editing. Unlike existing methods, Split&Splat tackles the problem by first segmenting the scene and then reconstructing each object individually. This design naturally supports downstream tasks and allows Split&Splat to achieve state-of-the-art performance on the ScanNetv2 segmentation benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01057v1">Radioactive 3D Gaussian Ray Tracing for Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged in computer vision as a promising rendering technique. By adapting the principles of Elliptical Weighted Average (EWA) splatting to a modern differentiable pipeline, 3DGS enables real-time, high-quality novel view synthesis. Building upon this, R2-Gaussian extended the 3DGS paradigm to tomographic reconstruction by rectifying integration bias, achieving state-of-the-art performance in computed tomography (CT). To enable differentiability, R2-Gaussian adopts a local affine approximation: each 3D Gaussian is locally mapped to a 2D Gaussian on the detector and composed via alpha blending to form projections. However, the affine approximation can degrade reconstruction quantitative accuracy and complicate the incorporation of nonlinear geometric corrections. To address these limitations, we propose a tomographic reconstruction framework based on 3D Gaussian ray tracing. Our approach provides two key advantages over splatting-based models: (i) it computes the line integral through 3D Gaussian primitives analytically, avoiding the local affine collapse and thus yielding a more physically consistent forward projection model; and (ii) the ray-tracing formulation gives explicit control over ray origins and directions, which facilitates the precise application of nonlinear geometric corrections, e.g., arc-correction used in positron emission tomography (PET). These properties extend the applicability of Gaussian-based reconstruction to a wider range of realistic tomography systems while improving projection accuracy.
    </details>
</div>
