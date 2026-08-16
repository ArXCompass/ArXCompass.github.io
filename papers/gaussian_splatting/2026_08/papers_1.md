# gaussian splatting - 2026_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13502v1">GS$^{2}$CI: Robust Gaussian Splatting For Snapshot Compressive Imaging via Large Vision Model Priors</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Snapshot Compressive Imaging (SCI) offers an efficient solution for high-speed video acquisition and, under exposure-time camera--scene relative motion, multi-view scene capture by compressing temporal or spatial information into a single 2D measurement. While recent studies have explored SCI for 3D scene reconstruction, existing methods struggle with significant challenges due to information loss, limited viewpoint diversity, and the computational burden of jointly optimizing 3D representations and camera poses. In this work, we propose a novel framework that reconstructs high-quality 3D scenes from a single SCI measurement by leveraging 3D Gaussian Splatting (3DGS) and the powerful priors of large-scale vision foundation models (VFMs). Our primary reconstruction combines measurement-derived 3D VFM initialization with SCI-aware Gaussian optimization. After coarse-stage convergence, an auxiliary 2D VFM provides pseudo-view supervision at synthesized viewpoints for local appearance refinement. To further address the instability caused by ambiguous SCI supervision during 3DGS optimization, we introduce Opacity-Guided Splitting and Growth Regulation (OSGR), an SCI-specific densification strategy that augments split candidates using local opacity statistics, discourages loss-compensating opacity inflation through mean-opacity regulation, and bounds representation growth with explicit candidate-ratio and Gaussian-count constraints. Extensive experiments across multiple benchmarks demonstrate that our method achieves the strongest overall performance, combining leading reconstruction quality and robustness to viewpoint variation with competitive computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13159v1">Splat-based Metal Artifact Reduction in Cone-Beam CT via Polychromatic Modeling</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Cone-beam computed tomography (CBCT) enables volumetric reconstruction from X-ray projections, but suffers from severe artifacts--especially beam hardening--when imaging materials with high attenuation such as metals. These artifacts arise from the polychromatic nature of X-rays and are not properly addressed by conventional monochromatic reconstruction algorithms. While recent neural representation-based methods offer improved reconstruction quality, they are computationally expensive and often impractical for deployment. We propose a novel physics-inspired, self-calibrating metal artifact reduction method that efficiently reconstructs 3D CBCT volumes while correcting beam hardening artifacts. Our method integrates a polychromatic X-ray projection model, material-dependent attenuation profiles, and system response modeling into a Gaussian Splatting framework. Unlike prior work, we eliminate the need for manual metal masks or strong prior assumptions, and we optimize both reconstruction parameters and X-ray spectral characteristics jointly during training. We further introduce a high-fidelity synthetic CBCT dataset generation pipeline validated on Monte-Carlo x-ray simulation toolbox and release new datasets with severe metal-induced artifacts to support the community. This is the first splat-based method for reducing beam hardening in CBCT. Extensive experiments on both synthetic and real-world datasets demonstrate that our method outperforms state-of-the-art approaches in artifact suppression and reconstruction accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.13143v1">ProbSplat: Efficient Probabilistic Hardware for Gaussian Splatting in 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 Accepted for publication in the 2026 IEEE Computer Society Annual Symposium on VLSI (ISVLSI)
    </div>
    <details class="paper-abstract">
      This paper presents ProbSplat, a Compute-in-Memory (CIM)-inspired architecture based on programmable and energy efficient floating-gate inverter columns for probabilistic computing. Improving upon our prior work, ProbSplat programs and stores both means and variances of Gaussian mixture components, and evaluates log-likelihood for gaussian splatting during scene reconstruction with high energy efficiency, suitable for robotics and augmented/virtual reality (AR/VR) at the edge. Our proposed scheme enables independent control of both mean and variance via deterministic adjustment of floating-gate MOSFET threshold voltages, increasing the fidelity of hardware to program probability distributions. The design is simulated in 180nm CMOS on 1.8 V at 50 MHz and achieves mean-variance independence with <2.4% deviation during 3-D Gaussian mixture modeling. Compared to conventional digital implementations, ProbSplat significantly reduces compute complexity, memory footprint, and power consumption. The scalable framework consumes 18pJ energy per log-likelihood inference with 4-bit precision while operating for 500 mixture functions in a 3-D GMM. Scene reconstruction with ProbSplat's characteristics gave satisfactory fidelity of 21.99 PSNR (dB) at 8-bit precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12860v1">HumanoidVLN: A Physics-Grounded Simulator and Benchmark for Vision-Language Navigation Across Diverse Humanoid Embodiments</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) for humanoid robots poses challenges existing benchmarks fail to address: bipedal locomotion imposes physical constraints absent from wheeled agents, humanoid morphologies vary across platforms, and egocentric observations are distorted by locomotion-induced camera dynamics. We present HumanoidVLN, a physics-grounded simulator and benchmark for VLN across diverse humanoid embodiments. Built on NVIDIA Isaac Sim, our platform supports an extensible set of humanoid configurations, demonstrated on four robots (Unitree G1, Unitree H1, Internal-A, Internal-B) spanning 10-12 lower-body DoF and heights from 1.17m to 1.80m, via a hierarchical control stack combining a reinforcement learning locomotion policy with interchangeable PD or MPC path trackers. New robots and VLN models integrate with minimal effort; we demonstrate compatibility with NaVILA, DualVLN, StreamVLN, and JanusVLN. Environments are drawn from artist-designed scenes and 3D Gaussian Splatting reconstructions, filtered for navigable areas exceeding 100 square meters. Instructions are generated by a dual generator-reviewer plus paraphraser multi-agent pipeline with human-in-the-loop verification, yielding 933 collision-aware reference episodes, each paired with one fine-grained instruction and three coarse-grained stylistic variants (formal, natural, casual). Across four models and four embodiments, JanusVLN achieves the highest mean success rate of 43.55% and nDTW of 48.38. In a 20-episode sim-to-real pilot with DualVLN and the Unitree G1, navigation errors correlate strongly (r=0.935), with a mean absolute difference of 0.68m and mean trajectory similarity of 0.782 (+/-0.188) nDTW. These results highlight the interaction between VLN models, controllers, and humanoid embodiments under physical execution. Code, benchmark, and data will be released upon acceptance at https://humanoid-vln.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12825v1">LocusGS: Spatially Grounded Tokens for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-13
    </div>
    <details class="paper-abstract">
      Recent query-based feed-forward 3DGS methods represent a scene using learnable queries, each aggregating multi-view evidence and decoding a group of Gaussians. Ideally, different queries should specialize in coherent local regions of the scene. However, we observe that Gaussians decoded from the same query often scatter across distant scene regions, resulting in weak query-level spatial coherence and poor alignment with the scene structure. We attribute this behavior to the purely latent representation of existing Gaussian queries. To address this limitation, we introduce LocusGS, which augments each Gaussian query with a 3D anchor state consisting of a center and a support radius. The anchor state is progressively refined across decoder layers and is used throughout query interaction, multi-view feature aggregation, and Gaussian generation. Specifically, an anchor-to-ray geometric bias guides each query toward spatially relevant image observations, while anchor-centered decoding organizes its Gaussians within a local region. Experiments on novel view synthesis benchmarks show that LocusGS improves rendering quality over query-based Gaussian token baselines under the same Gaussian budget. Further analysis shows that the learned anchors form coherent spatial layouts and lead to more structured Gaussian distributions, demonstrating that explicit anchor states improve the spatial organization. Our project page: https://leo-frank.github.io/LocusGS_viewer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11928v1">Seed2GS: Camera-Free, Training-Free Object Extraction from 3D Gaussian Scenes via a Single Reference-View Grounding</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Extracting a target object from a pre-built 3D Gaussian Splatting (3DGS) scene enables interactive 3D editing. Existing methods either train for tens of minutes per scene, sacrifice accuracy, or require original reconstruction cameras that pre-built assets may not include. We present Seed2GS, which achieves the highest reported LERF-MASK accuracy without original reconstruction cameras or scene-specific representation training. Its key insight is to separate target identity from 3D coverage. QD-SAM3 selects one reliable reference mask from several open-vocabulary candidates, fixing identity once. Seed lift and visibility-adaptive virtual orbits then expose the object from new viewpoints, while tracking propagates the seed without repeated detection. Because the scene remains frozen, these masks supervise only one temporary foreground logit per Gaussian. On LERF-MASK, Seed2GS reaches 92.1% mean intersection over union (mIoU) with a measured compute-only latency of 9.3 seconds, 3.7 points above the strongest scene-trained baseline and 7.6 points above the closest camera-free baseline. With one fixed test reference per scene, the complete pipeline retains 91.1% mIoU; replacing its predicted seed with a ground-truth mask improves mIoU by only 0.72 points. On 3D-OVS, Seed2GS reaches 95.7% mIoU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11150v2">CausalSplat: Towards Comprehensive Hierarchical Reasoning in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 Accepted to ACM MM 2026
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has advanced open vocabulary scene understanding, existing methods remain confined to explicit queries. They struggle to interpret implicit intents, complex spatial constraints, and commonsense reasoning required for practical embodied interactions. To address this gap, we introduce the task of reasoning 3D Gaussian segmentation and construct two benchmarks, Causal-LERF and Causal-ScanNet. These benchmarks systematically evaluate commonsense, spatial, affordance, and counterfactual reasoning. Evaluations reveal that current state of the art methods perform poorly on these reasoning challenges. Therefore, we propose CausalSplat, a framework that integrates vision-language models with 3D scene graphs to disentangle explicit structural perception from implicit logical inference. Extensive experiments demonstrate that CausalSplat achieves state of the art performance on our reasoning benchmarks while showing strong generalizability on standard referring and open vocabulary 3D segmentation tasks. Project Page: https://jiayuding031020.github.io/CausalSplat
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04348v2">SCAR-GS: Spatial Context Attention for Residuals in Progressive Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting have allowed for real-time, high-fidelity novel view synthesis. Nonetheless, these models have significant storage requirements for large and medium-sized scenes, hindering their deployment over cloud and streaming services. Some of the most recent progressive compression techniques for these models rely on progressive masking and scalar quantization techniques to reduce the bitrate of Gaussian attributes using spatial context models. While effective, scalar quantization may not optimally capture the correlations of high-dimensional feature vectors, which can potentially limit the rate-distortion performance. In this work, we introduce a novel progressive codec for 3D Gaussian Splatting that replaces traditional methods with a more powerful Residual Vector Quantization approach to compress the primitive features. Our key contribution is an auto-regressive entropy model, guided by a multi-resolution hash grid, that accurately predicts the conditional probability of each successive transmitted index, allowing for coarse and refinement layers to be compressed with high efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13794v2">BlitzGS: City-Scale Gaussian Splatting at Lightning Speed</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Large-scale 3D Gaussian Splatting underpins digital twins, simulation, and aerial mapping, yet city-scale training remains computationally expensive even with multi-GPU execution because every iteration must preprocess, communicate, and rasterize an overly dense set of primitives. At any given step, only a small fraction of these primitives contribute meaningfully to the loss; the rest incur redundant storage, communication, and rasterization costs. Existing approaches improve individual cost factors but do not fully address the underlying question: which Gaussians should be stored on each GPU, rendered for each view, and retained after early geometry formation? We present BlitzGS, a distributed 3DGS framework that reduces the active Gaussian workload at three coupled levels. At the system level, it shards Gaussians across GPUs by index parity rather than spatial blocks, mitigating the cross-block visibility redundancy of spatial partitioning, and distributes each render step through a single cross-GPU exchange that routes projected Gaussians to their tile owners. At the model level, it controls the population from both ends of densification. Scheduled importance-scoring passes prune redundant survivors, and a lightweight spawn gate withholds candidates predicted not to survive. The same importance signal also feeds back into density control. At the view level, a distance-based LOD gate and an importance-based mask trim each camera's active set. On large-scale benchmarks, BlitzGS matches the rendering quality of recent large-scale baselines while delivering nearly an order-of-magnitude speedup, training city-scale scenes in tens of minutes. Our code is available at https://github.com/AkierRaee/BlitzGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11100v1">WildFireGS: Physics-Based Wildfire Simulation in Large-Scale Semantics-Enriched Gaussian Splatting Forest Scenes</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Climate-driven environmental change is driving an increase in both the frequency and severity of wildfire events, making accurate simulation and prediction critical for effective risk mitigation and landscape management. While recent physics-based wildfire models achieve high realism by explicitly simulating combustion, heat transfer, and fuel dynamics, they remain largely restricted to synthetic environments with complete and idealized knowledge of forest structure, limiting their applicability to real-world environments captured via aerial imagery. To provide a pathway toward real-world wildfire digital twins derived directly from observational data, we present WildFireGS, a physics-based wildfire simulation framework operating directly on large-scale, semantics-enriched 3D Gaussian Splatting forest reconstructions. Our approach bridges learning-based scene reconstruction and environmental simulation by augmenting Gaussian primitives with semantics and material properties that encode vegetation type and fuel characteristics. We introduce a particle-based combustion model that operates natively on Gaussian representations, simulating ignition, heat transfer, combustion, and flame propagation across complex forest structures. This enables direct physics-based simulation of fire behavior on reconstructed real-world environments, without requiring conversion to explicit meshes or volumetric grids. We demonstrate the modularity of WildFireGS through a rain-driven cooling mechanism in terms of an energy-sink process to realistically model fire containment. Evaluations on synthetic scenes and real aerial forest captures show physically consistent wildfire behavior, reproducing characteristic dynamics including propagation scaling with vegetation density, wind velocity, and terrain slope. In addition, we validate our model through novel firebreak experiments and biomass loss estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10938v1">GS-CPE: Unified 6-Degree-of-Freedom Camera Pose Estimation via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 8 pages, accepted at IROS 2026
    </div>
    <details class="paper-abstract">
      Despite substantial progress in visual localization, from scene coordinate regression to direct camera pose regression, achieving both robust generalization and high accuracy remain challenging. This study introduces GS-CPE (Gaussian Splatting based Camera Pose Estimation), a coarse-to-fine framework for 6-DoF camera pose estimation that unifies geometry-based coarse pose estimation with robust 3D Gaussian Splatting (3DGS) warping based pose refinement. GS-CPE first estimates a coarse pose via retrieval-guided geometric pose estimation on a 3DGS scene representation, then refines it by minimizing a visibility aware masked RGB warping objective in a multi-scale optimization framework, with adaptive re-rendering. Extensive experiments on indoor and outdoor benchmarks including 7Scenes, Cambridge Landmarks, FAST-LIVO2 datasets, and a custom dataset demonstrate state-of-the-art performance, consistently outperforming in both accuracy and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10756v1">Embodied Multimodal Grounding for Open-Vocabulary Mobile Manipulation via Semantic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 9 pages, 11 figures. Accepted to ACM Multimedia 2026 (MM '26)
    </div>
    <details class="paper-abstract">
      Embodied mobile manipulation requires language, visual observations, three-dimensional scene structure, and action feasibility to be aligned before execution. We study open-vocabulary target grounding with few-shot manipulation in local household workspaces and present an embodied multimodal grounding framework that integrates active multi-view Semantic 3D Gaussian Splatting (Semantic-3DGS), reachability-aware base positioning, and a diffusion-based vision-language-action policy. A task-driven local Semantic-3DGS serves as a shared interface across active sensing, language-conditioned 3D localization, obstacle-aware scene reasoning, base preparation, and semantic conditioning of the action model. To preserve pretrained action priors, the 3D semantic cues are injected only into the late action-expert blocks. In expanded 50-trial real-robot evaluations against representative vision-language-action (VLA) approaches, the full system achieves 60% long-horizon success compared with 40% for PointVLA and 28% for DexVLA, and reaches 74% success in heavily cluttered manipulation compared with 52% for the single-view variant and 46% for PointVLA. It also maintains 75% success under a 75 cm height shift and eliminates photo-induced false grasps. These results indicate that explicit, refreshable 3D semantic grounding can improve robustness under clutter, occlusion, viewpoint variation, and embodiment constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10712v1">Compact Feed-Forward 3D Gaussians via Saliency-Guided Primitive Merging</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      3D scene reconstruction, modeling, and rendering are highly relevant for numerous tasks, and 3D Gaussian splatting has become a standard choice in this context. Its feed-forward variants provide fast reconstruction from sparse input views but often produce per-pixel primitives, leading to highly redundant and thus inefficient representations. We present a structure-aware merging pipeline that takes per-pixel primitives from any feed-forward method and consolidates them into a compact, content-adaptive Gaussian set while largely retaining visual quality at just $\frac{1}{20}^\text{th}$ of the Gaussians of a per-pixel method. We group spatially coherent Gaussians of similar appearance into variable-size clusters via adaptive superpixel segmentation guided by a saliency map, which allocates fine segments to textured regions and coarse segments to homogeneous areas. We compress each cluster into a compact latent representation through a learned encoder, then match and consolidate representations across views based on geometric overlap and feature similarity via a learned merger. A level-of-detail decoder then produces the final Gaussians at a controllable resolution, enabling a flexible quality-efficiency trade-off at inference. As a post-processing module, the pipeline is backbone-agnostic, leveraging the strengths of existing feed-forward methods. This leads to better and more robust quality than achieved by previous approaches that target a reduction in primitive count, while providing a highly compact representation, that can be rendered efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10682v1">Visual Geometry Foundation-Aware Gaussians for Single-Frame Surround-View Driving Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Single-frame surround-view reconstruction faces severe geometric instability and rendering artifacts due to minimal inter-camera overlap. While existing methods rely on complex decoders or auxiliary cues, they remain bottlenecked by the weak geometric capacity of upstream features. We argue that leveraging pretrained visual geometry priors strengthens upstream representations and alleviates the geometric ambiguity in sparse surround views. To this end, we propose VGGD, a visual geometry foundation-aware 3D Gaussian Splatting framework for feed-forward surround-view driving reconstruction, which shifts geometric modeling to the frontend and adapts foundation priors to the driving camera setting. First, VGGD leverages VGGT to provide transferable multi-view geometric prior tokens. Next, we introduce a Dual-Path Neck to decouple geometry-consistent and appearance-aware representations, improving appearance completion in weakly observed regions. We further apply Scale Warmup to stabilize early geometry learning and suppress scale drift under ego-pose changes. Finally, we use a hybrid pixel--volume Gaussian decoder to produce a renderable 3D Gaussian scene for novel-view synthesis. Experiments on the nuScenes single-frame benchmark show that VGGD achieves the best overall rendering quality among the compared methods and improves relative geometric consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10602v1">Gaussian Sculpting: End-to-End Controllable Surface Reconstruction via Field Optimization</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently enabled real-time novel view synthesis with impressive quality. However, it struggles to recover accurate surfaces under limited viewpoints and due to the inherent irregularity of Gaussian primitives. The resulting geometric errors are notoriously difficult to correct manually. To address these issues, we propose Gaussian Sculpting, a fully differentiable end-to-end framework for high-quality surface reconstruction. Our key insight is to anchor Gaussians onto an evolving differentiable surface, allowing them to guide signed distance field (SDF) optimization instead of extracting the surface only during post-processing. To enable stable gradient isolation during joint optimization, we design a bi-level training strategy in which the outer loop optimizes the geometry represented by the SDF, while the inner loop updates the Gaussians with the geometry fixed. We further impose constraints on Gaussian parameters to ensure consistency with the underlying surface, thereby improving both geometric and appearance fidelity during optimization. In addition, we introduce a multi-resolution subdivision scheme based on octree-like partitioning to preserve fine details while reducing memory consumption. Experiments on object-level scenes demonstrate that our method effectively removes redundant surfaces, recovers missing structures caused by limited viewpoints, and achieves strong reconstruction quality even at relatively low resolutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10345v1">CasDeblurGS: Cascaded 2D-to-3D Multi-View Consistency for 3D Gaussian Splatting from Two Blurry Images</a></div>
    <div class="paper-meta">
      📅 2026-08-11
      | 💬 Accepted to the ECCV 2026 MUSTCV Workshop. Project page: https://haeyun-choi.github.io/Cascaded2D3D_page/
    </div>
    <details class="paper-abstract">
      Free-viewpoint 3D scene media is increasingly important for immersive applications, yet practical capture often suffers from severe view sparsity and motion blur. Although neural rendering has advanced sparse-view synthesis, existing blur-aware methods typically require substantial multi-view redundancy, accurate camera poses, or costly per-scene optimization. We address a stringent yet practical setting: reconstructing a coherent 3D scene from only two motion-blurred images with known intrinsics, without input-view poses, auxiliary sharp images, or per-scene test-time optimization. To this end, we propose CasDeblurGS, a cascaded framework that progressively recovers reliable cross-view information from local 2D correspondences to global 3D guidance. Stage 1 constructs locally reliable guidance through occlusion-aware correspondence filtering, while Stage 2 aggregates the intermediate restorations into a provisional pose-free 3D Gaussian representation whose input-view re-renders provide dense global guidance for final restoration. The resulting views enable a more coherent 3D representation and higher-quality novel-view synthesis. Experiments on real-world and synthetic Deblur-NeRF scenes show consistent gains over strong baselines, improving PSNR by 1.19 dB and 2.11 dB, respectively. Progressive ablations, cross-view correspondence visualization, and camera reprojection analysis further demonstrate improvements in both rendering quality and multi-view geometric consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10286v1">TRACE-GS: On-Policy Trajectory Distillation with Privileged Geometric Conditioning for Sparse-View 3DGS Restoration</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      We present TRACE-GS, an on-policy trajectory distillation framework that leverages privileged geometric conditioning at training time, thereby adapting a diffusion prior to sparse-view 3D Gaussian Splatting (3DGS) restoration. Rather than pursuing increasingly sophisticated restoration architectures, we identify a more fundamental limitation shared by existing diffusion-based approaches: supervision at independently noised states does not cover those reached during inference. In sparse-view 3DGS, under-constrained geometry biases denoising from the outset, and the resulting deviations compound along the rollout. TRACE-GS instead performs on-policy trajectory distillation: a teacher conditioned on richer geometry from additional training views supplies targets along the sparse-view student's own rollout, aligning denoising directions and cross-view responses at each visited state. This training-only geometry places TRACE-GS in the learning using privileged information (LUPI) setting. At deployment, only the sparse-view student is retained, and its restored renderings serve as pseudo-observations for 3DGS refinement. To the best of our knowledge, TRACE-GS is the first to derive on-policy supervision from privileged geometry for sparse-view 3DGS restoration, achieving consistent gains and strong generalization across datasets and sparse-view settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10057v1">LEGO: Leveled Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 Accepted to ECCV 2026. Project page: https://pz0826.github.io/LEGO-Webpage/
    </div>
    <details class="paper-abstract">
      We introduce LEGO for advanced open-vocabulary scene understanding. Beyond basic concept recognition, its core innovation lies in capturing the intrinsic semantic hierarchies within the scene, such as the "flowerpot -> bouquet -> bud -> petal" lineage. While foundation models like SAM can identify multi-granular structures in 2D, their partitions are strictly perspective-bound and lack cross-view consensus. LEGO self-adaptively re-grades volatile multi-view SAM granularities into a unified, 3D-consistent hierarchy. This provides precise supervision for the structurally coherent, multi-level segmentation of 3D scenes. By grounding these segments with CLIP embeddings, LEGO recovers open-vocabulary semantic logic across hierarchical levels. Furthermore, by incorporating spatial relationships, we elevate these segments into level-wise language scene graphs, effectively empowering Large Language Models to perform complex, context-aware spatial reasoning and precise visual grounding. Experimental results demonstrate that LEGO establishes new state-of-the-art performance across both promptable and open-vocabulary 3D segmentation benchmarks, exhibiting advanced hierarchical scene decomposition and context-aware spatial reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01958v2">FAST-GS: Frequency Aware Space-time Gaussian Splatting for Photorealistic Dynamic Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 accepted by ICASSP2026
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) excels in dynamic 3D reconstruction and real-time novel view synthesis via efficient 4D Gaussian representations and parallelizable rendering. However, existing 4DGS approaches rely on a single polynomial to model motion, which limits performance in complex dynamic scenes where high-frequency motion components are prevalent, and fails to ensure long-term stability due to cumulative trajectory drift. To address these issues, we propose a Fourier Motion Modeling module: this paradigm decomposes motion into frequency-based sinusoidal components, capturing both low-frequency global trajectories and high-frequency local details to model complex motion patterns accurately. It retains the real-time rendering capability of 4DGS while improving complex motion fitting and long-term coherence. Additionally, we integrate a motion-aware regularization strategy into the loss function: it uses frequency-dependent weights to suppress high-frequency jitter while preserving low-frequency motion coherence. Extensive experiments on N3V and Google Immersive datasets from multiple scenarios demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06343v3">Uncertainty-Aware 4D Gaussian Splatting for Monocular Occluded Human Rendering</a></div>
    <div class="paper-meta">
      📅 2026-08-10
    </div>
    <details class="paper-abstract">
      High-fidelity rendering of dynamic humans from monocular videos typically degrades catastrophically under occlusions. Existing solutions incorporate external priors-either hallucinating missing content via generative models, which induces severe temporal flickering, or imposing rigid geometric heuristics that fail to capture diverse appearances. To this end, we reformulate the task as a Maximum A Posteriori estimation problem under heteroscedastic observation noise. In this paper, we propose U-4DGS, a framework integrating a Probabilistic Deformation Network and a Joint Rasterization pipeline. This architecture renders pixel-aligned uncertainty maps that act as an adaptive gradient modulator, automatically attenuating artifacts from unreliable observations. Furthermore, to prevent geometric drift in regions lacking reliable visual cues, we enforce Confidence-Aware Regularizations, which leverage the learned uncertainty to selectively propagate spatial-temporal validity. Extensive experiments on the ZJU-MoCap and OcMotion datasets demonstrate that U-4DGS achieves state-of-the-art rendering fidelity and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.08949v1">EndoMD-SLAM: Endoscopic Gaussian Splatting SLAM under Optical Degradation with Memory and Static-Transient Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-08-09
      | 💬 Project page: https://endomd-slam.github.io/
    </div>
    <details class="paper-abstract">
      Dense 3D reconstruction is critical for clinical endoscopic navigation and documentation. While Gaussian Splatting SLAM systems show promise in this domain, they fundamentally rely on strict multi-view photometric consistency. In routine procedures, this assumption is severely violated by intermittent optical degradations like moving debris and water flushing. Standard systems erroneously fuse these cameraattached artifacts into the persistent 3D geometry, causing severe tracking drift and irreversible map corruption. To address this limitation, we propose EndoMD-SLAM, a framework designed to maintain stability under optical degradation through specialized tracking and mapping mechanisms. On the tracking side, a memory-driven gating mechanism detects unreliable observations to suspend map updates and utilizes historical keyframes for drift-aware relocalization. On the mapping side, a self-supervised static-transient decomposition isolates visual contaminants into a dedicated transient field. This explicit separation prevents artifacts from structurally entangling with the persistent anatomical map. We curate a degradationfocused benchmark from colonoscopy videos to systematically evaluate these failure modes. Extensive experiments show that while standard baselines fail under severe optical degradation, EndoMD-SLAM preserves geometric integrity, reducing absolute trajectory error by 91% and improving rendering fidelity by 9.9 dB PSNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.08659v1">JSGS: JPEG State-Guided Supervision for 3D Gaussian Splatting from Mixed-Quality Views</a></div>
    <div class="paper-meta">
      📅 2026-08-09
    </div>
    <details class="paper-abstract">
      Standard 3D Gaussian Splatting (3DGS) assumes that every input image faithfully samples scene radiance. However, mixed-quality JPEG images violate this assumption because compression-induced blocking and ringing artifacts can corrupt updates to Gaussians shared across views. To address this problem, we propose JPEG State-Guided Supervision for 3D Gaussian Splatting from Mixed-Quality Views (JSGS). JSGS uses luminance and chrominance quantization tables stored in each JPEG file to construct a view-specific JPEG observation operator. This operator encodes and decodes each rendered view for domain-matched comparison with the corresponding decoded input image. The luminance quantization table supplies continuous weights within a fixed middle frequency band. A loss in the low frequency band anchors coarse structure, while the weighted middle frequency loss redistributes supervision among the selected DCT coordinates. The resulting block disagreement also guides the Gaussian Controller to regularize small primitives with high opacity in disagreement regions. Across seven scenes and three mixed-quality schedules, JSGS achieves the lowest mean LPIPS and the highest mean SSIM under every schedule while rendering at approximately 150 FPS. Code: https://github.com/Jayden-Cui/JSGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.08585v1">EvTrajGS: Accurate and Efficient 3D Gaussian Splatting from Unposed Event Streams</a></div>
    <div class="paper-meta">
      📅 2026-08-09
    </div>
    <details class="paper-abstract">
      Event cameras, with high temporal resolution, high dynamic range, and asynchronous sensing characteristics, have shown great potential for dense 3D reconstruction. Traditional reconstruction methods based on off-the-shelf pose estimates achieve high efficiency but produce low-fidelity results, as inaccurate pose initialization introduces cumulative reconstruction errors. In contrast, recent SLAM-style methods stabilize joint pose-scene optimization through incremental tracking and mapping, yielding higher reconstruction fidelity at the expense of considerable computational overhead. To address this trade-off, this paper presents EvTrajGS, an accurate and efficient 3D Gaussian Splatting framework for unposed event streams. Our method enables reliable joint pose-scene optimization initialized from coarse pose priors, eliminating the need for computationally expensive SLAM-style pipelines. EvTrajGS parameterizes camera motion as a continuous-time trajectory initialized from discrete camera poses, providing a unified representation for pose refinement. We then aggregate adjacent trajectory states into a temporally coupled pose, promoting temporally consistent pose updates during joint optimization. Additionally, we introduce a loss-reweighted event sampling strategy to adaptively emphasize temporally under-reconstructed intervals. Extensive experiments on both synthetic and real-world datasets demonstrate that EvTrajGS outperforms state-of-the-art methods in terms of both geometric reconstruction quality and pose estimation accuracy, achieving 3.8 dB higher PSNR, 0.1 higher SSIM, and over 40\% lower ATE RMSE while retaining high computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.08531v1">ERF-GS: Reconstructing Fast Motion from Disjoint Event-RGB Viewpoints</a></div>
    <div class="paper-meta">
      📅 2026-08-09
      | 💬 18 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Deep learning-driven representations such as neural radiance fields (NeRFs) and 3D Gaussian splatting (3DGS) have revolutionized the field of dynamic 3D scene reconstruction with improved visual precision and scalability. However, the reconstruction of fast-moving objects remains a challenge; existing methods based on conventional frame-based videos often struggle in scenarios such as sports events and animal videography. We propose an event-RGB fusion Gaussian splatting (ERF-GS) framework that integrates event information into both optimization and densification stages of the Gaussian splatting pipeline, taking advantage of novel event sensors with high frame-rate. Unlike many other event-assisted scene reconstruction methods, ERF-GS was developed using realistic simulation settings and realizes event-based learning detached from RGB inputs. This design enables its application beyond straightforward synthetic data into the realm of natural video with complex layout, low frame rates and severe motion blur. Our experiments show that ERF-GS outperforms both the 4DGS baseline and the concurrent E-D3DGS on different variants of the Neu3D and Nvidia datasets which include blurry RGB frames and disjoint RGB-event viewpoints. Our code is available at https://github.com/andrewbxy/ERF-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.07937v1">FlexSplat: Flexible Feed-Forward 3D Gaussian Splatting without Point Cloud Correspondence</a></div>
    <div class="paper-meta">
      📅 2026-08-08
      | 💬 Accepted to the British Machine Vision Conference (BMVC) 2026. Code: https://github.com/amir-sbg/FlexSplat
    </div>
    <details class="paper-abstract">
      We present FlexSplat, a feed-forward framework for novel view synthesis (NVS) from uncalibrated, object-centric multi-view image collections. A recent line of query-based methods reconstructs a compact set of 3D Gaussians by treating them as transformer queries that are refined with multi-view deformable attention; these methods, however, assume that camera poses are given. FlexSplat removes this assumption: a geometry transformer is trained jointly with the Gaussian decoder to predict per-image camera parameters and depth, which in turn ground a depth-guided Gaussian parameterization and a multi-view deformable cross-attention that aggregates evidence across all input views into a single, view-consistent set of primitives. An uncertainty-weighted depth-consistency objective lets the jointly trained geometry adapt to the reconstruction task, while the cross-view consensus formed during decoding absorbs the residual error of the estimated cameras and depth. The representation uses a compact Gaussian budget that is decoupled from the input resolution - unlike pixel-aligned methods, the primitive count does not grow with the image grid - and is not dictated by the number of views. On ShapeNet-SRN and Google Scanned Objects (GSO), FlexSplat matches or approaches posed state-of-the-art reconstructors while requiring neither camera poses nor ground-truth depth, and matches the best perceptual (LPIPS) quality among the compared methods on GSO. Our results indicate that a jointly trained geometry front-end is sufficient to bring calibration-free operation to query-based Gaussian reconstruction while staying within 0.7 dB PSNR of posed methods and matching their perceptual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04224v2">SplitGaussian: Reconstructing Dynamic Scenes via Visual Geometry Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-08-08
      | 💬 version 2. Accepted By ACM MM 2026
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular video remains fundamentally challenging due to the need to jointly infer motion, structure, and appearance from limited observations. Existing dynamic scene reconstruction methods based on Gaussian Splatting often entangle static and dynamic elements in a shared representation, leading to motion leakage, geometric distortions, and temporal flickering. We identify that the root cause lies in the coupled modeling of geometry and appearance across time, which hampers both stability and interpretability. To address this, we propose SplitGaussian, a novel framework that explicitly decomposes scene representations into static and dynamic components. By decoupling motion modeling from background geometry and allowing only the dynamic branch to deform over time, our method prevents motion artifacts in static regions while supporting view- and time-dependent appearance refinement. This disentangled design not only enhances temporal consistency and reconstruction fidelity but also accelerates convergence. Extensive experiments demonstrate that our approach outperforms prior state-of-the-art methods in rendering quality, geometric stability, and motion separation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.07760v1">XClipGS: Exact Half-Space Clipping for Medical Volume Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-07
    </div>
    <details class="paper-abstract">
      Gaussian-splatting proxies enable interactive rendering of volumetric medical scans, but a clipping plane exposes anatomy not constrained by external-view training and intersects primitives that conventional splatting can only keep or drop whole. We present XClipGS (eXact Clipping), which treats these as two separate problems: the render-time clip operator and supervision of the hidden interior. Under the local affine model used by EWA splatting, the ray integral of a half-space-restricted Gaussian factorizes exactly into its ordinary 2D footprint and a conditional Gaussian CDF whose argument is affine in pixel coordinates. The resulting closed-form per-pixel operator introduces no learned clipping parameters or auxiliary network and remains differentiable with respect to the primitive and plane. We use multi-distance reference views with varied clipping-plane axes and offsets to supervise the interior through the same operator. We also introduce a paired clipped/unclipped cut-face protocol with difference-referenced cut error (CDE) and culled-side leakage (Leak), because global image metrics dilute errors near the plane. On eight CT and MRI volumes with plane offsets not used for training, XClipGS attains the highest PSNR on every volume (33.56 versus 32.34 dB for ClipGS) while rendering at over 650 FPS, far above real time, versus 278 FPS. On voxel-axis cut-face views, it raises average band SSIM from 0.809 to 0.860 and leaks roughly 40 times less. Without retraining, it also achieves the best average across all four metrics on arbitrary-normal planes; on a fixed interior, it matches RaRa's face fidelity with about 16 times less leakage. Project page: https://gaozhongpai.github.io/XClipGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.07144v1">InstanceSplat: Instance-Aware Feed-Forward 3D Gaussian Splatting for Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2026-08-07
      | 💬 Project page: https://jamchaos.github.io/InsSplat/
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) enables efficient and generalizable 3D reconstruction, but current feed-forward 3DGS methods for scene understanding remain largely category-oriented. In contrast, instance-aware 3DGS methods typically rely on per-scene optimization and often decouple reconstruction from instance and semantic learning, limiting reciprocal interactions among them. We present InstanceSplat, a unified feed-forward 3DGS framework for generalizable 3D reconstruction and instance-aware scene understanding from pose-free multi-view images. In a single forward pass, InstanceSplat constructs an instance-aware Gaussian representation that jointly encodes appearance, geometry, instance identity, and language-aligned semantics. Shared 3D Gaussians ground instance identities across views, producing renderable and cross-view-consistent instance features. To allow reconstruction and scene understanding to benefit from each other, we further design an instance-centric learning strategy that connects reconstruction, instance learning, and semantic learning through shared instance structure. Specifically, instance cues guide reconstruction, language-aligned semantics strengthen the discrimination of confusing same-category instances, and instance regions aggregate semantic evidence into coherent object-level predictions. Experiments on novel-view synthesis, instance segmentation, and open-vocabulary semantic understanding under varying input-view settings and on an unseen dataset demonstrate state-of-the-art performance, practical efficiency, and strong generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06801v1">AdvTiles: Physical Adversarial Camouflage Clothing against Person Detectors via Learnable Tiles</a></div>
    <div class="paper-meta">
      📅 2026-08-07
    </div>
    <details class="paper-abstract">
      Physical adversarial attacks against person detectors have evolved from localized patches to full-body textures. However, achieving both visual naturalness and strong attack effectiveness remains challenging. Existing natural-looking methods typically optimize camouflage textures as a whole, limiting the flexibility to refine local adversarial patterns and their spatial arrangement. To address this issue, we propose AdvTiles, a physical adversarial camouflage framework built from learnable tiles, enabling strong attack performance while preserving a natural camouflage appearance. Specifically, we use a Straight-through (ST) Gumbel-Softmax estimator for differentiable tile selection, enabling joint optimization of tile patterns and spatial layouts. This design provides fine-grained control over adversarial texture generation. To improve robustness in diverse physical conditions, we further optimize the camouflage through differentiable 3D Gaussian Splatting rendering with variations in viewpoints, scales, illuminations and backgrounds. Extensive experiments across multiple detectors demonstrate that AdvTiles achieves an average ASR of 86.2%, outperforming existing state-of-the-art attack methods. We further fabricate the optimized camouflage into wearable adversarial clothing, validating its effectiveness in real-world scenarios across diverse distances, angles and backgrounds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11183v2">Versatile Video Representation via Feed-Forward 2D Gaussian Splatting Tokenization</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted by ACM MM 2026, Rio de Janeiro, Brazil
    </div>
    <details class="paper-abstract">
      Recent video representation methods that rely on fixed-grid, patch-wise tokenization often exhibit limited versatility.Spatially, uniformly allocating a fixed number of tokens often leads to over-encoding in low-information regions. Temporally, reducing redundancy remains challenging without explicitly distinguishing between static and dynamic content. In this work, we introduce the Gaussian Video Transformer (GVT), a versatile video representation framework built on a feed-forward 2D Gaussian Splatting (2DGS) tokenization scheme. We first extract latent rigid features from a video clip and represent them with a set of 2D Gaussians generated by our proposed Spatio-Temporal Gaussian Embedding (STGE) mechanism in a feed-forward manner. Such 2D Gaussians not only enhance spatial adaptability by assigning higher (resp., lower) rendering weights to regions with higher (resp., lower) information content during rasterization, but also improve generalization by avoiding per-video optimization. To enhance the temporal versatility, we introduce a Gaussian Set Partitioning (GSP) strategy that separates the 2D Gaussians into static and dynamic sets, which explicitly model static content shared across different time-steps and dynamic content specific to each time-step, enabling a compact representation. We evaluate GVT across four tasks: video reconstruction, video action recognition, video compression, and video generation, on the UCF101, Kinetics, and DAVIS datasets. The results demonstrate state-of-the-art reconstruction and compression performance, improved action recognition, and video generation performance comparable to the baseline MAGVIT-v2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06117v1">Confidence matters: Leveraging Multi-view Geometric Priors for GS-based Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has emerged as a widely-used tool for novel view synthesis, offering real-time rendering in a sparse representation. However, the method's reliance on structure-from-motion initialization and photometric optimization can lead to suboptimal geometric reconstruction, particularly for objects with high specularity. In this work, we investigate the integration of geometric priors, in the form of predicted normal and depth maps, into the 3DGS framework to improve the reconstruction quality. We analyze the effect of incorporating these priors into GS-based methods and our evaluation reveals that multi-view predictions, as they are done by the recent visual geometry grounded transformer (VGGT), outperform single-view alternatives. A major factor is the existence of a confidence map for the estimations, which comes as a by-product of multi-view models and which can significantly improve the effectiveness of priors by weighting each prediction appropriately. Extensive experiments on standard benchmarks show consistent improvement in reconstruction quality and significant gains in complex scenes including specular objects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05896v1">GSBF: Gaussian Splatting for Environment-Aware Beamforming</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Beamforming plays a key role in multiple-input-multiple-output (MIMO) communication systems. However, conventional beamforming design normally requires accurate instantaneous channel state information (CSI) and iterative optimization, which incur substantial pilot overhead and computational complexity. Recognizing that radio propagation is intrinsically governed by the physical geometry, we develop a 3D Gaussian splatting for environment-aware beamforming (GSBF) pipeline based on multi-modal data, which characterizes the environment through a persistent 3D Gaussian representation. Specifically, GSBF models the environmental scattering response with reciprocity-preserving bidirectional spherical Gaussian (Bi-SG) kernels and performs two-sided electromagnetic rasterization to render an angular propagator map. The rendered map is then aggregated through an over-complete array-manifold dictionary and projected to the constant-modulus beamformers, thereby synthesizing beams directly from the access point (AP) pose and user position without online instantaneous CSI. Simulations demonstrate that GSBF consistently outperforms baselines such as exhaustive beam alignment (EBA) with lower latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16278v3">RealityBridge: Bridging Editable 3D Gaussian Splatting Driving Simulations and Real-World Videos</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      Long-tail hazardous scenarios are essential for safety-oriented autonomous driving, yet they are difficult to collect at scale. Editable 3D Gaussian Splatting (3DGS) simulation offers a scalable alternative through real-scene reconstruction and controllable editing. However, edited 3DGS-rendered videos often exhibit a significant Sim-to-Real gap, manifested as rendering artifacts, degraded foreground assets, illumination mismatch, and temporal flickering. Addressing these coupled defects requires jointly restoring local appearance, harmonizing edited content, and maintaining temporal consistency, whereas existing methods typically address only a subset of these requirements. To fill this gap, we propose RealityBridge, a video restoration and harmonization framework that converts edited 3DGS renderings into realistic driving footage while preserving simulator-defined structure, edits, and dynamics. RealityBridge conditions a video foundation model on complementary modality signals, with a lightweight GateNet adaptively controlling their injection across backbone blocks. We further develop a task-oriented curation pipeline to construct training data, and design a four-stage supervised training strategy followed by reward-guided post-training. Extensive experiments demonstrate that RealityBridge outperforms existing methods in restoration and harmonization while preserving strong temporal consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05704v1">G$^2$ARD-GS: Geometry-Guided Anchor-Regularized Gaussian Splatting Distillation</a></div>
    <div class="paper-meta">
      📅 2026-08-06
    </div>
    <details class="paper-abstract">
      Dense colored LiDAR maps provide accurate city-scale geometry, but lifting them into 3D Gaussian Splatting (3DGS) retains millions of primitives, making the resulting models costly to store, transmit, render, and adapt. Aggressive primitive reduction alleviates this burden, but can remove the local surface support needed for stable novel-view synthesis and downstream geometric use. We introduce G$^2$ARD-GS, a geometry-guided distillation method that converts a dense Gaussian prior instantiated either as a training-free point-cloud lift or a trained GS model into a compact, reusable representation. G$^2$ARD-GS progressively consolidates the prior into surface-aware representatives, then recovers appearance on the resulting fixed topology under construction-time anchor constraints, with no primitives added or removed during recovery. Under limited supervision, geometry-aware view selection allocates the available view budget. On MatrixCity, G$^2$ARD-GS achieves the best PSNR, SSIM, and LPIPS across matched $5\times$--$30\times$ compression budgets, outperforming PUP by $3.2$--$6.8$,dB in PSNR. When reused as frozen geometry, the compact model improves off-trajectory appearance adaptation by $3.7$--$4.9$,dB over PUP 3D-GS and preserves image-to-model registration accuracy on Cambridge KingsCollege at $30\times$ compression. Project page: https://patrick1159.github.io/gardGS-page/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05564v1">ESVR: 3D Ellipsoid-based Sparse Volume Rendering via Structure-aware Primitive Learning and Per-primitive Ray Sampling</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 IEEE VIS 2026 accepted
    </div>
    <details class="paper-abstract">
      Efficient representation and rendering of large-scale sparse volumetric data remain challenging in scientific visualization, as meaningful structures often occupy only a small fraction of the spatial domain. While direct volume rendering (DVR) provides high-quality visualization, its computational and memory costs scale poorly with data size. Recent advances in 3D Gaussian Splatting (3DGS) address this challenge by representing volumetric scenes with compact geometric primitives, enabling efficient, high-fidelity rendering. However, existing 3DGS-based methods learn from DVR rendered images rather than raw volumes, leading to information loss and limiting flexible transfer function control for interactive exploration. To address these limitations, we propose ESVR, an ellipsoid-based sparse volume rendering framework that directly learns and renders volumetric data in 3D space. Our method combines differentiable ellipsoidal primitives with bounded support, structure-aware primitive learning with complementary pruning, and a per-primitive ray sampling strategy for fast and accurate transfer function mapping. To support large-scale datasets, we further introduce a chunk-based optimization scheme with ghost ellipsoids, providing boundary context during training. Across large sparse datasets, ESVR achieves up to four orders of magnitude compression and real-time rendering at 43-223 FPS while maintaining competitive reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02006v2">ASTRA: Asynchronous Spatio-Temporal Reconstruction via Trajectory Alignment</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 We wish to withdraw this preprint because the current statistical analysis of the experimental data is incomplete and requires re-verification. We plan to submit a revised and thoroughly checked version in the near future
    </div>
    <details class="paper-abstract">
      Dynamic 3D scene reconstruction has achieved remarkable success under the assumption of strictly synchronized multi-camera inputs. However, in real-world scenarios, temporal asynchrony among capturing devices remains a critical challenge, leading to severe motion blur and geometric artifacts. Existing asynchronous reconstruction methods typically estimate temporal offsets through photometric supervision, but appearance matching provides weak temporal cues under large offsets and complex motions. We attribute this limitation to two major bottlenecks: texture-induced collapse, where low-texture regions provide nearly vanishing alignment signals, and deformation-induced coupling, where temporal errors are absorbed into distorted geometry or motion rather than being explicitly corrected. To address these issues, we propose ASTRA (Asynchronous Spatio-Temporal Reconstruction via Trajectory Alignment), a framework that introduces 2D motion trajectories as explicit, texture-agnostic supervision for asynchronous dynamic reconstruction. Instead of synchronizing cameras solely through rendered color residuals, ASTRA jointly optimizes temporal offsets and dynamic 3D representations by aligning the projected motion of reconstructed 3D points with observed 2D trajectories, while using dynamic and certainty masking to suppress unreliable trajectory constraints. Extensive experiments on different dynamic Gaussian Splatting backbones show that ASTRA preserves high-frequency spatial details and sustains strong robustness even under severe asynchrony with up to 25-frame offsets, achieving approximately 1.4 dB PSNR improvement, reducing temporal-offset MAE by 54.0\%, and nearly quadrupling the synchronization success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02304v3">TRACE: Ergodic Trajectory Optimization for Active Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 11 pages, 7 figures, fixed a template bug in the Latex
    </div>
    <details class="paper-abstract">
      Existing active reconstruction systems with Gaussian-splatting maps select observations greedily, optimizing a single next-best-view (NBV) at each step and connecting the chosen views by short-horizon path planning. This greedy decoupling disregards the global structure of scene information, producing inefficient trajectories that waste sensing capacity in transit between selected views. In this work, we study active reconstruction as an ergodic coverage problem: the time-averaged spatial statistics of the sensor trajectory should match a target information distribution induced by the current map. Our approach derives this target distribution online from uncertainty and visibility, and calculates ergodic trajectories via a kernel-ergodic horizon planner with gradient flow and footprint depletion, closing the loop between mapping and trajectory optimization. We thoroughly evaluate TRACE on the Replica dataset against the Next-Best-View (NBV) baselines, improving PSNR by 1.5 dB. Code: https://github.com/spikelab-jhu/trace-active-reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05145v2">Objects as Audio-Visual Modal Sound Fields</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 ECCV 2026, Project page: https://zisenshao.github.io/AV-MSF/
    </div>
    <details class="paper-abstract">
      While modern 3D reconstruction excels at modeling object geometry and appearance, it largely ignores the rich acoustic cues revealed through physical interaction. Object impact sounds convey material, stiffness, and structural properties that complement vision, yet existing impact sound modeling approaches either rely on expensive physics-based simulation or require large datasets to generalize in a purely data-driven manner. We introduce Audio-Visual Modal Sound Field (AV-MSF), a novel object-level acoustic representation reconstructed from multi-view images and only a few impact sound recordings. AV-MSF builds on 3D Gaussian Splatting integrated with dense 3D visual feature to provide a strong geometry-aware prior, and represents the impact sound field using compact, physically meaningful modal parameters, enabling robust few-shot reconstruction. Experiments on two real-world datasets show that AV-MSF achieves state-of-the-art impact sound rendering, outperforming both physics-based and data-driven baselines. Furthermore, we demonstrate downstream applications enabled by our representation, including contact localization and object sound editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05482v1">CDSeg: A Renderable Gaussian Carrier for Image-to-3D Label Transfer</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Modern image models provide strong cues about \emph{what} should be segmented in each view, but their masks do not by themselves determine \emph{where} those labels should persist in 3D. We present Cross-Domain Segmentation via Gaussian Splatting (CDSeg), a label-transfer interface that requires no task-specific 3D segmentation training and uses Gaussian primitives as a renderable label carrier. An external mask source supplies the labels, while renderer-derived visibility determines which 3D primitives receive them. The carrier is instantiated either by completing each input point into one Gaussian, preserving its index, or by reusing the native primitives of an optimized Gaussian scene. CDSeg records pixel--primitive associations during rendering and fuses multi-view masks through voting and a local filter. The resulting labels can be returned to the original points, retained on the native Gaussian scene, or rendered into other views. CDSeg covers promptable, automatic instance, semantic, and LiDAR settings and processes scenes with millions of primitives in seconds. It obtains 92.35\% mIoU on DesktopObjects-360, 95.89\% on NeRDS-360, and 65.77\% on the full ScanNet-v2 validation split using the provided 2D semantic annotations. CDSeg thereby provides one interface for reusing 2D masks across point clouds, Gaussian scenes, and image views without a task-specific 3D segmentation network.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04842v1">RORA: Realistic Object Reconstruction with Articulation</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible
    </div>
    <details class="paper-abstract">
      Replicating real-world environments into simulation by realistic visual representation like NeRF and 3D Gaussian Splatting (3DGS) has emerged as an effective strategy to reduce the sim-to-real gap in robot learning. However, implementing object articulation during the real-to-sim process is still a challenging task. Existing motion tracking or learning based articulation methods shows low success rates on complex kinematic structures having multiple joints. Furthermore, those methods require scan of dynamic motion of objects, which makes reconstruction process much complicated. In this work, we propose the first end-to-end pipeline that reconstructs simulation-ready assets with accurate articulation from a single static object video input through suggestion based human-in-the-loop process. Our approach exports a hybrid representation combining 3DGS for photorealistic rendering and mesh-based geometry for physical interaction. In the reconstruction process, our pipeline performs convex decomposition followed by user grouping for intuitive part segmentation, subsequently binding 3D Gaussians to the corresponding mesh parts. An Automatic Joint Suggestion Algorithm then calculates candidate joint axes from local boundary geometries and presents them to users for efficient articulated asset reconstruction. We have shown that our method achieves precise articulation results on partnet-mobility-v0 dataset and real objects. Additionally we presented a potential usage of our framework on robot learning, deploying the reconstructed assets in Unreal Engine and NVIDIA Isaac Sim, demonstrating real-time dexterous hand manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25569v2">CORF-GS: Real-Time Wireless Radiance Field Reconstruction via Coupled Optical-RF Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 A collection of paper on 3DGS for Wireless Communications can be found at https://github.com/AI4Wireless/3DGS4Wireless
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS)-based wireless radiance field (WRF) reconstruction provide an efficient solution for wireless channel modeling. However, existing WRF reconstruction methods rely on pre-collected observations and offline optimization, and thus struggle to provide real-time channel knowledge. To bridge this gap, we propose CORF-GS, a real-time WRF reconstruction framework that processes sequential optical and radio frequency (RF) keyframes. Specifically, CORF-GS constructs a unified Gaussian representation for optical and RF with shared geometry and modality-specific appearance, allowing high-resolution optical images to provide structural priors for WRF reconstruction. When a new keyframe arrives, CORF-GS first employs optical-guided Gaussian sampling to densify the WRF in under-represented regions. Since light and radio waves may respond differently to the same object surfaces due to wavelength mismatch, relying solely on optical guidance may neglect RF-informative areas. Therefore, CORF-GS performs coupled optical-RF optimization to jointly refine the shared Gaussians. Compared with the existing two-stage training pipelines, this prevents WRF from passively adapting to a frozen optical geometry and encourages the shared Gaussians to adapt to both optical structures and RF power distributions. Simulations show that CORF-GS achieves state-of-the-art RF spectrum synthesis quality and reduces the reconstruction time by $6.4\times$ compared with existing WRF methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04701v1">UniWorld-View: Large-Baseline View Synthesis via Video Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Project Homepage: https://zhouhyocean.github.io/uniworld-view/ Code: https://github.com/PKU-YuanGroup/UniWorld-View
    </div>
    <details class="paper-abstract">
      The abundance of casually captured monocular videos and images on social media provides a valuable source for immersive content creation, where generating novel views from such sparse observations can greatly enhance user experiences. However, producing photorealistic and geometrically consistent views with precise camera control remains challenging when input coverage is extremely limited. Reconstruction-based approaches such as NeRF and 3D Gaussian Splatting (3DGS) deteriorate severely under sparse inputs and fail to explicitly handle occlusions. Generative methods ease data requirements but still struggle with large-baseline view synthesis due to inaccurate or implicit geometric guidance. To overcome these limitations, we introduce UniWorld-View, a unified framework for controllable large-baseline novel view synthesis from monocular inputs. UniWorld-View integrates explicit 3D guidance with generative diffusion modeling to enable precise camera control and geometrically consistent view generation. The geometric guidance is obtained through an occlusion-aware point cloud rendering strategy that resolves visibility ambiguities and provides accurate priors for diffusion-based synthesis. By coupling this rendering strategy with powerful video diffusion backbones, UniWorld-View achieves high-fidelity novel view generation even under extreme camera motions and wide-baseline changes, and can further provide multi-view videos for downstream dynamic 3DGS reconstruction. Experiments on the WorldScore benchmark and zero-shot NVS benchmarks demonstrate the effectiveness of UniWorld-View in controllability, geometric consistency, and visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.05218v1">PD-GS: Phoneme-Driven 3DGS for Audio-Driven Talking Heads</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Accepted to ACM MM 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables fast, photorealistic talking-head rendering, yet accurate lip articulation remains elusive: mouth motion is often over-smoothed and may violate hard articulatory constraints such as bilabial closures, producing the notorious ``leaky mouth'' artifact. A key difficulty is that brief, discrete articulatory events are inferred from a continuous acoustic embedding under a regression objective, which biases predictions toward averaged mouth configurations. While modern self-supervised speech encoders provide rich prosodic and phonetic cues, they do not provide an explicit, frame-aligned linguistic target that reliably disambiguates closure-level events. We propose \textbf{Phoneme-Driven Gaussian Splatting (PD-GS)}, which augments a 3DGS talker with time-aligned phoneme tokens obtained from an automatic ASR and forced-alignment pipeline. Our core component, the \textbf{Linguistic Fusion Module (LFM)}, adaptively fuses continuous audio context with discrete phoneme embeddings through a learned gate, allowing the model to preserve smooth audio-driven dynamics while strengthening phoneme guidance on articulation-critical segments. PD-GS is trained purely from monocular video using image reconstruction and lip landmark supervision. On HDTF, PD-GS achieves the best lip geometry among the compared baselines (LMD 2.66) and qualitatively reduces closure violations in challenging phoneme sequences, yielding more linguistically faithful neural avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04581v1">ACA-GS: Adaptive-Capacity Anchored Gaussian Splatting for Compact Dynamic Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 9 pages, 8 figures. Accepted to ACM Multimedia 2026
    </div>
    <details class="paper-abstract">
      Recent advances in 4D Gaussian Splatting (4DGS) enable high-fidelity, real-time spatiotemporal rendering, but expose a fundamental trade-off between motion expressiveness and storage efficiency. While anchor-based designs achieve compactness through anchor-level parameter sharing, their rigid uniform parametrization enforces fixed Neural Gaussian counts and feature budgets per anchor. Consequently, insufficient fidelity is addressed by excessive anchor density, rather than lightweight, targeted increases in Neural Gaussian count or feature capacity, resulting in memory waste. To overcome this rigidity, we introduce an adaptive-capacity anchor-based framework that dynamically allocates the representational capacity based on local spatiotemporal demands. Adaptive Anchor Cardinality varies the number of Neural Gaussians per anchor, concentrating primitives in regions of high geometric or motion complexity while suppressing redundancy. In parallel, Adaptive Anchor Feature Masking modulates anchor-level feature channels, assigning rich features to complex regions and lightweight representations to simpler ones. Experiments on MPEG, Panoptic Sports, and N3DV datasets demonstrate substantial storage reduction without degrading visual quality. Notably, on challenging MPEG sequences with complex motion, our method achieves up to 1.5x higher compression than state-of-the-art anchor-based methods while preserving comparable quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04560v1">OutLangSplat: 3D Language Gaussian Splatting for UAV Outdoor Scenes</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 9 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      3D Language Gaussian Splatting embeds open-vocabulary language features into 3D Gaussian Splatting, providing an efficient explicit representation for text-driven 3D scene understanding. However, existing methods are limited to indoor or small-scale scenes, and tend to fail in Unmanned Aerial Vehicle (UAV) outdoor scenes, where severe occlusions and long distance viewpoints often lead to incorrect semantic activations and missing target responses. In this paper, we present OutLangSplat which adapts language Gaussian representations to UAV outdoor scenes by improving feature representation and aggregation reliability. For the feature representation, a 2D-3D dual-branch representation with region-based alignment and fusion is designed to improve spatial consistency, reducing incomplete target responses and background misactivations. For the feature aggregation, we introduce a training-free contribution and consistency-aware Gaussian feature aggregation strategy that leverages pixel contribution reliability and cross-view semantic consistency to suppress unreliable responses from noisy viewpoints. A new dataset is provided by manually annotating various objects on four real-world public UAV outdoor scene datasets. To the best of our knowledge, it is the first accessible dataset of open-vocabulary 3D scene understanding for UAV outdoor scenes. Quantitative evaluations and ablation studies demonstrate that OutLangSplat outperforms SOTA methods on both open-vocabulary semantic segmentation and instance localization tasks. The datasets and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04475v1">Super-Gaussian: Interactive Scene Editing for 3D Gaussian Splatting and NLI-Based Volume Visualization in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 IEEE VIS 2026 accepted
    </div>
    <details class="paper-abstract">
      Despite the promise of virtual reality (VR) for intuitive spatial interaction, volume visualization (VolVis) in VR remains constrained by high rendering costs and motion discomfort. Recent advances have shown that representing volumetric scenes with 3D Gaussian splatting enables high-performance rendering, making this representation well-suited for VR. However, existing Gaussian-based scene editing workflows remain limited by slow offline segmentation and fatigue-inducing manual selection. To address these challenges, we present Super-Gaussian, a novel VolVis framework that enhances scene editing and interaction in VR through intuitive 3D Gaussian selection and natural language interaction (NLI). Our approach groups Gaussian primitives into higher-level units via feature-aware clustering, enabling efficient selection of complex volumetric regions, such as tumors in medical images or filaments in cosmological data, without point-by-point interaction. Building on this, we introduce a hierarchical select-and-refine workflow that combines random-walk-based region propagation, cluster selection, and point refinement, allowing users to progressively specify regions of interest with reduced effort. We further support on-the-fly text labeling of selected regions using NLI, allowing users to semantically query, interpret, and manipulate content within a visualization-perception-action loop. By integrating multimodal interaction, including speech, visual feedback, and spatial manipulation in VR, our framework supports intuitive exploration, editing, and scientific analysis of volumetric data. We demonstrate the effectiveness of Super-Gaussian through four case studies, quantitative selection benchmarks against existing Gaussian-based techniques, and system-level evaluations. Implementation details and experiments can be found on the project page: https://smin0136.github.io/super-gaussian-project/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.04004v3">Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2026-08-05
      | 💬 Accepted by IJRR
    </div>
    <details class="paper-abstract">
      This paper presents the first photo-realistic LiDAR-Inertial-Camera Gaussian Splatting SLAM system that simultaneously addresses visual quality, geometric accuracy, and real-time performance. The proposed method performs robust and accurate pose estimation within a continuous-time trajectory optimization framework, while incrementally reconstructing a 3D Gaussian map using camera and LiDAR data, all in real time. The resulting map enables high-quality, real-time novel view rendering of both RGB images and depth maps. To effectively address under-reconstruction in regions not covered by the LiDAR, we employ a lightweight zero-shot depth model that synergistically combines RGB appearance cues with sparse LiDAR measurements to generate dense depth maps. The depth completion enables reliable Gaussian initialization in LiDAR-blind areas, significantly improving system applicability for sparse LiDAR sensors. To enhance geometric accuracy, we use sparse but precise LiDAR depths to supervise Gaussian map optimization and accelerate it with carefully designed CUDA-accelerated strategies. Furthermore, we explore how the incrementally reconstructed Gaussian map can improve the robustness of odometry. By tightly incorporating photometric constraints from the Gaussian map into the continuous-time factor graph optimization, we demonstrate improved pose estimation under LiDAR degradation scenarios. We also showcase downstream applications via extending our elaborate system, including video frame interpolation and fast 3D mesh extraction. To support rigorous evaluation, we construct a dedicated LiDAR-Inertial-Camera dataset featuring ground-truth poses, depth maps, and extrapolated trajectories for assessing out-of-sequence novel view synthesis. Both the dataset and code will be made publicly available on project page https://xingxingzuo.github.io/gaussian_lic2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01659v2">StreamSplat: Streaming Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting enables efficient novel-view synthesis without per-scene optimization, but most existing methods assume a fixed set of context views and process them jointly. This limits their applicability to online scenarios where calibrated views arrive sequentially and the scene must be updated causally. We present \emph{StreamSplat}, a streaming feed-forward 3DGS framework that incrementally maintains a persistent geometry-grounded scene state and decodes it into renderable 3D Gaussians after each input chunk. StreamSplat centers on a \textbf{Voxel-Aligned Causal Cache (VACC)}, which stores historical 3D tokens in a memory-bounded voxel structure so that memory grows with explored scene geometry rather than stream length. To better reuse history during causal prediction, we introduce \textbf{History-Projected Depth Anchoring (HPDA)} to project cached geometry as depth guidance for current cost-volume estimation, and \textbf{Cache-Guided Feature Injection (CGFI)} to inject cached latent evidence into Gaussian-token regression. Experiments on DL3DV, RealEstate10K, and ScanNet show that StreamSplat remains competitive with state-of-the-art feed-forward 3DGS methods under sparse causal inputs, despite not using future views or full-scene context. More importantly, it scales to long input streams with 256, 512, and 1024 views where fixed-view baselines run out of memory, yielding sustained improvements in novel-view synthesis quality as more observations arrive. The code will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15169v4">MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent Diffusion</a></div>
    <div class="paper-meta">
      📅 2026-08-04
    </div>
    <details class="paper-abstract">
      Mesh models have become increasingly accessible for numerous cities; however, the lack of realistic textures restricts their application in virtual urban navigation and autonomous driving. To address this, this paper proposes MeSS (Meshbased Scene Synthesis) for generating high-quality, styleconsistent outdoor scenes with city mesh models serving as the geometric prior. While image and video diffusion models can leverage spatial layouts (such as depth maps or HD maps) as control conditions to generate street-level perspective views, they are not directly applicable to 3D scene generation. Video diffusion models excel at synthesizing consistent view sequences that depict scenes but often struggle to adhere to predefined camera paths or align accurately with rendered control videos. In contrast, image diffusion models, though unable to guarantee cross-view visual consistency, can produce more geometry-aligned results when combined with ControlNet. Building on this insight, our approach enhances image diffusion models by improving cross-view consistency. The pipeline comprises three key stages: first, we generate geometrically consistent sparse views using Cascaded Outpainting ControlNets; second, we propagate denser intermediate views via a component dubbed AGInpaint; and third, we globally eliminate visual inconsistencies (e.g., varying exposure) using the GCAlign module. Concurrently with generation, a 3D Gaussian Splatting (3DGS) scene is reconstructed by initializing Gaussian balls on the mesh surface. Our method outperforms existing approaches in both geometric alignment and generation quality. Once synthesized, the scene can be rendered in diverse styles through relighting and style transfer techniques. project page: https://albertchen98.github.io/mess/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25459v2">GS-Playground: A High-Throughput Photorealistic Simulator for Vision-Informed Robot Learning</a></div>
    <div class="paper-meta">
      📅 2026-08-04
      | 💬 Robotics: Science and Systems 2026
    </div>
    <details class="paper-abstract">
      Embodied AI research is undergoing a shift toward vision-centric perceptual paradigms. While massively parallel simulators have catalyzed breakthroughs in proprioception-based locomotion, their potential remains largely untapped for vision-informed tasks due to the prohibitive computational overhead of large-scale photorealistic rendering. Furthermore, the creation of simulation-ready 3D assets heavily relies on labor-intensive manual modeling, while the significant sim-to-real physical gap hinders the transfer of contact-rich manipulation policies. To address these bottlenecks, we propose GS-Playground, a multi-modal simulation framework designed to accelerate end-to-end perceptual learning. We develop a novel high-performance parallel physics engine, specifically designed to integrate with a batch 3D Gaussian Splatting (3DGS) rendering pipeline to ensure high-fidelity synchronization. Our system achieves a breakthrough throughput of 10^4 FPS at 640x480 resolution, significantly lowering the barrier for large-scale visual RL. Additionally, we introduce an automated Real2Sim workflow that reconstructs photorealistic, physically consistent, and memory-efficient environments, streamlining the generation of complex simulation-ready scenes. Extensive experiments on locomotion, navigation, and manipulation demonstrate that GS-Playground effectively bridges the perceptual and physical gaps across diverse embodied tasks. Project homepage: https://gsplayground.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17390v3">FGGS-LiDAR: Ultra-Fast, GPU-Accelerated Simulation from General 3DGS Models to LiDAR</a></div>
    <div class="paper-meta">
      📅 2026-08-04
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has emerged as a strong representation for photorealistic rendering, its vast ecosystem of assets remains difficult to use for high-performance LiDAR simulation, a critical tool for robotics and autonomous driving. We present \textbf{FGGS-LiDAR}, a geometry-first framework that bridges this gap in a plug-and-play manner. Our method converts pretrained 3DGS assets into watertight meshes directly from Gaussian parameters, without requiring LiDAR-specific supervision or architectural alterations, via volumetric discretization and Truncated Signed Distance Field (TSDF) extraction. We pair this with a GPU-accelerated ray-casting module that simulates LiDAR returns at over 500 FPS and supports batched multi-environment simulation with up to 4096 environments. In large-scale parallel settings, FGGS-LiDAR achieves an order-of-magnitude lower LiDAR simulation latency than Isaac Sim. We validate FGGS-LiDAR on both indoor and outdoor scenes, demonstrating high LiDAR-simulation fidelity. Furthermore, on COLMAP-posed indoor benchmarks, we compare against existing 3DGS-to-mesh baselines and report lower LiDAR-simulation error. Code is at https://github.com/discoverse-dev/FGGS-LiDAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.03279v1">3DGSI-Assessor: A Large-Scale Dataset and An LMM-based Method for 3D Gaussian Splatting Image Quality Assessment</a></div>
    <div class="paper-meta">
      📅 2026-08-04
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a dominant representation for real-time novel view synthesis (NVS), yet its storage footprint makes compression indispensable for practical deployment. 3DGS training and compression introduce representation-specific distortions such as floating artifacts and surface scattering, which conventional image quality assessment (IQA) metrics fail to capture. Moreover, the independent compression of geometric and color attributes may lead to decoupled dimension-specific distortions that must be diagnosed separately, yet existing metrics report only a single overall score. To address these gaps, we present 3DGS-IEval-15K+, a large-scale, multi-dimensional IQA dataset for compressed 3DGS, comprising 15,200 images from 10 diverse scenes, produced by 6 representative 3DGS algorithms at systematically designed compression levels and rendered from 20 strategically selected viewpoints spanning both training views and challenging novel views, annotated with 45,600 mean opinion scores (MOSs) across overall, geometry, and color quality. Based on 3DGS-IEval-15K+, we propose 3DGSI-Assessor, an all-in-one 3DGS IQA framework that integrates global semantic and dimension-specific local features within a large multimodal model (LMM), predicting all three dimensions in a single forward pass. 3DGSI-Assessor achieves state-of-the-art performance on 3DGS-IEval-15K+, and exhibits competitive generalization on other NVS benchmarks. Dataset and code will be released at https://github.com/YukeXing/3DGSI-Assessor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02437v2">InfiniSplat: Implicit Gaussian Decoding for Large-Baseline Monocular View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-08-04
      | 💬 Accepted to SIGGRAPH Asia 2026 (Journal Track). Code: https://github.com/zju3dv/InfiniSplat
    </div>
    <details class="paper-abstract">
      Single-image feed-forward 3D Gaussian Splatting (3DGS) aims to directly generate a renderable 3D scene representation from one input image, avoiding the cost of multi-view capture and per-scene optimization. However, existing methods are often constrained by a pixel-aligned representation, where Gaussians are predicted from fixed image-grid locations. Such pixel-aligned primitives can produce promising nearby-view renderings, but they remain weakly coupled to underlying scene surfaces and struggle to preserve coherent structures under large viewpoint shifts. We present InfiniSplat, a feed-forward single-image 3DGS framework that moves from a pixel-aligned representation toward a surface-aligned representation. InfiniSplat constructs this representation by first using geometry-guided sampling to place 2D supports according to depth-induced local surface structure, and then applying a query-conditioned implicit decoder to predict Gaussian attributes from the image features queried at these supports. By grounding support locations in geometry while decoupling Gaussian prediction from fixed pixel centers, InfiniSplat produces Gaussian layouts that better follow scene surfaces and reduce scattered primitives caused by grid discretization. Across multiple cross-dataset NVS evaluations, InfiniSplat achieves state-of-the-art performance compared with single-image feed-forward baselines, and demonstrates zero-shot generalization from Hypersim indoor synthetic training to complex open-world scenes. Project page: https://zju3dv.github.io/InfiniSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.03023v1">Standalone DINOv3 for Training-Free Open-Vocabulary Semantic Segmentation in Remote Sensing</a></div>
    <div class="paper-meta">
      📅 2026-08-04
    </div>
    <details class="paper-abstract">
      Remote sensing semantic segmentation is hindered by costly pixel-level annotations, motivating training-free open-vocabulary methods. Recently, the recent release of DINOv3 brings DINO.txt, which equips the standalone DINO backbone with image-text contrastive learning and thus opens up the possibility of open-vocabulary segmentation. We propose DinoSplat-OV, a training-free framework that adapts DINOv3 to remote sensing without fine-tuning or additional pretraining. Targeting the dense distribution, multi-scale nature, and large size of remote sensing imagery, we design two core modules. Its Text-aware Laplacian Propagation module de-noises patch-level predictions by combining textual semantic affinities with local visual similarity, improving regional consistency while preserving boundaries. Its Gaussian Splatting Upsampling module reconstructs pixel-level features through RGB-guided anisotropic aggregation and test-time optimization. A global-anchor sliding-window strategy further supports large-scale imagery. Experiments on UDD5, DOTA, and LoveDA demonstrate competitive or superior performance over existing training-free methods, effectively filling the gap of DINO-series models in training-free open-vocabulary segmentation and providing a viable new path for further advances in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02206v1">CLEAR: Conflict-aware Learning via Evidence-guided Adaptive Routing for Unified Sparse-View 3D Gaussian Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2026-08-03
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Sparse-view 3D Gaussian Splatting Super-resolution is highly challenging since the sparse and low-resolution (LR) inputs lack sufficient geometric and high-frequency information for accurate reconstruction. To achieve high-quality reconstruction, existing sparse-view super-resolution methods adhere to two-stage pipeline that performs LR Gaussian reconstruction and then high-resolution (HR) Gaussian refinement, which directly results in stage-wise Gaussian transfer and reconstruction error accumulation. To this end, we propose CLEAR, a Conflict-aware Learning via Evidence-guided Adaptive Routing, as the first unified single-stage framework for Sparse-view 3D Gaussian Splatting Super-resolution. Specifically, CLEAR performs joint the optimization of authentic LR observations and external HR priors within a unified Gaussian representation. To mitigate the gradient conflicts introduced by sparse supervision during training, we propose a Gaussian-wise conflict-aware optimization strategy that regards the LR gradient as a reliable anchor and applies evidence-conditioned soft correction only to severe HR conflicts. Moreover, to recover high-frequency details, we introduce an evidence-guided Patch-to-Gaussian routing mechanism which estimates patch reliability and detail demand, lifts them into Gaussian space, and selectively routes high-frequency gradients and densification. Finally, we employ shared Gaussian dropout and a detached mid-training anchoring to enhance the robustness of training framework. Extensive experiments on both synthetic and real-world $4\times$ super-resolution benchmarks demonstrate that CLEAR consistently achieves state-of-the-art rendering quality and superior geometric fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02191v1">DerainSplat: Feed-Forward Clean 3D Gaussian Splatting from Sparse Rainy Views</a></div>
    <div class="paper-meta">
      📅 2026-08-03
    </div>
    <details class="paper-abstract">
      Although image deraining has advanced substantially, existing methods mainly focus on 2D image restoration. As spatial intelligence applications such as embodied AI and autonomous driving continue to emerge, reconstructing clean 3D scenes from sparse rainy views in a feed-forward manner becomes increasingly important. Existing feed-forward 3D Gaussian Splatting (3DGS) methods often assume clean inputs and collapse under rainy conditions. To this end, we present \textbf{\textit{DerainSplat}}, a feed-forward framework that reconstructs clean 3D scenes from only a few rainy views. To support this task, we build a large-scale multi-view derain dataset through a four-stage synthesis pipeline that sequentially models overcast illumination, depth-dependent haze, rain streaks, and lens raindrops, producing privileged weather factors. We introduce a weather net that predicts the weather factors from rainy context and yields two support maps. Scene support modulates cross-view cost-volume matching, while radiance support drives depth-aligned appearance fusion to fill corrupted pixels. The derived geometry evidence further attenuates Gaussian opacity to reduce spurious structures. A rainy cycle consistency re-renders clean views using the predicted factors and aligns them with rainy inputs. Extensive experiments show that \textbf{\textit{DerainSplat}} outperforms existing methods on various datasets, including RealEstate10K, ACID, Mip-NeRF360, and real-world rainy scenes, with strong cross-dataset generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02177v1">GSRAIN: Physically Calibrated High-/Low-Frequency Rainfall Synthesis for 3D Gaussian Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2026-08-03
    </div>
    <details class="paper-abstract">
      Existing rainfall simulation methods for autonomous driving remain limited in physical controllability and multi-view consistency. This paper presents GSRAIN, a high-/low-frequency rainfall synthesis method for 3D Gaussian Splatting (3DGS) driving scenes. GSRAIN constructs a high-frequency raindrop model from measured rainfall data and generates low-frequency rainy appearance using a geometry-aware single-step diffusion model. The two effects are then fused in a unified 3DGS scene, enabling rainfall-intensity control over the range of 0--13~mm/h. The proposed method achieves a Fréchet Inception Distance (FID) of 149.09, outperforming CycleGAN-Turbo (155.71) and WeatherEdit (157.94). Object-detection and closed-loop driving experiments further show that the generated scenes expose scene-dependent performance changes of the evaluated algorithms under controllable rainfall. These results indicate that GSRAIN provides an effective approach for constructing physically controllable, repeatable, and closed-loop-compatible rainy-weather test scenes for autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02145v1">UniqueSplat: View-conditioned 3D Gaussian Splatting for Generalizable 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-08-03
    </div>
    <details class="paper-abstract">
      In this paper, we propose UniqueSplat, a view-conditioned feed-forward 3D Gaussian Splatting model to reconstruct customized 3D radiance fields for each view query. Existing feed-forward methods such as pixelSplat and MVSplat aim to generate fixed Gaussians across all views of each scene by minimizing the error between rendered views and ground-truth images. However, such fixed Gaussians generally render images from all views and lack the ability to adapt to specific viewpoints, as they do not incorporate target view information when predicting Gaussians. To address this, our UniqueSplat learns the view-conditioned information as a prior and incorporates this knowledge into network parameters, so that Gaussians are dynamically adjusted in accordance with different views. Specifically, we propose a two-branch view-conditioned hyperNetwork to simultaneously learn view-agnostic embeddings and view-specific knowledge, which not only explores the shareable knowledge from various views, but also adapts the model to specific views at test time. Extensive experiments on widely-used datasets including RealEstate10K, ACID and DTU demonstrate the superiority of UniqueSplat over the state-of-the-art methods. Moreover, UniqueSplat encouragingly outperforms existing methods in cross-dataset evaluation, showing its notable generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02099v1">DeGS: A Scalable 3DGS Architecture via Decoupled Workload Parsing and Reorganization</a></div>
    <div class="paper-meta">
      📅 2026-08-03
      | 💬 Accepted to the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO 2026)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading technique for real-time novel view synthesis, yet existing 3DGS accelerators suffer from poor architectural scalability: increasing the number of PEs leads to marginal performance improvement during rendering. We identify that the root cause is the tightly coupled ``checking-while-blending'' dataflow, which exacerbates PE underutilization caused by spatial redundancy from irregular Gaussian coverage and temporal redundancy from asynchronous pixel-wise termination under parallel execution. To address this issue, we propose DeGS, a scalable architecture for efficient 3DGS inference. To systematically eliminate the redundancies inherent in rendering, DeGS exploits a decoupled dataflow, restructuring the coupled $α$-checking, transmittance checking, and $α$-blending of the standard rendering process into consecutive workload parsing, reorganization, and blending stages. This allows the fragmented, length-variable, and temporal-dependent workloads to be reorganized into compact, conflict-free, and dense workloads prior to blending, thereby significantly improving PE utilization during parallel blending. Implemented in 28 nm technology, DeGS achieves 2.36$\times$--7.25$\times$ throughput, 1.82$\times$--6.02$\times$ end-to-end speedup, and 1.59$\times$--4.42$\times$ energy efficiency over state-of-the-art 3DGS accelerators (GSCore, GBU, GCC) across diverse scenes and resolutions (720p to 8K). Moreover, scaling from 16 to 1024 PEs, DeGS maintains over 80\% PE utilization at high resolutions, significantly outperforming existing accelerators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11500v2">HyperGS: Fast and Generalizable Gaussian Video Representation</a></div>
    <div class="paper-meta">
      📅 2026-08-03
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as an effective representation for video, but existing methods rely on per-video optimization. This leads to slow encoding and limits generalization across videos. To amortize this optimization, we propose HyperGS, a feedforward, optimization-free approach that directly predicts Gaussian representations from any video in a single forward pass, speeding up encoding and decoding by orders of magnitude while generalizing to out-of-distribution videos at higher resolutions. In HyperGS, we design a factorized spatiotemporal Transformer to extract tokens from video, and a learnable query-based Transformer to obtain 8-parameter Gaussian representations for each video frame. We find that naively predicting Gaussians across diverse videos induces a needle-like degeneration that collapses training, and address this with a rank-based geometric regularizer whose strength adapts dynamically to stabilize optimization. HyperGS achieves encoding at $10^4$--$10^5\times$ the speed of per-video Gaussian optimization at matched reconstruction quality while generalizing zero-shot to $720p$ video, enabling higher-resolution rendering without re-encoding. HyperGS improves PSNR by +2.9--3.1 dB over the prior video encoders on K400, SSv2, and UCF101 at a smaller video representation size. By predicting explicit 2D Gaussians in a single forward pass, HyperGS combines the fast, flexible rendering of Gaussian Splatting with the speed and generalization of feedforward prediction, advancing Gaussians as a practical direction for fast and generalizable video representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01969v1">3D Gaussian Splatting and Mesh-Based Digital Twins: An Exploratory Study for Virtual Reality Tourism</a></div>
    <div class="paper-meta">
      📅 2026-08-03
      | 💬 Accepted to the 1st Workshop on Hyperrealism in XR at ISMAR 2026
    </div>
    <details class="paper-abstract">
      Digital Twins (DTs) are increasingly used for immersive experiences in virtual tourism. Virtual Reality (VR) enables remote visits to replicated locations for promotional purposes or access to fragile and rural cultural heritage sites. However, developing high-fidelity DTs of tourist destinations is costly, due to the manual creation of 3D environments. Novel 3D rendering techniques, such as 3D Gaussian splatting (3DGS), pose a promising approach to creating immersive experiences. This study investigates the user experience (UX) of a 3D-mesh-based scene and a 3DGS-based scene within a VR tourism application. In a laboratory study, 20 participants engaged with both versions and rated UX, cybersickness, presence and affect through standardized questionnaires. A custom questionnaire was created to measure the perception of the DTs. The collected data suggests that both versions were enjoyed and induced positive affect, with the Mesh version receiving good UX ratings. While the 3DGS version scored higher in terms of experienced realism, it showed clear weaknesses in pragmatic quality. Further, the results suggest that the feeling of presence could be enhanced and cybersickness reduced in both versions. Overall, the study contributes to the understanding of UX in VR tourism applications by implementing mesh-based and 3DGS-based DTs and raising important questions about the perception of realism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01958v1">FAST-GS: Frequency Aware Space-time Gaussian Splatting for Photorealistic Dynamic Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-08-03
      | 💬 accepted by ICASSP2026
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) excels in dynamic 3D reconstruction and real-time novel view synthesis via efficient 4D Gaussian representations and parallelizable rendering. However, existing 4DGS approaches rely on a single polynomial to model motion, which limits performance in complex dynamic scenes where high-frequency motion components are prevalent, and fails to ensure long-term stability due to cumulative trajectory drift. To address these issues, we propose a Fourier Motion Modeling module: this paradigm decomposes motion into frequency-based sinusoidal components, capturing both low-frequency global trajectories and high-frequency local details to model complex motion patterns accurately. It retains the real-time rendering capability of 4DGS while improving complex motion fitting and long-term coherence. Additionally, we integrate a motion-aware regularization strategy into the loss function: it uses frequency-dependent weights to suppress high-frequency jitter while preserving low-frequency motion coherence. Extensive experiments on N3V and Google Immersive datasets from multiple scenarios demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01761v1">DecoupleGS: Interactive 3D Gaussian Splatting for End-to-End Autonomous Driving Testing</a></div>
    <div class="paper-meta">
      📅 2026-08-03
      | 💬 Accepted to ECCV 2026
    </div>
    <details class="paper-abstract">
      End-to-end (E2E) autonomous driving algorithms require rigorous closed-loop validation in simulation environments offering high visual fidelity, strong interactivity, and real-time performance. Existing approaches, from game engines to static neural rendering, inherently trade off these requirements and struggle with the dynamic scene composition essential for E2E testing. To bridge this gap, we propose a novel decoupled 3D Gaussian Splatting (3DGS) framework tailored for large-scale E2E evaluation. We fundamentally decompose scenes into a high-fidelity static background and manipulable dynamic agents using an object-centric canonical representation. To resolve resulting representational conflicts, we introduce three targeted modules: (1) asset compression via perceptual pruning and vector quantization for real-time traffic rendering; (2) map-guided geometric registration leveraging semantic topology to strictly align trajectories; and (3) proxy-based relighting transferring ambient illumination for seamless photometric integration. Extensive experiments demonstrate that DecoupleGS achieves a balanced fidelity-efficiency trade-off, improves metric and photometric consistency, and provides a practical closed-loop sensor simulation platform for E2E autonomous driving evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01726v1">G-Skin: Learning to Bind 3D Gaussians with Generative Visual Priors</a></div>
    <div class="paper-meta">
      📅 2026-08-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has achieved remarkable success in photorealistic and efficient rendering, leading to a rapid increase in 3D assets represented by 3D Gaussian primitives. Directly rigging these assets with arbitrary skeleton topologies is highly desirable. However, training a feed-forward skinning framework is infeasible due to the lack of high-quality 3D Gaussian rigging datasets. An alternative solution is to transfer mesh-based techniques to 3D Gaussian-based representation, but 3D Gaussian primitives are not restricted to the surface and lack explicit topological connectivity. Moreover, this kind of method suffers from poor generalization to unseen data due to its strong dependence on training data, while acquiring high-quality rigging data is prohibitively expensive. To address this challenging problem, we propose G-Skin, a novel generative skinning framework designed for expressive and high-fidelity animation with 3D Gaussian representation. To overcome this 3D data scarcity, we introduce a skeleton-controllable image generation model leveraging 2D vision foundation models to distill powerful motion priors into pseudo-guidance. Guided by these priors, we formulate an optimization pipeline incorporating geometry-aware regularizations, which stabilizes the learning process and ensures smooth, structurally coherent skinning weights. G-Skin also generalizes flexibly to the augmented variants of 3D Gaussian representation designed to mitigate animation-induced rendering artifacts. Extensive experiments validate the effectiveness of our approach, demonstrating clear advantages over state-of-the-art methods. Project page: https://yaoyx689.github.io/GSkin.html.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05950v3">CLONE: Continuous Latent Optimization for Normal Estimation via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-03
    </div>
    <details class="paper-abstract">
      We propose CLONE, a Continuous Latent Optimization framework for Normal Estimation via 3D Gaussian splatting. The core idea is to construct an image-geometry-image consistency loop that unifies explicit geometric representation with differentiable rendering, thereby enabling weakly supervised learning without normal ground truth. Specifically, CLONE comprises four components. First, by introducing a differentiable light interaction model with a learnable modulation kernel, we perform a unified reparameterization of the 3DGS parameter space, establishing an explicit and stable mapping between 3DGS geometric parameters and surface normals and turning the photometric loss into an internal supervision signal. Second, the conditional single-step deterministic refinement network integrates denoising architectures with differentiable reprojection constraints to refine the initial normals, thereby adaptively recovering the high-frequency details erased by the inherently smooth Gaussian primitives. Third, the cross-domain gating fusion mechanism adaptively combines the two complementary normal estimates while imposing multi-view reprojection consistency and implicit geometric regularization, reconciling the geometrically consistent yet over-smooth 3DGS estimate with the detailed yet potentially geometry-inconsistent refinement. Finally, all components are jointly optimized under a unified photometric reprojection objective with geometric consistency regularizations in a fully differentiable pathway, and the directional regularization aligns the learnable principal directions with the geometric normals, achieving an end-to-end optimization closed loop without relying on external normal labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01588v1">D^2-4DGS: Dual-Depth Guided Sparse-Camera 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-03
    </div>
    <details class="paper-abstract">
      Dynamic 4D Gaussian Splatting has emerged as an efficient representation for dynamic novel view synthesis through explicit scene modeling and real-time rendering. However, existing methods typically require dense multi-view videos for sufficient geometric constraints, making capture expensive and limiting sparse-camera deployment. Reducing input views lowers acquisition cost but weakens geometry supervision, often causing missing structures and floating Gaussians. Depth priors provide geometric cues, yet no single source offers both dense coverage and reliable geometry. Monocular depth provides dense structure but is scale-ambiguous and locally biased, whereas multi-view geometric depth provides incomplete anchors consistent with the reconstruction coordinate system. To exploit their complementarity, we propose D$^2$-4DGS, a sparse-camera dynamic 4D Gaussian Splatting framework guided by dual-source depth priors. We align monocular estimates with valid multi-view geometric depths and verify their consistency to identify reliable geometric anchors. These verified anchors support consistency-aware pruning and depth supervision, while verified geometric depths and aligned mono-only estimates provide candidate geometry for densification in under-reconstructed regions. Finally, RGB-D joint optimization improves appearance fidelity and geometric consistency under sparse-view supervision. Across all nine dataset--view settings, D$^2$-4DGS achieves the highest PSNR, improving by 1.33 dB on average over the best competing method in each setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06404v1">UAV3DCrop: Benchmarking 3D Reconstruction in Repeated Multi-Angle UAV Crop Surveys</a></div>
    <div class="paper-meta">
      📅 2026-08-03
      | 💬 22 pages, 7 figures. Dataset and project page: https://link-dev.github.io/UAV3DCrop/
    </div>
    <details class="paper-abstract">
      Accurate 3D crop monitoring underpins data-driven precision agriculture by enabling field-scale analysis of plant structure, growth dynamics, and management response. Modern 3D reconstruction methods perform strongly on generic benchmarks, but rendered appearance may not translate into metrically and agronomically useful geometry in crop fields. We introduce UAV3DCrop, a public benchmark of repeated multi-angle unmanned aerial vehicle (UAV) crop surveys. It contains 88,830 RGB images at $5280 \times 3956$ pixels, with a ground sampling distance of 3.6-5.8 mm, from 91 scenes spanning corn, soybean, wheat, and oat. Track A evaluates seven scene-optimized methods -- Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) variants -- on held-out views, photogrammetry-referenced depth, and canopy-height recovery. Track B tests four pretrained feed-forward models on zero-shot camera-pose and geometry estimation. The scene-optimized methods rank differently across the three targets: Splatfacto-big leads appearance, whereas Scaffold-GS leads depth and is statistically tied with Splatfacto for canopy height. Among feed-forward models, MapAnything leads on seven of the eight metrics, while the remaining models vary more across crops and fail severely on absolute scale in a way that alignment conceals. Repeated acquisitions reveal further sensitivities that differ by output type and by model, associated with position within the acquisition sequence and with tie-point multiplicity. Current 3D reconstruction methods are therefore not yet interchangeable for agronomic use: no single method wins on appearance, geometry, and canopy height at once, and only one of four feed-forward models recovers usable metric scale. The dataset is publicly available at https://link-dev.github.io/UAV3DCrop/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01492v1">GaussianSelector: Lightweight Human-Guided Object Selection in 3D Gaussian Splatting with Graph Optimization</a></div>
    <div class="paper-meta">
      📅 2026-08-02
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Selecting a complete 3D object from a reconstructed scene with minimal user effort is essential for practical scene editing and embodied interaction. Existing 3DGS-based methods either retrain the Gaussian representation to embed per-object labels, or build dense multi-view SAM observations, both requiring heavy computation and dense viewpoint coverage that is rarely available in practice. We present GaussianSelector, a training-free framework for interactive 3D object selection from sparse views and sparse scribble guidance. Operating directly on native Gaussian primitives, we coarsen dense Gaussians into geometrically coherent superpoints and construct a continuity-weighted graph using appearance and spatial cues. Sparse user scribbles are lifted into 3D via visibility-aware transmittance coverage, and selection is solved as a global graph-cut energy minimization that propagates sparse evidence to a complete 3D object. This design naturally supports multi-round refinement, where users iteratively correct the selection from additional viewpoints to progressively improve the result. Experiments demonstrate that GaussianSelector achieves competitive selection quality against state-of-the-art multi-view SAM-based methods, while requiring significantly fewer interaction views and substantially lower computational overhead. These properties make it well suited for human-in-the-loop 3D scene editing and 3D asset extraction in real-world deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01186v1">QuerySplat: Decoupling Geometry and Appearance Representations in 3DGS Prediction</a></div>
    <div class="paper-meta">
      📅 2026-08-02
    </div>
    <details class="paper-abstract">
      While feed-forward 3D Gaussian Splatting (3DGS) enables efficient 3D reconstruction, achieving high-fidelity rendering remains challenging. Existing pixel-aligned approaches suffer from spatial inflexibility and massive structural redundancy, whereas query-based methods lack 3D priors and entangle geometry with appearance, yielding blurry, pose-dependent results. To overcome these deficiencies, we propose \textbf{QuerySplat}, a feed-forward 3DGS framework driven by geometric priors and explicit appearance decoupling. Specifically, we design a dual-branch query-based decoder: the geometry branch leverages a pretrained Vision Geometric Model for spatial understanding, which intrinsically endows QuerySplat with pose-free modeling capabilities, while the appearance branch recovers high-frequency details through a dedicated pathway separated from geometric attribute regression. Extensive experiments demonstrate that QuerySplat mitigates the blurry rendering issues of early query-based models and consistently outperforms pixel-aligned approaches in rendering fidelity. On the challenging DL3DV benchmark, it achieves state-of-the-art novel view synthesis performance, with average PSNR gains of 2.30 dB and 1.04 dB over the best pose-free and pose-required baselines, respectively. Project Page: https://inspatio.github.io/querysplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01178v1">DynActiveGS: Active Gaussian Splatting for Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-08-02
      | 💬 Accepted to ACM Multimedia 2026
    </div>
    <details class="paper-abstract">
      We present DynActiveGS, a dynamic-aware active reconstruction framework based on 3D Gaussian Splatting (3DGS) for autonomous exploration in dynamic environments. The framework incrementally reconstructs a 3D Gaussian scene representation while suppressing motion-corrupted observations through online uncertainty prediction and uncertainty-weighted Gaussian optimization. A key component of DynActiveGS is the explicit decomposition of uncertainty into structural uncertainty and motion-induced uncertainty, which enables the system to distinguish under-reconstructed static regions from dynamically unreliable areas. Based on these uncertainty fields, DynActiveGS performs dynamic-aware viewpoint selection and dynamic-constrained path planning to favor informative yet stable observations during exploration. The resulting system forms a unified closed-loop pipeline for robust active reconstruction in dynamic scenes. Extensive experiments on challenging dynamic benchmarks demonstrate consistent improvements over existing active reconstruction baselines in reconstruction accuracy, completeness, rendering quality, and exploration efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25129v2">AirSplat: Alignment and Rating for Robust Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-08-02
      | 💬 Project page: https://kaist-viclab.github.io/airsplat-site, accepted to ECCV 2026
    </div>
    <details class="paper-abstract">
      While 3D Vision Foundation Models (3DVFMs) have demonstrated remarkable zero-shot capabilities in visual geometry estimation, their direct application to generalizable novel view synthesis (NVS) remains challenging. In this paper, we propose AirSplat, a novel training framework that effectively adapts the robust geometric priors of 3DVFMs into high-fidelity, pose-free NVS. Our approach introduces two key technical contributions: (1) Self-Consistent Pose Alignment (SCPA), a training-time feedback loop that ensures pixel-aligned supervision to resolve pose-geometry discrepancy; and (2) Rating-based Opacity Matching (ROM), which leverages the local 3D geometry consistency knowledge from a sparse-view NVS teacher model to filter out degraded primitives. Experimental results on large-scale benchmarks demonstrate that our method significantly outperforms state-of-the-art pose-free NVS approaches in reconstruction quality. Our AirSplat highlights the potential of adapting 3DVFMs to enable simultaneous visual geometry estimation and high-quality view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.00950v1">Swimm3R: Splatting with Medium-aware SfM for Underwater 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-08-02
      | 💬 Project Page: https://mnseong.github.io/swimm3r.github.io
    </div>
    <details class="paper-abstract">
      We propose Swimm3R, a unified framework that combines medium-aware structure-from-motion (SfM) with Underwater Beta Splatting to address scattering- and attenuation-induced failures in underwater 3D reconstruction. Swimm3R distills in-air geometric priors into a feed-forward backbone and uses a physics head to regress underwater image-formation parameters, camera poses, and restored point clouds. Additionally, we introduce Underwater Beta Splatting, which extends Gaussian splatting with Beta primitives and scattering-aware geometric gradients for stable underwater geometry representation. We further establish the Barbados underwater video dataset to demonstrate the effectiveness of our method in challenging underwater environments. On this dataset, Swimm3R robustly recovers underwater scene structure under challenging scattering conditions, yielding coherent seafloor geometry. Using these predicted point clouds, the proposed Underwater Beta Splatting improves average PSNR by $1.47$ dB over WaterSplatting while increasing downstream localization performance by $2.0$ and $2.4$ percentage points in RRA@15 and RTA@15, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.00931v1">Stipple: Real-Time Incremental Gaussian Splatting with Visual-Inertial Tracking</a></div>
    <div class="paper-meta">
      📅 2026-08-02
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) provides efficient rendering of photo-realistic scenes, but its heavy preprocessing and training steps make it a poor fit for applications that require real-time reconstruction in robotics or XR. This capability is important since it allows immediate feedback and interaction with new environments. Visual-inertial odometry (VIO) and simultaneous localization and mapping (VI-SLAM) systems, on the other hand, specifically target these real-time applications, which makes them a good choice for integration with 3DGS. We propose a new method that tracks and reconstructs simultaneously in real-time by leveraging an efficient visual-inertial tracking system based on Basalt together with a novel incremental method built on top of Brush, an efficient Rust-based GPU-vendor-agnostic implementation of 3D Gaussian Splatting. We show that many of the heavy preprocessing and training steps of 3DGS can be replaced with a more efficient incremental training strategy that has direct access to the information generated by the visual-inertial tracking system. Furthermore, we propose and combine multiple practical improvements to increase the efficiency of the training pipeline and adapt it to run in real-time, parallel to the tracking thread. This work highlights the value of exploiting the complementary nature of SLAM and 3DGS, and how that can lead to promising results for real-time 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.00463v1">Scene2Sound: Auditory-Grounded Soundscape Generation for 3D Gaussian Worlds</a></div>
    <div class="paper-meta">
      📅 2026-08-01
      | 💬 14 pages. Project page: https://masaki-lmd.github.io/scene2sound/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) turns captured or generated imagery into photorealistic 3D world simulations that users can freely explore, yet these worlds remain silent. Because existing audio generation methods condition on a single image or viewpoint, their sound is tied to that observation and cannot stay consistent while a listener moves. We introduce the task of generating a spatially consistent soundscape for a given 3DGS world through auditory grounding, identifying which objects in the world should emit sound and anchoring each to a persistent 3D position, and present Scene2Sound, a training-free framework built on this grounding. From the input world alone, our pipeline selects viewpoints that jointly cover the scene, identifies sound-emitting objects with a vision-language model, and associates the multi-view detections into 3D instances through Gaussian set matching, which measures the overlap between the Gaussian sets that render each detection. Each source then receives generated audio that a standard object-based audio engine spatializes in real time at arbitrary listener poses. We further propose two spatial-consistency metrics, one testing whether rendered audio responds consistently to listener motion and one testing whether the claimed sources are supported by views held out from their placement. On a curated set of generated 3DGS worlds and on 3DGS scenes generated from real-world 360-degree captures, Scene2Sound preserves the audio quality of strong per-viewpoint baselines while remaining spatially consistent where per-viewpoint and single-panorama pipelines do not, and a user study confirms the perceptual benefit. Project page: https://masaki-lmd.github.io/scene2sound/.
    </details>
</div>
