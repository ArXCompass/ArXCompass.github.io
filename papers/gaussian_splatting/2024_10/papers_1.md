# gaussian splatting - 2024_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00239v1">Aquatic-GS: A Hybrid 3D Representation for Underwater Scenes</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 13 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Representing underwater 3D scenes is a valuable yet complex task, as attenuation and scattering effects during underwater imaging significantly couple the information of the objects and the water. This coupling presents a significant challenge for existing methods in effectively representing both the objects and the water medium simultaneously. To address this challenge, we propose Aquatic-GS, a hybrid 3D representation approach for underwater scenes that effectively represents both the objects and the water medium. Specifically, we construct a Neural Water Field (NWF) to implicitly model the water parameters, while extending the latest 3D Gaussian Splatting (3DGS) to model the objects explicitly. Both components are integrated through a physics-based underwater image formation model to represent complex underwater scenes. Moreover, to construct more precise scene geometry and details, we design a Depth-Guided Optimization (DGO) mechanism that uses a pseudo-depth map as auxiliary guidance. After optimization, Aquatic-GS enables the rendering of novel underwater viewpoints and supports restoring the true appearance of underwater scenes, as if the water medium were absent. Extensive experiments on both simulated and real-world datasets demonstrate that Aquatic-GS surpasses state-of-the-art underwater 3D representation methods, achieving better rendering quality and real-time rendering performance with a 410x increase in speed. Furthermore, regarding underwater image restoration, Aquatic-GS outperforms representative dewatering methods in color correction, detail recovery, and stability. Our models, code, and datasets can be accessed at https://aquaticgs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.24207v1">No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 Project page: https://noposplat.github.io/
    </div>
    <details class="paper-abstract">
      We introduce NoPoSplat, a feed-forward model capable of reconstructing 3D scenes parameterized by 3D Gaussians from \textit{unposed} sparse multi-view images. Our model, trained exclusively with photometric loss, achieves real-time 3D Gaussian reconstruction during inference. To eliminate the need for accurate pose input during reconstruction, we anchor one input view's local camera coordinates as the canonical space and train the network to predict Gaussian primitives for all views within this space. This approach obviates the need to transform Gaussian primitives from local coordinates into a global coordinate system, thus avoiding errors associated with per-frame Gaussians and pose estimation. To resolve scale ambiguity, we design and compare various intrinsic embedding methods, ultimately opting to convert camera intrinsics into a token embedding and concatenate it with image tokens as input to the model, enabling accurate scene scale prediction. We utilize the reconstructed 3D Gaussians for novel view synthesis and pose estimation tasks and propose a two-stage coarse-to-fine pipeline for accurate pose estimation. Experimental results demonstrate that our pose-free approach can achieve superior novel view synthesis quality compared to pose-required methods, particularly in scenarios with limited input image overlap. For pose estimation, our method, trained without ground truth depth or explicit matching loss, significantly outperforms the state-of-the-art methods with substantial improvements. This work makes significant advances in pose-free generalizable 3D reconstruction and demonstrates its applicability to real-world scenarios. Code and trained models are available at https://noposplat.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08447v2">WildGaussians: 3D Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 NeurIPS 2024; Project page: https://wild-gaussians.github.io/
    </div>
    <details class="paper-abstract">
      While the field of 3D scene reconstruction is dominated by NeRFs due to their photorealistic quality, 3D Gaussian Splatting (3DGS) has recently emerged, offering similar quality with real-time rendering speeds. However, both methods primarily excel with well-controlled 3D scenes, while in-the-wild data - characterized by occlusions, dynamic objects, and varying illumination - remains challenging. NeRFs can adapt to such conditions easily through per-image embedding vectors, but 3DGS struggles due to its explicit representation and lack of shared parameters. To address this, we introduce WildGaussians, a novel approach to handle occlusions and appearance changes with 3DGS. By leveraging robust DINO features and integrating an appearance modeling module within 3DGS, our method achieves state-of-the-art results. We demonstrate that WildGaussians matches the real-time rendering speed of 3DGS while surpassing both 3DGS and NeRF baselines in handling in-the-wild data, all within a simple architectural framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12282v2">Subsurface Scattering for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 Project page: https://sss.jdihlmann.com/
    </div>
    <details class="paper-abstract">
      3D reconstruction and relighting of objects made from scattering materials present a significant challenge due to the complex light transport beneath the surface. 3D Gaussian Splatting introduced high-quality novel view synthesis at real-time speeds. While 3D Gaussians efficiently approximate an object's surface, they fail to capture the volumetric properties of subsurface scattering. We propose a framework for optimizing an object's shape together with the radiance transfer field given multi-view OLAT (one light at a time) data. Our method decomposes the scene into an explicit surface represented as 3D Gaussians, with a spatially varying BRDF, and an implicit volumetric representation of the scattering component. A learned incident light field accounts for shadowing. We optimize all parameters jointly via ray-traced differentiable rendering. Our approach enables material editing, relighting and novel view synthesis at interactive rates. We show successful application on synthetic data and introduce a newly acquired multi-view multi-light dataset of objects in a light-stage setup. Compared to previous work we achieve comparable or better results at a fraction of optimization and rendering time while enabling detailed control over material attributes. Project page https://sss.jdihlmann.com/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23718v1">GaussianMarker: Uncertainty-Aware Copyright Protection of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a crucial method for acquiring 3D assets. To protect the copyright of these assets, digital watermarking techniques can be applied to embed ownership information discreetly within 3DGS models. However, existing watermarking methods for meshes, point clouds, and implicit radiance fields cannot be directly applied to 3DGS models, as 3DGS models use explicit 3D Gaussians with distinct structures and do not rely on neural networks. Naively embedding the watermark on a pre-trained 3DGS can cause obvious distortion in rendered images. In our work, we propose an uncertainty-based method that constrains the perturbation of model parameters to achieve invisible watermarking for 3DGS. At the message decoding stage, the copyright messages can be reliably extracted from both 3D Gaussians and 2D rendered images even under various forms of 3D and 2D distortions. We conduct extensive experiments on the Blender, LLFF and MipNeRF-360 datasets to validate the effectiveness of our proposed method, demonstrating state-of-the-art performance on both message decoding accuracy and view synthesis quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22817v2">Epipolar-Free 3D Gaussian Splatting for Generalizable Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Generalizable 3D Gaussian splitting (3DGS) can reconstruct new scenes from sparse-view observations in a feed-forward inference manner, eliminating the need for scene-specific retraining required in conventional 3DGS. However, existing methods rely heavily on epipolar priors, which can be unreliable in complex realworld scenes, particularly in non-overlapping and occluded regions. In this paper, we propose eFreeSplat, an efficient feed-forward 3DGS-based model for generalizable novel view synthesis that operates independently of epipolar line constraints. To enhance multiview feature extraction with 3D perception, we employ a selfsupervised Vision Transformer (ViT) with cross-view completion pre-training on large-scale datasets. Additionally, we introduce an Iterative Cross-view Gaussians Alignment method to ensure consistent depth scales across different views. Our eFreeSplat represents an innovative approach for generalizable novel view synthesis. Different from the existing pure geometry-free methods, eFreeSplat focuses more on achieving epipolar-free feature matching and encoding by providing 3D priors through cross-view pretraining. We evaluate eFreeSplat on wide-baseline novel view synthesis tasks using the RealEstate10K and ACID datasets. Extensive experiments demonstrate that eFreeSplat surpasses state-of-the-art baselines that rely on epipolar priors, achieving superior geometry reconstruction and novel view synthesis quality. Project page: https://tatakai1.github.io/efreesplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23658v1">GS-Blur: A 3D Scene-Based Dataset for Realistic Image Deblurring</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 Accepted at NeurIPS 2024 Datasets & Benchmarks Track
    </div>
    <details class="paper-abstract">
      To train a deblurring network, an appropriate dataset with paired blurry and sharp images is essential. Existing datasets collect blurry images either synthetically by aggregating consecutive sharp frames or using sophisticated camera systems to capture real blur. However, these methods offer limited diversity in blur types (blur trajectories) or require extensive human effort to reconstruct large-scale datasets, failing to fully reflect real-world blur scenarios. To address this, we propose GS-Blur, a dataset of synthesized realistic blurry images created using a novel approach. To this end, we first reconstruct 3D scenes from multi-view images using 3D Gaussian Splatting (3DGS), then render blurry images by moving the camera view along the randomly generated motion trajectories. By adopting various camera trajectories in reconstructing our GS-Blur, our dataset contains realistic and diverse types of blur, offering a large-scale dataset that generalizes well to real-world blur. Using GS-Blur with various deblurring methods, we demonstrate its ability to generalize effectively compared to previous synthetic or real blur datasets, showing significant improvements in deblurring performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10242v2">GeoGS3D: Single-view 3D Reconstruction via Geometric-aware Diffusion Model and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      We introduce GeoGS3D, a novel two-stage framework for reconstructing detailed 3D objects from single-view images. Inspired by the success of pre-trained 2D diffusion models, our method incorporates an orthogonal plane decomposition mechanism to extract 3D geometric features from the 2D input, facilitating the generation of multi-view consistent images. During the following Gaussian Splatting, these images are fused with epipolar attention, fully utilizing the geometric correlations across views. Moreover, we propose a novel metric, Gaussian Divergence Significance (GDS), to prune unnecessary operations during optimization, significantly accelerating the reconstruction process. Extensive experiments demonstrate that GeoGS3D generates images with high consistency across views and reconstructs high-quality 3D objects, both qualitatively and quantitatively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.19657v2">DiffGS: Functional Gaussian Splatting Diffusion</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 Accepted by NeurIPS 2024. Project page: https://junshengzhou.github.io/DiffGS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown convincing performance in rendering speed and fidelity, yet the generation of Gaussian Splatting remains a challenge due to its discreteness and unstructured nature. In this work, we propose DiffGS, a general Gaussian generator based on latent diffusion models. DiffGS is a powerful and efficient 3D generative model which is capable of generating Gaussian primitives at arbitrary numbers for high-fidelity rendering with rasterization. The key insight is to represent Gaussian Splatting in a disentangled manner via three novel functions to model Gaussian probabilities, colors and transforms. Through the novel disentanglement of 3DGS, we represent the discrete and unstructured 3DGS with continuous Gaussian Splatting functions, where we then train a latent diffusion model with the target of generating these Gaussian Splatting functions both unconditionally and conditionally. Meanwhile, we introduce a discretization algorithm to extract Gaussians at arbitrary numbers from the generated functions via octree-guided sampling and optimization. We explore DiffGS for various tasks, including unconditional generation, conditional generation from text, image, and partial 3DGS, as well as Point-to-Gaussian generation. We believe that DiffGS provides a new direction for flexibly modeling and generating Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2405.15196v2">DisC-GS: Discontinuity-aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Recently, Gaussian Splatting, a method that represents a 3D scene as a collection of Gaussian distributions, has gained significant attention in addressing the task of novel view synthesis. In this paper, we highlight a fundamental limitation of Gaussian Splatting: its inability to accurately render discontinuities and boundaries in images due to the continuous nature of Gaussian distributions. To address this issue, we propose a novel framework enabling Gaussian Splatting to perform discontinuity-aware image rendering. Additionally, we introduce a Bézier-boundary gradient approximation strategy within our framework to keep the "differentiability" of the proposed discontinuity-aware rendering process. Extensive experiments demonstrate the efficacy of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15196v2">DisC-GS: Discontinuity-aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Recently, Gaussian Splatting, a method that represents a 3D scene as a collection of Gaussian distributions, has gained significant attention in addressing the task of novel view synthesis. In this paper, we highlight a fundamental limitation of Gaussian Splatting: its inability to accurately render discontinuities and boundaries in images due to the continuous nature of Gaussian distributions. To address this issue, we propose a novel framework enabling Gaussian Splatting to perform discontinuity-aware image rendering. Additionally, we introduce a B\'ezier-boundary gradient approximation strategy within our framework to keep the "differentiability" of the proposed discontinuity-aware rendering process. Extensive experiments demonstrate the efficacy of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23213v1">ELMGS: Enhancing memory and computation scaLability through coMpression for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      3D models have recently been popularized by the potentiality of end-to-end training offered first by Neural Radiance Fields and most recently by 3D Gaussian Splatting models. The latter has the big advantage of naturally providing fast training convergence and high editability. However, as the research around these is still in its infancy, there is still a gap in the literature regarding the model's scalability. In this work, we propose an approach enabling both memory and computation scalability of such models. More specifically, we propose an iterative pruning strategy that removes redundant information encoded in the model. We also enhance compressibility for the model by including in the optimization strategy a differentiable quantization and entropy coding estimator. Our results on popular benchmarks showcase the effectiveness of the proposed approach and open the road to the broad deployability of such a solution even on resource-constrained devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12459v2">HumanSplat: Generalizable Single-Image Human Gaussian Splatting with Structure Priors</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      Despite recent advancements in high-fidelity human reconstruction techniques, the requirements for densely captured images or time-consuming per-instance optimization significantly hinder their applications in broader scenarios. To tackle these issues, we present HumanSplat which predicts the 3D Gaussian Splatting properties of any human from a single input image in a generalizable manner. In particular, HumanSplat comprises a 2D multi-view diffusion model and a latent reconstruction transformer with human structure priors that adeptly integrate geometric priors and semantic features within a unified framework. A hierarchical loss that incorporates human semantic information is further designed to achieve high-fidelity texture modeling and better constrain the estimated multiple views. Comprehensive experiments on standard benchmarks and in-the-wild images demonstrate that HumanSplat surpasses existing state-of-the-art methods in achieving photorealistic novel-view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06613v2">ES-Gaussian: Gaussian Splatting Mapping via Error Space-Based Gaussian Completion</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 This preprint has been withdrawn due to concerns regarding the originality of certain technical elements, as well as its basis in a company project report that was intended solely for internal discussions. To avoid any potential misunderstandings, we have decided to withdraw this submission from public access. We apologize for any confusion this may have caused
    </div>
    <details class="paper-abstract">
      Accurate and affordable indoor 3D reconstruction is critical for effective robot navigation and interaction. Traditional LiDAR-based mapping provides high precision but is costly, heavy, and power-intensive, with limited ability for novel view rendering. Vision-based mapping, while cost-effective and capable of capturing visual data, often struggles with high-quality 3D reconstruction due to sparse point clouds. We propose ES-Gaussian, an end-to-end system using a low-altitude camera and single-line LiDAR for high-quality 3D indoor reconstruction. Our system features Visual Error Construction (VEC) to enhance sparse point clouds by identifying and correcting areas with insufficient geometric detail from 2D error maps. Additionally, we introduce a novel 3DGS initialization method guided by single-line LiDAR, overcoming the limitations of traditional multi-view setups and enabling effective reconstruction in resource-constrained environments. Extensive experimental results on our new Dreame-SR dataset and a publicly available dataset demonstrate that ES-Gaussian outperforms existing methods, particularly in challenging scenarios. The project page is available at https://chenlu-china.github.io/ES-Gaussian/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05774v2">VCR-GauS: View Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Project page: https://hlinchen.github.io/projects/VCR-GauS/
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting has been widely studied because of its realistic and efficient novel-view synthesis, it is still challenging to extract a high-quality surface from the point-based representation. Previous works improve the surface by incorporating geometric priors from the off-the-shelf normal estimator. However, there are two main limitations: 1) Supervising normals rendered from 3D Gaussians effectively updates the rotation parameter but is less effective for other geometric parameters; 2) The inconsistency of predicted normal maps across multiple views may lead to severe reconstruction artifacts. In this paper, we propose a Depth-Normal regularizer that directly couples normal with other geometric parameters, leading to full updates of the geometric parameters from normal regularization. We further propose a confidence term to mitigate inconsistencies of normal predictions across multiple views. Moreover, we also introduce a densification and splitting strategy to regularize the size and distribution of 3D Gaussians for more accurate surface modeling. Compared with Gaussian-based baselines, experiments show that our approach obtains better reconstruction quality and maintains competitive appearance quality at faster training speed and 100+ FPS rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22705v1">Geometry Cloak: Preventing TGS-based 3D Reconstruction from Copyrighted Images</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Single-view 3D reconstruction methods like Triplane Gaussian Splatting (TGS) have enabled high-quality 3D model generation from just a single image input within seconds. However, this capability raises concerns about potential misuse, where malicious users could exploit TGS to create unauthorized 3D models from copyrighted images. To prevent such infringement, we propose a novel image protection approach that embeds invisible geometry perturbations, termed "geometry cloaks", into images before supplying them to TGS. These carefully crafted perturbations encode a customized message that is revealed when TGS attempts 3D reconstructions of the cloaked image. Unlike conventional adversarial attacks that simply degrade output quality, our method forces TGS to fail the 3D reconstruction in a specific way - by generating an identifiable customized pattern that acts as a watermark. This watermark allows copyright holders to assert ownership over any attempted 3D reconstructions made from their protected images. Extensive experiments have verified the effectiveness of our geometry cloak. Our project is available at https://qsong2001.github.io/geometry_cloak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19657v2">DiffGS: Functional Gaussian Splatting Diffusion</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Accepted by NeurIPS 2024. Project page: https://junshengzhou.github.io/DiffGS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown convincing performance in rendering speed and fidelity, yet the generation of Gaussian Splatting remains a challenge due to its discreteness and unstructured nature. In this work, we propose DiffGS, a general Gaussian generator based on latent diffusion models. DiffGS is a powerful and efficient 3D generative model which is capable of generating Gaussian primitives at arbitrary numbers for high-fidelity rendering with rasterization. The key insight is to represent Gaussian Splatting in a disentangled manner via three novel functions to model Gaussian probabilities, colors and transforms. Through the novel disentanglement of 3DGS, we represent the discrete and unstructured 3DGS with continuous Gaussian Splatting functions, where we then train a latent diffusion model with the target of generating these Gaussian Splatting functions both unconditionally and conditionally. Meanwhile, we introduce a discretization algorithm to extract Gaussians at arbitrary numbers from the generated functions via octree-guided sampling and optimization. We explore DiffGS for various tasks, including unconditional generation, conditional generation from text, image, and partial 3DGS, as well as Point-to-Gaussian generation. We believe that DiffGS provides a new direction for flexibly modeling and generating Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.19657v2">DiffGS: Functional Gaussian Splatting Diffusion</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Accepted by NeurIPS 2024. Project page: https://junshengzhou.github.io/DiffGS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown convincing performance in rendering speed and fidelity, yet the generation of Gaussian Splatting remains a challenge due to its discreteness and unstructured nature. In this work, we propose DiffGS, a general Gaussian generator based on latent diffusion models. DiffGS is a powerful and efficient 3D generative model which is capable of generating Gaussian primitives at arbitrary numbers for high-fidelity rendering with rasterization. The key insight is to represent Gaussian Splatting in a disentangled manner via three novel functions to model Gaussian probabilities, colors and transforms. Through the novel disentanglement of 3DGS, we represent the discrete and unstructured 3DGS with continuous Gaussian Splatting functions, where we then train a latent diffusion model with the target of generating these Gaussian Splatting functions both unconditionally and conditionally. Meanwhile, we introduce a discretization algorithm to extract Gaussians at arbitrary numbers from the generated functions via octree-guided sampling and optimization. We explore DiffGS for various tasks, including unconditional generation, conditional generation from text, image, and partial 3DGS, as well as Point-to-Gaussian generation. We believe that DiffGS provides a new direction for flexibly modeling and generating Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.15196v2">DisC-GS: Discontinuity-aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Recently, Gaussian Splatting, a method that represents a 3D scene as a collection of Gaussian distributions, has gained significant attention in addressing the task of novel view synthesis. In this paper, we highlight a fundamental limitation of Gaussian Splatting: its inability to accurately render discontinuities and boundaries in images due to the continuous nature of Gaussian distributions. To address this issue, we propose a novel framework enabling Gaussian Splatting to perform discontinuity-aware image rendering. Additionally, we introduce a Bézier-boundary gradient approximation strategy within our framework to keep the "differentiability" of the proposed discontinuity-aware rendering process. Extensive experiments demonstrate the efficacy of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01804v5">EVER: Exact Volumetric Ellipsoid Rendering for Real-time View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 Project page: https://half-potato.gitlab.io/posts/ever
    </div>
    <details class="paper-abstract">
      We present Exact Volumetric Ellipsoid Rendering (EVER), a method for real-time differentiable emission-only volume rendering. Unlike recent rasterization based approach by 3D Gaussian Splatting (3DGS), our primitive based representation allows for exact volume rendering, rather than alpha compositing 3D Gaussian billboards. As such, unlike 3DGS our formulation does not suffer from popping artifacts and view dependent density, but still achieves frame rates of $\sim\!30$ FPS at 720p on an NVIDIA RTX4090. Since our approach is built upon ray tracing it enables effects such as defocus blur and camera distortion (e.g. such as from fisheye cameras), which are difficult to achieve by rasterization. We show that our method is more accurate with fewer blending issues than 3DGS and follow-up work on view-consistent rendering, especially on the challenging large-scale scenes from the Zip-NeRF dataset where it achieves sharpest results among real-time techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12954v2">GStex: Per-Primitive Texturing of 2D Gaussian Splatting for Decoupled Appearance and Geometry Modeling</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 Project page: https://lessvrong.com/cs/gstex. Updated Oct. 29 to correct Table 1 numbers. Please see https://github.com/victor-rong/GStex?tab=readme-ov-file#errata for details
    </div>
    <details class="paper-abstract">
      Gaussian splatting has demonstrated excellent performance for view synthesis and scene reconstruction. The representation achieves photorealistic quality by optimizing the position, scale, color, and opacity of thousands to millions of 2D or 3D Gaussian primitives within a scene. However, since each Gaussian primitive encodes both appearance and geometry, these attributes are strongly coupled--thus, high-fidelity appearance modeling requires a large number of Gaussian primitives, even when the scene geometry is simple (e.g., for a textured planar surface). We propose to texture each 2D Gaussian primitive so that even a single Gaussian can be used to capture appearance details. By employing per-primitive texturing, our appearance representation is agnostic to the topology and complexity of the scene's geometry. We show that our approach, GStex, yields improved visual quality over prior work in texturing Gaussian splats. Furthermore, we demonstrate that our decoupling enables improved novel view synthesis performance compared to 2D Gaussian splatting when reducing the number of Gaussian primitives, and that GStex can be used for scene appearance editing and re-texturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04237v4">GSD: View-Guided Gaussian Splatting Diffusion for 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 ECCV 2024
    </div>
    <details class="paper-abstract">
      We present GSD, a diffusion model approach based on Gaussian Splatting (GS) representation for 3D object reconstruction from a single view. Prior works suffer from inconsistent 3D geometry or mediocre rendering quality due to improper representations. We take a step towards resolving these shortcomings by utilizing the recent state-of-the-art 3D explicit representation, Gaussian Splatting, and an unconditional diffusion model. This model learns to generate 3D objects represented by sets of GS ellipsoids. With these strong generative 3D priors, though learning unconditionally, the diffusion model is ready for view-guided reconstruction without further model fine-tuning. This is achieved by propagating fine-grained 2D features through the efficient yet flexible splatting function and the guided denoising sampling process. In addition, a 2D diffusion model is further employed to enhance rendering fidelity, and improve reconstructed GS quality by polishing and re-using the rendered images. The final reconstructed objects explicitly come with high-quality 3D structure and texture, and can be efficiently rendered in arbitrary views. Experiments on the challenging real-world CO3D dataset demonstrate the superiority of our approach. Project page: https://yxmu.foo/GSD/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22128v1">PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 project page: https://cvlab-kaist.github.io/PF3plat/
    </div>
    <details class="paper-abstract">
      We consider the problem of novel view synthesis from unposed images in a single feed-forward. Our framework capitalizes on fast speed, scalability, and high-quality 3D reconstruction and view synthesis capabilities of 3DGS, where we further extend it to offer a practical solution that relaxes common assumptions such as dense image views, accurate camera poses, and substantial image overlaps. We achieve this through identifying and addressing unique challenges arising from the use of pixel-aligned 3DGS: misaligned 3D Gaussians across different views induce noisy or sparse gradients that destabilize training and hinder convergence, especially when above assumptions are not met. To mitigate this, we employ pre-trained monocular depth estimation and visual correspondence models to achieve coarse alignments of 3D Gaussians. We then introduce lightweight, learnable modules to refine depth and pose estimates from the coarse alignments, improving the quality of 3D reconstruction and novel view synthesis. Furthermore, the refined estimates are leveraged to estimate geometry confidence scores, which assess the reliability of 3D Gaussian centers and condition the prediction of Gaussian parameters accordingly. Extensive evaluations on large-scale real-world datasets demonstrate that PF3plat sets a new state-of-the-art across all benchmarks, supported by comprehensive ablation studies validating our design choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22070v1">FreeGaussian: Guidance-free Controllable 3D Gaussian Splats with Flow Derivatives</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Reconstructing controllable Gaussian splats from monocular video is a challenging task due to its inherently insufficient constraints. Widely adopted approaches supervise complex interactions with additional masks and control signal annotations, limiting their real-world applications. In this paper, we propose an annotation guidance-free method, dubbed FreeGaussian, that mathematically derives dynamic Gaussian motion from optical flow and camera motion using novel dynamic Gaussian constraints. By establishing a connection between 2D flows and 3D Gaussian dynamic control, our method enables self-supervised optimization and continuity of dynamic Gaussian motions from flow priors. Furthermore, we introduce a 3D spherical vector controlling scheme, which represents the state with a 3D Gaussian trajectory, thereby eliminating the need for complex 1D control signal calculations and simplifying controllable Gaussian modeling. Quantitative and qualitative evaluations on extensive experiments demonstrate the state-of-the-art visual performance and control capability of our method. Project page: https://freegaussian.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13943v2">DOGS: Distributed-Oriented Gaussian Splatting for Large-Scale 3D Reconstruction Via Gaussian Consensus</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 Accepted to NeurIPS 2024
    </div>
    <details class="paper-abstract">
      The recent advances in 3D Gaussian Splatting (3DGS) show promising results on the novel view synthesis (NVS) task. With its superior rendering performance and high-fidelity rendering quality, 3DGS is excelling at its previous NeRF counterparts. The most recent 3DGS method focuses either on improving the instability of rendering efficiency or reducing the model size. On the other hand, the training efficiency of 3DGS on large-scale scenes has not gained much attention. In this work, we propose DoGaussian, a method that trains 3DGS distributedly. Our method first decomposes a scene into K blocks and then introduces the Alternating Direction Method of Multipliers (ADMM) into the training procedure of 3DGS. During training, our DOGS maintains one global 3DGS model on the master node and K local 3DGS models on the slave nodes. The K local 3DGS models are dropped after training and we only query the global 3DGS model during inference. The training time is reduced by scene decomposition, and the training convergence and stability are guaranteed through the consensus on the shared 3D Gaussians. Our method accelerates the training of 3DGS by 6+ times when evaluated on large-scale scenes while concurrently achieving state-of-the-art rendering quality. Our code is publicly available at https://github.com/AIBluefisher/DOGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21955v1">ActiveSplat: High-Fidelity Scene Reconstruction through Active Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      We propose ActiveSplat, an autonomous high-fidelity reconstruction system leveraging Gaussian splatting. Taking advantage of efficient and realistic rendering, the system establishes a unified framework for online mapping, viewpoint selection, and path planning. The key to ActiveSplat is a hybrid map representation that integrates both dense information about the environment and a sparse abstraction of the workspace. Therefore, the system leverages sparse topology for efficient viewpoint sampling and path planning, while exploiting view-dependent dense prediction for viewpoint selection, facilitating efficient decision-making with promising accuracy and completeness. A hierarchical planning strategy based on the topological map is adopted to mitigate repetitive trajectories and improve local granularity given limited budgets, ensuring high-fidelity reconstruction with photorealistic view synthesis. Extensive experiments and ablation studies validate the efficacy of the proposed method in terms of reconstruction accuracy, data coverage, and exploration efficiency. Project page: https://li-yuetao.github.io/ActiveSplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00434v2">MoDGS: Dynamic Gaussian Splatting from Casually-captured Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 project page: https://modgs.github.io
    </div>
    <details class="paper-abstract">
      In this paper, we propose MoDGS, a new pipeline to render novel views of dy namic scenes from a casually captured monocular video. Previous monocular dynamic NeRF or Gaussian Splatting methods strongly rely on the rapid move ment of input cameras to construct multiview consistency but struggle to recon struct dynamic scenes on casually captured input videos whose cameras are either static or move slowly. To address this challenging task, MoDGS adopts recent single-view depth estimation methods to guide the learning of the dynamic scene. Then, a novel 3D-aware initialization method is proposed to learn a reasonable deformation field and a new robust depth loss is proposed to guide the learning of dynamic scene geometry. Comprehensive experiments demonstrate that MoDGS is able to render high-quality novel view images of dynamic scenes from just a casually captured monocular video, which outperforms state-of-the-art meth ods by a significant margin. The code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17958v3">FreeSplat: Generalizable 3D Gaussian Splatting Towards Free-View Synthesis of Indoor Scenes</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Empowering 3D Gaussian Splatting with generalization ability is appealing. However, existing generalizable 3D Gaussian Splatting methods are largely confined to narrow-range interpolation between stereo images due to their heavy backbones, thus lacking the ability to accurately localize 3D Gaussian and support free-view synthesis across wide view range. In this paper, we present a novel framework FreeSplat that is capable of reconstructing geometrically consistent 3D scenes from long sequence input towards free-view synthesis.Specifically, we firstly introduce Low-cost Cross-View Aggregation achieved by constructing adaptive cost volumes among nearby views and aggregating features using a multi-scale structure. Subsequently, we present the Pixel-wise Triplet Fusion to eliminate redundancy of 3D Gaussians in overlapping view regions and to aggregate features observed across multiple views. Additionally, we propose a simple but effective free-view training strategy that ensures robust view synthesis across broader view range regardless of the number of views. Our empirical results demonstrate state-of-the-art novel view synthesis peformances in both novel view rendered color maps quality and depth maps accuracy across different numbers of input views. We also show that FreeSplat performs inference more efficiently and can effectively reduce redundant Gaussians, offering the possibility of feed-forward large scene reconstruction without depth priors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21566v1">MVSDet: Multi-View Indoor 3D Object Detection via Efficient Plane Sweeps</a></div>
    <div class="paper-meta">
      📅 2024-10-28
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      The key challenge of multi-view indoor 3D object detection is to infer accurate geometry information from images for precise 3D detection. Previous method relies on NeRF for geometry reasoning. However, the geometry extracted from NeRF is generally inaccurate, which leads to sub-optimal detection performance. In this paper, we propose MVSDet which utilizes plane sweep for geometry-aware 3D object detection. To circumvent the requirement for a large number of depth planes for accurate depth prediction, we design a probabilistic sampling and soft weighting mechanism to decide the placement of pixel features on the 3D volume. We select multiple locations that score top in the probability volume for each pixel and use their probability score to indicate the confidence. We further apply recent pixel-aligned Gaussian Splatting to regularize depth prediction and improve detection performance with little computation overhead. Extensive experiments on ScanNet and ARKitScenes datasets are conducted to show the superiority of our model. Our code is available at https://github.com/Pixie8888/MVSDet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20789v1">LoDAvatar: Hierarchical Embedding and Adaptive Levels of Detail with Gaussian Splatting for Enhanced Human Avatars</a></div>
    <div class="paper-meta">
      📅 2024-10-28
      | 💬 9 pages, 7 figures, submitted to IEEE VR 2025
    </div>
    <details class="paper-abstract">
      With the advancement of virtual reality, the demand for 3D human avatars is increasing. The emergence of Gaussian Splatting technology has enabled the rendering of Gaussian avatars with superior visual quality and reduced computational costs. Despite numerous methods researchers propose for implementing drivable Gaussian avatars, limited attention has been given to balancing visual quality and computational costs. In this paper, we introduce LoDAvatar, a method that introduces levels of detail into Gaussian avatars through hierarchical embedding and selective detail enhancement methods. The key steps of LoDAvatar encompass data preparation, Gaussian embedding, Gaussian optimization, and selective detail enhancement. We conducted experiments involving Gaussian avatars at various levels of detail, employing both objective assessments and subjective evaluations. The outcomes indicate that incorporating levels of detail into Gaussian avatars can decrease computational costs during rendering while upholding commendable visual quality, thereby enhancing runtime frame rates. We advocate adopting LoDAvatar to render multiple dynamic Gaussian avatars or extensive Gaussian scenes to balance visual quality and computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20723v1">CompGS: Unleashing 2D Compositionality for Compositional Text-to-3D via Dynamically Optimizing 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-10-28
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in text-guided image generation have significantly advanced the field of 3D generation. While generating a single high-quality 3D object is now feasible, generating multiple objects with reasonable interactions within a 3D space, a.k.a. compositional 3D generation, presents substantial challenges. This paper introduces CompGS, a novel generative framework that employs 3D Gaussian Splatting (GS) for efficient, compositional text-to-3D content generation. To achieve this goal, two core designs are proposed: (1) 3D Gaussians Initialization with 2D compositionality: We transfer the well-established 2D compositionality to initialize the Gaussian parameters on an entity-by-entity basis, ensuring both consistent 3D priors for each entity and reasonable interactions among multiple entities; (2) Dynamic Optimization: We propose a dynamic strategy to optimize 3D Gaussians using Score Distillation Sampling (SDS) loss. CompGS first automatically decomposes 3D Gaussians into distinct entity parts, enabling optimization at both the entity and composition levels. Additionally, CompGS optimizes across objects of varying scales by dynamically adjusting the spatial parameters of each entity, enhancing the generation of fine-grained details, particularly in smaller entities. Qualitative comparisons and quantitative evaluations on T3Bench demonstrate the effectiveness of CompGS in generating compositional 3D objects with superior image quality and semantic alignment over existing methods. CompGS can also be easily extended to controllable 3D editing, facilitating scene generation. We hope CompGS will provide new insights to the compositional 3D generation. Project page: https://chongjiange.github.io/compgs.html.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20686v1">ODGS: 3D Scene Reconstruction from Omnidirectional Images with 3D Gaussian Splattings</a></div>
    <div class="paper-meta">
      📅 2024-10-28
    </div>
    <details class="paper-abstract">
      Omnidirectional (or 360-degree) images are increasingly being used for 3D applications since they allow the rendering of an entire scene with a single image. Existing works based on neural radiance fields demonstrate successful 3D reconstruction quality on egocentric videos, yet they suffer from long training and rendering times. Recently, 3D Gaussian splatting has gained attention for its fast optimization and real-time rendering. However, directly using a perspective rasterizer to omnidirectional images results in severe distortion due to the different optical properties between two image domains. In this work, we present ODGS, a novel rasterization pipeline for omnidirectional images, with geometric interpretation. For each Gaussian, we define a tangent plane that touches the unit sphere and is perpendicular to the ray headed toward the Gaussian center. We then leverage a perspective camera rasterizer to project the Gaussian onto the corresponding tangent plane. The projected Gaussians are transformed and combined into the omnidirectional image, finalizing the omnidirectional rasterization process. This interpretation reveals the implicit assumptions within the proposed pipeline, which we verify through mathematical proofs. The entire rasterization process is parallelized using CUDA, achieving optimization and rendering speeds 100 times faster than NeRF-based methods. Our comprehensive experiments highlight the superiority of ODGS by delivering the best reconstruction and perceptual quality across various datasets. Additionally, results on roaming datasets demonstrate that ODGS restores fine details effectively, even when reconstructing large 3D scenes. The source code is available on our project page (https://github.com/esw0116/ODGS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20593v1">Normal-GS: 3D Gaussian Splatting with Normal-Involved Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-27
      | 💬 9 pages, 5 figures, accepted at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Rendering and reconstruction are long-standing topics in computer vision and graphics. Achieving both high rendering quality and accurate geometry is a challenge. Recent advancements in 3D Gaussian Splatting (3DGS) have enabled high-fidelity novel view synthesis at real-time speeds. However, the noisy and discrete nature of 3D Gaussian primitives hinders accurate surface estimation. Previous attempts to regularize 3D Gaussian normals often degrade rendering quality due to the fundamental disconnect between normal vectors and the rendering pipeline in 3DGS-based methods. Therefore, we introduce Normal-GS, a novel approach that integrates normal vectors into the 3DGS rendering pipeline. The core idea is to model the interaction between normals and incident lighting using the physically-based rendering equation. Our approach re-parameterizes surface colors as the product of normals and a designed Integrated Directional Illumination Vector (IDIV). To optimize memory usage and simplify optimization, we employ an anchor-based 3DGS to implicitly encode locally-shared IDIVs. Additionally, Normal-GS leverages optimized normals and Integrated Directional Encoding (IDE) to accurately model specular effects, enhancing both rendering quality and surface normal precision. Extensive experiments demonstrate that Normal-GS achieves near state-of-the-art visual quality while obtaining accurate surface normals and preserving real-time rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20693v2">R$^2$-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-10-27
      | 💬 Accepted to NeurIPS 2024. Project page: https://github.com/Ruyi-Zha/r2_gaussian
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has shown promising results in image rendering and surface reconstruction. However, its potential in volumetric reconstruction tasks, such as X-ray computed tomography, remains under-explored. This paper introduces R$^2$-Gaussian, the first 3DGS-based framework for sparse-view tomographic reconstruction. By carefully deriving X-ray rasterization functions, we discover a previously unknown integration bias in the standard 3DGS formulation, which hampers accurate volume retrieval. To address this issue, we propose a novel rectification technique via refactoring the projection from 3D to 2D Gaussians. Our new method presents three key innovations: (1) introducing tailored Gaussian kernels, (2) extending rasterization to X-ray imaging, and (3) developing a CUDA-based differentiable voxelizer. Experiments on synthetic and real-world datasets demonstrate that our method outperforms state-of-the-art approaches in accuracy and efficiency. Crucially, it delivers high-quality results in 4 minutes, which is 12$\times$ faster than NeRF-based methods and on par with traditional algorithms. Code and models are available on the project page https://github.com/Ruyi-Zha/r2_gaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18822v2">Binocular-Guided 3D Gaussian Splatting with View Consistency for Sparse View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-10-27
      | 💬 Accepted by NeurIPS 2024. Project page: https://hanl2010.github.io/Binocular3DGS/
    </div>
    <details class="paper-abstract">
      Novel view synthesis from sparse inputs is a vital yet challenging task in 3D computer vision. Previous methods explore 3D Gaussian Splatting with neural priors (e.g. depth priors) as an additional supervision, demonstrating promising quality and efficiency compared to the NeRF based methods. However, the neural priors from 2D pretrained models are often noisy and blurry, which struggle to precisely guide the learning of radiance fields. In this paper, We propose a novel method for synthesizing novel views from sparse views with Gaussian Splatting that does not require external prior as supervision. Our key idea lies in exploring the self-supervisions inherent in the binocular stereo consistency between each pair of binocular images constructed with disparity-guided image warping. To this end, we additionally introduce a Gaussian opacity constraint which regularizes the Gaussian locations and avoids Gaussian redundancy for improving the robustness and efficiency of inferring 3D Gaussians from sparse views. Extensive experiments on the LLFF, DTU, and Blender datasets demonstrate that our method significantly outperforms the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20220v1">Neural Fields in Robotics: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-10-26
      | 💬 20 pages, 20 figures. Project Page: https://robonerf.github.io
    </div>
    <details class="paper-abstract">
      Neural Fields have emerged as a transformative approach for 3D scene representation in computer vision and robotics, enabling accurate inference of geometry, 3D semantics, and dynamics from posed 2D data. Leveraging differentiable rendering, Neural Fields encompass both continuous implicit and explicit neural representations enabling high-fidelity 3D reconstruction, integration of multi-modal sensor data, and generation of novel viewpoints. This survey explores their applications in robotics, emphasizing their potential to enhance perception, planning, and control. Their compactness, memory efficiency, and differentiability, along with seamless integration with foundation and generative models, make them ideal for real-time applications, improving robot adaptability and decision-making. This paper provides a thorough review of Neural Fields in robotics, categorizing applications across various domains and evaluating their strengths and limitations, based on over 200 papers. First, we present four key Neural Fields frameworks: Occupancy Networks, Signed Distance Fields, Neural Radiance Fields, and Gaussian Splatting. Second, we detail Neural Fields' applications in five major robotics domains: pose estimation, manipulation, navigation, physics, and autonomous driving, highlighting key works and discussing takeaways and open challenges. Finally, we outline the current limitations of Neural Fields in robotics and propose promising directions for future research. Project page: https://robonerf.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15125v4">HDR-GS: Efficient High Dynamic Range Novel View Synthesis at 1000x Speed via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-26
      | 💬 NeurIPS 2024; The first 3D Gaussian Splatting-based method for HDR imaging
    </div>
    <details class="paper-abstract">
      High dynamic range (HDR) novel view synthesis (NVS) aims to create photorealistic images from novel viewpoints using HDR imaging techniques. The rendered HDR images capture a wider range of brightness levels containing more details of the scene than normal low dynamic range (LDR) images. Existing HDR NVS methods are mainly based on NeRF. They suffer from long training time and slow inference speed. In this paper, we propose a new framework, High Dynamic Range Gaussian Splatting (HDR-GS), which can efficiently render novel HDR views and reconstruct LDR images with a user input exposure time. Specifically, we design a Dual Dynamic Range (DDR) Gaussian point cloud model that uses spherical harmonics to fit HDR color and employs an MLP-based tone-mapper to render LDR color. The HDR and LDR colors are then fed into two Parallel Differentiable Rasterization (PDR) processes to reconstruct HDR and LDR views. To establish the data foundation for the research of 3D Gaussian splatting-based methods in HDR NVS, we recalibrate the camera parameters and compute the initial positions for Gaussian point clouds. Experiments demonstrate that our HDR-GS surpasses the state-of-the-art NeRF-based method by 3.84 and 1.91 dB on LDR and HDR NVS while enjoying 1000x inference speed and only requiring 6.3% training time. Code and recalibrated data will be publicly available at https://github.com/caiyuanhao1998/HDR-GS . A brief video introduction of our work is available at https://youtu.be/wtU7Kcwe7ck
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04116v3">Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-10-26
      | 💬 ECCV 2024; The first 3D Gaussian Splatting-based method for X-ray 3D reconstruction
    </div>
    <details class="paper-abstract">
      X-ray is widely applied for transmission imaging due to its stronger penetration than natural light. When rendering novel view X-ray projections, existing methods mainly based on NeRF suffer from long training time and slow inference speed. In this paper, we propose a 3D Gaussian splatting-based framework, namely X-Gaussian, for X-ray novel view synthesis. Firstly, we redesign a radiative Gaussian point cloud model inspired by the isotropic nature of X-ray imaging. Our model excludes the influence of view direction when learning to predict the radiation intensity of 3D points. Based on this model, we develop a Differentiable Radiative Rasterization (DRR) with CUDA implementation. Secondly, we customize an Angle-pose Cuboid Uniform Initialization (ACUI) strategy that directly uses the parameters of the X-ray scanner to compute the camera information and then uniformly samples point positions within a cuboid enclosing the scanned object. Experiments show that our X-Gaussian outperforms state-of-the-art methods by 6.5 dB while enjoying less than 15% training time and over 73x inference speed. The application on sparse-view CT reconstruction also reveals the practical values of our method. Code is publicly available at https://github.com/caiyuanhao1998/X-Gaussian . A video demo of the training process visualization is at https://www.youtube.com/watch?v=gDVf_Ngeghg .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19564v1">Robotic Learning in your Backyard: A Neural Simulator from Open Source Components</a></div>
    <div class="paper-meta">
      📅 2024-10-25
      | 💬 Accepted for Oral Presentation at IEEE International Conference on Robotic Computing (IRC)
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting for fast and high-quality novel view synthesize has opened up the possibility to construct photo-realistic simulations from video for robotic reinforcement learning. While the approach has been demonstrated in several research papers, the software tools used to build such a simulator remain unavailable or proprietary. We present SplatGym, an open source neural simulator for training data-driven robotic control policies. The simulator creates a photorealistic virtual environment from a single video. It supports ego camera view generation, collision detection, and virtual object in-painting. We demonstrate training several visual navigation policies via reinforcement learning. SplatGym represents a notable first step towards an open-source general-purpose neural environment for robotic learning. It broadens the range of applications that can effectively utilise reinforcement learning by providing convenient and unrestricted tooling, and by eliminating the need for the manual development of conventional 3D environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19483v1">Content-Aware Radiance Fields: Aligning Model Complexity with Scene Intricacy Through Learned Bitwidth Quantization</a></div>
    <div class="paper-meta">
      📅 2024-10-25
      | 💬 accepted by ECCV2024
    </div>
    <details class="paper-abstract">
      The recent popular radiance field models, exemplified by Neural Radiance Fields (NeRF), Instant-NGP and 3D Gaussian Splatting, are designed to represent 3D content by that training models for each individual scene. This unique characteristic of scene representation and per-scene training distinguishes radiance field models from other neural models, because complex scenes necessitate models with higher representational capacity and vice versa. In this paper, we propose content-aware radiance fields, aligning the model complexity with the scene intricacies through Adversarial Content-Aware Quantization (A-CAQ). Specifically, we make the bitwidth of parameters differentiable and trainable, tailored to the unique characteristics of specific scenes and requirements. The proposed framework has been assessed on Instant-NGP, a well-known NeRF variant and evaluated using various datasets. Experimental results demonstrate a notable reduction in computational complexity, while preserving the requisite reconstruction and rendering quality, making it beneficial for practical deployment of radiance fields models. Codes are available at https://github.com/WeihangLiu2024/Content_Aware_NeRF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21310v1">ArCSEM: Artistic Colorization of SEM Images via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-25
      | 💬 presented and published at AI for Visual Arts Workshop and Challenges (AI4VA) in conjunction with European Conference on Computer Vision (ECCV) 2024, Milano, Italy
    </div>
    <details class="paper-abstract">
      Scanning Electron Microscopes (SEMs) are widely renowned for their ability to analyze the surface structures of microscopic objects, offering the capability to capture highly detailed, yet only grayscale, images. To create more expressive and realistic illustrations, these images are typically manually colorized by an artist with the support of image editing software. This task becomes highly laborious when multiple images of a scanned object require colorization. We propose facilitating this process by using the underlying 3D structure of the microscopic scene to propagate the color information to all the captured images, from as little as one colorized view. We explore several scene representation techniques and achieve high-quality colorized novel view synthesis of a SEM scene. In contrast to prior work, there is no manual intervention or labelling involved in obtaining the 3D representation. This enables an artist to color a single or few views of a sequence and automatically retrieve a fully colored scene or video. Project page: https://ronly2460.github.io/ArCSEM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18974v1">3D-Adapter: Geometry-Consistent Multi-View Diffusion for High-Quality 3D Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 Project page: https://lakonik.github.io/3d-adapter/
    </div>
    <details class="paper-abstract">
      Multi-view image diffusion models have significantly advanced open-domain 3D object generation. However, most existing models rely on 2D network architectures that lack inherent 3D biases, resulting in compromised geometric consistency. To address this challenge, we introduce 3D-Adapter, a plug-in module designed to infuse 3D geometry awareness into pretrained image diffusion models. Central to our approach is the idea of 3D feedback augmentation: for each denoising step in the sampling loop, 3D-Adapter decodes intermediate multi-view features into a coherent 3D representation, then re-encodes the rendered RGBD views to augment the pretrained base model through feature addition. We study two variants of 3D-Adapter: a fast feed-forward version based on Gaussian splatting and a versatile training-free version utilizing neural fields and meshes. Our extensive experiments demonstrate that 3D-Adapter not only greatly enhances the geometry quality of text-to-multi-view models such as Instant3D and Zero123++, but also enables high-quality 3D generation using the plain text-to-image Stable Diffusion. Furthermore, we showcase the broad application potential of 3D-Adapter by presenting high quality results in text-to-3D, image-to-3D, text-to-texture, and text-to-avatar tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18931v1">Sort-free Gaussian Splatting via Weighted Sum Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-24
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has emerged as a significant advancement in 3D scene reconstruction, attracting considerable attention due to its ability to recover high-fidelity details while maintaining low complexity. Despite the promising results achieved by 3DGS, its rendering performance is constrained by its dependence on costly non-commutative alpha-blending operations. These operations mandate complex view dependent sorting operations that introduce computational overhead, especially on the resource-constrained platforms such as mobile phones. In this paper, we propose Weighted Sum Rendering, which approximates alpha blending with weighted sums, thereby removing the need for sorting. This simplifies implementation, delivers superior performance, and eliminates the "popping" artifacts caused by sorting. Experimental results show that optimizing a generalized Gaussian splatting formulation to the new differentiable rendering yields competitive image quality. The method was implemented and tested in a mobile device GPU, achieving on average $1.23\times$ faster rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18912v1">Dynamic 3D Gaussian Tracking for Graph-Based Neural Dynamics Modeling</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 Project Page: https://gs-dynamics.github.io
    </div>
    <details class="paper-abstract">
      Videos of robots interacting with objects encode rich information about the objects' dynamics. However, existing video prediction approaches typically do not explicitly account for the 3D information from videos, such as robot actions and objects' 3D states, limiting their use in real-world robotic applications. In this work, we introduce a framework to learn object dynamics directly from multi-view RGB videos by explicitly considering the robot's action trajectories and their effects on scene dynamics. We utilize the 3D Gaussian representation of 3D Gaussian Splatting (3DGS) to train a particle-based dynamics model using Graph Neural Networks. This model operates on sparse control particles downsampled from the densely tracked 3D Gaussian reconstructions. By learning the neural dynamics model on offline robot interaction data, our method can predict object motions under varying initial configurations and unseen robot actions. The 3D transformations of Gaussians can be interpolated from the motions of control particles, enabling the rendering of predicted future object states and achieving action-conditioned video prediction. The dynamics model can also be applied to model-based planning frameworks for object manipulation tasks. We conduct experiments on various kinds of deformable materials, including ropes, clothes, and stuffed animals, demonstrating our framework's ability to model complex shapes and dynamics. Our project page is available at https://gs-dynamics.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13607v2">DN-4DGS: Denoised Deformable Network with Temporal-Spatial Aggregation for Dynamic Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Dynamic scenes rendering is an intriguing yet challenging problem. Although current methods based on NeRF have achieved satisfactory performance, they still can not reach real-time levels. Recently, 3D Gaussian Splatting (3DGS) has garnered researchers attention due to their outstanding rendering quality and real-time speed. Therefore, a new paradigm has been proposed: defining a canonical 3D gaussians and deforming it to individual frames in deformable fields. However, since the coordinates of canonical 3D gaussians are filled with noise, which can transfer noise into the deformable fields, and there is currently no method that adequately considers the aggregation of 4D information. Therefore, we propose Denoised Deformable Network with Temporal-Spatial Aggregation for Dynamic Scene Rendering (DN-4DGS). Specifically, a Noise Suppression Strategy is introduced to change the distribution of the coordinates of the canonical 3D gaussians and suppress noise. Additionally, a Decoupled Temporal-Spatial Aggregation Module is designed to aggregate information from adjacent points and frames. Extensive experiments on various real-world datasets demonstrate that our method achieves state-of-the-art rendering quality under a real-time level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17932v1">VR-Splatting: Foveated Radiance Field Rendering via 3D Gaussian Splatting and Neural Points</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      Recent advances in novel view synthesis (NVS), particularly neural radiance fields (NeRF) and Gaussian splatting (3DGS), have demonstrated impressive results in photorealistic scene rendering. These techniques hold great potential for applications in virtual tourism and teleportation, where immersive realism is crucial. However, the high-performance demands of virtual reality (VR) systems present challenges in directly utilizing even such fast-to-render scene representations like 3DGS due to latency and computational constraints. In this paper, we propose foveated rendering as a promising solution to these obstacles. We analyze state-of-the-art NVS methods with respect to their rendering performance and compatibility with the human visual system. Our approach introduces a novel foveated rendering approach for Virtual Reality, that leverages the sharp, detailed output of neural point rendering for the foveal region, fused with a smooth rendering of 3DGS for the peripheral vision. Our evaluation confirms that perceived sharpness and detail-richness are increased by our approach compared to a standard VR-ready 3DGS configuration. Our system meets the necessary performance requirements for real-time VR interactions, ultimately enhancing the user's immersive experience. Project page: https://lfranke.github.io/vr_splatting
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17505v1">PLGS: Robust Panoptic Lifting with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      Previous methods utilize the Neural Radiance Field (NeRF) for panoptic lifting, while their training and rendering speed are unsatisfactory. In contrast, 3D Gaussian Splatting (3DGS) has emerged as a prominent technique due to its rapid training and rendering speed. However, unlike NeRF, the conventional 3DGS may not satisfy the basic smoothness assumption as it does not rely on any parameterized structures to render (e.g., MLPs). Consequently, the conventional 3DGS is, in nature, more susceptible to noisy 2D mask supervision. In this paper, we propose a new method called PLGS that enables 3DGS to generate consistent panoptic segmentation masks from noisy 2D segmentation masks while maintaining superior efficiency compared to NeRF-based methods. Specifically, we build a panoptic-aware structured 3D Gaussian model to introduce smoothness and design effective noise reduction strategies. For the semantic field, instead of initialization with structure from motion, we construct reliable semantic anchor points to initialize the 3D Gaussians. We then use these anchor points as smooth regularization during training. Additionally, we present a self-training approach using pseudo labels generated by merging the rendered masks with the noisy masks to enhance the robustness of PLGS. For the instance field, we project the 2D instance masks into 3D space and match them with oriented bounding boxes to generate cross-view consistent instance masks for supervision. Experiments on various benchmarks demonstrate that our method outperforms previous state-of-the-art methods in terms of both segmentation quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.15629v2">Fully Explicit Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 Accepted at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has shown fast and high-quality rendering results in static scenes by leveraging dense 3D prior and explicit representations. Unfortunately, the benefits of the prior and representation do not involve novel view synthesis for dynamic motions. Ironically, this is because the main barrier is the reliance on them, which requires increasing training and rendering times to account for dynamic motions. In this paper, we design a Explicit 4D Gaussian Splatting(Ex4DGS). Our key idea is to firstly separate static and dynamic Gaussians during training, and to explicitly sample positions and rotations of the dynamic Gaussians at sparse timestamps. The sampled positions and rotations are then interpolated to represent both spatially and temporally continuous motions of objects in dynamic scenes as well as reducing computational cost. Additionally, we introduce a progressive training scheme and a point-backtracking technique that improves Ex4DGS's convergence. We initially train Ex4DGS using short timestamps and progressively extend timestamps, which makes it work well with a few point clouds. The point-backtracking is used to quantify the cumulative error of each Gaussian over time, enabling the detection and removal of erroneous Gaussians in dynamic scenes. Comprehensive experiments on various scenes demonstrate the state-of-the-art rendering quality from our method, achieving fast rendering of 62 fps on a single 2080Ti GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15392v2">EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 Project Page: https://lbh666.github.io/ef-3dgs/
    </div>
    <details class="paper-abstract">
      Scene reconstruction from casually captured videos has wide applications in real-world scenarios. With recent advancements in differentiable rendering techniques, several methods have attempted to simultaneously optimize scene representations (NeRF or 3DGS) and camera poses. Despite recent progress, existing methods relying on traditional camera input tend to fail in high-speed (or equivalently low-frame-rate) scenarios. Event cameras, inspired by biological vision, record pixel-wise intensity changes asynchronously with high temporal resolution, providing valuable scene and motion information in blind inter-frame intervals. In this paper, we introduce the event camera to aid scene construction from a casually captured video for the first time, and propose Event-Aided Free-Trajectory 3DGS, called EF-3DGS, which seamlessly integrates the advantages of event cameras into 3DGS through three key components. First, we leverage the Event Generation Model (EGM) to fuse events and frames, supervising the rendered views observed by the event stream. Second, we adopt the Contrast Maximization (CMax) framework in a piece-wise manner to extract motion information by maximizing the contrast of the Image of Warped Events (IWE), thereby calibrating the estimated poses. Besides, based on the Linear Event Generation Model (LEGM), the brightness information encoded in the IWE is also utilized to constrain the 3DGS in the gradient domain. Third, to mitigate the absence of color information of events, we introduce photometric bundle adjustment (PBA) to ensure view consistency across events and frames. We evaluate our method on the public Tanks and Temples benchmark and a newly collected real-world dataset, RealEv-DAVIS. Our project page is https://lbh666.github.io/ef-3dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16995v1">E-3DGS: Gaussian Splatting with Exposure and Motion Events</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 The source code and dataset will be available at https://github.com/MasterHow/E-3DGS
    </div>
    <details class="paper-abstract">
      Estimating Neural Radiance Fields (NeRFs) from images captured under optimal conditions has been extensively explored in the vision community. However, robotic applications often face challenges such as motion blur, insufficient illumination, and high computational overhead, which adversely affect downstream tasks like navigation, inspection, and scene visualization. To address these challenges, we propose E-3DGS, a novel event-based approach that partitions events into motion (from camera or object movement) and exposure (from camera exposure), using the former to handle fast-motion scenes and using the latter to reconstruct grayscale images for high-quality training and optimization of event-based 3D Gaussian Splatting (3DGS). We introduce a novel integration of 3DGS with exposure events for high-quality reconstruction of explicit scene representations. Our versatile framework can operate on motion events alone for 3D reconstruction, enhance quality using exposure events, or adopt a hybrid mode that balances quality and effectiveness by optimizing with initial exposure events followed by high-speed motion events. We also introduce EME-3D, a real-world 3D dataset with exposure events, motion events, camera calibration parameters, and sparse point clouds. Our method is faster and delivers better reconstruction quality than event-based NeRF while being more cost-effective than NeRF methods that combine event and RGB data by using a single event sensor. By combining motion and exposure events, E-3DGS sets a new benchmark for event-based 3D reconstruction with robust performance in challenging conditions and lower hardware demands. The source code and dataset will be available at https://github.com/MasterHow/E-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16978v1">Multi-Layer Gaussian Splatting for Immersive Anatomy Visualization</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      In medical image visualization, path tracing of volumetric medical data like CT scans produces lifelike three-dimensional visualizations. Immersive VR displays can further enhance the understanding of complex anatomies. Going beyond the diagnostic quality of traditional 2D slices, they enable interactive 3D evaluation of anatomies, supporting medical education and planning. Rendering high-quality visualizations in real-time, however, is computationally intensive and impractical for compute-constrained devices like mobile headsets. We propose a novel approach utilizing GS to create an efficient but static intermediate representation of CT scans. We introduce a layered GS representation, incrementally including different anatomical structures while minimizing overlap and extending the GS training to remove inactive Gaussians. We further compress the created model with clustering across layers. Our approach achieves interactive frame rates while preserving anatomical structures, with quality adjustable to the target hardware. Compared to standard GS, our representation retains some of the explorative qualities initially enabled by immersive path tracing. Selective activation and clipping of layers are possible at rendering time, adding a degree of interactivity to otherwise static GS models. This could enable scenarios where high computational demands would otherwise prohibit using path-traced medical volumes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15629v2">Fully Explicit Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 Accepted at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has shown fast and high-quality rendering results in static scenes by leveraging dense 3D prior and explicit representations. Unfortunately, the benefits of the prior and representation do not involve novel view synthesis for dynamic motions. Ironically, this is because the main barrier is the reliance on them, which requires increasing training and rendering times to account for dynamic motions. In this paper, we design a Explicit 4D Gaussian Splatting(Ex4DGS). Our key idea is to firstly separate static and dynamic Gaussians during training, and to explicitly sample positions and rotations of the dynamic Gaussians at sparse timestamps. The sampled positions and rotations are then interpolated to represent both spatially and temporally continuous motions of objects in dynamic scenes as well as reducing computational cost. Additionally, we introduce a progressive training scheme and a point-backtracking technique that improves Ex4DGS's convergence. We initially train Ex4DGS using short timestamps and progressively extend timestamps, which makes it work well with a few point clouds. The point-backtracking is used to quantify the cumulative error of each Gaussian over time, enabling the detection and removal of erroneous Gaussians in dynamic scenes. Comprehensive experiments on various scenes demonstrate the state-of-the-art rendering quality from our method, achieving fast rendering of 62 fps on a single 2080Ti GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.15629v2">Fully Explicit Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 Accepted at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has shown fast and high-quality rendering results in static scenes by leveraging dense 3D prior and explicit representations. Unfortunately, the benefits of the prior and representation do not involve novel view synthesis for dynamic motions. Ironically, this is because the main barrier is the reliance on them, which requires increasing training and rendering times to account for dynamic motions. In this paper, we design a Explicit 4D Gaussian Splatting(Ex4DGS). Our key idea is to firstly separate static and dynamic Gaussians during training, and to explicitly sample positions and rotations of the dynamic Gaussians at sparse timestamps. The sampled positions and rotations are then interpolated to represent both spatially and temporally continuous motions of objects in dynamic scenes as well as reducing computational cost. Additionally, we introduce a progressive training scheme and a point-backtracking technique that improves Ex4DGS's convergence. We initially train Ex4DGS using short timestamps and progressively extend timestamps, which makes it work well with a few point clouds. The point-backtracking is used to quantify the cumulative error of each Gaussian over time, enabling the detection and removal of erroneous Gaussians in dynamic scenes. Comprehensive experiments on various scenes demonstrate the state-of-the-art rendering quality from our method, achieving fast rendering of 62 fps on a single 2080Ti GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16266v1">3DGS-Enhancer: Enhancing Unbounded 3D Gaussian Splatting with View-consistent 2D Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 Accepted by NeurIPS 2024 Spotlight
    </div>
    <details class="paper-abstract">
      Novel-view synthesis aims to generate novel views of a scene from multiple input images or videos, and recent advancements like 3D Gaussian splatting (3DGS) have achieved notable success in producing photorealistic renderings with efficient pipelines. However, generating high-quality novel views under challenging settings, such as sparse input views, remains difficult due to insufficient information in under-sampled areas, often resulting in noticeable artifacts. This paper presents 3DGS-Enhancer, a novel pipeline for enhancing the representation quality of 3DGS representations. We leverage 2D video diffusion priors to address the challenging 3D view consistency problem, reformulating it as achieving temporal consistency within a video generation process. 3DGS-Enhancer restores view-consistent latent features of rendered novel views and integrates them with the input views through a spatial-temporal decoder. The enhanced views are then used to fine-tune the initial 3DGS model, significantly improving its rendering performance. Extensive experiments on large-scale datasets of unbounded scenes demonstrate that 3DGS-Enhancer yields superior reconstruction performance and high-fidelity rendering results compared to state-of-the-art methods. The project webpage is https://xiliu8006.github.io/3DGS-Enhancer-project .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15730v1">MSGField: A Unified Scene Representation Integrating Motion, Semantics, and Geometry for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      Combining accurate geometry with rich semantics has been proven to be highly effective for language-guided robotic manipulation. Existing methods for dynamic scenes either fail to update in real-time or rely on additional depth sensors for simple scene editing, limiting their applicability in real-world. In this paper, we introduce MSGField, a representation that uses a collection of 2D Gaussians for high-quality reconstruction, further enhanced with attributes to encode semantic and motion information. Specially, we represent the motion field compactly by decomposing each primitive's motion into a combination of a limited set of motion bases. Leveraging the differentiable real-time rendering of Gaussian splatting, we can quickly optimize object motion, even for complex non-rigid motions, with image supervision from only two camera views. Additionally, we designed a pipeline that utilizes object priors to efficiently obtain well-defined semantics. In our challenging dataset, which includes flexible and extremely small objects, our method achieve a success rate of 79.2% in static and 63.3% in dynamic environments for language-guided manipulation. For specified object grasping, we achieve a success rate of 90%, on par with point cloud-based methods. Code and dataset will be released at:https://shengyu724.github.io/MSGField.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01597v2">End-to-End Rate-Distortion Optimized 3D Gaussian Representation</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 ECCV 2024
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become an emerging technique with remarkable potential in 3D representation and image rendering. However, the substantial storage overhead of 3DGS significantly impedes its practical applications. In this work, we formulate the compact 3D Gaussian learning as an end-to-end Rate-Distortion Optimization (RDO) problem and propose RDO-Gaussian that can achieve flexible and continuous rate control. RDO-Gaussian addresses two main issues that exist in current schemes: 1) Different from prior endeavors that minimize the rate under the fixed distortion, we introduce dynamic pruning and entropy-constrained vector quantization (ECVQ) that optimize the rate and distortion at the same time. 2) Previous works treat the colors of each Gaussian equally, while we model the colors of different regions and materials with learnable numbers of parameters. We verify our method on both real and synthetic scenes, showcasing that RDO-Gaussian greatly reduces the size of 3D Gaussian over 40x, and surpasses existing methods in rate-distortion performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2408.03753v1">3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 The 18th European Conference on Computer Vision ECCV 2024
    </div>
    <details class="paper-abstract">
      The use of 3D Gaussians as representation of radiance fields has enabled high quality novel view synthesis at real-time rendering speed. However, the choice of optimising the outgoing radiance of each Gaussian independently as spherical harmonics results in unsatisfactory view dependent effects. In response to these limitations, our work, Factorised Tensorial Illumination for 3D Gaussian Splatting, or 3iGS, improves upon 3D Gaussian Splatting (3DGS) rendering quality. Instead of optimising a single outgoing radiance parameter, 3iGS enhances 3DGS view-dependent effects by expressing the outgoing radiance as a function of a local illumination field and Bidirectional Reflectance Distribution Function (BRDF) features. We optimise a continuous incident illumination field through a Tensorial Factorisation representation, while separately fine-tuning the BRDF features of each 3D Gaussian relative to this illumination field. Our methodology significantly enhances the rendering quality of specular view-dependent effects of 3DGS, while maintaining rapid training and rendering speeds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04354v2">StreetSurfGS: Scalable Urban Street Surface Reconstruction with Planar-based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-19
    </div>
    <details class="paper-abstract">
      Reconstructing urban street scenes is crucial due to its vital role in applications such as autonomous driving and urban planning. These scenes are characterized by long and narrow camera trajectories, occlusion, complex object relationships, and data sparsity across multiple scales. Despite recent advancements, existing surface reconstruction methods, which are primarily designed for object-centric scenarios, struggle to adapt effectively to the unique characteristics of street scenes. To address this challenge, we introduce StreetSurfGS, the first method to employ Gaussian Splatting specifically tailored for scalable urban street scene surface reconstruction. StreetSurfGS utilizes a planar-based octree representation and segmented training to reduce memory costs, accommodate unique camera characteristics, and ensure scalability. Additionally, to mitigate depth inaccuracies caused by object overlap, we propose a guided smoothing strategy within regularization to eliminate inaccurate boundary points and outliers. Furthermore, to address sparse views and multi-scale challenges, we use a dual-step matching strategy that leverages adjacent and long-term information. Extensive experiments validate the efficacy of StreetSurfGS in both novel view synthesis and surface reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17084v1">GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 15 pages, 13 figures
    </div>
    <details class="paper-abstract">
      In this paper, we introduce GS-LIVM, a real-time photo-realistic LiDAR-Inertial-Visual mapping framework with Gaussian Splatting tailored for outdoor scenes. Compared to existing methods based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), our approach enables real-time photo-realistic mapping while ensuring high-quality image rendering in large-scale unbounded outdoor environments. In this work, Gaussian Process Regression (GPR) is employed to mitigate the issues resulting from sparse and unevenly distributed LiDAR observations. The voxel-based 3D Gaussians map representation facilitates real-time dense mapping in large outdoor environments with acceleration governed by custom CUDA kernels. Moreover, the overall framework is designed in a covariance-centered manner, where the estimated covariance is used to initialize the scale and rotation of 3D Gaussians, as well as update the parameters of the GPR. We evaluate our algorithm on several outdoor datasets, and the results demonstrate that our method achieves state-of-the-art performance in terms of mapping efficiency and rendering quality. The source code is available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08107v2">IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 Code Page: https://github.com/wu-cvgl/IncEventGS
    </div>
    <details class="paper-abstract">
      Implicit neural representation and explicit 3D Gaussian Splatting (3D-GS) for novel view synthesis have achieved remarkable progress with frame-based camera (e.g. RGB and RGB-D cameras) recently. Compared to frame-based camera, a novel type of bio-inspired visual sensor, i.e. event camera, has demonstrated advantages in high temporal resolution, high dynamic range, low power consumption and low latency. Due to its unique asynchronous and irregular data capturing process, limited work has been proposed to apply neural representation or 3D Gaussian splatting for an event camera. In this work, we present IncEventGS, an incremental 3D Gaussian Splatting reconstruction algorithm with a single event camera. To recover the 3D scene representation incrementally, we exploit the tracking and mapping paradigm of conventional SLAM pipelines for IncEventGS. Given the incoming event stream, the tracker firstly estimates an initial camera motion based on prior reconstructed 3D-GS scene representation. The mapper then jointly refines both the 3D scene representation and camera motion based on the previously estimated motion trajectory from the tracker. The experimental results demonstrate that IncEventGS delivers superior performance compared to prior NeRF-based methods and other related baselines, even we do not have the ground-truth camera poses. Furthermore, our method can also deliver better performance compared to state-of-the-art event visual odometry methods in terms of camera motion estimation. Code is publicly available at: https://github.com/wu-cvgl/IncEventGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19525v3">MicroDreamer: Efficient 3D Generation in $\sim$20 Seconds by Score-based Iterative Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Optimization-based approaches, such as score distillation sampling (SDS), show promise in zero-shot 3D generation but suffer from low efficiency, primarily due to the high number of function evaluations (NFEs) required for each sample and the limitation of optimization confined to latent space. This paper introduces score-based iterative reconstruction (SIR), an efficient and general algorithm mimicking a differentiable 3D reconstruction process to reduce the NFEs and enable optimization in pixel space. Given a single set of images sampled from a multi-view score-based diffusion model, SIR repeatedly optimizes 3D parameters, unlike the single-step optimization in SDS. With other improvements in training, we present an efficient approach called MicroDreamer that generally applies to various 3D representations and 3D generation tasks. In particular, MicroDreamer is 5-20 times faster than SDS in generating neural radiance field while retaining a comparable performance and takes about 20 seconds to create meshes from 3D Gaussian splatting on a single A100 GPU, halving the time of the fastest optimization-based baseline DreamGaussian with significantly superior performance compared to the measurement standard deviation. Our code is available at https://github.com/ML-GSAI/MicroDreamer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14189v1">Neural Signed Distance Function Inference through Splatting 3D Gaussians Pulled on Zero-Level Set</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 Accepted by NeurIPS 2024. Project page: https://wen-yuan-zhang.github.io/GS-Pull/
    </div>
    <details class="paper-abstract">
      It is vital to infer a signed distance function (SDF) in multi-view based surface reconstruction. 3D Gaussian splatting (3DGS) provides a novel perspective for volume rendering, and shows advantages in rendering efficiency and quality. Although 3DGS provides a promising neural rendering option, it is still hard to infer SDFs for surface reconstruction with 3DGS due to the discreteness, the sparseness, and the off-surface drift of 3D Gaussians. To resolve these issues, we propose a method that seamlessly merge 3DGS with the learning of neural SDFs. Our key idea is to more effectively constrain the SDF inference with the multi-view consistency. To this end, we dynamically align 3D Gaussians on the zero-level set of the neural SDF using neural pulling, and then render the aligned 3D Gaussians through the differentiable rasterization. Meanwhile, we update the neural SDF by pulling neighboring space to the pulled 3D Gaussians, which progressively refine the signed distance field near the surface. With both differentiable pulling and splatting, we jointly optimize 3D Gaussians and the neural SDF with both RGB and geometry constraints, which recovers more accurate, smooth, and complete surfaces with more geometry details. Our numerical and visual comparisons show our superiority over the state-of-the-art results on the widely used benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14169v1">DaRePlane: Direction-aware Representations for Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 arXiv admin note: substantial text overlap with arXiv:2403.02265
    </div>
    <details class="paper-abstract">
      Numerous recent approaches to modeling and re-rendering dynamic scenes leverage plane-based explicit representations, addressing slow training times associated with models like neural radiance fields (NeRF) and Gaussian splatting (GS). However, merely decomposing 4D dynamic scenes into multiple 2D plane-based representations is insufficient for high-fidelity re-rendering of scenes with complex motions. In response, we present DaRePlane, a novel direction-aware representation approach that captures scene dynamics from six different directions. This learned representation undergoes an inverse dual-tree complex wavelet transformation (DTCWT) to recover plane-based information. Within NeRF pipelines, DaRePlane computes features for each space-time point by fusing vectors from these recovered planes, then passed to a tiny MLP for color regression. When applied to Gaussian splatting, DaRePlane computes the features of Gaussian points, followed by a tiny multi-head MLP for spatial-time deformation prediction. Notably, to address redundancy introduced by the six real and six imaginary direction-aware wavelet coefficients, we introduce a trainable masking approach, mitigating storage issues without significant performance decline. To demonstrate the generality and efficiency of DaRePlane, we test it on both regular and surgical dynamic scenes, for both NeRF and GS systems. Extensive experiments show that DaRePlane yields state-of-the-art performance in novel view synthesis for various complex dynamic scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13851v1">Differentiable Robot Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 Project Page: https://drrobot.cs.columbia.edu/
    </div>
    <details class="paper-abstract">
      Vision foundation models trained on massive amounts of visual data have shown unprecedented reasoning and planning skills in open-world settings. A key challenge in applying them to robotic tasks is the modality gap between visual data and action data. We introduce differentiable robot rendering, a method allowing the visual appearance of a robot body to be directly differentiable with respect to its control parameters. Our model integrates a kinematics-aware deformable model and Gaussians Splatting and is compatible with any robot form factors and degrees of freedom. We demonstrate its capability and usage in applications including reconstruction of robot poses from images and controlling robots through vision language models. Quantitative and qualitative results show that our differentiable rendering model provides effective gradients for robotic control directly from pixels, setting the foundation for the future applications of vision foundation models in robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13613v1">MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) has recently emerged as a promising technique for capturing complex dynamic 3D scenes with high fidelity. It utilizes a 4D Gaussian representation and a GPU-friendly rasterizer, enabling rapid rendering speeds. Despite its advantages, 4DGS faces significant challenges, notably the requirement of millions of 4D Gaussians, each with extensive associated attributes, leading to substantial memory and storage cost. This paper introduces a memory-efficient framework for 4DGS. We streamline the color attribute by decomposing it into a per-Gaussian direct color component with only 3 parameters and a shared lightweight alternating current color predictor. This approach eliminates the need for spherical harmonics coefficients, which typically involve up to 144 parameters in classic 4DGS, thereby creating a memory-efficient 4D Gaussian representation. Furthermore, we introduce an entropy-constrained Gaussian deformation technique that uses a deformation field to expand the action range of each Gaussian and integrates an opacity-based entropy loss to limit the number of Gaussians, thus forcing our model to use as few Gaussians as possible to fit a dynamic scene well. With simple half-precision storage and zip compression, our framework achieves a storage reduction by approximately 190$\times$ and 125$\times$ on the Technicolor and Neural 3D Video datasets, respectively, compared to the original 4DGS. Meanwhile, it maintains comparable rendering speeds and scene representation quality, setting a new standard in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13349v1">GlossyGS: Inverse Rendering of Glossy Objects with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Reconstructing objects from posed images is a crucial and complex task in computer graphics and computer vision. While NeRF-based neural reconstruction methods have exhibited impressive reconstruction ability, they tend to be time-comsuming. Recent strategies have adopted 3D Gaussian Splatting (3D-GS) for inverse rendering, which have led to quick and effective outcomes. However, these techniques generally have difficulty in producing believable geometries and materials for glossy objects, a challenge that stems from the inherent ambiguities of inverse rendering. To address this, we introduce GlossyGS, an innovative 3D-GS-based inverse rendering framework that aims to precisely reconstruct the geometry and materials of glossy objects by integrating material priors. The key idea is the use of micro-facet geometry segmentation prior, which helps to reduce the intrinsic ambiguities and improve the decomposition of geometries and materials. Additionally, we introduce a normal map prefiltering strategy to more accurately simulate the normal distribution of reflective surfaces. These strategies are integrated into a hybrid geometry and material representation that employs both explicit and implicit methods to depict glossy objects. We demonstrate through quantitative analysis and qualitative visualization that the proposed method is effective to reconstruct high-fidelity geometries and materials of glossy objects, and performs favorably against state-of-the-arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17898v2">Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 Project page: https://city-super.github.io/octree-gs/
    </div>
    <details class="paper-abstract">
      The recent 3D Gaussian splatting (3D-GS) has shown remarkable rendering fidelity and efficiency compared to NeRF-based neural scene representations. While demonstrating the potential for real-time rendering, 3D-GS encounters rendering bottlenecks in large scenes with complex details due to an excessive number of Gaussian primitives located within the viewing frustum. This limitation is particularly noticeable in zoom-out views and can lead to inconsistent rendering speeds in scenes with varying details. Moreover, it often struggles to capture the corresponding level of details at different scales with its heuristic density control operation. Inspired by the Level-of-Detail (LOD) techniques, we introduce Octree-GS, featuring an LOD-structured 3D Gaussian approach supporting level-of-detail decomposition for scene representation that contributes to the final rendering results. Our model dynamically selects the appropriate level from the set of multi-resolution anchor points, ensuring consistent rendering performance with adaptive LOD adjustments while maintaining high-fidelity rendering results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.14166v3">Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      In this study, we explore the challenge of efficiently representing scenes with a constrained number of Gaussians. Our analysis shifts from traditional graphics and 2D computer vision to the perspective of point clouds, highlighting the inefficient spatial distribution of Gaussian representation as a key limitation in model performance. To address this, we introduce strategies for densification including blur split and depth reinitialization, and simplification through intersection preserving and sampling. These techniques reorganize the spatial positions of the Gaussians, resulting in significant improvements across various datasets and benchmarks in terms of rendering quality, resource consumption, and storage compression. Our Mini-Splatting integrates seamlessly with the original rasterization pipeline, providing a strong baseline for future research in Gaussian-Splatting-based works. \href{https://github.com/fatPeter/mini-splatting}{Code is available}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15149v2">Gaussian Splatting to Real World Flight Navigation Transfer with Liquid Networks</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Simulators are powerful tools for autonomous robot learning as they offer scalable data generation, flexible design, and optimization of trajectories. However, transferring behavior learned from simulation data into the real world proves to be difficult, usually mitigated with compute-heavy domain randomization methods or further model fine-tuning. We present a method to improve generalization and robustness to distribution shifts in sim-to-real visual quadrotor navigation tasks. To this end, we first build a simulator by integrating Gaussian Splatting with quadrotor flight dynamics, and then, train robust navigation policies using Liquid neural networks. In this way, we obtain a full-stack imitation learning protocol that combines advances in 3D Gaussian splatting radiance field rendering, crafty programming of expert demonstration training data, and the task understanding capabilities of Liquid networks. Through a series of quantitative flight tests, we demonstrate the robust transfer of navigation skills learned in a single simulation scene directly to the real world. We further show the ability to maintain performance beyond the training environment under drastic distribution and physical environment changes. Our learned Liquid policies, trained on single target manoeuvres curated from a photorealistic simulated indoor flight only, generalize to multi-step hikes onboard a real hardware platform outdoors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12781v1">Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      We propose Long-LRM, a generalizable 3D Gaussian reconstruction model that is capable of reconstructing a large scene from a long sequence of input images. Specifically, our model can process 32 source images at 960x540 resolution within only 1.3 seconds on a single A100 80G GPU. Our architecture features a mixture of the recent Mamba2 blocks and the classical transformer blocks which allowed many more tokens to be processed than prior work, enhanced by efficient token merging and Gaussian pruning steps that balance between quality and efficiency. Unlike previous feed-forward models that are limited to processing 1~4 input images and can only reconstruct a small portion of a large scene, Long-LRM reconstructs the entire scene in a single feed-forward step. On large-scale scene datasets such as DL3DV-140 and Tanks and Temples, our method achieves performance comparable to optimization-based approaches while being two orders of magnitude more efficient. Project page: https://arthurhero.github.io/projects/llrm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03394v2">Gaussian Primitives for Deformable Image Registration</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Deformable Image Registration (DIR) is essential for aligning medical images that exhibit anatomical variations, facilitating applications such as disease tracking and radiotherapy planning. While classical iterative methods and deep learning approaches have achieved success in DIR, they are often hindered by computational inefficiency or poor generalization. In this paper, we introduce GaussianDIR, a novel, case-specific optimization DIR method inspired by 3D Gaussian splatting. In general, GaussianDIR represents image deformations using a sparse set of mobile and flexible Gaussian primitives, each defined by a center position, covariance, and local rigid transformation. This compact and explicit representation reduces noise and computational overhead while improving interpretability. Furthermore, the movement of individual voxel is derived via blending the local rigid transformation of the neighboring Gaussian primitives. By this, GaussianDIR captures both global smoothness and local rigidity as well as reduces the computational burden. To address varying levels of deformation complexity, GaussianDIR also integrates an adaptive density control mechanism that dynamically adjusts the density of Gaussian primitives. Additionally, we employ multi-scale Gaussian primitives to capture both coarse and fine deformations, reducing optimization to local minima. Experimental results on brain MRI, lung CT, and cardiac MRI datasets demonstrate that GaussianDIR outperforms existing DIR methods in both accuracy and efficiency, highlighting its potential for clinical applications. Finally, as a training-free approach, it challenges the stereotype that iterative methods are inherently slow and transcend the limitations of poor generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.14166v3">Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      In this study, we explore the challenge of efficiently representing scenes with a constrained number of Gaussians. Our analysis shifts from traditional graphics and 2D computer vision to the perspective of point clouds, highlighting the inefficient spatial distribution of Gaussian representation as a key limitation in model performance. To address this, we introduce strategies for densification including blur split and depth reinitialization, and simplification through intersection preserving and sampling. These techniques reorganize the spatial positions of the Gaussians, resulting in significant improvements across various datasets and benchmarks in terms of rendering quality, resource consumption, and storage compression. Our Mini-Splatting integrates seamlessly with the original rasterization pipeline, providing a strong baseline for future research in Gaussian-Splatting-based works. \href{https://github.com/fatPeter/mini-splatting}{Code is available}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.11419v1">GS^3: Efficient Relighting with Triple Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Accepted to SIGGRAPH Asia 2024. Project page: https://gsrelight.github.io/
    </div>
    <details class="paper-abstract">
      We present a spatial and angular Gaussian based representation and a triple splatting process, for real-time, high-quality novel lighting-and-view synthesis from multi-view point-lit input images. To describe complex appearance, we employ a Lambertian plus a mixture of angular Gaussians as an effective reflectance function for each spatial Gaussian. To generate self-shadow, we splat all spatial Gaussians towards the light source to obtain shadow values, which are further refined by a small multi-layer perceptron. To compensate for other effects like global illumination, another network is trained to compute and add a per-spatial-Gaussian RGB tuple. The effectiveness of our representation is demonstrated on 30 samples with a wide variation in geometry (from solid to fluffy) and appearance (from translucent to anisotropic), as well as using different forms of input data, including rendered images of synthetic/reconstructed objects, photographs captured with a handheld camera and a flash, or from a professional lightstage. We achieve a training time of 40-70 minutes and a rendering speed of 90 fps on a single commodity GPU. Our results compare favorably with state-of-the-art techniques in terms of quality/performance. Our code and data are publicly available at https://GSrelight.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.11080v1">Few-shot Novel View Synthesis using Depth Aware 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Presented in ECCV 2024 workshop S3DSGR
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has surpassed neural radiance field methods in novel view synthesis by achieving lower computational costs and real-time high-quality rendering. Although it produces a high-quality rendering with a lot of input views, its performance drops significantly when only a few views are available. In this work, we address this by proposing a depth-aware Gaussian splatting method for few-shot novel view synthesis. We use monocular depth prediction as a prior, along with a scale-invariant depth loss, to constrain the 3D shape under just a few input views. We also model color using lower-order spherical harmonics to avoid overfitting. Further, we observe that removing splats with lower opacity periodically, as performed in the original work, leads to a very sparse point cloud and, hence, a lower-quality rendering. To mitigate this, we retain all the splats, leading to a better reconstruction in a few view settings. Experimental results show that our method outperforms the traditional 3D Gaussian splatting methods by achieving improvements of 10.5% in peak signal-to-noise ratio, 6% in structural similarity index, and 14.1% in perceptual similarity, thereby validating the effectiveness of our approach. The code will be made available at: https://github.com/raja-kumar/depth-aware-3DGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.14166v3">Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      In this study, we explore the challenge of efficiently representing scenes with a constrained number of Gaussians. Our analysis shifts from traditional graphics and 2D computer vision to the perspective of point clouds, highlighting the inefficient spatial distribution of Gaussian representation as a key limitation in model performance. To address this, we introduce strategies for densification including blur split and depth reinitialization, and simplification through intersection preserving and sampling. These techniques reorganize the spatial positions of the Gaussians, resulting in significant improvements across various datasets and benchmarks in terms of rendering quality, resource consumption, and storage compression. Our Mini-Splatting integrates seamlessly with the original rasterization pipeline, providing a strong baseline for future research in Gaussian-Splatting-based works. \href{https://github.com/fatPeter/mini-splatting}{Code is available}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12080v1">SplatPose+: Real-time Image-Based Pose-Agnostic 3D Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Image-based Pose-Agnostic 3D Anomaly Detection is an important task that has emerged in industrial quality control. This task seeks to find anomalies from query images of a tested object given a set of reference images of an anomaly-free object. The challenge is that the query views (a.k.a poses) are unknown and can be different from the reference views. Currently, new methods such as OmniposeAD and SplatPose have emerged to bridge the gap by synthesizing pseudo reference images at the query views for pixel-to-pixel comparison. However, none of these methods can infer in real-time, which is critical in industrial quality control for massive production. For this reason, we propose SplatPose+, which employs a hybrid representation consisting of a Structure from Motion (SfM) model for localization and a 3D Gaussian Splatting (3DGS) model for Novel View Synthesis. Although our proposed pipeline requires the computation of an additional SfM model, it offers real-time inference speeds and faster training compared to SplatPose. Quality-wise, we achieved a new SOTA on the Pose-agnostic Anomaly Detection benchmark with the Multi-Pose Anomaly Detection (MAD-SIM) dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11505v1">LoGS: Visual Localization via Gaussian Splatting with Fewer Training Images</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Visual localization involves estimating a query image's 6-DoF (degrees of freedom) camera pose, which is a fundamental component in various computer vision and robotic tasks. This paper presents LoGS, a vision-based localization pipeline utilizing the 3D Gaussian Splatting (GS) technique as scene representation. This novel representation allows high-quality novel view synthesis. During the mapping phase, structure-from-motion (SfM) is applied first, followed by the generation of a GS map. During localization, the initial position is obtained through image retrieval, local feature matching coupled with a PnP solver, and then a high-precision pose is achieved through the analysis-by-synthesis manner on the GS map. Experimental results on four large-scale datasets demonstrate the proposed approach's SoTA accuracy in estimating camera poses and robustness under challenging few-shot conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10719v2">4-LEGS: 4D Language Embedded Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Project webpage: https://tau-vailab.github.io/4-LEGS/
    </div>
    <details class="paper-abstract">
      The emergence of neural representations has revolutionized our means for digitally viewing a wide range of 3D scenes, enabling the synthesis of photorealistic images rendered from novel views. Recently, several techniques have been proposed for connecting these low-level representations with the high-level semantics understanding embodied within the scene. These methods elevate the rich semantic understanding from 2D imagery to 3D representations, distilling high-dimensional spatial features onto 3D space. In our work, we are interested in connecting language with a dynamic modeling of the world. We show how to lift spatio-temporal features to a 4D representation based on 3D Gaussian Splatting. This enables an interactive interface where the user can spatiotemporally localize events in the video from text prompts. We demonstrate our system on public 3D video datasets of people and animals performing various actions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11419v1">GS^3: Efficient Relighting with Triple Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Accepted to SIGGRAPH Asia 2024. Project page: https://gsrelight.github.io/
    </div>
    <details class="paper-abstract">
      We present a spatial and angular Gaussian based representation and a triple splatting process, for real-time, high-quality novel lighting-and-view synthesis from multi-view point-lit input images. To describe complex appearance, we employ a Lambertian plus a mixture of angular Gaussians as an effective reflectance function for each spatial Gaussian. To generate self-shadow, we splat all spatial Gaussians towards the light source to obtain shadow values, which are further refined by a small multi-layer perceptron. To compensate for other effects like global illumination, another network is trained to compute and add a per-spatial-Gaussian RGB tuple. The effectiveness of our representation is demonstrated on 30 samples with a wide variation in geometry (from solid to fluffy) and appearance (from translucent to anisotropic), as well as using different forms of input data, including rendered images of synthetic/reconstructed objects, photographs captured with a handheld camera and a flash, or from a professional lightstage. We achieve a training time of 40-70 minutes and a rendering speed of 90 fps on a single commodity GPU. Our results compare favorably with state-of-the-art techniques in terms of quality/performance. Our code and data are publicly available at https://GSrelight.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11394v1">MCGS: Multiview Consistency Enhancement for Sparse-View 3D Gaussian Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Radiance fields represented by 3D Gaussians excel at synthesizing novel views, offering both high training efficiency and fast rendering. However, with sparse input views, the lack of multi-view consistency constraints results in poorly initialized point clouds and unreliable heuristics for optimization and densification, leading to suboptimal performance. Existing methods often incorporate depth priors from dense estimation networks but overlook the inherent multi-view consistency in input images. Additionally, they rely on multi-view stereo (MVS)-based initialization, which limits the efficiency of scene representation. To overcome these challenges, we propose a view synthesis framework based on 3D Gaussian Splatting, named MCGS, enabling photorealistic scene reconstruction from sparse input views. The key innovations of MCGS in enhancing multi-view consistency are as follows: i) We introduce an initialization method by leveraging a sparse matcher combined with a random filling strategy, yielding a compact yet sufficient set of initial points. This approach enhances the initial geometry prior, promoting efficient scene representation. ii) We develop a multi-view consistency-guided progressive pruning strategy to refine the Gaussian field by strengthening consistency and eliminating low-contribution Gaussians. These modular, plug-and-play strategies enhance robustness to sparse input views, accelerate rendering, and reduce memory consumption, making MCGS a practical and efficient framework for 3D Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12771v2">Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3D-GS) has achieved impressive results in novel view synthesis, demonstrating high fidelity and efficiency. However, it easily exhibits needle-like artifacts, especially when increasing the sampling rate. Mip-Splatting tries to remove these artifacts with a 3D smoothing filter for frequency constraints and a 2D Mip filter for approximated supersampling. Unfortunately, it tends to produce over-blurred results, and sometimes needle-like Gaussians still persist. Our spectral analysis of the covariance matrix during optimization and densification reveals that current 3D-GS lacks shape awareness, relying instead on spectral radius and view positional gradients to determine splitting. As a result, needle-like Gaussians with small positional gradients and low spectral entropy fail to split and overfit high-frequency details. Furthermore, both the filters used in 3D-GS and Mip-Splatting reduce the spectral entropy and increase the condition number during zooming in to synthesize novel view, causing view inconsistencies and more pronounced artifacts. Our Spectral-GS, based on spectral analysis, introduces 3D shape-aware splitting and 2D view-consistent filtering strategies, effectively addressing these issues, enhancing 3D-GS's capability to represent high-frequency details without noticeable artifacts, and achieving high-quality photorealistic rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11285v1">Scalable Indoor Novel-View Synthesis using Drone-Captured 360 Imagery with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Accepted to ECCV 2024 S3DSGR Workshop
    </div>
    <details class="paper-abstract">
      Scene reconstruction and novel-view synthesis for large, complex, multi-story, indoor scenes is a challenging and time-consuming task. Prior methods have utilized drones for data capture and radiance fields for scene reconstruction, both of which present certain challenges. First, in order to capture diverse viewpoints with the drone's front-facing camera, some approaches fly the drone in an unstable zig-zag fashion, which hinders drone-piloting and generates motion blur in the captured data. Secondly, most radiance field methods do not easily scale to arbitrarily large number of images. This paper proposes an efficient and scalable pipeline for indoor novel-view synthesis from drone-captured 360 videos using 3D Gaussian Splatting. 360 cameras capture a wide set of viewpoints, allowing for comprehensive scene capture under a simple straightforward drone trajectory. To scale our method to large scenes, we devise a divide-and-conquer strategy to automatically split the scene into smaller blocks that can be reconstructed individually and in parallel. We also propose a coarse-to-fine alignment strategy to seamlessly match these blocks together to compose the entire scene. Our experiments demonstrate marked improvement in both reconstruction quality, i.e. PSNR and SSIM, and computation time compared to prior approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.10412v1">4DStyleGaussian: Zero-shot 4D Style Transfer with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      3D neural style transfer has gained significant attention for its potential to provide user-friendly stylization with spatial consistency. However, existing 3D style transfer methods often fall short in terms of inference efficiency, generalization ability, and struggle to handle dynamic scenes with temporal consistency. In this paper, we introduce 4DStyleGaussian, a novel 4D style transfer framework designed to achieve real-time stylization of arbitrary style references while maintaining reasonable content affinity, multi-view consistency, and temporal coherence. Our approach leverages an embedded 4D Gaussian Splatting technique, which is trained using a reversible neural network for reducing content loss in the feature distillation process. Utilizing the 4D embedded Gaussians, we predict a 4D style transformation matrix that facilitates spatially and temporally consistent style transfer with Gaussian Splatting. Experiments demonstrate that our method can achieve high-quality and zero-shot stylization for 4D scenarios with enhanced efficiency and spatial-temporal consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.11419v1">GS^3: Efficient Relighting with Triple Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Accepted to SIGGRAPH Asia 2024. Project page: https://gsrelight.github.io/
    </div>
    <details class="paper-abstract">
      We present a spatial and angular Gaussian based representation and a triple splatting process, for real-time, high-quality novel lighting-and-view synthesis from multi-view point-lit input images. To describe complex appearance, we employ a Lambertian plus a mixture of angular Gaussians as an effective reflectance function for each spatial Gaussian. To generate self-shadow, we splat all spatial Gaussians towards the light source to obtain shadow values, which are further refined by a small multi-layer perceptron. To compensate for other effects like global illumination, another network is trained to compute and add a per-spatial-Gaussian RGB tuple. The effectiveness of our representation is demonstrated on 30 samples with a wide variation in geometry (from solid to fluffy) and appearance (from translucent to anisotropic), as well as using different forms of input data, including rendered images of synthetic/reconstructed objects, photographs captured with a handheld camera and a flash, or from a professional lightstage. We achieve a training time of 40-70 minutes and a rendering speed of 90 fps on a single commodity GPU. Our results compare favorably with state-of-the-art techniques in terms of quality/performance. Our code and data are publicly available at https://GSrelight.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11080v1">Few-shot Novel View Synthesis using Depth Aware 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 Presented in ECCV 2024 workshop S3DSGR
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has surpassed neural radiance field methods in novel view synthesis by achieving lower computational costs and real-time high-quality rendering. Although it produces a high-quality rendering with a lot of input views, its performance drops significantly when only a few views are available. In this work, we address this by proposing a depth-aware Gaussian splatting method for few-shot novel view synthesis. We use monocular depth prediction as a prior, along with a scale-invariant depth loss, to constrain the 3D shape under just a few input views. We also model color using lower-order spherical harmonics to avoid overfitting. Further, we observe that removing splats with lower opacity periodically, as performed in the original work, leads to a very sparse point cloud and, hence, a lower-quality rendering. To mitigate this, we retain all the splats, leading to a better reconstruction in a few view settings. Experimental results show that our method outperforms the traditional 3D Gaussian splatting methods by achieving improvements of 10.5% in peak signal-to-noise ratio, 6% in structural similarity index, and 14.1% in perceptual similarity, thereby validating the effectiveness of our approach. The code will be made available at: https://github.com/raja-kumar/depth-aware-3DGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10412v1">4DStyleGaussian: Zero-shot 4D Style Transfer with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      3D neural style transfer has gained significant attention for its potential to provide user-friendly stylization with spatial consistency. However, existing 3D style transfer methods often fall short in terms of inference efficiency, generalization ability, and struggle to handle dynamic scenes with temporal consistency. In this paper, we introduce 4DStyleGaussian, a novel 4D style transfer framework designed to achieve real-time stylization of arbitrary style references while maintaining reasonable content affinity, multi-view consistency, and temporal coherence. Our approach leverages an embedded 4D Gaussian Splatting technique, which is trained using a reversible neural network for reducing content loss in the feature distillation process. Utilizing the 4D embedded Gaussians, we predict a 4D style transformation matrix that facilitates spatially and temporally consistent style transfer with Gaussian Splatting. Experiments demonstrate that our method can achieve high-quality and zero-shot stylization for 4D scenarios with enhanced efficiency and spatial-temporal consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13711v2">SceneDreamer360: Text-Driven 3D-Consistent Scene Generation with Panoramic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Text-driven 3D scene generation has seen significant advancements recently. However, most existing methods generate single-view images using generative models and then stitch them together in 3D space. This independent generation for each view often results in spatial inconsistency and implausibility in the 3D scenes. To address this challenge, we proposed a novel text-driven 3D-consistent scene generation model: SceneDreamer360. Our proposed method leverages a text-driven panoramic image generation model as a prior for 3D scene generation and employs 3D Gaussian Splatting (3DGS) to ensure consistency across multi-view panoramic images. Specifically, SceneDreamer360 enhances the fine-tuned Panfusion generator with a three-stage panoramic enhancement, enabling the generation of high-resolution, detail-rich panoramic images. During the 3D scene construction, a novel point cloud fusion initialization method is used, producing higher quality and spatially consistent point clouds. Our extensive experiments demonstrate that compared to other methods, SceneDreamer360 with its panoramic image generation and 3DGS can produce higher quality, spatially consistent, and visually appealing 3D scenes from any text prompt. Our codes are available at \url{https://github.com/liwrui/SceneDreamer360}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02972v4">Event3DGS: Event-Based 3D Gaussian Splatting for High-Speed Robot Egomotion</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 In the 8th Annual Conference on Robot Learning (CoRL 2024)
    </div>
    <details class="paper-abstract">
      By combining differentiable rendering with explicit point-based scene representations, 3D Gaussian Splatting (3DGS) has demonstrated breakthrough 3D reconstruction capabilities. However, to date 3DGS has had limited impact on robotics, where high-speed egomotion is pervasive: Egomotion introduces motion blur and leads to artifacts in existing frame-based 3DGS reconstruction methods. To address this challenge, we introduce Event3DGS, an {\em event-based} 3DGS framework. By exploiting the exceptional temporal resolution of event cameras, Event3GDS can reconstruct high-fidelity 3D structure and appearance under high-speed egomotion. Extensive experiments on multiple synthetic and real-world datasets demonstrate the superiority of Event3DGS compared with existing event-based dense 3D scene reconstruction frameworks; Event3DGS substantially improves reconstruction quality (+3dB) while reducing computational costs by 95\%. Our framework also allows one to incorporate a few motion-blurred frame-based measurements into the reconstruction process to further improve appearance fidelity without loss of structural accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.08941v1">MeshGS: Adaptive Mesh-Aligned Gaussian Splatting for High-Quality Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 ACCV (Asian Conference on Computer Vision) 2024
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting has gained attention for its capability to generate high-fidelity rendering results. At the same time, most applications such as games, animation, and AR/VR use mesh-based representations to represent and render 3D scenes. We propose a novel approach that integrates mesh representation with 3D Gaussian splats to perform high-quality rendering of reconstructed real-world scenes. In particular, we introduce a distance-based Gaussian splatting technique to align the Gaussian splats with the mesh surface and remove redundant Gaussian splats that do not contribute to the rendering. We consider the distance between each Gaussian splat and the mesh surface to distinguish between tightly-bound and loosely-bound Gaussian splats. The tightly-bound splats are flattened and aligned well with the mesh geometry. The loosely-bound Gaussian splats are used to account for the artifacts in reconstructed 3D meshes in terms of rendering. We present a training strategy of binding Gaussian splats to the mesh geometry, and take into account both types of splats. In this context, we introduce several regularization techniques aimed at precisely aligning tightly-bound Gaussian splats with the mesh surface during the training process. We validate the effectiveness of our method on large and unbounded scene from mip-NeRF 360 and Deep Blending datasets. Our method surpasses recent mesh-based neural rendering techniques by achieving a 2dB higher PSNR, and outperforms mesh-based Gaussian splatting methods by 1.3 dB PSNR, particularly on the outdoor mip-NeRF 360 dataset, demonstrating better rendering quality. We provide analyses for each type of Gaussian splat and achieve a reduction in the number of Gaussian splats by 30% compared to the original 3D Gaussian splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.11080v1">Few-shot Novel View Synthesis using Depth Aware 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 Presented in ECCV 2024 workshop S3DSGR
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has surpassed neural radiance field methods in novel view synthesis by achieving lower computational costs and real-time high-quality rendering. Although it produces a high-quality rendering with a lot of input views, its performance drops significantly when only a few views are available. In this work, we address this by proposing a depth-aware Gaussian splatting method for few-shot novel view synthesis. We use monocular depth prediction as a prior, along with a scale-invariant depth loss, to constrain the 3D shape under just a few input views. We also model color using lower-order spherical harmonics to avoid overfitting. Further, we observe that removing splats with lower opacity periodically, as performed in the original work, leads to a very sparse point cloud and, hence, a lower-quality rendering. To mitigate this, we retain all the splats, leading to a better reconstruction in a few view settings. Experimental results show that our method outperforms the traditional 3D Gaussian splatting methods by achieving improvements of 10.5% in peak signal-to-noise ratio, 6% in structural similarity index, and 14.1% in perceptual similarity, thereby validating the effectiveness of our approach. The code will be made available at: https://github.com/raja-kumar/depth-aware-3DGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.10412v1">4DStyleGaussian: Zero-shot 4D Style Transfer with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      3D neural style transfer has gained significant attention for its potential to provide user-friendly stylization with spatial consistency. However, existing 3D style transfer methods often fall short in terms of inference efficiency, generalization ability, and struggle to handle dynamic scenes with temporal consistency. In this paper, we introduce 4DStyleGaussian, a novel 4D style transfer framework designed to achieve real-time stylization of arbitrary style references while maintaining reasonable content affinity, multi-view consistency, and temporal coherence. Our approach leverages an embedded 4D Gaussian Splatting technique, which is trained using a reversible neural network for reducing content loss in the feature distillation process. Utilizing the 4D embedded Gaussians, we predict a 4D style transformation matrix that facilitates spatially and temporally consistent style transfer with Gaussian Splatting. Experiments demonstrate that our method can achieve high-quality and zero-shot stylization for 4D scenarios with enhanced efficiency and spatial-temporal consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16964v2">GSDF: 3DGS Meets SDF for Improved Rendering and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 Accepted to NeurIPS 2024. Project page: https://city-super.github.io/GSDF
    </div>
    <details class="paper-abstract">
      Presenting a 3D scene from multiview images remains a core and long-standing challenge in computer vision and computer graphics. Two main requirements lie in rendering and reconstruction. Notably, SOTA rendering quality is usually achieved with neural volumetric rendering techniques, which rely on aggregated point/primitive-wise color and neglect the underlying scene geometry. Learning of neural implicit surfaces is sparked from the success of neural rendering. Current works either constrain the distribution of density fields or the shape of primitives, resulting in degraded rendering quality and flaws on the learned scene surfaces. The efficacy of such methods is limited by the inherent constraints of the chosen neural representation, which struggles to capture fine surface details, especially for larger, more intricate scenes. To address these issues, we introduce GSDF, a novel dual-branch architecture that combines the benefits of a flexible and efficient 3D Gaussian Splatting (3DGS) representation with neural Signed Distance Fields (SDF). The core idea is to leverage and enhance the strengths of each branch while alleviating their limitation through mutual guidance and joint supervision. We show on diverse scenes that our design unlocks the potential for more accurate and detailed surface reconstructions, and at the meantime benefits 3DGS rendering with structures that are more aligned with the underlying geometry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09740v1">Gaussian Splatting Visual MPC for Granular Media Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 project website https://weichengtseng.github.io/gs-granular-mani/
    </div>
    <details class="paper-abstract">
      Recent advancements in learned 3D representations have enabled significant progress in solving complex robotic manipulation tasks, particularly for rigid-body objects. However, manipulating granular materials such as beans, nuts, and rice, remains challenging due to the intricate physics of particle interactions, high-dimensional and partially observable state, inability to visually track individual particles in a pile, and the computational demands of accurate dynamics prediction. Current deep latent dynamics models often struggle to generalize in granular material manipulation due to a lack of inductive biases. In this work, we propose a novel approach that learns a visual dynamics model over Gaussian splatting representations of scenes and leverages this model for manipulating granular media via Model-Predictive Control. Our method enables efficient optimization for complex manipulation tasks on piles of granular media. We evaluate our approach in both simulated and real-world settings, demonstrating its ability to solve unseen planning tasks and generalize to new environments in a zero-shot transfer. We also show significant prediction and manipulation performance improvements compared to existing granular media manipulation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09292v1">SurgicalGS: Dynamic 3D Gaussian Splatting for Accurate Robotic-Assisted Surgical Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      Accurate 3D reconstruction of dynamic surgical scenes from endoscopic video is essential for robotic-assisted surgery. While recent 3D Gaussian Splatting methods have shown promise in achieving high-quality reconstructions with fast rendering speeds, their use of inverse depth loss functions compresses depth variations. This can lead to a loss of fine geometric details, limiting their ability to capture precise 3D geometry and effectiveness in intraoperative application. To address these challenges, we present SurgicalGS, a dynamic 3D Gaussian Splatting framework specifically designed for surgical scene reconstruction with improved geometric accuracy. Our approach first initialises a Gaussian point cloud using depth priors, employing binary motion masks to identify pixels with significant depth variations and fusing point clouds from depth maps across frames for initialisation. We use the Flexible Deformation Model to represent dynamic scene and introduce a normalised depth regularisation loss along with an unsupervised depth smoothness constraint to ensure more accurate geometric reconstruction. Extensive experiments on two real surgical datasets demonstrate that SurgicalGS achieves state-of-the-art reconstruction quality, especially in terms of accurate geometry, advancing the usability of 3D Gaussian Splatting in robotic-assisted surgery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05468v2">PH-Dropout: Practical Epistemic Uncertainty Quantification for View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 21 pages, in submision
    </div>
    <details class="paper-abstract">
      View synthesis using Neural Radiance Fields (NeRF) and Gaussian Splatting (GS) has demonstrated impressive fidelity in rendering real-world scenarios. However, practical methods for accurate and efficient epistemic Uncertainty Quantification (UQ) in view synthesis are lacking. Existing approaches for NeRF either introduce significant computational overhead (e.g., ``10x increase in training time" or ``10x repeated training") or are limited to specific uncertainty conditions or models. Notably, GS models lack any systematic approach for comprehensive epistemic UQ. This capability is crucial for improving the robustness and scalability of neural view synthesis, enabling active model updates, error estimation, and scalable ensemble modeling based on uncertainty. In this paper, we revisit NeRF and GS-based methods from a function approximation perspective, identifying key differences and connections in 3D representation learning. Building on these insights, we introduce PH-Dropout (Post hoc Dropout), the first real-time and accurate method for epistemic uncertainty estimation that operates directly on pre-trained NeRF and GS models. Extensive evaluations validate our theoretical findings and demonstrate the effectiveness of PH-Dropout.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08941v1">MeshGS: Adaptive Mesh-Aligned Gaussian Splatting for High-Quality Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 ACCV (Asian Conference on Computer Vision) 2024
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting has gained attention for its capability to generate high-fidelity rendering results. At the same time, most applications such as games, animation, and AR/VR use mesh-based representations to represent and render 3D scenes. We propose a novel approach that integrates mesh representation with 3D Gaussian splats to perform high-quality rendering of reconstructed real-world scenes. In particular, we introduce a distance-based Gaussian splatting technique to align the Gaussian splats with the mesh surface and remove redundant Gaussian splats that do not contribute to the rendering. We consider the distance between each Gaussian splat and the mesh surface to distinguish between tightly-bound and loosely-bound Gaussian splats. The tightly-bound splats are flattened and aligned well with the mesh geometry. The loosely-bound Gaussian splats are used to account for the artifacts in reconstructed 3D meshes in terms of rendering. We present a training strategy of binding Gaussian splats to the mesh geometry, and take into account both types of splats. In this context, we introduce several regularization techniques aimed at precisely aligning tightly-bound Gaussian splats with the mesh surface during the training process. We validate the effectiveness of our method on large and unbounded scene from mip-NeRF 360 and Deep Blending datasets. Our method surpasses recent mesh-based neural rendering techniques by achieving a 2dB higher PSNR, and outperforms mesh-based Gaussian splatting methods by 1.3 dB PSNR, particularly on the outdoor mip-NeRF 360 dataset, demonstrating better rendering quality. We provide analyses for each type of Gaussian splat and achieve a reduction in the number of Gaussian splats by 30% compared to the original 3D Gaussian splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08017v2">Fast Feedforward 3D Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 Project Page: https://yihangchen-ee.github.io/project_fcgs/ Code: https://github.com/yihangchen-ee/fcgs/
    </div>
    <details class="paper-abstract">
      With 3D Gaussian Splatting (3DGS) advancing real-time and high-fidelity rendering for novel view synthesis, storage requirements pose challenges for their widespread adoption. Although various compression techniques have been proposed, previous art suffers from a common limitation: for any existing 3DGS, per-scene optimization is needed to achieve compression, making the compression sluggish and slow. To address this issue, we introduce Fast Compression of 3D Gaussian Splatting (FCGS), an optimization-free model that can compress 3DGS representations rapidly in a single feed-forward pass, which significantly reduces compression time from minutes to seconds. To enhance compression efficiency, we propose a multi-path entropy module that assigns Gaussian attributes to different entropy constraint paths for balance between size and fidelity. We also carefully design both inter- and intra-Gaussian context models to remove redundancies among the unstructured Gaussian blobs. Overall, FCGS achieves a compression ratio of over 20X while maintaining fidelity, surpassing most per-scene SOTA optimization-based methods. Our code is available at: https://github.com/YihangChen-ee/FCGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08840v1">Learning Interaction-aware 3D Gaussian Splatting for One-shot Hand Avatars</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 Accepted to NeurIPS 2024
    </div>
    <details class="paper-abstract">
      In this paper, we propose to create animatable avatars for interacting hands with 3D Gaussian Splatting (GS) and single-image inputs. Existing GS-based methods designed for single subjects often yield unsatisfactory results due to limited input views, various hand poses, and occlusions. To address these challenges, we introduce a novel two-stage interaction-aware GS framework that exploits cross-subject hand priors and refines 3D Gaussians in interacting areas. Particularly, to handle hand variations, we disentangle the 3D presentation of hands into optimization-based identity maps and learning-based latent geometric features and neural texture maps. Learning-based features are captured by trained networks to provide reliable priors for poses, shapes, and textures, while optimization-based identity maps enable efficient one-shot fitting of out-of-distribution hands. Furthermore, we devise an interaction-aware attention module and a self-adaptive Gaussian refinement module. These modules enhance image rendering quality in areas with intra- and inter-hand interactions, overcoming the limitations of existing GS-based methods. Our proposed method is validated via extensive experiments on the large-scale InterHand2.6M dataset, and it significantly improves the state-of-the-art performance in image quality. Project Page: \url{https://github.com/XuanHuang0/GuassianHand}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08743v1">Look Gauss, No Pose: Novel View Synthesis using Gaussian Splatting without Accurate Pose Initialization</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 Accepted in IROS 2024
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently emerged as a powerful tool for fast and accurate novel-view synthesis from a set of posed input images. However, like most novel-view synthesis approaches, it relies on accurate camera pose information, limiting its applicability in real-world scenarios where acquiring accurate camera poses can be challenging or even impossible. We propose an extension to the 3D Gaussian Splatting framework by optimizing the extrinsic camera parameters with respect to photometric residuals. We derive the analytical gradients and integrate their computation with the existing high-performance CUDA implementation. This enables downstream tasks such as 6-DoF camera pose estimation as well as joint reconstruction and camera refinement. In particular, we achieve rapid convergence and high accuracy for pose estimation on real-world scenes. Our method enables fast reconstruction of 3D scenes without requiring accurate pose information by jointly optimizing geometry and camera poses, while achieving state-of-the-art results in novel-view synthesis. Our approach is considerably faster to optimize than most competing methods, and several times faster in rendering. We show results on real-world scenes and complex trajectories through simulated environments, achieving state-of-the-art results on LLFF while reducing runtime by two to four times compared to the most efficient competing method. Source code will be available at https://github.com/Schmiddo/noposegs .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01579v2">Tetrahedron Splatting for 3D Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 Project page: https://fudan-zvg.github.io/tet-splatting/
    </div>
    <details class="paper-abstract">
      3D representation is essential to the significant advance of 3D generation with 2D diffusion priors. As a flexible representation, NeRF has been first adopted for 3D representation. With density-based volumetric rendering, it however suffers both intensive computational overhead and inaccurate mesh extraction. Using a signed distance field and Marching Tetrahedra, DMTet allows for precise mesh extraction and real-time rendering but is limited in handling large topological changes in meshes, leading to optimization challenges. Alternatively, 3D Gaussian Splatting (3DGS) is favored in both training and rendering efficiency while falling short in mesh extraction. In this work, we introduce a novel 3D representation, Tetrahedron Splatting (TeT-Splatting), that supports easy convergence during optimization, precise mesh extraction, and real-time rendering simultaneously. This is achieved by integrating surface-based volumetric rendering within a structured tetrahedral grid while preserving the desired ability of precise mesh extraction, and a tile-based differentiable tetrahedron rasterizer. Furthermore, we incorporate eikonal and normal consistency regularization terms for the signed distance field to improve generation quality and stability. Critically, our representation can be trained without mesh extraction, making the optimization process easier to converge. Our TeT-Splatting can be readily integrated in existing 3D generation pipelines, along with polygonal mesh for texture optimization. Extensive experiments show that our TeT-Splatting strikes a superior tradeoff among convergence speed, render efficiency, and mesh quality as compared to previous alternatives under varying 3D generation settings.
    </details>
</div>
