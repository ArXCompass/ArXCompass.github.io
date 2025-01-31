# gaussian splatting - 2024_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15491v2">GSDeformer: Direct, Real-time and Extensible Cage-based Deformation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-11
      | 💬 Project Page: https://jhuangbu.github.io/gsdeformer, Video: https://www.youtube.com/watch?v=-ecrj48-MqM
    </div>
    <details class="paper-abstract">
      We present GSDeformer, a method that achieves cage-based deformation on 3D Gaussian Splatting (3DGS). Our method bridges cage-based deformation and 3DGS using a proxy point cloud representation. The point cloud is created from 3DGS, and deformations on the point cloud translate to transformations on the 3D Gaussians that comprise 3DGS. To handle potential bending from deformation, we employ a splitting process to approximate it. Our method does not extend or modify the core architecture of 3DGS; thus, it can work with any existing trained vanilla 3DGS as well as its variants. We also automated cage construction from 3DGS for convenience. Experiments show that GSDeformer produces superior deformation results than current methods, is robust under extreme deformations, does not require retraining for editing, runs in real-time(60FPS), and can extend to other 3DGS variants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04844v2">Discretized Gaussian Representation for Tomographic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-11
    </div>
    <details class="paper-abstract">
      Computed Tomography (CT) is a widely used imaging technique that provides detailed cross-sectional views of objects. Over the past decade, Deep Learning-based Reconstruction (DLR) methods have led efforts to enhance image quality and reduce noise, yet they often require large amounts of data and are computationally intensive. Inspired by recent advancements in scene reconstruction, some approaches have adapted NeRF and 3D Gaussian Splatting (3DGS) techniques for CT reconstruction. However, these methods are not ideal for direct 3D volume reconstruction. In this paper, we reconsider the representation of CT reconstruction and propose a novel Discretized Gaussian Representation (DGR) specifically designed for CT. Unlike the popular 3D Gaussian Splatting, our representation directly reconstructs the 3D volume using a set of discretized Gaussian functions in an end-to-end manner. Additionally, we introduce a Fast Volume Reconstruction technique that efficiently aggregates the contributions of these Gaussians into a discretized volume. Extensive experiments on both real-world and synthetic datasets demonstrate the effectiveness of our method in improving reconstruction quality and computational efficiency. Our code has been provided for review purposes and will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02571v3">SuperGS: Super-Resolution 3D Gaussian Splatting Enhanced by Variational Residual Features and Uncertainty-Augmented Learning</a></div>
    <div class="paper-meta">
      📅 2024-12-11
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has exceled in novel view synthesis (NVS) with its real-time rendering capabilities and superior quality. However, it faces challenges for high-resolution novel view synthesis (HRNVS) due to the coarse nature of primitives derived from low-resolution input views. To address this issue, we propose Super-Resolution 3DGS (SuperGS), which is an expansion of 3DGS designed with a two-stage coarse-to-fine training framework. In this framework, we use a latent feature field to represent the low-resolution scene, serving as both the initialization and foundational information for super-resolution optimization. Additionally, we introduce variational residual features to enhance high-resolution details, using their variance as uncertainty estimates to guide the densification process and loss computation. Furthermore, the introduction of a multi-view joint learning approach helps mitigate ambiguities caused by multi-view inconsistencies in the pseudo labels. Extensive experiments demonstrate that SuperGS surpasses state-of-the-art HRNVS methods on both real-world and synthetic datasets using only low-resolution inputs. Code is available at https://github.com/SYXieee/SuperGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08331v1">SLGaussian: Fast Language Gaussian Splatting in Sparse Views</a></div>
    <div class="paper-meta">
      📅 2024-12-11
    </div>
    <details class="paper-abstract">
      3D semantic field learning is crucial for applications like autonomous navigation, AR/VR, and robotics, where accurate comprehension of 3D scenes from limited viewpoints is essential. Existing methods struggle under sparse view conditions, relying on inefficient per-scene multi-view optimizations, which are impractical for many real-world tasks. To address this, we propose SLGaussian, a feed-forward method for constructing 3D semantic fields from sparse viewpoints, allowing direct inference of 3DGS-based scenes. By ensuring consistent SAM segmentations through video tracking and using low-dimensional indexing for high-dimensional CLIP features, SLGaussian efficiently embeds language information in 3D space, offering a robust solution for accurate 3D scene understanding under sparse view conditions. In experiments on two-view sparse 3D object querying and segmentation in the LERF and 3D-OVS datasets, SLGaussian outperforms existing methods in chosen IoU, Localization Accuracy, and mIoU. Moreover, our model achieves scene inference in under 30 seconds and open-vocabulary querying in just 0.011 seconds per query.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09648v1">DSplats: 3D Generation by Denoising Splats-Based Multiview Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2024-12-11
    </div>
    <details class="paper-abstract">
      Generating high-quality 3D content requires models capable of learning robust distributions of complex scenes and the real-world objects within them. Recent Gaussian-based 3D reconstruction techniques have achieved impressive results in recovering high-fidelity 3D assets from sparse input images by predicting 3D Gaussians in a feed-forward manner. However, these techniques often lack the extensive priors and expressiveness offered by Diffusion Models. On the other hand, 2D Diffusion Models, which have been successfully applied to denoise multiview images, show potential for generating a wide range of photorealistic 3D outputs but still fall short on explicit 3D priors and consistency. In this work, we aim to bridge these two approaches by introducing DSplats, a novel method that directly denoises multiview images using Gaussian Splat-based Reconstructors to produce a diverse array of realistic 3D assets. To harness the extensive priors of 2D Diffusion Models, we incorporate a pretrained Latent Diffusion Model into the reconstructor backbone to predict a set of 3D Gaussians. Additionally, the explicit 3D representation embedded in the denoising network provides a strong inductive bias, ensuring geometrically consistent novel view generation. Our qualitative and quantitative experiments demonstrate that DSplats not only produces high-quality, spatially consistent outputs, but also sets a new standard in single-image to 3D reconstruction. When evaluated on the Google Scanned Objects dataset, DSplats achieves a PSNR of 20.38, an SSIM of 0.842, and an LPIPS of 0.109.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08152v1">ProGDF: Progressive Gaussian Differential Field for Controllable and Flexible 3D Editing</a></div>
    <div class="paper-meta">
      📅 2024-12-11
    </div>
    <details class="paper-abstract">
      3D editing plays a crucial role in editing and reusing existing 3D assets, thereby enhancing productivity. Recently, 3DGS-based methods have gained increasing attention due to their efficient rendering and flexibility. However, achieving desired 3D editing results often requires multiple adjustments in an iterative loop, resulting in tens of minutes of training time cost for each attempt and a cumbersome trial-and-error cycle for users. This in-the-loop training paradigm results in a poor user experience. To address this issue, we introduce the concept of process-oriented modelling for 3D editing and propose the Progressive Gaussian Differential Field (ProGDF), an out-of-loop training approach that requires only a single training session to provide users with controllable editing capability and variable editing results through a user-friendly interface in real-time. ProGDF consists of two key components: Progressive Gaussian Splatting (PGS) and Gaussian Differential Field (GDF). PGS introduces the progressive constraint to extract the diverse intermediate results of the editing process and employs rendering quality regularization to improve the quality of these results. Based on these intermediate results, GDF leverages a lightweight neural network to model the editing process. Extensive results on two novel applications, namely controllable 3D editing and flexible fine-grained 3D manipulation, demonstrate the effectiveness, practicality and flexibility of the proposed ProGDF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01168v2">Mirror-3DGS: Incorporating Mirror Reflections into 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-11
      | 💬 IEEE International Conference on Visual Communications and Image Processing (VCIP 2024, Oral)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has significantly advanced 3D scene reconstruction and novel view synthesis. However, like Neural Radiance Fields (NeRF), 3DGS struggles with accurately modeling physical reflections, particularly in mirrors, leading to incorrect reconstructions and inconsistent reflective properties. To address this challenge, we introduce Mirror-3DGS, a novel framework designed to accurately handle mirror geometries and reflections, thereby generating realistic mirror reflections. By incorporating mirror attributes into 3DGS and leveraging plane mirror imaging principles, Mirror-3DGS simulates a mirrored viewpoint from behind the mirror, enhancing the realism of scene renderings. Extensive evaluations on both synthetic and real-world scenes demonstrate that our method can render novel views with improved fidelity in real-time, surpassing the state-of-the-art Mirror-NeRF, especially in mirror regions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04955v2">MixedGaussianAvatar: Realistically and Geometrically Accurate Head Avatar via Mixed 2D-3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-11
      | 💬 Project: https://chenvoid.github.io/MGA/
    </div>
    <details class="paper-abstract">
      Reconstructing high-fidelity 3D head avatars is crucial in various applications such as virtual reality. The pioneering methods reconstruct realistic head avatars with Neural Radiance Fields (NeRF), which have been limited by training and rendering speed. Recent methods based on 3D Gaussian Splatting (3DGS) significantly improve the efficiency of training and rendering. However, the surface inconsistency of 3DGS results in subpar geometric accuracy; later, 2DGS uses 2D surfels to enhance geometric accuracy at the expense of rendering fidelity. To leverage the benefits of both 2DGS and 3DGS, we propose a novel method named MixedGaussianAvatar for realistically and geometrically accurate head avatar reconstruction. Our main idea is to utilize 2D Gaussians to reconstruct the surface of the 3D head, ensuring geometric accuracy. We attach the 2D Gaussians to the triangular mesh of the FLAME model and connect additional 3D Gaussians to those 2D Gaussians where the rendering quality of 2DGS is inadequate, creating a mixed 2D-3D Gaussian representation. These 2D-3D Gaussians can then be animated using FLAME parameters. We further introduce a progressive training strategy that first trains the 2D Gaussians and then fine-tunes the mixed 2D-3D Gaussians. We demonstrate the superiority of MixedGaussianAvatar through comprehensive experiments. The code will be released at: https://github.com/ChenVoid/MGA/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07984v1">Diffusion-Based Attention Warping for Consistent 3D Scene Editing</a></div>
    <div class="paper-meta">
      📅 2024-12-10
    </div>
    <details class="paper-abstract">
      We present a novel method for 3D scene editing using diffusion models, designed to ensure view consistency and realism across perspectives. Our approach leverages attention features extracted from a single reference image to define the intended edits. These features are warped across multiple views by aligning them with scene geometry derived from Gaussian splatting depth estimates. Injecting these warped features into other viewpoints enables coherent propagation of edits, achieving high fidelity and spatial alignment in 3D space. Extensive evaluations demonstrate the effectiveness of our method in generating versatile edits of 3D scenes, significantly advancing the capabilities of scene manipulation compared to the existing methods. Project page: \url{https://attention-warp.github.io}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07739v1">GASP: Gaussian Avatars with Synthetic Priors</a></div>
    <div class="paper-meta">
      📅 2024-12-10
      | 💬 Project page: https://microsoft.github.io/GASP/
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has changed the game for real-time photo-realistic rendering. One of the most popular applications of Gaussian Splatting is to create animatable avatars, known as Gaussian Avatars. Recent works have pushed the boundaries of quality and rendering efficiency but suffer from two main limitations. Either they require expensive multi-camera rigs to produce avatars with free-view rendering, or they can be trained with a single camera but only rendered at high quality from this fixed viewpoint. An ideal model would be trained using a short monocular video or image from available hardware, such as a webcam, and rendered from any view. To this end, we propose GASP: Gaussian Avatars with Synthetic Priors. To overcome the limitations of existing datasets, we exploit the pixel-perfect nature of synthetic data to train a Gaussian Avatar prior. By fitting this prior model to a single photo or video and fine-tuning it, we get a high-quality Gaussian Avatar, which supports 360$^\circ$ rendering. Our prior is only required for fitting, not inference, enabling real-time application. Through our method, we obtain high-quality, animatable Avatars from limited data which can be animated and rendered at 70fps on commercial hardware. See our project page (https://microsoft.github.io/GASP/) for results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07660v1">Proc-GS: Procedural Building Generation for City Assembly with 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-12-10
      | 💬 Project page: https://city-super.github.io/procgs/
    </div>
    <details class="paper-abstract">
      Buildings are primary components of cities, often featuring repeated elements such as windows and doors. Traditional 3D building asset creation is labor-intensive and requires specialized skills to develop design rules. Recent generative models for building creation often overlook these patterns, leading to low visual fidelity and limited scalability. Drawing inspiration from procedural modeling techniques used in the gaming and visual effects industry, our method, Proc-GS, integrates procedural code into the 3D Gaussian Splatting (3D-GS) framework, leveraging their advantages in high-fidelity rendering and efficient asset management from both worlds. By manipulating procedural code, we can streamline this process and generate an infinite variety of buildings. This integration significantly reduces model size by utilizing shared foundational assets, enabling scalable generation with precise control over building assembly. We showcase the potential for expansive cityscape generation while maintaining high rendering fidelity and precise control on both real and synthetic cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07608v1">Faster and Better 3D Splatting via Group Training</a></div>
    <div class="paper-meta">
      📅 2024-12-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, demonstrating remarkable capability in high-fidelity scene reconstruction through its Gaussian primitive representations. However, the computational overhead induced by the massive number of primitives poses a significant bottleneck to training efficiency. To overcome this challenge, we propose Group Training, a simple yet effective strategy that organizes Gaussian primitives into manageable groups, optimizing training efficiency and improving rendering quality. This approach shows universal compatibility with existing 3DGS frameworks, including vanilla 3DGS and Mip-Splatting, consistently achieving accelerated training while maintaining superior synthesis quality. Extensive experiments reveal that our straightforward Group Training strategy achieves up to 30% faster convergence and improved rendering quality across diverse scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07494v1">ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery</a></div>
    <div class="paper-meta">
      📅 2024-12-10
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3D-GS) has prevailed in novel view synthesis, achieving high fidelity and efficiency. However, it often struggles to capture rich details and complete geometry. Our analysis highlights a key limitation of 3D-GS caused by the fixed threshold in densification, which balances geometry coverage against detail recovery as the threshold varies. To address this, we introduce a novel densification method, residual split, which adds a downscaled Gaussian as a residual. Our approach is capable of adaptively retrieving details and complementing missing geometry while enabling progressive refinement. To further support this method, we propose a pipeline named ResGS. Specifically, we integrate a Gaussian image pyramid for progressive supervision and implement a selection scheme that prioritizes the densification of coarse Gaussians over time. Extensive experiments demonstrate that our method achieves SOTA rendering quality. Consistent performance improvements can be achieved by applying our residual split on various 3D-GS variants, underscoring its versatility and potential for broader application in 3D-GS-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07293v1">EventSplat: 3D Gaussian Splatting from Moving Event Cameras for Real-time Rendering</a></div>
    <div class="paper-meta">
      📅 2024-12-10
    </div>
    <details class="paper-abstract">
      We introduce a method for using event camera data in novel view synthesis via Gaussian Splatting. Event cameras offer exceptional temporal resolution and a high dynamic range. Leveraging these capabilities allows us to effectively address the novel view synthesis challenge in the presence of fast camera motion. For initialization of the optimization process, our approach uses prior knowledge encoded in an event-to-video model. We also use spline interpolation for obtaining high quality poses along the event camera trajectory. This enhances the reconstruction quality from fast-moving cameras while overcoming the computational limitations traditionally associated with event-based Neural Radiance Field (NeRF) methods. Our experimental evaluation demonstrates that our results achieve higher visual fidelity and better performance than existing event-based NeRF approaches while being an order of magnitude faster to render.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03844v2">HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-10
      | 💬 Project page: https://gujiaqivadin.github.io/hybridgs/
    </div>
    <details class="paper-abstract">
      Generating high-quality novel view renderings of 3D Gaussian Splatting (3DGS) in scenes featuring transient objects is challenging. We propose a novel hybrid representation, termed as HybridGS, using 2D Gaussians for transient objects per image and maintaining traditional 3D Gaussians for the whole static scenes. Note that, the 3DGS itself is better suited for modeling static scenes that assume multi-view consistency, but the transient objects appear occasionally and do not adhere to the assumption, thus we model them as planar objects from a single view, represented with 2D Gaussians. Our novel representation decomposes the scene from the perspective of fundamental viewpoint consistency, making it more reasonable. Additionally, we present a novel multi-view regulated supervision method for 3DGS that leverages information from co-visible regions, further enhancing the distinctions between the transients and statics. Then, we propose a straightforward yet effective multi-stage training strategy to ensure robust training and high-quality view synthesis across various settings. Experiments on benchmark datasets show our state-of-the-art performance of novel view synthesis in both indoor and outdoor scenes, even in the presence of distracting elements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05256v2">Extrapolated Urban View Synthesis Benchmark</a></div>
    <div class="paper-meta">
      📅 2024-12-10
      | 💬 Project page: https://ai4ce.github.io/EUVS-Benchmark/
    </div>
    <details class="paper-abstract">
      Photorealistic simulators are essential for the training and evaluation of vision-centric autonomous vehicles (AVs). At their core is Novel View Synthesis (NVS), a crucial capability that generates diverse unseen viewpoints to accommodate the broad and continuous pose distribution of AVs. Recent advances in radiance fields, such as 3D Gaussian Splatting, achieve photorealistic rendering at real-time speeds and have been widely used in modeling large-scale driving scenes. However, their performance is commonly evaluated using an interpolated setup with highly correlated training and test views. In contrast, extrapolation, where test views largely deviate from training views, remains underexplored, limiting progress in generalizable simulation technology. To address this gap, we leverage publicly available AV datasets with multiple traversals, multiple vehicles, and multiple cameras to build the first Extrapolated Urban View Synthesis (EUVS) benchmark. Meanwhile, we conduct quantitative and qualitative evaluations of state-of-the-art Gaussian Splatting methods across different difficulty levels. Our results show that Gaussian Splatting is prone to overfitting to training views. Besides, incorporating diffusion priors and improving geometry cannot fundamentally improve NVS under large view changes, highlighting the need for more robust approaches and large-scale training. We have released our data to help advance self-driving and urban robotics simulation technology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06974v1">MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      Recent sparse multi-view scene reconstruction advances like DUSt3R and MASt3R no longer require camera calibration and camera pose estimation. However, they only process a pair of views at a time to infer pixel-aligned pointmaps. When dealing with more than two views, a combinatorial number of error prone pairwise reconstructions are usually followed by an expensive global optimization, which often fails to rectify the pairwise reconstruction errors. To handle more views, reduce errors, and improve inference time, we propose the fast single-stage feed-forward network MV-DUSt3R. At its core are multi-view decoder blocks which exchange information across any number of views while considering one reference view. To make our method robust to reference view selection, we further propose MV-DUSt3R+, which employs cross-reference-view blocks to fuse information across different reference view choices. To further enable novel view synthesis, we extend both by adding and jointly training Gaussian splatting heads. Experiments on multi-view stereo reconstruction, multi-view pose estimation, and novel view synthesis confirm that our methods improve significantly upon prior art. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15235v2">Learning-based Multi-View Stereo: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      3D reconstruction aims to recover the dense 3D structure of a scene. It plays an essential role in various applications such as Augmented/Virtual Reality (AR/VR), autonomous driving and robotics. Leveraging multiple views of a scene captured from different viewpoints, Multi-View Stereo (MVS) algorithms synthesize a comprehensive 3D representation, enabling precise reconstruction in complex environments. Due to its efficiency and effectiveness, MVS has become a pivotal method for image-based 3D reconstruction. Recently, with the success of deep learning, many learning-based MVS methods have been proposed, achieving impressive performance against traditional methods. We categorize these learning-based methods as: depth map-based, voxel-based, NeRF-based, 3D Gaussian Splatting-based, and large feed-forward methods. Among these, we focus significantly on depth map-based methods, which are the main family of MVS due to their conciseness, flexibility and scalability. In this survey, we provide a comprehensive review of the literature at the time of this writing. We investigate these learning-based methods, summarize their performances on popular benchmarks, and discuss promising future research directions in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02518v2">DDGS-CT: Direction-Disentangled Gaussian Splatting for Realistic Volume Rendering</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 Accepted by NeurIPS2024
    </div>
    <details class="paper-abstract">
      Digitally reconstructed radiographs (DRRs) are simulated 2D X-ray images generated from 3D CT volumes, widely used in preoperative settings but limited in intraoperative applications due to computational bottlenecks, especially for accurate but heavy physics-based Monte Carlo methods. While analytical DRR renderers offer greater efficiency, they overlook anisotropic X-ray image formation phenomena, such as Compton scattering. We present a novel approach that marries realistic physics-inspired X-ray simulation with efficient, differentiable DRR generation using 3D Gaussian splatting (3DGS). Our direction-disentangled 3DGS (DDGS) method separates the radiosity contribution into isotropic and direction-dependent components, approximating complex anisotropic interactions without intricate runtime simulations. Additionally, we adapt the 3DGS initialization to account for tomography data properties, enhancing accuracy and efficiency. Our method outperforms state-of-the-art techniques in image accuracy. Furthermore, our DDGS shows promise for intraoperative applications and inverse problems such as pose registration, delivering superior registration accuracy and runtime performance compared to analytical DRR methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01583v2">3DSceneEditor: Controllable 3D Scene Editing with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 Project Page: https://ziyangyan.github.io/3DSceneEditor
    </div>
    <details class="paper-abstract">
      The creation of 3D scenes has traditionally been both labor-intensive and costly, requiring designers to meticulously configure 3D assets and environments. Recent advancements in generative AI, including text-to-3D and image-to-3D methods, have dramatically reduced the complexity and cost of this process. However, current techniques for editing complex 3D scenes continue to rely on generally interactive multi-step, 2D-to-3D projection methods and diffusion-based techniques, which often lack precision in control and hamper real-time performance. In this work, we propose 3DSceneEditor, a fully 3D-based paradigm for real-time, precise editing of intricate 3D scenes using Gaussian Splatting. Unlike conventional methods, 3DSceneEditor operates through a streamlined 3D pipeline, enabling direct manipulation of Gaussians for efficient, high-quality edits based on input prompts.The proposed framework (i) integrates a pre-trained instance segmentation model for semantic labeling; (ii) employs a zero-shot grounding approach with CLIP to align target objects with user prompts; and (iii) applies scene modifications, such as object addition, repositioning, recoloring, replacing, and deletion directly on Gaussians. Extensive experimental results show that 3DSceneEditor achieves superior editing precision and speed with respect to current SOTA 3D scene editing approaches, establishing a new benchmark for efficient and interactive 3D scene customization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06424v1">Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Recent 4D reconstruction methods have yielded impressive results but rely on sharp videos as supervision. However, motion blur often occurs in videos due to camera shake and object movement, while existing methods render blurry results when using such videos for reconstructing 4D models. Although a few NeRF-based approaches attempted to address the problem, they struggled to produce high-quality results, due to the inaccuracy in estimating continuous dynamic representations within the exposure time. Encouraged by recent works in 3D motion trajectory modeling using 3D Gaussian Splatting (3DGS), we suggest taking 3DGS as the scene representation manner, and propose the first 4D Gaussian Splatting framework to reconstruct a high-quality 4D model from blurry monocular video, named Deblur4DGS. Specifically, we transform continuous dynamic representations estimation within an exposure time into the exposure time estimation. Moreover, we introduce exposure regularization to avoid trivial solutions, as well as multi-frame and multi-resolution consistency ones to alleviate artifacts. Furthermore, to better represent objects with large motion, we suggest blur-aware variable canonical Gaussians. Beyond novel-view synthesis, Deblur4DGS can be applied to improve blurry video from multiple perspectives, including deblurring, frame interpolation, and video stabilization. Extensive experiments on the above four tasks show that Deblur4DGS outperforms state-of-the-art 4D reconstruction methods. The codes are available at https://github.com/ZcsrenlongZ/Deblur4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06299v1">4D Gaussian Splatting with Scale-aware Residual Field and Adaptive Optimization for Real-time Rendering of Temporally Complex Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic scenes from video sequences is a highly promising task in the multimedia domain. While previous methods have made progress, they often struggle with slow rendering and managing temporal complexities such as significant motion and object appearance/disappearance. In this paper, we propose SaRO-GS as a novel dynamic scene representation capable of achieving real-time rendering while effectively handling temporal complexities in dynamic scenes. To address the issue of slow rendering speed, we adopt a Gaussian primitive-based representation and optimize the Gaussians in 4D space, which facilitates real-time rendering with the assistance of 3D Gaussian Splatting. Additionally, to handle temporally complex dynamic scenes, we introduce a Scale-aware Residual Field. This field considers the size information of each Gaussian primitive while encoding its residual feature and aligns with the self-splitting behavior of Gaussian primitives. Furthermore, we propose an Adaptive Optimization Schedule, which assigns different optimization strategies to Gaussian primitives based on their distinct temporal properties, thereby expediting the reconstruction of dynamic regions. Through evaluations on monocular and multi-view datasets, our method has demonstrated state-of-the-art performance. Please see our project page at https://yjb6.github.io/SaRO-GS.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06250v1">Splatter-360: Generalizable 360$^{\circ}$ Gaussian Splatting for Wide-baseline Panoramic Images</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 Project page:https://3d-aigc.github.io/Splatter-360/. Code: https://github.com/thucz/splatter360
    </div>
    <details class="paper-abstract">
      Wide-baseline panoramic images are frequently used in applications like VR and simulations to minimize capturing labor costs and storage needs. However, synthesizing novel views from these panoramic images in real time remains a significant challenge, especially due to panoramic imagery's high resolution and inherent distortions. Although existing 3D Gaussian splatting (3DGS) methods can produce photo-realistic views under narrow baselines, they often overfit the training views when dealing with wide-baseline panoramic images due to the difficulty in learning precise geometry from sparse 360$^{\circ}$ views. This paper presents \textit{Splatter-360}, a novel end-to-end generalizable 3DGS framework designed to handle wide-baseline panoramic images. Unlike previous approaches, \textit{Splatter-360} performs multi-view matching directly in the spherical domain by constructing a spherical cost volume through a spherical sweep algorithm, enhancing the network's depth perception and geometry estimation. Additionally, we introduce a 3D-aware bi-projection encoder to mitigate the distortions inherent in panoramic images and integrate cross-view attention to improve feature interactions across multiple viewpoints. This enables robust 3D-aware feature representations and real-time rendering capabilities. Experimental results on the HM3D~\cite{hm3d} and Replica~\cite{replica} demonstrate that \textit{Splatter-360} significantly outperforms state-of-the-art NeRF and 3DGS methods (e.g., PanoGRF, MVSplat, DepthSplat, and HiSplat) in both synthesis quality and generalization performance for wide-baseline panoramic images. Code and trained models are available at \url{https://3d-aigc.github.io/Splatter-360/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11672v3">Effective Rank Analysis and Regularization for Enhanced 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-08
      | 💬 project page: https://junhahyung.github.io/erankgs.github.io
    </div>
    <details class="paper-abstract">
      3D reconstruction from multi-view images is one of the fundamental challenges in computer vision and graphics. Recently, 3D Gaussian Splatting (3DGS) has emerged as a promising technique capable of real-time rendering with high-quality 3D reconstruction. This method utilizes 3D Gaussian representation and tile-based splatting techniques, bypassing the expensive neural field querying. Despite its potential, 3DGS encounters challenges such as needle-like artifacts, suboptimal geometries, and inaccurate normals caused by the Gaussians converging into anisotropic shapes with one dominant variance. We propose using the effective rank analysis to examine the shape statistics of 3D Gaussian primitives, and identify the Gaussians indeed converge into needle-like shapes with the effective rank 1. To address this, we introduce the effective rank as a regularization, which constrains the structure of the Gaussians. Our new regularization method enhances normal and geometry reconstruction while reducing needle-like artifacts. The approach can be integrated as an add-on module to other 3DGS variants, improving their quality without compromising visual fidelity. The project page is available at https://junhahyung.github.io/erankgs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05908v1">GBR: Generative Bundle Refinement for High-fidelity Gaussian Splatting and Meshing</a></div>
    <div class="paper-meta">
      📅 2024-12-08
    </div>
    <details class="paper-abstract">
      Gaussian splatting has gained attention for its efficient representation and rendering of 3D scenes using continuous Gaussian primitives. However, it struggles with sparse-view inputs due to limited geometric and photometric information, causing ambiguities in depth, shape, and texture. we propose GBR: Generative Bundle Refinement, a method for high-fidelity Gaussian splatting and meshing using only 4-6 input views. GBR integrates a neural bundle adjustment module to enhance geometry accuracy and a generative depth refinement module to improve geometry fidelity. More specifically, the neural bundle adjustment module integrates a foundation network to produce initial 3D point maps and point matches from unposed images, followed by bundle adjustment optimization to improve multiview consistency and point cloud accuracy. The generative depth refinement module employs a diffusion-based strategy to enhance geometric details and fidelity while preserving the scale. Finally, for Gaussian splatting optimization, we propose a multimodal loss function incorporating depth and normal consistency, geometric regularization, and pseudo-view supervision, providing robust guidance under sparse-view conditions. Experiments on widely used datasets show that GBR significantly outperforms existing methods under sparse-view inputs. Additionally, GBR demonstrates the ability to reconstruct and render large-scale real-world scenes, such as the Pavilion of Prince Teng and the Great Wall, with remarkable details using only 6 views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03923v2">CRiM-GS: Continuous Rigid Motion-Aware Gaussian Splatting from Motion-Blurred Images</a></div>
    <div class="paper-meta">
      📅 2024-12-08
      | 💬 Project Page : https://jho-yonsei.github.io/CRiM-Gaussian/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for their high-quality novel view rendering, motivating research to address real-world challenges. A critical issue is the camera motion blur caused by movement during exposure, which hinders accurate 3D scene reconstruction. In this study, we propose CRiM-GS, a \textbf{C}ontinuous \textbf{Ri}gid \textbf{M}otion-aware \textbf{G}aussian \textbf{S}platting that reconstructs precise 3D scenes from motion-blurred images while maintaining real-time rendering speed. Considering the complex motion patterns inherent in real-world camera movements, we predict continuous camera trajectories using neural ordinary differential equations (ODE). To ensure accurate modeling, we employ rigid body transformations with proper regularization, preserving object shape and size. Additionally, we introduce an adaptive distortion-aware transformation to compensate for potential nonlinear distortions, such as rolling shutter effects, and unpredictable camera movements. By revisiting fundamental camera theory and leveraging advanced neural training techniques, we achieve precise modeling of continuous camera trajectories. Extensive experiments demonstrate state-of-the-art performance both quantitatively and qualitatively on benchmark datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17249v2">SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes</a></div>
    <div class="paper-meta">
      📅 2024-12-07
      | 💬 Project page: https://cdfan0627.github.io/spectromotion/
    </div>
    <details class="paper-abstract">
      We present SpectroMotion, a novel approach that combines 3D Gaussian Splatting (3DGS) with physically-based rendering (PBR) and deformation fields to reconstruct dynamic specular scenes. Previous methods extending 3DGS to model dynamic scenes have struggled to represent specular surfaces accurately. Our method addresses this limitation by introducing a residual correction technique for accurate surface normal computation during deformation, complemented by a deformable environment map that adapts to time-varying lighting conditions. We implement a coarse-to-fine training strategy that significantly enhances scene geometry and specular color prediction. It is the only existing 3DGS method capable of synthesizing photorealistic real-world dynamic specular scenes, outperforming state-of-the-art methods in rendering complex, dynamic, and specular scenes. See our project page for video results at https://cdfan0627.github.io/spectromotion/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.19495v2">CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-12-07
      | 💬 ECCV2024, Project page: https://people.engr.tamu.edu/nimak/Papers/CoherentGS, Code: https://github.com/avinashpaliwal/CoherentGS
    </div>
    <details class="paper-abstract">
      The field of 3D reconstruction from images has rapidly evolved in the past few years, first with the introduction of Neural Radiance Field (NeRF) and more recently with 3D Gaussian Splatting (3DGS). The latter provides a significant edge over NeRF in terms of the training and inference speed, as well as the reconstruction quality. Although 3DGS works well for dense input images, the unstructured point-cloud like representation quickly overfits to the more challenging setup of extremely sparse input images (e.g., 3 images), creating a representation that appears as a jumble of needles from novel views. To address this issue, we propose regularized optimization and depth-based initialization. Our key idea is to introduce a structured Gaussian representation that can be controlled in 2D image space. We then constraint the Gaussians, in particular their position, and prevent them from moving independently during optimization. Specifically, we introduce single and multiview constraints through an implicit convolutional decoder and a total variation loss, respectively. With the coherency introduced to the Gaussians, we further constrain the optimization through a flow-based loss function. To support our regularized optimization, we propose an approach to initialize the Gaussians using monocular depth estimates at each input view. We demonstrate significant improvements compared to the state-of-the-art sparse-view NeRF-based approaches on a variety of scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05700v1">Temporally Compressed 3D Gaussian Splatting for Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2024-12-07
      | 💬 Code will be released soon
    </div>
    <details class="paper-abstract">
      Recent advancements in high-fidelity dynamic scene reconstruction have leveraged dynamic 3D Gaussians and 4D Gaussian Splatting for realistic scene representation. However, to make these methods viable for real-time applications such as AR/VR, gaming, and rendering on low-power devices, substantial reductions in memory usage and improvements in rendering efficiency are required. While many state-of-the-art methods prioritize lightweight implementations, they struggle in handling scenes with complex motions or long sequences. In this work, we introduce Temporally Compressed 3D Gaussian Splatting (TC3DGS), a novel technique designed specifically to effectively compress dynamic 3D Gaussian representations. TC3DGS selectively prunes Gaussians based on their temporal relevance and employs gradient-aware mixed-precision quantization to dynamically compress Gaussian parameters. It additionally relies on a variation of the Ramer-Douglas-Peucker algorithm in a post-processing step to further reduce storage by interpolating Gaussian trajectories across frames. Our experiments across multiple datasets demonstrate that TC3DGS achieves up to 67$\times$ compression with minimal or no degradation in visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05695v1">WATER-GS: Toward Copyright Protection for 3D Gaussian Splatting via Universal Watermarking</a></div>
    <div class="paper-meta">
      📅 2024-12-07
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a pivotal technique for 3D scene representation, providing rapid rendering speeds and high fidelity. As 3DGS gains prominence, safeguarding its intellectual property becomes increasingly crucial since 3DGS could be used to imitate unauthorized scene creations and raise copyright issues. Existing watermarking methods for implicit NeRFs cannot be directly applied to 3DGS due to its explicit representation and real-time rendering process, leaving watermarking for 3DGS largely unexplored. In response, we propose WATER-GS, a novel method designed to protect 3DGS copyrights through a universal watermarking strategy. First, we introduce a pre-trained watermark decoder, treating raw 3DGS generative modules as potential watermark encoders to ensure imperceptibility. Additionally, we implement novel 3D distortion layers to enhance the robustness of the embedded watermark against common real-world distortions of point cloud data. Comprehensive experiments and ablation studies demonstrate that WATER-GS effectively embeds imperceptible and robust watermarks into 3DGS without compromising rendering efficiency and quality. Our experiments indicate that the 3D distortion layers can yield up to a 20% improvement in accuracy rate. Notably, our method is adaptable to different 3DGS variants, including 3DGS compression frameworks and 2D Gaussian splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04474v2">LumiGauss: Relightable Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2024-12-07
      | 💬 Accepted at WACV2025
    </div>
    <details class="paper-abstract">
      Decoupling lighting from geometry using unconstrained photo collections is notoriously challenging. Solving it would benefit many users as creating complex 3D assets takes days of manual labor. Many previous works have attempted to address this issue, often at the expense of output fidelity, which questions the practicality of such methods. We introduce LumiGauss - a technique that tackles 3D reconstruction of scenes and environmental lighting through 2D Gaussian Splatting. Our approach yields high-quality scene reconstructions and enables realistic lighting synthesis under novel environment maps. We also propose a method for enhancing the quality of shadows, common in outdoor scenes, by exploiting spherical harmonics properties. Our approach facilitates seamless integration with game engines and enables the use of fast precomputed radiance transfer. We validate our method on the NeRF-OSR dataset, demonstrating superior performance over baseline methods. Moreover, LumiGauss can synthesize realistic images for unseen environment maps. Our code: https://github.com/joaxkal/lumigauss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05570v1">Template-free Articulated Gaussian Splatting for Real-time Reposable Dynamic View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-12-07
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      While novel view synthesis for dynamic scenes has made significant progress, capturing skeleton models of objects and re-posing them remains a challenging task. To tackle this problem, in this paper, we propose a novel approach to automatically discover the associated skeleton model for dynamic objects from videos without the need for object-specific templates. Our approach utilizes 3D Gaussian Splatting and superpoints to reconstruct dynamic objects. Treating superpoints as rigid parts, we can discover the underlying skeleton model through intuitive cues and optimize it using the kinematic model. Besides, an adaptive control strategy is applied to avoid the emergence of redundant superpoints. Extensive experiments demonstrate the effectiveness and efficiency of our method in obtaining re-posable 3D objects. Not only can our approach achieve excellent visual fidelity, but it also allows for the real-time rendering of high-resolution images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00623v2">A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision</a></div>
    <div class="paper-meta">
      📅 2024-12-07
    </div>
    <details class="paper-abstract">
      We introduce a diffusion model for Gaussian Splats, SplatDiffusion, to enable generation of three-dimensional structures from single images, addressing the ill-posed nature of lifting 2D inputs to 3D. Existing methods rely on deterministic, feed-forward predictions, which limit their ability to handle the inherent ambiguity of 3D inference from 2D data. Diffusion models have recently shown promise as powerful generative models for 3D data, including Gaussian splats; however, standard diffusion frameworks typically require the target signal and denoised signal to be in the same modality, which is challenging given the scarcity of 3D data. To overcome this, we propose a novel training strategy that decouples the denoised modality from the supervision modality. By using a deterministic model as a noisy teacher to create the noised signal and transitioning from single-step to multi-step denoising supervised by an image rendering loss, our approach significantly enhances performance compared to the deterministic teacher. Additionally, our method is flexible, as it can learn from various 3D Gaussian Splat (3DGS) teachers with minimal adaptation; we demonstrate this by surpassing the performance of two different deterministic models as teachers, highlighting the potential generalizability of our framework. Our approach further incorporates a guidance mechanism to aggregate information from multiple views, enhancing reconstruction quality when more than one view is available. Experimental results on object-level and scene-level datasets demonstrate the effectiveness of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05560v1">Text-to-3D Gaussian Splatting with Physics-Grounded Motion Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-07
    </div>
    <details class="paper-abstract">
      Text-to-3D generation is a valuable technology in virtual reality and digital content creation. While recent works have pushed the boundaries of text-to-3D generation, producing high-fidelity 3D objects with inefficient prompts and simulating their physics-grounded motion accurately still remain unsolved challenges. To address these challenges, we present an innovative framework that utilizes the Large Language Model (LLM)-refined prompts and diffusion priors-guided Gaussian Splatting (GS) for generating 3D models with accurate appearances and geometric structures. We also incorporate a continuum mechanics-based deformation map and color regularization to synthesize vivid physics-grounded motion for the generated 3D Gaussians, adhering to the conservation of mass and momentum. By integrating text-to-3D generation with physics-grounded motion synthesis, our framework renders photo-realistic 3D objects that exhibit physics-aware motion, accurately reflecting the behaviors of the objects under various forces and constraints across different materials. Extensive experiments demonstrate that our approach achieves high-quality 3D generations with realistic physics-grounded motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18197v2">Make-It-Animatable: An Efficient Framework for Authoring Animation-Ready 3D Characters</a></div>
    <div class="paper-meta">
      📅 2024-12-07
      | 💬 Project Page: https://jasongzy.github.io/Make-It-Animatable/
    </div>
    <details class="paper-abstract">
      3D characters are essential to modern creative industries, but making them animatable often demands extensive manual work in tasks like rigging and skinning. Existing automatic rigging tools face several limitations, including the necessity for manual annotations, rigid skeleton topologies, and limited generalization across diverse shapes and poses. An alternative approach is to generate animatable avatars pre-bound to a rigged template mesh. However, this method often lacks flexibility and is typically limited to realistic human shapes. To address these issues, we present Make-It-Animatable, a novel data-driven method to make any 3D humanoid model ready for character animation in less than one second, regardless of its shapes and poses. Our unified framework generates high-quality blend weights, bones, and pose transformations. By incorporating a particle-based shape autoencoder, our approach supports various 3D representations, including meshes and 3D Gaussian splats. Additionally, we employ a coarse-to-fine representation and a structure-aware modeling strategy to ensure both accuracy and robustness, even for characters with non-standard skeleton structures. We conducted extensive experiments to validate our framework's effectiveness. Compared to existing methods, our approach demonstrates significant improvements in both quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05546v1">Radiant: Large-scale 3D Gaussian Rendering based on Hierarchical Framework</a></div>
    <div class="paper-meta">
      📅 2024-12-07
    </div>
    <details class="paper-abstract">
      With the advancement of computer vision, the recently emerged 3D Gaussian Splatting (3DGS) has increasingly become a popular scene reconstruction algorithm due to its outstanding performance. Distributed 3DGS can efficiently utilize edge devices to directly train on the collected images, thereby offloading computational demands and enhancing efficiency. However, traditional distributed frameworks often overlook computational and communication challenges in real-world environments, hindering large-scale deployment and potentially posing privacy risks. In this paper, we propose Radiant, a hierarchical 3DGS algorithm designed for large-scale scene reconstruction that considers system heterogeneity, enhancing the model performance and training efficiency. Via extensive empirical study, we find that it is crucial to partition the regions for each edge appropriately and allocate varying camera positions to each device for image collection and training. The core of Radiant is partitioning regions based on heterogeneous environment information and allocating workloads to each device accordingly. Furthermore, we provide a 3DGS model aggregation algorithm that enhances the quality and ensures the continuity of models' boundaries. Finally, we develop a testbed, and experiments demonstrate that Radiant improved reconstruction quality by up to 25.7\% and reduced up to 79.6\% end-to-end latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04433v2">PBDyG: Position Based Dynamic Gaussians for Motion-Aware Clothed Human Avatars</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      This paper introduces a novel clothed human model that can be learned from multiview RGB videos, with a particular emphasis on recovering physically accurate body and cloth movements. Our method, Position Based Dynamic Gaussians (PBDyG), realizes ``movement-dependent'' cloth deformation via physical simulation, rather than merely relying on ``pose-dependent'' rigid transformations. We model the clothed human holistically but with two distinct physical entities in contact: clothing modeled as 3D Gaussians, which are attached to a skinned SMPL body that follows the movement of the person in the input videos. The articulation of the SMPL body also drives physically-based simulation of the clothes' Gaussians to transform the avatar to novel poses. In order to run position based dynamics simulation, physical properties including mass and material stiffness are estimated from the RGB videos through Dynamic 3D Gaussian Splatting. Experiments demonstrate that our method not only accurately reproduces appearance but also enables the reconstruction of avatars wearing highly deformable garments, such as skirts or coats, which have been challenging to reconstruct using existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02058v2">OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 NeurIPS2024
    </div>
    <details class="paper-abstract">
      This paper introduces OpenGaussian, a method based on 3D Gaussian Splatting (3DGS) capable of 3D point-level open vocabulary understanding. Our primary motivation stems from observing that existing 3DGS-based open vocabulary methods mainly focus on 2D pixel-level parsing. These methods struggle with 3D point-level tasks due to weak feature expressiveness and inaccurate 2D-3D feature associations. To ensure robust feature presentation and 3D point-level understanding, we first employ SAM masks without cross-frame associations to train instance features with 3D consistency. These features exhibit both intra-object consistency and inter-object distinction. Then, we propose a two-stage codebook to discretize these features from coarse to fine levels. At the coarse level, we consider the positional information of 3D points to achieve location-based clustering, which is then refined at the fine level. Finally, we introduce an instance-level 3D-2D feature association method that links 3D points to 2D masks, which are further associated with 2D CLIP features. Extensive experiments, including open vocabulary-based 3D object selection, 3D point cloud understanding, click-based 3D object selection, and ablation studies, demonstrate the effectiveness of our proposed method. The source code is available at our project page: https://3d-aigc.github.io/OpenGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20224v3">EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 Project Page: https://www.falcary.com/EvaGaussians/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) has demonstrated exceptional capabilities in 3D scene reconstruction and novel view synthesis. However, its training heavily depends on high-quality, sharp images and accurate camera poses. Fulfilling these requirements can be challenging in non-ideal real-world scenarios, where motion-blurred images are commonly encountered in high-speed moving cameras or low-light environments that require long exposure times. To address these challenges, we introduce Event Stream Assisted Gaussian Splatting (EvaGaussians), a novel approach that integrates event streams captured by an event camera to assist in reconstructing high-quality 3D-GS from blurry images. Capitalizing on the high temporal resolution and dynamic range offered by the event camera, we leverage the event streams to explicitly model the formation process of motion-blurred images and guide the deblurring reconstruction of 3D-GS. By jointly optimizing the 3D-GS parameters and recovering camera motion trajectories during the exposure time, our method can robustly facilitate the acquisition of high-fidelity novel views with intricate texture details. We comprehensively evaluated our method and compared it with previous state-of-the-art deblurring rendering methods. Both qualitative and quantitative comparisons demonstrate that our method surpasses existing techniques in restoring fine details from blurry images and producing high-fidelity novel views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05526v3">Benchmarking Neural Radiance Fields for Autonomous Robots: An Overview</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 32 pages, 5 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRF) have emerged as a powerful paradigm for 3D scene representation, offering high-fidelity renderings and reconstructions from a set of sparse and unstructured sensor data. In the context of autonomous robotics, where perception and understanding of the environment are pivotal, NeRF holds immense promise for improving performance. In this paper, we present a comprehensive survey and analysis of the state-of-the-art techniques for utilizing NeRF to enhance the capabilities of autonomous robots. We especially focus on the perception, localization and navigation, and decision-making modules of autonomous robots and delve into tasks crucial for autonomous operation, including 3D reconstruction, segmentation, pose estimation, simultaneous localization and mapping (SLAM), navigation and planning, and interaction. Our survey meticulously benchmarks existing NeRF-based methods, providing insights into their strengths and limitations. Moreover, we explore promising avenues for future research and development in this domain. Notably, we discuss the integration of advanced techniques such as 3D Gaussian splatting (3DGS), large language models (LLM), and generative AIs, envisioning enhanced reconstruction efficiency, scene understanding, decision-making capabilities. This survey serves as a roadmap for researchers seeking to leverage NeRFs to empower autonomous robots, paving the way for innovative solutions that can navigate and interact seamlessly in complex environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04887v1">Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has demonstrated notable success in large-scale scene reconstruction, but challenges persist due to high training memory consumption and storage overhead. Hybrid representations that integrate implicit and explicit features offer a way to mitigate these limitations. However, when applied in parallelized block-wise training, two critical issues arise since reconstruction accuracy deteriorates due to reduced data diversity when training each block independently, and parallel training restricts the number of divided blocks to the available number of GPUs. To address these issues, we propose Momentum-GS, a novel approach that leverages momentum-based self-distillation to promote consistency and accuracy across the blocks while decoupling the number of blocks from the physical GPU count. Our method maintains a teacher Gaussian decoder updated with momentum, ensuring a stable reference during training. This teacher provides each block with global guidance in a self-distillation manner, promoting spatial consistency in reconstruction. To further ensure consistency across the blocks, we incorporate block weighting, dynamically adjusting each block's weight according to its reconstruction accuracy. Extensive experiments on large-scale scenes show that our method consistently outperforms existing techniques, achieving a 12.8% improvement in LPIPS over CityGaussian with much fewer divided blocks and establishing a new state of the art. Project page: https://jixuan-fan.github.io/Momentum-GS_Page/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04832v1">WRF-GS: Wireless Radiation Field Reconstruction with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 accepted to the IEEE International Conference on Computer Communications (INFOCOM 2025)
    </div>
    <details class="paper-abstract">
      Wireless channel modeling plays a pivotal role in designing, analyzing, and optimizing wireless communication systems. Nevertheless, developing an effective channel modeling approach has been a longstanding challenge. This issue has been escalated due to the denser network deployment, larger antenna arrays, and wider bandwidth in 5G and beyond networks. To address this challenge, we put forth WRF-GS, a novel framework for channel modeling based on wireless radiation field (WRF) reconstruction using 3D Gaussian splatting. WRF-GS employs 3D Gaussian primitives and neural networks to capture the interactions between the environment and radio signals, enabling efficient WRF reconstruction and visualization of the propagation characteristics. The reconstructed WRF can then be used to synthesize the spatial spectrum for comprehensive wireless channel characterization. Notably, with a small number of measurements, WRF-GS can synthesize new spatial spectra within milliseconds for a given scene, thereby enabling latency-sensitive applications. Experimental results demonstrate that WRF-GS outperforms existing methods for spatial spectrum synthesis, such as ray tracing and other deep-learning approaches. Moreover, WRF-GS achieves superior performance in the channel state information prediction task, surpassing existing methods by a significant margin of more than 2.43 dB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04826v1">Pushing Rendering Boundaries: Hard Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive Novel View Synthesis (NVS) results in a real-time rendering manner. During training, it relies heavily on the average magnitude of view-space positional gradients to grow Gaussians to reduce rendering loss. However, this average operation smooths the positional gradients from different viewpoints and rendering errors from different pixels, hindering the growth and optimization of many defective Gaussians. This leads to strong spurious artifacts in some areas. To address this problem, we propose Hard Gaussian Splatting, dubbed HGS, which considers multi-view significant positional gradients and rendering errors to grow hard Gaussians that fill the gaps of classical Gaussian Splatting on 3D scenes, thus achieving superior NVS results. In detail, we present positional gradient driven HGS, which leverages multi-view significant positional gradients to uncover hard Gaussians. Moreover, we propose rendering error guided HGS, which identifies noticeable pixel rendering errors and potentially over-large Gaussians to jointly mine hard Gaussians. By growing and optimizing these hard Gaussians, our method helps to resolve blurring and needle-like artifacts. Experiments on various datasets demonstrate that our method achieves state-of-the-art rendering quality while maintaining real-time efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.07668v2">ChromaDistill: Colorizing Monochrome Radiance Fields with Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 WACV 2025, AI3DCC @ ICCV 2023
    </div>
    <details class="paper-abstract">
      Colorization is a well-explored problem in the domains of image and video processing. However, extending colorization to 3D scenes presents significant challenges. Recent Neural Radiance Field (NeRF) and Gaussian-Splatting(3DGS) methods enable high-quality novel-view synthesis for multi-view images. However, the question arises: How can we colorize these 3D representations? This work presents a method for synthesizing colorized novel views from input grayscale multi-view images. Using image or video colorization methods to colorize novel views from these 3D representations naively will yield output with severe inconsistencies. We introduce a novel method to use powerful image colorization models for colorizing 3D representations. We propose a distillation-based method that transfers color from these networks trained on natural images to the target 3D representation. Notably, this strategy does not add any additional weights or computational overhead to the original representation during inference. Extensive experiments demonstrate that our method produces high-quality colorized views for indoor and outdoor scenes, showcasing significant cross-view consistency advantages over baseline approaches. Our method is agnostic to the underlying 3D representation and easily generalizable to NeRF and 3DGS methods. Further, we validate the efficacy of our approach in several diverse applications: 1.) Infra-Red (IR) multi-view images and 2.) Legacy grayscale multi-view image sequences. Project Webpage: https://val.cds.iisc.ac.in/chroma-distill.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04470v1">Turbo3D: Ultra-fast Text-to-3D Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 project page: https://turbo-3d.github.io/
    </div>
    <details class="paper-abstract">
      We present Turbo3D, an ultra-fast text-to-3D system capable of generating high-quality Gaussian splatting assets in under one second. Turbo3D employs a rapid 4-step, 4-view diffusion generator and an efficient feed-forward Gaussian reconstructor, both operating in latent space. The 4-step, 4-view generator is a student model distilled through a novel Dual-Teacher approach, which encourages the student to learn view consistency from a multi-view teacher and photo-realism from a single-view teacher. By shifting the Gaussian reconstructor's inputs from pixel space to latent space, we eliminate the extra image decoding time and halve the transformer sequence length for maximum efficiency. Our method demonstrates superior 3D generation results compared to previous baselines, while operating in a fraction of their runtime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04469v1">QUEEN: QUantized Efficient ENcoding of Dynamic Gaussians for Streaming Free-viewpoint Videos</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 Accepted at NeurIPS 2024, Project website: https://research.nvidia.com/labs/amri/projects/queen
    </div>
    <details class="paper-abstract">
      Online free-viewpoint video (FVV) streaming is a challenging problem, which is relatively under-explored. It requires incremental on-the-fly updates to a volumetric representation, fast training and rendering to satisfy real-time constraints and a small memory footprint for efficient transmission. If achieved, it can enhance user experience by enabling novel applications, e.g., 3D video conferencing and live volumetric video broadcast, among others. In this work, we propose a novel framework for QUantized and Efficient ENcoding (QUEEN) for streaming FVV using 3D Gaussian Splatting (3D-GS). QUEEN directly learns Gaussian attribute residuals between consecutive frames at each time-step without imposing any structural constraints on them, allowing for high quality reconstruction and generalizability. To efficiently store the residuals, we further propose a quantization-sparsity framework, which contains a learned latent-decoder for effectively quantizing attribute residuals other than Gaussian positions and a learned gating module to sparsify position residuals. We propose to use the Gaussian viewspace gradient difference vector as a signal to separate the static and dynamic content of the scene. It acts as a guide for effective sparsity learning and speeds up training. On diverse FVV benchmarks, QUEEN outperforms the state-of-the-art online FVV methods on all metrics. Notably, for several highly dynamic scenes, it reduces the model size to just 0.7 MB per frame while training in under 5 sec and rendering at 350 FPS. Project website is at https://research.nvidia.com/labs/amri/projects/queen
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04459v1">Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 Code release in progress
    </div>
    <details class="paper-abstract">
      We propose an efficient radiance field rendering algorithm that incorporates a rasterization process on sparse voxels without neural networks or 3D Gaussians. There are two key contributions coupled with the proposed system. The first is to render sparse voxels in the correct depth order along pixel rays by using dynamic Morton ordering. This avoids the well-known popping artifact found in Gaussian splatting. Second, we adaptively fit sparse voxels to different levels of detail within scenes, faithfully reproducing scene details while achieving high rendering frame rates. Our method improves the previous neural-free voxel grid representation by over 4db PSNR and more than 10x rendering FPS speedup, achieving state-of-the-art comparable novel-view synthesis results. Additionally, our neural-free sparse voxels are seamlessly compatible with grid-based 3D processing algorithms. We achieve promising mesh reconstruction accuracy by integrating TSDF-Fusion and Marching Cubes into our sparse grid system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04457v1">Monocular Dynamic Gaussian Splatting is Fast and Brittle but Smooth Motion Helps</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 37 pages, 39 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Gaussian splatting methods are emerging as a popular approach for converting multi-view image data into scene representations that allow view synthesis. In particular, there is interest in enabling view synthesis for dynamic scenes using only monocular input data -- an ill-posed and challenging problem. The fast pace of work in this area has produced multiple simultaneous papers that claim to work best, which cannot all be true. In this work, we organize, benchmark, and analyze many Gaussian-splatting-based methods, providing apples-to-apples comparisons that prior works have lacked. We use multiple existing datasets and a new instructive synthetic dataset designed to isolate factors that affect reconstruction quality. We systematically categorize Gaussian splatting methods into specific motion representation types and quantify how their differences impact performance. Empirically, we find that their rank order is well-defined in synthetic data, but the complexity of real-world data currently overwhelms the differences. Furthermore, the fast rendering speed of all Gaussian-based methods comes at the cost of brittleness in optimization. We summarize our experiments into a list of findings that can help to further progress in this lively problem setting. Project Webpage: https://lynl7130.github.io/MonoDyGauBench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00112v2">DynMF: Neural Motion Factorization for Real-time Dynamic View Synthesis with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 Project page: https://agelosk.github.io/dynmf/
    </div>
    <details class="paper-abstract">
      Accurately and efficiently modeling dynamic scenes and motions is considered so challenging a task due to temporal dynamics and motion complexity. To address these challenges, we propose DynMF, a compact and efficient representation that decomposes a dynamic scene into a few neural trajectories. We argue that the per-point motions of a dynamic scene can be decomposed into a small set of explicit or learned trajectories. Our carefully designed neural framework consisting of a tiny set of learned basis queried only in time allows for rendering speed similar to 3D Gaussian Splatting, surpassing 120 FPS, while at the same time, requiring only double the storage compared to static scenes. Our neural representation adequately constrains the inherently underconstrained motion field of a dynamic scene leading to effective and fast optimization. This is done by biding each point to motion coefficients that enforce the per-point sharing of basis trajectories. By carefully applying a sparsity loss to the motion coefficients, we are able to disentangle the motions that comprise the scene, independently control them, and generate novel motion combinations that have never been seen before. We can reach state-of-the-art render quality within just 5 minutes of training and in less than half an hour, we can synthesize novel views of dynamic scenes with superior photorealistic quality. Our representation is interpretable, efficient, and expressive enough to offer real-time view synthesis of complex dynamic scene motions, in monocular and multi-view scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03911v1">Multi-View Pose-Agnostic Change Localization with Zero Labels</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      Autonomous agents often require accurate methods for detecting and localizing changes in their environment, particularly when observations are captured from unconstrained and inconsistent viewpoints. We propose a novel label-free, pose-agnostic change detection method that integrates information from multiple viewpoints to construct a change-aware 3D Gaussian Splatting (3DGS) representation of the scene. With as few as 5 images of the post-change scene, our approach can learn additional change channels in a 3DGS and produce change masks that outperform single-view techniques. Our change-aware 3D scene representation additionally enables the generation of accurate change masks for unseen viewpoints. Experimental results demonstrate state-of-the-art performance in complex multi-object scenes, achieving a 1.7$\times$ and 1.6$\times$ improvement in Mean Intersection Over Union and F1 score respectively over other baselines. We also contribute a new real-world dataset to benchmark change detection in diverse challenging scenes in the presence of lighting variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00333v2">Gaussians on their Way: Wasserstein-Constrained 4D Gaussian Splatting with State-Space Modeling</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      Dynamic scene rendering has taken a leap forward with the rise of 4D Gaussian Splatting, but there's still one elusive challenge: how to make 3D Gaussians move through time as naturally as they would in the real world, all while keeping the motion smooth and consistent. In this paper, we unveil a fresh approach that blends state-space modeling with Wasserstein geometry, paving the way for a more fluid and coherent representation of dynamic scenes. We introduce a State Consistency Filter that merges prior predictions with the current observations, enabling Gaussians to stay true to their way over time. We also employ Wasserstein distance regularization to ensure smooth, consistent updates of Gaussian parameters, reducing motion artifacts. Lastly, we leverage Wasserstein geometry to capture both translational motion and shape deformations, creating a more physically plausible model for dynamic scenes. Our approach guides Gaussians along their natural way in the Wasserstein space, achieving smoother, more realistic motion and stronger temporal coherence. Experimental results show significant improvements in rendering quality and efficiency, outperforming current state-of-the-art techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17422v2">Multimodal LLM Guided Exploration and Active Mapping using Fisher Information</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      We present an active mapping system that could plan for long-horizon exploration goals and short-term actions with a 3D Gaussian Splatting (3DGS) representation. Existing methods either did not take advantage of recent developments in multimodal Large Language Models (LLM) or did not consider challenges in localization uncertainty, which is critical in embodied agents. We propose employing multimodal LLMs for long-horizon planning in conjunction with detailed motion planning using our information-based algorithm. By leveraging high-quality view synthesis from our 3DGS representation, our method employs a multimodal LLM as a zero-shot planner for long-horizon exploration goals from the semantic perspective. We also introduce an uncertainty-aware path proposal and selection algorithm that balances the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10219v2">PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Recent advances in novel view synthesis have enabled real-time rendering speeds with high reconstruction accuracy. 3D Gaussian Splatting (3D-GS), a foundational point-based parametric 3D scene representation, models scenes as large sets of 3D Gaussians. However, complex scenes can consist of millions of Gaussians, resulting in high storage and memory requirements that limit the viability of 3D-GS on devices with limited resources. Current techniques for compressing these pretrained models by pruning Gaussians rely on combining heuristics to determine which Gaussians to remove. At high compression ratios, these pruned scenes suffer from heavy degradation of visual fidelity and loss of foreground details. In this paper, we propose a principled sensitivity pruning score that preserves visual fidelity and foreground details at significantly higher compression ratios than existing approaches. It is computed as a second-order approximation of the reconstruction error on the training views with respect to the spatial parameters of each Gaussian. Additionally, we propose a multi-round prune-refine pipeline that can be applied to any pretrained 3D-GS model without changing its training pipeline. After pruning 90% of Gaussians, a substantially higher percentage than previous methods, our PUP 3D-GS pipeline increases average rendering speed by 3.56$\times$ while retaining more salient foreground information and achieving higher image quality metrics than existing techniques on scenes from the Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03526v1">Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Project website: https://research.nvidia.com/labs/toronto-ai/bullet-timer/
    </div>
    <details class="paper-abstract">
      Recent advancements in static feed-forward scene reconstruction have demonstrated significant progress in high-quality novel view synthesis. However, these models often struggle with generalizability across diverse environments and fail to effectively handle dynamic content. We present BTimer (short for BulletTimer), the first motion-aware feed-forward model for real-time reconstruction and novel view synthesis of dynamic scenes. Our approach reconstructs the full scene in a 3D Gaussian Splatting representation at a given target ('bullet') timestamp by aggregating information from all the context frames. Such a formulation allows BTimer to gain scalability and generalization by leveraging both static and dynamic scene datasets. Given a casual monocular dynamic video, BTimer reconstructs a bullet-time scene within 150ms while reaching state-of-the-art performance on both static and dynamic scene datasets, even compared with optimization-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03518v1">Dense Scene Reconstruction from Light-Field Images Affected by Rolling Shutter</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      This paper presents a dense depth estimation approach from light-field (LF) images that is able to compensate for strong rolling shutter (RS) effects. Our method estimates RS compensated views and dense RS compensated disparity maps. We present a two-stage method based on a 2D Gaussians Splatting that allows for a ``render and compare" strategy with a point cloud formulation. In the first stage, a subset of sub-aperture images is used to estimate an RS agnostic 3D shape that is related to the scene target shape ``up to a motion". In the second stage, the deformation of the 3D shape is computed by estimating an admissible camera motion. We demonstrate the effectiveness and advantages of this approach through several experiments conducted for different scenes and types of motions. Due to lack of suitable datasets for evaluation, we also present a new carefully designed synthetic dataset of RS LF images. The source code, trained models and dataset will be made publicly available at: https://github.com/ICB-Vision-AI/DenseRSLF
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03473v1">Urban4D: Semantic-Guided 4D Gaussian Splatting for Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic urban scenes presents significant challenges due to their intrinsic geometric structures and spatiotemporal dynamics. Existing methods that attempt to model dynamic urban scenes without leveraging priors on potentially moving regions often produce suboptimal results. Meanwhile, approaches based on manual 3D annotations yield improved reconstruction quality but are impractical due to labor-intensive labeling. In this paper, we revisit the potential of 2D semantic maps for classifying dynamic and static Gaussians and integrating spatial and temporal dimensions for urban scene representation. We introduce Urban4D, a novel framework that employs a semantic-guided decomposition strategy inspired by advances in deep 2D semantic map generation. Our approach distinguishes potentially dynamic objects through reliable semantic Gaussians. To explicitly model dynamic objects, we propose an intuitive and effective 4D Gaussian splatting (4DGS) representation that aggregates temporal information through learnable time embeddings for each Gaussian, predicting their deformations at desired timestamps using a multilayer perceptron (MLP). For more accurate static reconstruction, we also design a k-nearest neighbor (KNN)-based consistency regularization to handle the ground surface due to its low-texture characteristic. Extensive experiments on real-world datasets demonstrate that Urban4D not only achieves comparable or better quality than previous state-of-the-art methods but also effectively captures dynamic objects while maintaining high visual fidelity for static elements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14108v2">GaussianBeV: 3D Gaussian Representation meets Perception Models for BeV Segmentation</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Accepted to WACV 2025
    </div>
    <details class="paper-abstract">
      The Bird's-eye View (BeV) representation is widely used for 3D perception from multi-view camera images. It allows to merge features from different cameras into a common space, providing a unified representation of the 3D scene. The key component is the view transformer, which transforms image views into the BeV. However, actual view transformer methods based on geometry or cross-attention do not provide a sufficiently detailed representation of the scene, as they use a sub-sampling of the 3D space that is non-optimal for modeling the fine structures of the environment. In this paper, we propose GaussianBeV, a novel method for transforming image features to BeV by finely representing the scene using a set of 3D gaussians located and oriented in 3D space. This representation is then splattered to produce the BeV feature map by adapting recent advances in 3D representation rendering based on gaussian splatting. GaussianBeV is the first approach to use this 3D gaussian modeling and 3D scene rendering process online, i.e. without optimizing it on a specific scene and directly integrated into a single stage model for BeV scene understanding. Experiments show that the proposed representation is highly effective and place GaussianBeV as the new state-of-the-art on the BeV semantic segmentation task on the nuScenes dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03428v1">2DGS-Room: Seed-Guided 2D Gaussian Splatting with Geometric Constrains for High-Fidelity Indoor Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      The reconstruction of indoor scenes remains challenging due to the inherent complexity of spatial structures and the prevalence of textureless regions. Recent advancements in 3D Gaussian Splatting have improved novel view synthesis with accelerated processing but have yet to deliver comparable performance in surface reconstruction. In this paper, we introduce 2DGS-Room, a novel method leveraging 2D Gaussian Splatting for high-fidelity indoor scene reconstruction. Specifically, we employ a seed-guided mechanism to control the distribution of 2D Gaussians, with the density of seed points dynamically optimized through adaptive growth and pruning mechanisms. To further improve geometric accuracy, we incorporate monocular depth and normal priors to provide constraints for details and textureless regions respectively. Additionally, multi-view consistency constraints are employed to mitigate artifacts and further enhance reconstruction quality. Extensive experiments on ScanNet and ScanNet++ datasets demonstrate that our method achieves state-of-the-art performance in indoor scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03371v1">SGSST: Scaling Gaussian Splatting StyleTransfer</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Applying style transfer to a full 3D environment is a challenging task that has seen many developments since the advent of neural rendering. 3D Gaussian splatting (3DGS) has recently pushed further many limits of neural rendering in terms of training speed and reconstruction quality. This work introduces SGSST: Scaling Gaussian Splatting Style Transfer, an optimization-based method to apply style transfer to pretrained 3DGS scenes. We demonstrate that a new multiscale loss based on global neural statistics, that we name SOS for Simultaneously Optimized Scales, enables style transfer to ultra-high resolution 3D scenes. Not only SGSST pioneers 3D scene style transfer at such high image resolutions, it also produces superior visual quality as assessed by thorough qualitative, quantitative and perceptual comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02245v2">SparseLGS: Sparse View Language Embedded Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Project Page: https://ustc3dv.github.io/SparseLGS
    </div>
    <details class="paper-abstract">
      Recently, several studies have combined Gaussian Splatting to obtain scene representations with language embeddings for open-vocabulary 3D scene understanding. While these methods perform well, they essentially require very dense multi-view inputs, limiting their applicability in real-world scenarios. In this work, we propose SparseLGS to address the challenge of 3D scene understanding with pose-free and sparse view input images. Our method leverages a learning-based dense stereo model to handle pose-free and sparse inputs, and a three-step region matching approach to address the multi-view semantic inconsistency problem, which is especially important for sparse inputs. Different from directly learning high-dimensional CLIP features, we extract low-dimensional information and build bijections to avoid excessive learning and storage costs. We introduce a reconstruction loss during semantic training to improve Gaussian positions and shapes. To the best of our knowledge, we are the first to address the 3D semantic field problem with sparse pose-free inputs. Experimental results show that SparseLGS achieves comparable quality when reconstructing semantic fields with fewer inputs (3-4 views) compared to previous SOTA methods with dense input. Besides, when using the same sparse input, SparseLGS leads significantly in quality and heavily improves the computation speed (5$\times$speedup). Project page: https://ustc3dv.github.io/SparseLGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03263v1">NeRF and Gaussian Splatting SLAM in the Wild</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 5 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Navigating outdoor environments with visual Simultaneous Localization and Mapping (SLAM) systems poses significant challenges due to dynamic scenes, lighting variations, and seasonal changes, requiring robust solutions. While traditional SLAM methods struggle with adaptability, deep learning-based approaches and emerging neural radiance fields as well as Gaussian Splatting-based SLAM methods, offer promising alternatives. However, these methods have primarily been evaluated in controlled indoor environments with stable conditions, leaving a gap in understanding their performance in unstructured and variable outdoor settings. This study addresses this gap by evaluating these methods in natural outdoor environments, focusing on camera tracking accuracy, robustness to environmental factors, and computational efficiency, highlighting distinct trade-offs. Extensive evaluations demonstrate that neural SLAM methods achieve superior robustness, particularly under challenging conditions such as low light, but at a high computational cost. At the same time, traditional methods perform the best across seasons but are highly sensitive to variations in lighting conditions. The code of the benchmark is publicly available at https://github.com/iis-esslingen/nerf-3dgs-benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03121v1">Splats in Splats: Embedding Invisible 3D Watermark within Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has demonstrated impressive 3D reconstruction performance with explicit scene representations. Given the widespread application of 3DGS in 3D reconstruction and generation tasks, there is an urgent need to protect the copyright of 3DGS assets. However, existing copyright protection techniques for 3DGS overlook the usability of 3D assets, posing challenges for practical deployment. Here we describe WaterGS, the first 3DGS watermarking framework that embeds 3D content in 3DGS itself without modifying any attributes of the vanilla 3DGS. To achieve this, we take a deep insight into spherical harmonics (SH) and devise an importance-graded SH coefficient encryption strategy to embed the hidden SH coefficients. Furthermore, we employ a convolutional autoencoder to establish a mapping between the original Gaussian primitives' opacity and the hidden Gaussian primitives' opacity. Extensive experiments indicate that WaterGS significantly outperforms existing 3D steganography techniques, with 5.31% higher scene fidelity and 3X faster rendering speed, while ensuring security, robustness, and user experience. Codes and data will be released at https://water-gs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03077v1">RoDyGS: Robust Dynamic Gaussian Splatting for Casual Videos</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Project Page: https://rodygs.github.io/
    </div>
    <details class="paper-abstract">
      Dynamic view synthesis (DVS) has advanced remarkably in recent years, achieving high-fidelity rendering while reducing computational costs. Despite the progress, optimizing dynamic neural fields from casual videos remains challenging, as these videos do not provide direct 3D information, such as camera trajectories or the underlying scene geometry. In this work, we present RoDyGS, an optimization pipeline for dynamic Gaussian Splatting from casual videos. It effectively learns motion and underlying geometry of scenes by separating dynamic and static primitives, and ensures that the learned motion and geometry are physically plausible by incorporating motion and geometric regularization terms. We also introduce a comprehensive benchmark, Kubric-MRig, that provides extensive camera and object motion along with simultaneous multi-view captures, features that are absent in previous benchmarks. Experimental results demonstrate that the proposed method significantly outperforms previous pose-free dynamic neural fields and achieves competitive rendering quality compared to existing pose-free static neural fields. The code and data are publicly available at https://rodygs.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01853v3">GVKF: Gaussian Voxel Kernel Functions for Highly Efficient Surface Reconstruction in Open Scenes</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      In this paper we present a novel method for efficient and effective 3D surface reconstruction in open scenes. Existing Neural Radiance Fields (NeRF) based works typically require extensive training and rendering time due to the adopted implicit representations. In contrast, 3D Gaussian splatting (3DGS) uses an explicit and discrete representation, hence the reconstructed surface is built by the huge number of Gaussian primitives, which leads to excessive memory consumption and rough surface details in sparse Gaussian areas. To address these issues, we propose Gaussian Voxel Kernel Functions (GVKF), which establish a continuous scene representation based on discrete 3DGS through kernel regression. The GVKF integrates fast 3DGS rasterization and highly effective scene implicit representations, achieving high-fidelity open scene surface reconstruction. Experiments on challenging scene datasets demonstrate the efficiency and effectiveness of our proposed GVKF, featuring with high reconstruction quality, real-time rendering speed, significant savings in storage and training memory consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01217v2">RGBDS-SLAM: A RGB-D Semantic Dense SLAM Based on 3D Multi Level Pyramid Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      High-quality reconstruction is crucial for dense SLAM. Recent popular approaches utilize 3D Gaussian Splatting (3D GS) techniques for RGB, depth, and semantic reconstruction of scenes. However, these methods often overlook issues of detail and consistency in different parts of the scene. To address this, we propose RGBDS-SLAM, a RGB-D semantic dense SLAM system based on 3D multi-level pyramid gaussian splatting, which enables high-quality dense reconstruction of scene RGB, depth, and semantics.In this system, we introduce a 3D multi-level pyramid gaussian splatting method that restores scene details by extracting multi-level image pyramids for gaussian splatting training, ensuring consistency in RGB, depth, and semantic reconstructions. Additionally, we design a tightly-coupled multi-features reconstruction optimization mechanism, allowing the reconstruction accuracy of RGB, depth, and semantic maps to mutually enhance each other during the rendering optimization process. Extensive quantitative, qualitative, and ablation experiments on the Replica and ScanNet public datasets demonstrate that our proposed method outperforms current state-of-the-art methods. The open-source code will be available at: https://github.com/zhenzhongcao/RGBDS-SLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02803v1">Gaussian Splatting Under Attack: Investigating Adversarial Noise in 3D Objects</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 Accepted to Safe Generative AI Workshop @ NeurIPS 2024: https://neurips.cc/virtual/2024/workshop/84705
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has advanced radiance field reconstruction, enabling high-quality view synthesis and fast rendering in 3D modeling. While adversarial attacks on object detection models are well-studied for 2D images, their impact on 3D models remains underexplored. This work introduces the Masked Iterative Fast Gradient Sign Method (M-IFGSM), designed to generate adversarial noise targeting the CLIP vision-language model. M-IFGSM specifically alters the object of interest by focusing perturbations on masked regions, degrading the performance of CLIP's zero-shot object detection capability when applied to 3D models. Using eight objects from the Common Objects 3D (CO3D) dataset, we demonstrate that our method effectively reduces the accuracy and confidence of the model, with adversarial noise being nearly imperceptible to human observers. The top-1 accuracy in original model renders drops from 95.4\% to 12.5\% for train images and from 91.2\% to 35.4\% for test images, with confidence levels reflecting this shift from true classification to misclassification, underscoring the risks of adversarial attacks on 3D models in applications such as autonomous driving, robotics, and surveillance. The significance of this research lies in its potential to expose vulnerabilities in modern 3D vision models, including radiance fields, prompting the development of more robust defenses and security measures in critical real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02684v1">AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 Project Page: https://lingtengqiu.github.io/2024/AniGS/
    </div>
    <details class="paper-abstract">
      Generating animatable human avatars from a single image is essential for various digital human modeling applications. Existing 3D reconstruction methods often struggle to capture fine details in animatable models, while generative approaches for controllable animation, though avoiding explicit 3D modeling, suffer from viewpoint inconsistencies in extreme poses and computational inefficiencies. In this paper, we address these challenges by leveraging the power of generative models to produce detailed multi-view canonical pose images, which help resolve ambiguities in animatable human reconstruction. We then propose a robust method for 3D reconstruction of inconsistent images, enabling real-time rendering during inference. Specifically, we adapt a transformer-based video generation model to generate multi-view canonical pose images and normal maps, pretraining on a large-scale video dataset to improve generalization. To handle view inconsistencies, we recast the reconstruction problem as a 4D task and introduce an efficient 3D modeling approach using 4D Gaussian Splatting. Experiments demonstrate that our method achieves photorealistic, real-time animation of 3D human avatars from in-the-wild images, showcasing its effectiveness and generalization capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14276v3">D-MiSo: Editing Dynamic 3D Scenes using Multi-Gaussians Soup</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Over the past years, we have observed an abundance of approaches for modeling dynamic 3D scenes using Gaussian Splatting (GS). Such solutions use GS to represent the scene's structure and the neural network to model dynamics. Such approaches allow fast rendering and extracting each element of such a dynamic scene. However, modifying such objects over time is challenging. SC-GS (Sparse Controlled Gaussian Splatting) enhanced with Deformed Control Points partially solves this issue. However, this approach necessitates selecting elements that need to be kept fixed, as well as centroids that should be adjusted throughout editing. Moreover, this task poses additional difficulties regarding the re-productivity of such editing. To address this, we propose Dynamic Multi-Gaussian Soup (D-MiSo), which allows us to model the mesh-inspired representation of dynamic GS. Additionally, we propose a strategy of linking parameterized Gaussian splats, forming a Triangle Soup with the estimated mesh. Consequently, we can separately construct new trajectories for the 3D objects composing the scene. Thus, we can make the scene's dynamic editable over time or while maintaining partial dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02493v1">RelayGS: Reconstructing Dynamic Scenes with Large-Scale and Complex Motions via Relay Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 Technical Report. GitHub: https://github.com/gqk/RelayGS
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic scenes with large-scale and complex motions remains a significant challenge. Recent techniques like Neural Radiance Fields and 3D Gaussian Splatting (3DGS) have shown promise but still struggle with scenes involving substantial movement. This paper proposes RelayGS, a novel method based on 3DGS, specifically designed to represent and reconstruct highly dynamic scenes. Our RelayGS learns a complete 4D representation with canonical 3D Gaussians and a compact motion field, consisting of three stages. First, we learn a fundamental 3DGS from all frames, ignoring temporal scene variations, and use a learnable mask to separate the highly dynamic foreground from the minimally moving background. Second, we replicate multiple copies of the decoupled foreground Gaussians from the first stage, each corresponding to a temporal segment, and optimize them using pseudo-views constructed from multiple frames within each segment. These Gaussians, termed Relay Gaussians, act as explicit relay nodes, simplifying and breaking down large-scale motion trajectories into smaller, manageable segments. Finally, we jointly learn the scene's temporal motion and refine the canonical Gaussians learned from the first two stages. We conduct thorough experiments on two dynamic scene datasets featuring large and complex motions, where our RelayGS outperforms state-of-the-arts by more than 1 dB in PSNR, and successfully reconstructs real-world basketball game scenes in a much more complete and coherent manner, whereas previous methods usually struggle to capture the complex motion of players. Code will be publicly available at https://github.com/gqk/RelayGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07266v5">Spiking GS: Towards High-Accuracy and Low-Cost Surface Reconstruction via Spiking Neuron-based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting is capable of reconstructing 3D scenes in minutes. Despite recent advances in improving surface reconstruction accuracy, the reconstructed results still exhibit bias and suffer from inefficiency in storage and training. This paper provides a different observation on the cause of the inefficiency and the reconstruction bias, which is attributed to the integration of the low-opacity parts (LOPs) of the generated Gaussians. We show that LOPs consist of Gaussians with overall low-opacity (LOGs) and the low-opacity tails (LOTs) of Gaussians. We propose Spiking GS to reduce such two types of LOPs by integrating spiking neurons into the Gaussian Splatting pipeline. Specifically, we introduce global and local full-precision integrate-and-fire spiking neurons to the opacity and representation function of flattened 3D Gaussians, respectively. Furthermore, we enhance the density control strategy with spiking neurons' thresholds and a new criterion on the scale of Gaussians. Our method can represent more accurate reconstructed surfaces at a lower cost. The supplementary material and code are available at https://github.com/zju-bmi-lab/SpikingGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02421v1">TimeWalker: Personalized Neural Space for Lifelong Head Avatars</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 Project Page: https://timewalker2024.github.io/timewalker.github.io/ , Video: https://www.youtube.com/watch?v=x8cpOVMY_ko
    </div>
    <details class="paper-abstract">
      We present TimeWalker, a novel framework that models realistic, full-scale 3D head avatars of a person on lifelong scale. Unlike current human head avatar pipelines that capture identity at the momentary level(e.g., instant photography or short videos), TimeWalker constructs a person's comprehensive identity from unstructured data collection over his/her various life stages, offering a paradigm to achieve full reconstruction and animation of that person at different moments of life. At the heart of TimeWalker's success is a novel neural parametric model that learns personalized representation with the disentanglement of shape, expression, and appearance across ages. Central to our methodology are the concepts of two aspects: (1) We track back to the principle of modeling a person's identity in an additive combination of average head representation in the canonical space, and moment-specific head attribute representations driven from a set of neural head basis. To learn the set of head basis that could represent the comprehensive head variations in a compact manner, we propose a Dynamic Neural Basis-Blending Module (Dynamo). It dynamically adjusts the number and blend weights of neural head bases, according to both shared and specific traits of the target person over ages. (2) Dynamic 2D Gaussian Splatting (DNA-2DGS), an extension of Gaussian splatting representation, to model head motion deformations like facial expressions without losing the realism of rendering and reconstruction. DNA-2DGS includes a set of controllable 2D oriented planar Gaussian disks that utilize the priors from parametric model, and move/rotate with the change of expression. Through extensive experimental evaluations, we show TimeWalker's ability to reconstruct and animate avatars across decoupled dimensions with realistic rendering effects, demonstrating a way to achieve personalized 'time traveling' in a breeze.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02267v1">GSGTrack: Gaussian Splatting-Guided Object Pose Tracking from RGB Videos</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Tracking the 6DoF pose of unknown objects in monocular RGB video sequences is crucial for robotic manipulation. However, existing approaches typically rely on accurate depth information, which is non-trivial to obtain in real-world scenarios. Although depth estimation algorithms can be employed, geometric inaccuracy can lead to failures in RGBD-based pose tracking methods. To address this challenge, we introduce GSGTrack, a novel RGB-based pose tracking framework that jointly optimizes geometry and pose. Specifically, we adopt 3D Gaussian Splatting to create an optimizable 3D representation, which is learned simultaneously with a graph-based geometry optimization to capture the object's appearance features and refine its geometry. However, the joint optimization process is susceptible to perturbations from noisy pose and geometry data. Thus, we propose an object silhouette loss to address the issue of pixel-wise loss being overly sensitive to pose noise during tracking. To mitigate the geometric ambiguities caused by inaccurate depth information, we propose a geometry-consistent image pair selection strategy, which filters out low-confidence pairs and ensures robust geometric optimization. Extensive experiments on the OnePose and HO3D datasets demonstrate the effectiveness of GSGTrack in both 6DoF pose tracking and object reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02249v1">Multi-robot autonomous 3D reconstruction using Gaussian splatting with Semantic guidance</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Implicit neural representations and 3D Gaussian splatting (3DGS) have shown great potential for scene reconstruction. Recent studies have expanded their applications in autonomous reconstruction through task assignment methods. However, these methods are mainly limited to single robot, and rapid reconstruction of large-scale scenes remains challenging. Additionally, task-driven planning based on surface uncertainty is prone to being trapped in local optima. To this end, we propose the first 3DGS-based centralized multi-robot autonomous 3D reconstruction framework. To further reduce time cost of task generation and improve reconstruction quality, we integrate online open-vocabulary semantic segmentation with surface uncertainty of 3DGS, focusing view sampling on regions with high instance uncertainty. Finally, we develop a multi-robot collaboration strategy with mode and task assignments improving reconstruction quality while ensuring planning efficiency. Our method demonstrates the highest reconstruction quality among all planning methods and superior planning efficiency compared to existing multi-robot methods. We deploy our method on multiple robots, and results show that it can effectively plan view paths and reconstruct scenes with high quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02225v1">How to Use Diffusion Priors under Sparse Views?</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Novel view synthesis under sparse views has been a long-term important challenge in 3D reconstruction. Existing works mainly rely on introducing external semantic or depth priors to supervise the optimization of 3D representations. However, the diffusion model, as an external prior that can directly provide visual supervision, has always underperformed in sparse-view 3D reconstruction using Score Distillation Sampling (SDS) due to the low information entropy of sparse views compared to text, leading to optimization challenges caused by mode deviation. To this end, we present a thorough analysis of SDS from the mode-seeking perspective and propose Inline Prior Guided Score Matching (IPSM), which leverages visual inline priors provided by pose relationships between viewpoints to rectify the rendered image distribution and decomposes the original optimization objective of SDS, thereby offering effective diffusion visual guidance without any fine-tuning or pre-training. Furthermore, we propose the IPSM-Gaussian pipeline, which adopts 3D Gaussian Splatting as the backbone and supplements depth and geometry consistency regularization based on IPSM to further improve inline priors and rectified distribution. Experimental results on different public datasets show that our method achieves state-of-the-art reconstruction quality. The code is released at https://github.com/iCVTEAM/IPSM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01552v2">GFreeDet: Exploiting Gaussian Splatting and Foundation Models for Model-free Unseen Object Detection in the BOP Challenge 2024</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      In this report, we provide the technical details of the submitted method GFreeDet, which exploits Gaussian splatting and vision Foundation models for the model-free unseen object Detection track in the BOP 2024 Challenge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02140v1">SparseGrasp: Robotic Grasping via 3D Semantic Gaussian Splatting from Sparse Multi-View RGB Images</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Language-guided robotic grasping is a rapidly advancing field where robots are instructed using human language to grasp specific objects. However, existing methods often depend on dense camera views and struggle to quickly update scenes, limiting their effectiveness in changeable environments. In contrast, we propose SparseGrasp, a novel open-vocabulary robotic grasping system that operates efficiently with sparse-view RGB images and handles scene updates fastly. Our system builds upon and significantly enhances existing computer vision modules in robotic learning. Specifically, SparseGrasp utilizes DUSt3R to generate a dense point cloud as the initialization for 3D Gaussian Splatting (3DGS), maintaining high fidelity even under sparse supervision. Importantly, SparseGrasp incorporates semantic awareness from recent vision foundation models. To further improve processing efficiency, we repurpose Principal Component Analysis (PCA) to compress features from 2D models. Additionally, we introduce a novel render-and-compare strategy that ensures rapid scene updates, enabling multi-turn grasping in changeable environments. Experimental results show that SparseGrasp significantly outperforms state-of-the-art methods in terms of both speed and adaptability, providing a robust solution for multi-turn grasping in changeable environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02075v1">Gaussian Object Carver: Object-Compositional Gaussian Splatting with surfaces completion</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      3D scene reconstruction is a foundational problem in computer vision. Despite recent advancements in Neural Implicit Representations (NIR), existing methods often lack editability and compositional flexibility, limiting their use in scenarios requiring high interactivity and object-level manipulation. In this paper, we introduce the Gaussian Object Carver (GOC), a novel, efficient, and scalable framework for object-compositional 3D scene reconstruction. GOC leverages 3D Gaussian Splatting (GS), enriched with monocular geometry priors and multi-view geometry regularization, to achieve high-quality and flexible reconstruction. Furthermore, we propose a zero-shot Object Surface Completion (OSC) model, which uses 3D priors from 3d object data to reconstruct unobserved surfaces, ensuring object completeness even in occluded areas. Experimental results demonstrate that GOC improves reconstruction efficiency and geometric fidelity. It holds promise for advancing the practical application of digital twins in embodied AI, AR/VR, and interactive simulation environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01459v4">GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) is a novel, state-of-the-art technique for rendering points in a 3D scene by approximating their contribution to image pixels through Gaussian distributions, warranting fast training and real-time rendering. The main drawback of GS is the absence of a well-defined approach for its conditioning due to the necessity of conditioning several hundred thousand Gaussian components. To solve this, we introduce the Gaussian Mesh Splatting (GaMeS) model, which allows modification of Gaussian components in a similar way as meshes. We parameterize each Gaussian component by the vertices of the mesh face. Furthermore, our model needs mesh initialization on input or estimated mesh during training. We also define Gaussian splats solely based on their location on the mesh, allowing for automatic adjustments in position, scale, and rotation during animation. As a result, we obtain a real-time rendering of editable GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01931v1">Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025
    </div>
    <details class="paper-abstract">
      This paper presents Planar Gaussian Splatting (PGS), a novel neural rendering approach to learn the 3D geometry and parse the 3D planes of a scene, directly from multiple RGB images. The PGS leverages Gaussian primitives to model the scene and employ a hierarchical Gaussian mixture approach to group them. Similar Gaussians are progressively merged probabilistically in the tree-structured Gaussian mixtures to identify distinct 3D plane instances and form the overall 3D scene geometry. In order to enable the grouping, the Gaussian primitives contain additional parameters, such as plane descriptors derived by lifting 2D masks from a general 2D segmentation model and surface normals. Experiments show that the proposed PGS achieves state-of-the-art performance in 3D planar reconstruction without requiring either 3D plane labels or depth supervision. In contrast to existing supervised methods that have limited generalizability and struggle under domain shift, PGS maintains its performance across datasets thanks to its neural rendering and scene-specific optimization mechanism, while also being significantly faster than existing optimization-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01823v1">HDGS: Textured 2D Gaussian Splatting for Enhanced Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project Page: https://timsong412.github.io/HDGS-ProjPage/
    </div>
    <details class="paper-abstract">
      Recent advancements in neural rendering, particularly 2D Gaussian Splatting (2DGS), have shown promising results for jointly reconstructing fine appearance and geometry by leveraging 2D Gaussian surfels. However, current methods face significant challenges when rendering at arbitrary viewpoints, such as anti-aliasing for down-sampled rendering, and texture detail preservation for high-resolution rendering. We proposed a novel method to align the 2D surfels with texture maps and augment it with per-ray depth sorting and fisher-based pruning for rendering consistency and efficiency. With correct order, per-surfel texture maps significantly improve the capabilities to capture fine details. Additionally, to render high-fidelity details in varying viewpoints, we designed a frustum-based sampling method to mitigate the aliasing artifacts. Experimental results on benchmarks and our custom texture-rich dataset demonstrate that our method surpasses existing techniques, particularly in detail preservation and anti-aliasing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01807v1">Occam's LGS: A Simple Approach for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project Page: https://insait-institute.github.io/OccamLGS/
    </div>
    <details class="paper-abstract">
      TL;DR: Gaussian Splatting is a widely adopted approach for 3D scene representation that offers efficient, high-quality 3D reconstruction and rendering. A major reason for the success of 3DGS is its simplicity of representing a scene with a set of Gaussians, which makes it easy to interpret and adapt. To enhance scene understanding beyond the visual representation, approaches have been developed that extend 3D Gaussian Splatting with semantic vision-language features, especially allowing for open-set tasks. In this setting, the language features of 3D Gaussian Splatting are often aggregated from multiple 2D views. Existing works address this aggregation problem using cumbersome techniques that lead to high computational cost and training time. In this work, we show that the sophisticated techniques for language-grounded 3D Gaussian Splatting are simply unnecessary. Instead, we apply Occam's razor to the task at hand and perform weighted multi-view feature aggregation using the weights derived from the standard rendering process, followed by a simple heuristic-based noisy Gaussian filtration. Doing so offers us state-of-the-art results with a speed-up of two orders of magnitude. We showcase our results in two commonly used benchmark datasets: LERF and 3D-OVS. Our simple approach allows us to perform reasoning directly in the language features, without any compression whatsoever. Such modeling in turn offers easy scene manipulation, unlike the existing methods -- which we illustrate using an application of object insertion in the scene. Furthermore, we provide a thorough discussion regarding the significance of our contributions within the context of the current literature. Project Page: https://insait-institute.github.io/OccamLGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01792v1">CTRL-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project page: https://ihe-kaii.github.io/CTRL-D/
    </div>
    <details class="paper-abstract">
      Recent advances in 3D representations, such as Neural Radiance Fields and 3D Gaussian Splatting, have greatly improved realistic scene modeling and novel-view synthesis. However, achieving controllable and consistent editing in dynamic 3D scenes remains a significant challenge. Previous work is largely constrained by its editing backbones, resulting in inconsistent edits and limited controllability. In our work, we introduce a novel framework that first fine-tunes the InstructPix2Pix model, followed by a two-stage optimization of the scene based on deformable 3D Gaussians. Our fine-tuning enables the model to "learn" the editing ability from a single edited reference image, transforming the complex task of dynamic scene editing into a simple 2D image editing process. By directly learning editing regions and styles from the reference, our approach enables consistent and precise local edits without the need for tracking desired editing regions, effectively addressing key challenges in dynamic scene editing. Then, our two-stage optimization progressively edits the trained dynamic scene, using a designed edited image buffer to accelerate convergence and improve temporal consistency. Compared to state-of-the-art methods, our approach offers more flexible and controllable local scene editing, achieving high-quality and consistent results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19895v2">GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project page: https://narcissusex.github.io/GuardSplat and Code: https://github.com/NarcissusEx/GuardSplat
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently created impressive assets for various applications. However, the copyright of these assets is not well protected as existing watermarking methods are not suited for 3DGS considering security, capacity, and invisibility. Besides, these methods often require hours or even days for optimization, limiting the application scenarios. In this paper, we propose GuardSplat, an innovative and efficient framework that effectively protects the copyright of 3DGS assets. Specifically, 1) We first propose a CLIP-guided Message Decoupling Optimization module for training the message decoder, leveraging CLIP's aligning capability and rich representations to achieve a high extraction accuracy with minimal optimization costs, presenting exceptional capability and efficiency. 2) Then, we propose a Spherical-harmonic-aware (SH-aware) Message Embedding module tailored for 3DGS, which employs a set of SH offsets to seamlessly embed the message into the SH features of each 3D Gaussian while maintaining the original 3D structure. It enables the 3DGS assets to be watermarked with minimal fidelity trade-offs and prevents malicious users from removing the messages from the model files, meeting the demands for invisibility and security. 3) We further propose an Anti-distortion Message Extraction module to improve robustness against various visual distortions. Extensive experiments demonstrate that GuardSplat outperforms the state-of-the-art methods and achieves fast optimization speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01745v1">Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Seamless integration of both aerial and street view images remains a significant challenge in neural scene reconstruction and rendering. Existing methods predominantly focus on single domain, limiting their applications in immersive environments, which demand extensive free view exploration with large view changes both horizontally and vertically. We introduce Horizon-GS, a novel approach built upon Gaussian Splatting techniques, tackles the unified reconstruction and rendering for aerial and street views. Our method addresses the key challenges of combining these perspectives with a new training strategy, overcoming viewpoint discrepancies to generate high-fidelity scenes. We also curate a high-quality aerial-to-ground views dataset encompassing both synthetic and real-world scene to advance further research. Experiments across diverse urban scene datasets confirm the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01718v1">HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Our project page is at https://xdimlab.github.io/HUGSIM
    </div>
    <details class="paper-abstract">
      In the past few decades, autonomous driving algorithms have made significant progress in perception, planning, and control. However, evaluating individual components does not fully reflect the performance of entire systems, highlighting the need for more holistic assessment methods. This motivates the development of HUGSIM, a closed-loop, photo-realistic, and real-time simulator for evaluating autonomous driving algorithms. We achieve this by lifting captured 2D RGB images into the 3D space via 3D Gaussian Splatting, improving the rendering quality for closed-loop scenarios, and building the closed-loop environment. In terms of rendering, We tackle challenges of novel view synthesis in closed-loop scenarios, including viewpoint extrapolation and 360-degree vehicle rendering. Beyond novel view synthesis, HUGSIM further enables the full closed simulation loop, dynamically updating the ego and actor states and observations based on control commands. Moreover, HUGSIM offers a comprehensive benchmark across more than 70 sequences from KITTI-360, Waymo, nuScenes, and PandaSet, along with over 400 varying scenarios, providing a fair and realistic evaluation platform for existing autonomous driving algorithms. HUGSIM not only serves as an intuitive evaluation benchmark but also unlocks the potential for fine-tuning autonomous driving algorithms in a photorealistic closed-loop setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01717v1">Driving Scene Synthesis on Free-form Trajectories with Generative Prior</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Driving scene synthesis along free-form trajectories is essential for driving simulations to enable closed-loop evaluation of end-to-end driving policies. While existing methods excel at novel view synthesis on recorded trajectories, they face challenges with novel trajectories due to limited views of driving videos and the vastness of driving environments. To tackle this challenge, we propose a novel free-form driving view synthesis approach, dubbed DriveX, by leveraging video generative prior to optimize a 3D model across a variety of trajectories. Concretely, we crafted an inverse problem that enables a video diffusion model to be utilized as a prior for many-trajectory optimization of a parametric 3D model (e.g., Gaussian splatting). To seamlessly use the generative prior, we iteratively conduct this process during optimization. Our resulting model can produce high-fidelity virtual driving environments outside the recorded trajectory, enabling free-form trajectory driving simulation. Beyond real driving scenes, DriveX can also be utilized to simulate virtual driving worlds from AI-generated videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06390v2">Deepfake for the Good: Generating Avatars through Face-Swapping with Implicit Deepfake Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Numerous emerging deep-learning techniques have had a substantial impact on computer graphics. Among the most promising breakthroughs are the rise of Neural Radiance Fields (NeRFs) and Gaussian Splatting (GS). NeRFs encode the object's shape and color in neural network weights using a handful of images with known camera positions to generate novel views. In contrast, GS provides accelerated training and inference without a decrease in rendering quality by encoding the object's characteristics in a collection of Gaussian distributions. These two techniques have found many use cases in spatial computing and other domains. On the other hand, the emergence of deepfake methods has sparked considerable controversy. Deepfakes refers to artificial intelligence-generated videos that closely mimic authentic footage. Using generative models, they can modify facial features, enabling the creation of altered identities or expressions that exhibit a remarkably realistic appearance to a real person. Despite these controversies, deepfake can offer a next-generation solution for avatar creation and gaming when of desirable quality. To that end, we show how to combine all these emerging technologies to obtain a more plausible outcome. Our ImplicitDeepfake uses the classical deepfake algorithm to modify all training images separately and then train NeRF and GS on modified faces. Such simple strategies can produce plausible 3D deepfake-based avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01553v1">SfM-Free 3D Gaussian Splatting via Hierarchical Training</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Standard 3D Gaussian Splatting (3DGS) relies on known or pre-computed camera poses and a sparse point cloud, obtained from structure-from-motion (SfM) preprocessing, to initialize and grow 3D Gaussians. We propose a novel SfM-Free 3DGS (SFGS) method for video input, eliminating the need for known camera poses and SfM preprocessing. Our approach introduces a hierarchical training strategy that trains and merges multiple 3D Gaussian representations -- each optimized for specific scene regions -- into a single, unified 3DGS model representing the entire scene. To compensate for large camera motions, we leverage video frame interpolation models. Additionally, we incorporate multi-source supervision to reduce overfitting and enhance representation. Experimental results reveal that our approach significantly surpasses state-of-the-art SfM-free novel view synthesis methods. On the Tanks and Temples dataset, we improve PSNR by an average of 2.25dB, with a maximum gain of 3.72dB in the best scene. On the CO3D-V2 dataset, we achieve an average PSNR boost of 1.74dB, with a top gain of 3.90dB. The code is available at https://github.com/jibo27/3DGS_Hierarchical_Training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01543v1">6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Efficient and accurate object pose estimation is an essential component for modern vision systems in many applications such as Augmented Reality, autonomous driving, and robotics. While research in model-based 6D object pose estimation has delivered promising results, model-free methods are hindered by the high computational load in rendering and inferring consistent poses of arbitrary objects in a live RGB-D video stream. To address this issue, we present 6DOPE-GS, a novel method for online 6D object pose estimation \& tracking with a single RGB-D camera by effectively leveraging advances in Gaussian Splatting. Thanks to the fast differentiable rendering capabilities of Gaussian Splatting, 6DOPE-GS can simultaneously optimize for 6D object poses and 3D object reconstruction. To achieve the necessary efficiency and accuracy for live tracking, our method uses incremental 2D Gaussian Splatting with an intelligent dynamic keyframe selection procedure to achieve high spatial object coverage and prevent erroneous pose updates. We also propose an opacity statistic-based pruning mechanism for adaptive Gaussian density control, to ensure training stability and efficiency. We evaluate our method on the HO3D and YCBInEOAT datasets and show that 6DOPE-GS matches the performance of state-of-the-art baselines for model-free simultaneous 6D pose tracking and reconstruction while providing a 5$\times$ speedup. We also demonstrate the method's suitability for live, dynamic object tracking and reconstruction in a real-world setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12440v3">Beyond Gaussians: Fast and High-Fidelity 3D Splatting with Linear Kernels</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have substantially improved novel view synthesis, enabling high-quality reconstruction and real-time rendering. However, blurring artifacts, such as floating primitives and over-reconstruction, remain challenging. Current methods address these issues by refining scene structure, enhancing geometric representations, addressing blur in training images, improving rendering consistency, and optimizing density control, yet the role of kernel design remains underexplored. We identify the soft boundaries of Gaussian ellipsoids as one of the causes of these artifacts, limiting detail capture in high-frequency regions. To bridge this gap, we introduce 3D Linear Splatting (3DLS), which replaces Gaussian kernels with linear kernels to achieve sharper and more precise results, particularly in high-frequency regions. Through evaluations on three datasets, 3DLS demonstrates state-of-the-art fidelity and accuracy, along with a 30% FPS improvement over baseline 3DGS. The implementation will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01402v1">ULSR-GS: Ultra Large-scale Surface Reconstruction Gaussian Splatting with Multi-View Geometric Consistency</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project page: https://ulsrgs.github.io
    </div>
    <details class="paper-abstract">
      While Gaussian Splatting (GS) demonstrates efficient and high-quality scene rendering and small area surface extraction ability, it falls short in handling large-scale aerial image surface extraction tasks. To overcome this, we present ULSR-GS, a framework dedicated to high-fidelity surface extraction in ultra-large-scale scenes, addressing the limitations of existing GS-based mesh extraction methods. Specifically, we propose a point-to-photo partitioning approach combined with a multi-view optimal view matching principle to select the best training images for each sub-region. Additionally, during training, ULSR-GS employs a densification strategy based on multi-view geometric consistency to enhance surface extraction details. Experimental results demonstrate that ULSR-GS outperforms other state-of-the-art GS-based works on large-scale aerial photogrammetry benchmark datasets, significantly improving surface extraction accuracy in complex urban environments. Project page: https://ulsrgs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19454v2">GausSurf: Geometry-Guided 3D Gaussian Splatting for Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project page: https://jiepengwang.github.io/GausSurf/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has achieved impressive performance in novel view synthesis with real-time rendering capabilities. However, reconstructing high-quality surfaces with fine details using 3D Gaussians remains a challenging task. In this work, we introduce GausSurf, a novel approach to high-quality surface reconstruction by employing geometry guidance from multi-view consistency in texture-rich areas and normal priors in texture-less areas of a scene. We observe that a scene can be mainly divided into two primary regions: 1) texture-rich and 2) texture-less areas. To enforce multi-view consistency at texture-rich areas, we enhance the reconstruction quality by incorporating a traditional patch-match based Multi-View Stereo (MVS) approach to guide the geometry optimization in an iterative scheme. This scheme allows for mutual reinforcement between the optimization of Gaussians and patch-match refinement, which significantly improves the reconstruction results and accelerates the training process. Meanwhile, for the texture-less areas, we leverage normal priors from a pre-trained normal estimation model to guide optimization. Extensive experiments on the DTU and Tanks and Temples datasets demonstrate that our method surpasses state-of-the-art methods in terms of reconstruction quality and computation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16323v2">LeanGaussian: Breaking Pixel or Point Cloud Correspondence in Modeling 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Rencently, Gaussian splatting has demonstrated significant success in novel view synthesis. Current methods often regress Gaussians with pixel or point cloud correspondence, linking each Gaussian with a pixel or a 3D point. This leads to the redundancy of Gaussians being used to overfit the correspondence rather than the objects represented by the 3D Gaussians themselves, consequently wasting resources and lacking accurate geometries or textures. In this paper, we introduce LeanGaussian, a novel approach that treats each query in deformable Transformer as one 3D Gaussian ellipsoid, breaking the pixel or point cloud correspondence constraints. We leverage deformable decoder to iteratively refine the Gaussians layer-by-layer with the image features as keys and values. Notably, the center of each 3D Gaussian is defined as 3D reference points, which are then projected onto the image for deformable attention in 2D space. On both the ShapeNet SRN dataset (category level) and the Google Scanned Objects dataset (open-category level, trained with the Objaverse dataset), our approach, outperforms prior methods by approximately 6.1\%, achieving a PSNR of 25.44 and 22.36, respectively. Additionally, our method achieves a 3D reconstruction speed of 7.2 FPS and rendering speed 500 FPS. The code will be released at https://github.com/jwubz123/DIG3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00905v1">Ref-GS: Directional Factorization for 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-01
      | 💬 Project page: https://ref-gs.github.io/
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Ref-GS, a novel approach for directional light factorization in 2D Gaussian splatting, which enables photorealistic view-dependent appearance rendering and precise geometry recovery. Ref-GS builds upon the deferred rendering of Gaussian splatting and applies directional encoding to the deferred-rendered surface, effectively reducing the ambiguity between orientation and viewing angle. Next, we introduce a spherical Mip-grid to capture varying levels of surface roughness, enabling roughness-aware Gaussian shading. Additionally, we propose a simple yet efficient geometry-lighting factorization that connects geometry and lighting via the vector outer product, significantly reducing renderer overhead when integrating volumetric attributes. Our method achieves superior photorealistic rendering for a range of open-world scenes while also accurately recovering geometry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00851v1">DynSUP: Dynamic Gaussian Splatting from An Unposed Image Pair</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting have shown promising results. Existing methods typically assume static scenes and/or multiple images with prior poses. Dynamics, sparse views, and unknown poses significantly increase the problem complexity due to insufficient geometric constraints. To overcome this challenge, we propose a method that can use only two images without prior poses to fit Gaussians in dynamic environments. To achieve this, we introduce two technical contributions. First, we propose an object-level two-view bundle adjustment. This strategy decomposes dynamic scenes into piece-wise rigid components, and jointly estimates the camera pose and motions of dynamic objects. Second, we design an SE(3) field-driven Gaussian training method. It enables fine-grained motion modeling through learnable per-Gaussian transformations. Our method leads to high-fidelity novel view synthesis of dynamic scenes while accurately preserving temporal consistency and object motion. Experiments on both synthetic and real-world datasets demonstrate that our method significantly outperforms state-of-the-art approaches designed for the cases of static environments, multiple images, and/or known poses. Our project page is available at https://colin-de.github.io/DynSUP/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00734v1">ChatSplat: 3D Conversational Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      Humans naturally interact with their 3D surroundings using language, and modeling 3D language fields for scene understanding and interaction has gained growing interest. This paper introduces ChatSplat, a system that constructs a 3D language field, enabling rich chat-based interaction within 3D space. Unlike existing methods that primarily use CLIP-derived language features focused solely on segmentation, ChatSplat facilitates interaction on three levels: objects, views, and the entire 3D scene. For view-level interaction, we designed an encoder that encodes the rendered feature map of each view into tokens, which are then processed by a large language model (LLM) for conversation. At the scene level, ChatSplat combines multi-view tokens, enabling interactions that consider the entire scene. For object-level interaction, ChatSplat uses a patch-wise language embedding, unlike LangSplat's pixel-wise language embedding that implicitly includes mask and embedding. Here, we explicitly decouple the language embedding into separate mask and feature map representations, allowing more flexible object-level interaction. To address the challenge of learning 3D Gaussians posed by the complex and diverse distribution of language embeddings used in the LLM, we introduce a learnable normalization technique to standardize these embeddings, facilitating effective learning. Extensive experimental results demonstrate that ChatSplat supports multi-level interactions -- object, view, and scene -- within 3D space, enhancing both understanding and engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00682v1">FlashSLAM: Accelerated RGB-D SLAM for Real-Time 3D Scene Reconstruction with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-01
      | 💬 16 pages, 9 figures, 13 tables
    </div>
    <details class="paper-abstract">
      We present FlashSLAM, a novel SLAM approach that leverages 3D Gaussian Splatting for efficient and robust 3D scene reconstruction. Existing 3DGS-based SLAM methods often fall short in sparse view settings and during large camera movements due to their reliance on gradient descent-based optimization, which is both slow and inaccurate. FlashSLAM addresses these limitations by combining 3DGS with a fast vision-based camera tracking technique, utilizing a pretrained feature matching model and point cloud registration for precise pose estimation in under 80 ms - a 90% reduction in tracking time compared to SplaTAM - without costly iterative rendering. In sparse settings, our method achieves up to a 92% improvement in average tracking accuracy over previous methods. Additionally, it accounts for noise in depth sensors, enhancing robustness when using unspecialized devices such as smartphones. Extensive experiments show that FlashSLAM performs reliably across both sparse and dense settings, in synthetic and real-world environments. Evaluations on benchmark datasets highlight its superior accuracy and efficiency, establishing FlashSLAM as a versatile and high-performance solution for SLAM, advancing the state-of-the-art in 3D reconstruction across diverse applications.
    </details>
</div>
