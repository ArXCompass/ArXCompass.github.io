# gaussian splatting - 2024_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.14937v3">SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2024-03-31
      | 💬 Code link: https://github.com/yihua7/SC-GS
    </div>
    <details class="paper-abstract">
      Novel view synthesis for dynamic scenes is still a challenging problem in computer vision and graphics. Recently, Gaussian splatting has emerged as a robust technique to represent static scenes and enable high-quality and real-time novel view synthesis. Building upon this technique, we propose a new representation that explicitly decomposes the motion and appearance of dynamic scenes into sparse control points and dense Gaussians, respectively. Our key idea is to use sparse control points, significantly fewer in number than the Gaussians, to learn compact 6 DoF transformation bases, which can be locally interpolated through learned interpolation weights to yield the motion field of 3D Gaussians. We employ a deformation MLP to predict time-varying 6 DoF transformations for each control point, which reduces learning complexities, enhances learning abilities, and facilitates obtaining temporal and spatial coherent motion patterns. Then, we jointly learn the 3D Gaussians, the canonical space locations of control points, and the deformation MLP to reconstruct the appearance, geometry, and dynamics of 3D scenes. During learning, the location and number of control points are adaptively adjusted to accommodate varying motion complexities in different regions, and an ARAP loss following the principle of as rigid as possible is developed to enforce spatial continuity and local rigidity of learned motions. Finally, thanks to the explicit sparse motion representation and its decomposition from appearance, our method can enable user-controlled motion editing while retaining high-fidelity appearances. Extensive experiments demonstrate that our approach outperforms existing approaches on novel view synthesis with a high rendering speed and enables novel appearance-preserved motion editing applications. Project page: https://yihua7.github.io/SC-GS-web/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.16084v2">LangSplat: 3D Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-31
      | 💬 CVPR 2024. Project Page: https://langsplat.github.io
    </div>
    <details class="paper-abstract">
      Humans live in a 3D world and commonly use natural language to interact with a 3D scene. Modeling a 3D language field to support open-ended language queries in 3D has gained increasing attention recently. This paper introduces LangSplat, which constructs a 3D language field that enables precise and efficient open-vocabulary querying within 3D spaces. Unlike existing methods that ground CLIP language embeddings in a NeRF model, LangSplat advances the field by utilizing a collection of 3D Gaussians, each encoding language features distilled from CLIP, to represent the language field. By employing a tile-based splatting technique for rendering language features, we circumvent the costly rendering process inherent in NeRF. Instead of directly learning CLIP embeddings, LangSplat first trains a scene-wise language autoencoder and then learns language features on the scene-specific latent space, thereby alleviating substantial memory demands imposed by explicit modeling. Existing methods struggle with imprecise and vague 3D language fields, which fail to discern clear boundaries between objects. We delve into this issue and propose to learn hierarchical semantics using SAM, thereby eliminating the need for extensively querying the language field across various scales and the regularization of DINO features. Extensive experimental results show that LangSplat significantly outperforms the previous state-of-the-art method LERF by a large margin. Notably, LangSplat is extremely efficient, achieving a 199 $\times$ speedup compared to LERF at the resolution of 1440 $\times$ 1080. We strongly recommend readers to check out our video results at https://langsplat.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20275v1">Snap-it, Tap-it, Splat-it: Tactile-Informed 3D Gaussian Splatting for Reconstructing Challenging Surfaces</a></div>
    <div class="paper-meta">
      📅 2024-03-29
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Touch and vision go hand in hand, mutually enhancing our ability to understand the world. From a research perspective, the problem of mixing touch and vision is underexplored and presents interesting challenges. To this end, we propose Tactile-Informed 3DGS, a novel approach that incorporates touch data (local depth maps) with multi-view vision data to achieve surface reconstruction and novel view synthesis. Our method optimises 3D Gaussian primitives to accurately model the object's geometry at points of contact. By creating a framework that decreases the transmittance at touch locations, we achieve a refined surface reconstruction, ensuring a uniformly smooth depth map. Touch is particularly useful when considering non-Lambertian objects (e.g. shiny or reflective surfaces) since contemporary methods tend to fail to reconstruct with fidelity specular highlights. By combining vision and tactile sensing, we achieve more accurate geometry reconstructions with fewer images than prior methods. We conduct evaluation on objects with glossy and reflective surfaces and demonstrate the effectiveness of our approach, offering significant improvements in reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20159v1">HGS-Mapping: Online Dense Mapping Using Hybrid Gaussian Representation in Urban Scenes</a></div>
    <div class="paper-meta">
      📅 2024-03-29
    </div>
    <details class="paper-abstract">
      Online dense mapping of urban scenes forms a fundamental cornerstone for scene understanding and navigation of autonomous vehicles. Recent advancements in mapping methods are mainly based on NeRF, whose rendering speed is too slow to meet online requirements. 3D Gaussian Splatting (3DGS), with its rendering speed hundreds of times faster than NeRF, holds greater potential in online dense mapping. However, integrating 3DGS into a street-view dense mapping framework still faces two challenges, including incomplete reconstruction due to the absence of geometric information beyond the LiDAR coverage area and extensive computation for reconstruction in large urban scenes. To this end, we propose HGS-Mapping, an online dense mapping framework in unbounded large-scale scenes. To attain complete construction, our framework introduces Hybrid Gaussian Representation, which models different parts of the entire scene using Gaussians with distinct properties. Furthermore, we employ a hybrid Gaussian initialization mechanism and an adaptive update method to achieve high-fidelity and rapid reconstruction. To the best of our knowledge, we are the first to integrate Gaussian representation into online dense mapping of urban scenes. Our approach achieves SOTA reconstruction accuracy while only employing 66% number of Gaussians, leading to 20% faster reconstruction speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20079v1">SGD: Street View Synthesis with Gaussian Splatting and Diffusion Prior</a></div>
    <div class="paper-meta">
      📅 2024-03-29
    </div>
    <details class="paper-abstract">
      Novel View Synthesis (NVS) for street scenes play a critical role in the autonomous driving simulation. The current mainstream technique to achieve it is neural rendering, such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Although thrilling progress has been made, when handling street scenes, current methods struggle to maintain rendering quality at the viewpoint that deviates significantly from the training viewpoints. This issue stems from the sparse training views captured by a fixed camera on a moving vehicle. To tackle this problem, we propose a novel approach that enhances the capacity of 3DGS by leveraging prior from a Diffusion Model along with complementary multi-modal data. Specifically, we first fine-tune a Diffusion Model by adding images from adjacent frames as condition, meanwhile exploiting depth data from LiDAR point clouds to supply additional spatial information. Then we apply the Diffusion Model to regularize the 3DGS at unseen views during training. Experimental results validate the effectiveness of our method compared with current state-of-the-art models, and demonstrate its advance in rendering images from broader views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.16653v2">DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation</a></div>
    <div class="paper-meta">
      📅 2024-03-29
      | 💬 Camera-ready version. Project page: https://dreamgaussian.github.io/
    </div>
    <details class="paper-abstract">
      Recent advances in 3D content creation mostly leverage optimization-based 3D generation via score distillation sampling (SDS). Though promising results have been exhibited, these methods often suffer from slow per-sample optimization, limiting their practical usage. In this paper, we propose DreamGaussian, a novel 3D content generation framework that achieves both efficiency and quality simultaneously. Our key insight is to design a generative 3D Gaussian Splatting model with companioned mesh extraction and texture refinement in UV space. In contrast to the occupancy pruning used in Neural Radiance Fields, we demonstrate that the progressive densification of 3D Gaussians converges significantly faster for 3D generative tasks. To further enhance the texture quality and facilitate downstream applications, we introduce an efficient algorithm to convert 3D Gaussians into textured meshes and apply a fine-tuning stage to refine the details. Extensive experiments demonstrate the superior efficiency and competitive generation quality of our proposed approach. Notably, DreamGaussian produces high-quality textured meshes in just 2 minutes from a single-view image, achieving approximately 10 times acceleration compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20032v1">HO-Gaussian: Hybrid Optimization of 3D Gaussian Splatting for Urban Scenes</a></div>
    <div class="paper-meta">
      📅 2024-03-29
    </div>
    <details class="paper-abstract">
      The rapid growth of 3D Gaussian Splatting (3DGS) has revolutionized neural rendering, enabling real-time production of high-quality renderings. However, the previous 3DGS-based methods have limitations in urban scenes due to reliance on initial Structure-from-Motion(SfM) points and difficulties in rendering distant, sky and low-texture areas. To overcome these challenges, we propose a hybrid optimization method named HO-Gaussian, which combines a grid-based volume with the 3DGS pipeline. HO-Gaussian eliminates the dependency on SfM point initialization, allowing for rendering of urban scenes, and incorporates the Point Densitification to enhance rendering quality in problematic regions during training. Furthermore, we introduce Gaussian Direction Encoding as an alternative for spherical harmonics in the rendering pipeline, which enables view-dependent color representation. To account for multi-camera systems, we introduce neural warping to enhance object consistency across different cameras. Experimental results on widely used autonomous driving datasets demonstrate that HO-Gaussian achieves photo-realistic rendering in real-time on multi-camera urban datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11461v2">GAvatar: Animatable 3D Gaussian Avatars with Implicit Mesh Learning</a></div>
    <div class="paper-meta">
      📅 2024-03-29
      | 💬 CVPR 2024. Project website: https://nvlabs.github.io/GAvatar
    </div>
    <details class="paper-abstract">
      Gaussian splatting has emerged as a powerful 3D representation that harnesses the advantages of both explicit (mesh) and implicit (NeRF) 3D representations. In this paper, we seek to leverage Gaussian splatting to generate realistic animatable avatars from textual descriptions, addressing the limitations (e.g., flexibility and efficiency) imposed by mesh or NeRF-based representations. However, a naive application of Gaussian splatting cannot generate high-quality animatable avatars and suffers from learning instability; it also cannot capture fine avatar geometries and often leads to degenerate body parts. To tackle these problems, we first propose a primitive-based 3D Gaussian representation where Gaussians are defined inside pose-driven primitives to facilitate animation. Second, to stabilize and amortize the learning of millions of Gaussians, we propose to use neural implicit fields to predict the Gaussian attributes (e.g., colors). Finally, to capture fine avatar geometries and extract detailed meshes, we propose a novel SDF-based implicit mesh learning approach for 3D Gaussians that regularizes the underlying geometries and extracts highly detailed textured meshes. Our proposed method, GAvatar, enables the large-scale generation of diverse animatable avatars using only text prompts. GAvatar significantly surpasses existing methods in terms of both appearance and geometry quality, and achieves extremely fast rendering (100 fps) at 1K resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.19615v1">SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing</a></div>
    <div class="paper-meta">
      📅 2024-03-29
      | 💬 Project page: https://kevinsong729.github.io/project-pages/SA-GS/ Code: https://github.com/zsy1987/SA-GS
    </div>
    <details class="paper-abstract">
      In this paper, we present a Scale-adaptive method for Anti-aliasing Gaussian Splatting (SA-GS). While the state-of-the-art method Mip-Splatting needs modifying the training procedure of Gaussian splatting, our method functions at test-time and is training-free. Specifically, SA-GS can be applied to any pretrained Gaussian splatting field as a plugin to significantly improve the field's anti-alising performance. The core technique is to apply 2D scale-adaptive filters to each Gaussian during test time. As pointed out by Mip-Splatting, observing Gaussians at different frequencies leads to mismatches between the Gaussian scales during training and testing. Mip-Splatting resolves this issue using 3D smoothing and 2D Mip filters, which are unfortunately not aware of testing frequency. In this work, we show that a 2D scale-adaptive filter that is informed of testing frequency can effectively match the Gaussian scale, thus making the Gaussian primitive distribution remain consistent across different testing frequencies. When scale inconsistency is eliminated, sampling rates smaller than the scene frequency result in conventional jaggedness, and we propose to integrate the projected 2D Gaussian within each pixel during testing. This integration is actually a limiting case of super-sampling, which significantly improves anti-aliasing performance over vanilla Gaussian Splatting. Through extensive experiments using various settings and both bounded and unbounded scenes, we show SA-GS performs comparably with or better than Mip-Splatting. Note that super-sampling and integration are only effective when our scale-adaptive filtering is activated. Our codes, data and models are available at https://github.com/zsy1987/SA-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.19632v1">GauStudio: A Modular Framework for 3D Gaussian Splatting and Beyond</a></div>
    <div class="paper-meta">
      📅 2024-03-29
      | 💬 Code: https://github.com/GAP-LAB-CUHK-SZ/gaustudio
    </div>
    <details class="paper-abstract">
      We present GauStudio, a novel modular framework for modeling 3D Gaussian Splatting (3DGS) to provide standardized, plug-and-play components for users to easily customize and implement a 3DGS pipeline. Supported by our framework, we propose a hybrid Gaussian representation with foreground and skyball background models. Experiments demonstrate this representation reduces artifacts in unbounded outdoor scenes and improves novel view synthesis. Finally, we propose Gaussian Splatting Surface Reconstruction (GauS), a novel render-then-fuse approach for high-fidelity mesh reconstruction from 3DGS inputs without fine-tuning. Overall, our GauStudio framework, hybrid representation, and GauS approach enhance 3DGS modeling and rendering capabilities, enabling higher-quality novel view synthesis and surface reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.19632v1">GauStudio: A Modular Framework for 3D Gaussian Splatting and Beyond</a></div>
    <div class="paper-meta">
      📅 2024-03-28
      | 💬 Code: https://github.com/GAP-LAB-CUHK-SZ/gaustudio
    </div>
    <details class="paper-abstract">
      We present GauStudio, a novel modular framework for modeling 3D Gaussian Splatting (3DGS) to provide standardized, plug-and-play components for users to easily customize and implement a 3DGS pipeline. Supported by our framework, we propose a hybrid Gaussian representation with foreground and skyball background models. Experiments demonstrate this representation reduces artifacts in unbounded outdoor scenes and improves novel view synthesis. Finally, we propose Gaussian Splatting Surface Reconstruction (GauS), a novel render-then-fuse approach for high-fidelity mesh reconstruction from 3DGS inputs without fine-tuning. Overall, our GauStudio framework, hybrid representation, and GauS approach enhance 3DGS modeling and rendering capabilities, enabling higher-quality novel view synthesis and surface reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.19615v1">SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing</a></div>
    <div class="paper-meta">
      📅 2024-03-28
      | 💬 Project page: https://kevinsong729.github.io/project-pages/SA-GS/ Code: https://github.com/zsy1987/SA-GS
    </div>
    <details class="paper-abstract">
      In this paper, we present a Scale-adaptive method for Anti-aliasing Gaussian Splatting (SA-GS). While the state-of-the-art method Mip-Splatting needs modifying the training procedure of Gaussian splatting, our method functions at test-time and is training-free. Specifically, SA-GS can be applied to any pretrained Gaussian splatting field as a plugin to significantly improve the field's anti-alising performance. The core technique is to apply 2D scale-adaptive filters to each Gaussian during test time. As pointed out by Mip-Splatting, observing Gaussians at different frequencies leads to mismatches between the Gaussian scales during training and testing. Mip-Splatting resolves this issue using 3D smoothing and 2D Mip filters, which are unfortunately not aware of testing frequency. In this work, we show that a 2D scale-adaptive filter that is informed of testing frequency can effectively match the Gaussian scale, thus making the Gaussian primitive distribution remain consistent across different testing frequencies. When scale inconsistency is eliminated, sampling rates smaller than the scene frequency result in conventional jaggedness, and we propose to integrate the projected 2D Gaussian within each pixel during testing. This integration is actually a limiting case of super-sampling, which significantly improves anti-aliasing performance over vanilla Gaussian Splatting. Through extensive experiments using various settings and both bounded and unbounded scenes, we show SA-GS performs comparably with or better than Mip-Splatting. Note that super-sampling and integration are only effective when our scale-adaptive filtering is activated. Our codes, data and models are available at https://github.com/zsy1987/SA-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17113v2">Human Gaussian Splatting: Real-time Rendering of Animatable Avatars</a></div>
    <div class="paper-meta">
      📅 2024-03-28
      | 💬 Accepted to CVPR 2024
    </div>
    <details class="paper-abstract">
      This work addresses the problem of real-time rendering of photorealistic human body avatars learned from multi-view videos. While the classical approaches to model and render virtual humans generally use a textured mesh, recent research has developed neural body representations that achieve impressive visual quality. However, these models are difficult to render in real-time and their quality degrades when the character is animated with body poses different than the training observations. We propose an animatable human model based on 3D Gaussian Splatting, that has recently emerged as a very efficient alternative to neural radiance fields. The body is represented by a set of gaussian primitives in a canonical space which is deformed with a coarse to fine approach that combines forward skinning and local non-rigid refinement. We describe how to learn our Human Gaussian Splatting (HuGS) model in an end-to-end fashion from multi-view observations, and evaluate it against the state-of-the-art approaches for novel pose synthesis of clothed body. Our method achieves 1.5 dB PSNR improvement over the state-of-the-art on THuman4 dataset while being able to render in real-time (80 fps for 512x512 resolution).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02137v2">MANUS: Markerless Grasp Capture using Articulated 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-03-28
      | 💬 IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR) 2024
    </div>
    <details class="paper-abstract">
      Understanding how we grasp objects with our hands has important applications in areas like robotics and mixed reality. However, this challenging problem requires accurate modeling of the contact between hands and objects. To capture grasps, existing methods use skeletons, meshes, or parametric models that does not represent hand shape accurately resulting in inaccurate contacts. We present MANUS, a method for Markerless Hand-Object Grasp Capture using Articulated 3D Gaussians. We build a novel articulated 3D Gaussians representation that extends 3D Gaussian splatting for high-fidelity representation of articulating hands. Since our representation uses Gaussian primitives, it enables us to efficiently and accurately estimate contacts between the hand and the object. For the most accurate results, our method requires tens of camera views that current datasets do not provide. We therefore build MANUS-Grasps, a new dataset that contains hand-object grasps viewed from 50+ cameras across 30+ scenes, 3 subjects, and comprising over 7M frames. In addition to extensive qualitative results, we also show that our method outperforms others on a quantitative contact evaluation method that uses paint transfer from the object to the hand.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02069v2">GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-03-28
      | 💬 Project page: https://shenhanqian.github.io/gaussian-avatars
    </div>
    <details class="paper-abstract">
      We introduce GaussianAvatars, a new method to create photorealistic head avatars that are fully controllable in terms of expression, pose, and viewpoint. The core idea is a dynamic 3D representation based on 3D Gaussian splats that are rigged to a parametric morphable face model. This combination facilitates photorealistic rendering while allowing for precise animation control via the underlying parametric model, e.g., through expression transfer from a driving sequence or by manually changing the morphable model parameters. We parameterize each splat by a local coordinate frame of a triangle and optimize for explicit displacement offset to obtain a more accurate geometric representation. During avatar reconstruction, we jointly optimize for the morphable model parameters and Gaussian splat parameters in an end-to-end fashion. We demonstrate the animation capabilities of our photorealistic avatar in several challenging scenarios. For instance, we show reenactments from a driving video, where our method outperforms existing works by a significant margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16473v3">GS-IR: 3D Gaussian Splatting for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2024-03-28
    </div>
    <details class="paper-abstract">
      We propose GS-IR, a novel inverse rendering approach based on 3D Gaussian Splatting (GS) that leverages forward mapping volume rendering to achieve photorealistic novel view synthesis and relighting results. Unlike previous works that use implicit neural representations and volume rendering (e.g. NeRF), which suffer from low expressive power and high computational complexity, we extend GS, a top-performance representation for novel view synthesis, to estimate scene geometry, surface material, and environment illumination from multi-view images captured under unknown lighting conditions. There are two main problems when introducing GS to inverse rendering: 1) GS does not support producing plausible normal natively; 2) forward mapping (e.g. rasterization and splatting) cannot trace the occlusion like backward mapping (e.g. ray tracing). To address these challenges, our GS-IR proposes an efficient optimization scheme that incorporates a depth-derivation-based regularization for normal estimation and a baking-based occlusion to model indirect lighting. The flexible and expressive GS representation allows us to achieve fast and compact geometry reconstruction, photorealistic novel view synthesis, and effective physically-based rendering. We demonstrate the superiority of our method over baseline methods through qualitative and quantitative evaluations on various challenging scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.18476v1">Modeling uncertainty for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-28
    </div>
    <details class="paper-abstract">
      We present Stochastic Gaussian Splatting (SGS): the first framework for uncertainty estimation using Gaussian Splatting (GS). GS recently advanced the novel-view synthesis field by achieving impressive reconstruction quality at a fraction of the computational cost of Neural Radiance Fields (NeRF). However, contrary to the latter, it still lacks the ability to provide information about the confidence associated with their outputs. To address this limitation, in this paper, we introduce a Variational Inference-based approach that seamlessly integrates uncertainty prediction into the common rendering pipeline of GS. Additionally, we introduce the Area Under Sparsification Error (AUSE) as a new term in the loss function, enabling optimization of uncertainty estimation alongside image reconstruction. Experimental results on the LLFF dataset demonstrate that our method outperforms existing approaches in terms of both image rendering quality and uncertainty estimation accuracy. Overall, our framework equips practitioners with valuable insights into the reliability of synthesized views, facilitating safer decision-making in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.19615v1">SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing</a></div>
    <div class="paper-meta">
      📅 2024-03-28
      | 💬 Project page: https://kevinsong729.github.io/project-pages/SA-GS/ Code: https://github.com/zsy1987/SA-GS
    </div>
    <details class="paper-abstract">
      In this paper, we present a Scale-adaptive method for Anti-aliasing Gaussian Splatting (SA-GS). While the state-of-the-art method Mip-Splatting needs modifying the training procedure of Gaussian splatting, our method functions at test-time and is training-free. Specifically, SA-GS can be applied to any pretrained Gaussian splatting field as a plugin to significantly improve the field's anti-alising performance. The core technique is to apply 2D scale-adaptive filters to each Gaussian during test time. As pointed out by Mip-Splatting, observing Gaussians at different frequencies leads to mismatches between the Gaussian scales during training and testing. Mip-Splatting resolves this issue using 3D smoothing and 2D Mip filters, which are unfortunately not aware of testing frequency. In this work, we show that a 2D scale-adaptive filter that is informed of testing frequency can effectively match the Gaussian scale, thus making the Gaussian primitive distribution remain consistent across different testing frequencies. When scale inconsistency is eliminated, sampling rates smaller than the scene frequency result in conventional jaggedness, and we propose to integrate the projected 2D Gaussian within each pixel during testing. This integration is actually a limiting case of super-sampling, which significantly improves anti-aliasing performance over vanilla Gaussian Splatting. Through extensive experiments using various settings and both bounded and unbounded scenes, we show SA-GS performs comparably with or better than Mip-Splatting. Note that super-sampling and integration are only effective when our scale-adaptive filtering is activated. Our codes, data and models are available at https://github.com/zsy1987/SA-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.19632v1">GauStudio: A Modular Framework for 3D Gaussian Splatting and Beyond</a></div>
    <div class="paper-meta">
      📅 2024-03-28
      | 💬 Code: https://github.com/GAP-LAB-CUHK-SZ/gaustudio
    </div>
    <details class="paper-abstract">
      We present GauStudio, a novel modular framework for modeling 3D Gaussian Splatting (3DGS) to provide standardized, plug-and-play components for users to easily customize and implement a 3DGS pipeline. Supported by our framework, we propose a hybrid Gaussian representation with foreground and skyball background models. Experiments demonstrate this representation reduces artifacts in unbounded outdoor scenes and improves novel view synthesis. Finally, we propose Gaussian Splatting Surface Reconstruction (GauS), a novel render-then-fuse approach for high-fidelity mesh reconstruction from 3DGS inputs without fine-tuning. Overall, our GauStudio framework, hybrid representation, and GauS approach enhance 3DGS modeling and rendering capabilities, enabling higher-quality novel view synthesis and surface reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18476v1">Modeling uncertainty for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-27
    </div>
    <details class="paper-abstract">
      We present Stochastic Gaussian Splatting (SGS): the first framework for uncertainty estimation using Gaussian Splatting (GS). GS recently advanced the novel-view synthesis field by achieving impressive reconstruction quality at a fraction of the computational cost of Neural Radiance Fields (NeRF). However, contrary to the latter, it still lacks the ability to provide information about the confidence associated with their outputs. To address this limitation, in this paper, we introduce a Variational Inference-based approach that seamlessly integrates uncertainty prediction into the common rendering pipeline of GS. Additionally, we introduce the Area Under Sparsification Error (AUSE) as a new term in the loss function, enabling optimization of uncertainty estimation alongside image reconstruction. Experimental results on the LLFF dataset demonstrate that our method outperforms existing approaches in terms of both image rendering quality and uncertainty estimation accuracy. Overall, our framework equips practitioners with valuable insights into the reliability of synthesized views, facilitating safer decision-making in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.18476v1">Modeling uncertainty for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-27
    </div>
    <details class="paper-abstract">
      We present Stochastic Gaussian Splatting (SGS): the first framework for uncertainty estimation using Gaussian Splatting (GS). GS recently advanced the novel-view synthesis field by achieving impressive reconstruction quality at a fraction of the computational cost of Neural Radiance Fields (NeRF). However, contrary to the latter, it still lacks the ability to provide information about the confidence associated with their outputs. To address this limitation, in this paper, we introduce a Variational Inference-based approach that seamlessly integrates uncertainty prediction into the common rendering pipeline of GS. Additionally, we introduce the Area Under Sparsification Error (AUSE) as a new term in the loss function, enabling optimization of uncertainty estimation alongside image reconstruction. Experimental results on the LLFF dataset demonstrate that our method outperforms existing approaches in terms of both image rendering quality and uncertainty estimation accuracy. Overall, our framework equips practitioners with valuable insights into the reliability of synthesized views, facilitating safer decision-making in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06003v2">TRIPS: Trilinear Point Splatting for Real-Time Radiance Field Rendering</a></div>
    <div class="paper-meta">
      📅 2024-03-26
    </div>
    <details class="paper-abstract">
      Point-based radiance field rendering has demonstrated impressive results for novel view synthesis, offering a compelling blend of rendering quality and computational efficiency. However, also latest approaches in this domain are not without their shortcomings. 3D Gaussian Splatting [Kerbl and Kopanas et al. 2023] struggles when tasked with rendering highly detailed scenes, due to blurring and cloudy artifacts. On the other hand, ADOP [R\"uckert et al. 2022] can accommodate crisper images, but the neural reconstruction network decreases performance, it grapples with temporal instability and it is unable to effectively address large gaps in the point cloud. In this paper, we present TRIPS (Trilinear Point Splatting), an approach that combines ideas from both Gaussian Splatting and ADOP. The fundamental concept behind our novel technique involves rasterizing points into a screen-space image pyramid, with the selection of the pyramid layer determined by the projected point size. This approach allows rendering arbitrarily large points using a single trilinear write. A lightweight neural network is then used to reconstruct a hole-free image including detail beyond splat resolution. Importantly, our render pipeline is entirely differentiable, allowing for automatic optimization of both point sizes and positions. Our evaluation demonstrate that TRIPS surpasses existing state-of-the-art methods in terms of rendering quality while maintaining a real-time frame rate of 60 frames per second on readily available hardware. This performance extends to challenging scenarios, such as scenes featuring intricate geometry, expansive landscapes, and auto-exposed footage. The project page is located at: https://lfranke.github.io/trips/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17237v1">DreamPolisher: Towards High-Quality Text-to-3D Generation via Geometric Diffusion</a></div>
    <div class="paper-meta">
      📅 2024-03-25
      | 💬 Project webpage: https://yuanze-lin.me/DreamPolisher_page/
    </div>
    <details class="paper-abstract">
      We present DreamPolisher, a novel Gaussian Splatting based method with geometric guidance, tailored to learn cross-view consistency and intricate detail from textual descriptions. While recent progress on text-to-3D generation methods have been promising, prevailing methods often fail to ensure view-consistency and textural richness. This problem becomes particularly noticeable for methods that work with text input alone. To address this, we propose a two-stage Gaussian Splatting based approach that enforces geometric consistency among views. Initially, a coarse 3D generation undergoes refinement via geometric optimization. Subsequently, we use a ControlNet driven refiner coupled with the geometric consistency term to improve both texture fidelity and overall consistency of the generated 3D asset. Empirical evaluations across diverse textual prompts spanning various object categories demonstrate the efficacy of DreamPolisher in generating consistent and realistic 3D objects, aligning closely with the semantics of the textual instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04926v2">BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling</a></div>
    <div class="paper-meta">
      📅 2024-03-24
    </div>
    <details class="paper-abstract">
      Recent efforts in using 3D Gaussians for scene reconstruction and novel view synthesis can achieve impressive results on curated benchmarks; however, images captured in real life are often blurry. In this work, we analyze the robustness of Gaussian-Splatting-based methods against various image blur, such as motion blur, defocus blur, downscaling blur, \etc. Under these degradations, Gaussian-Splatting-based methods tend to overfit and produce worse results than Neural-Radiance-Field-based methods. To address this issue, we propose Blur Agnostic Gaussian Splatting (BAGS). BAGS introduces additional 2D modeling capacities such that a 3D-consistent and high quality scene can be reconstructed despite image-wise blur. Specifically, we model blur by estimating per-pixel convolution kernels from a Blur Proposal Network (BPN). BPN is designed to consider spatial, color, and depth variations of the scene to maximize modeling capacity. Additionally, BPN also proposes a quality-assessing mask, which indicates regions where blur occur. Finally, we introduce a coarse-to-fine kernel optimization scheme; this optimization scheme is fast and avoids sub-optimal solutions due to a sparse point cloud initialization, which often occurs when we apply Structure-from-Motion on blurry images. We demonstrate that BAGS achieves photorealistic renderings under various challenging blur conditions and imaging geometry, while significantly improving upon existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06912v3">DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization</a></div>
    <div class="paper-meta">
      📅 2024-03-24
      | 💬 Accepted at CVPR 2024. Project page: https://fictionarry.github.io/DNGaussian/
    </div>
    <details class="paper-abstract">
      Radiance fields have demonstrated impressive performance in synthesizing novel views from sparse input views, yet prevailing methods suffer from high training costs and slow inference speed. This paper introduces DNGaussian, a depth-regularized framework based on 3D Gaussian radiance fields, offering real-time and high-quality few-shot novel view synthesis at low costs. Our motivation stems from the highly efficient representation and surprising quality of the recent 3D Gaussian Splatting, despite it will encounter a geometry degradation when input views decrease. In the Gaussian radiance fields, we find this degradation in scene geometry primarily lined to the positioning of Gaussian primitives and can be mitigated by depth constraint. Consequently, we propose a Hard and Soft Depth Regularization to restore accurate scene geometry under coarse monocular depth supervision while maintaining a fine-grained color appearance. To further refine detailed geometry reshaping, we introduce Global-Local Depth Normalization, enhancing the focus on small local depth changes. Extensive experiments on LLFF, DTU, and Blender datasets demonstrate that DNGaussian outperforms state-of-the-art methods, achieving comparable or better results with significantly reduced memory cost, a $25 \times$ reduction in training time, and over $3000 \times$ faster rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16095v1">CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field</a></div>
    <div class="paper-meta">
      📅 2024-03-24
      | 💬 Project Page: https://zju3dv.github.io/cg-slam
    </div>
    <details class="paper-abstract">
      Recently neural radiance fields (NeRF) have been widely exploited as 3D representations for dense simultaneous localization and mapping (SLAM). Despite their notable successes in surface modeling and novel view synthesis, existing NeRF-based methods are hindered by their computationally intensive and time-consuming volume rendering pipeline. This paper presents an efficient dense RGB-D SLAM system, i.e., CG-SLAM, based on a novel uncertainty-aware 3D Gaussian field with high consistency and geometric stability. Through an in-depth analysis of Gaussian Splatting, we propose several techniques to construct a consistent and stable 3D Gaussian field suitable for tracking and mapping. Additionally, a novel depth uncertainty model is proposed to ensure the selection of valuable Gaussian primitives during optimization, thereby improving tracking efficiency and accuracy. Experiments on various datasets demonstrate that CG-SLAM achieves superior tracking and mapping performance with a notable tracking speed of up to 15 Hz. We will make our source code publicly available. Project page: https://zju3dv.github.io/cg-slam.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15530v1">Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive novel view synthesis results while advancing real-time rendering performance. However, it relies heavily on the quality of the initial point cloud, resulting in blurring and needle-like artifacts in areas with insufficient initializing points. This is mainly attributed to the point cloud growth condition in 3DGS that only considers the average gradient magnitude of points from observable views, thereby failing to grow for large Gaussians that are observable for many viewpoints while many of them are only covered in the boundaries. To this end, we propose a novel method, named Pixel-GS, to take into account the number of pixels covered by the Gaussian in each view during the computation of the growth condition. We regard the covered pixel numbers as the weights to dynamically average the gradients from different views, such that the growth of large Gaussians can be prompted. As a result, points within the areas with insufficient initializing points can be grown more effectively, leading to a more accurate and detailed reconstruction. In addition, we propose a simple yet effective strategy to scale the gradient field according to the distance to the camera, to suppress the growth of floaters near the camera. Extensive experiments both qualitatively and quantitatively demonstrate that our method achieves state-of-the-art rendering quality while maintaining real-time rendering speed, on the challenging Mip-NeRF 360 and Tanks & Temples datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.10070v2">Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-22
    </div>
    <details class="paper-abstract">
      We present a dense simultaneous localization and mapping (SLAM) method that uses 3D Gaussians as a scene representation. Our approach enables interactive-time reconstruction and photo-realistic rendering from real-world single-camera RGBD videos. To this end, we propose a novel effective strategy for seeding new Gaussians for newly explored areas and their effective online optimization that is independent of the scene size and thus scalable to larger scenes. This is achieved by organizing the scene into sub-maps which are independently optimized and do not need to be kept in memory. We further accomplish frame-to-model camera tracking by minimizing photometric and geometric losses between the input and rendered frames. The Gaussian representation allows for high-quality photo-realistic real-time rendering of real-world scenes. Evaluation on synthetic and real-world datasets demonstrates competitive or superior performance in mapping, tracking, and rendering compared to existing neural dense SLAM methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12550v2">RGBD GS-ICP SLAM</a></div>
    <div class="paper-meta">
      📅 2024-03-22
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) with dense representation plays a key role in robotics, Virtual Reality (VR), and Augmented Reality (AR) applications. Recent advancements in dense representation SLAM have highlighted the potential of leveraging neural scene representation and 3D Gaussian representation for high-fidelity spatial representation. In this paper, we propose a novel dense representation SLAM approach with a fusion of Generalized Iterative Closest Point (G-ICP) and 3D Gaussian Splatting (3DGS). In contrast to existing methods, we utilize a single Gaussian map for both tracking and mapping, resulting in mutual benefits. Through the exchange of covariances between tracking and mapping processes with scale alignment techniques, we minimize redundant computations and achieve an efficient system. Additionally, we enhance tracking accuracy and mapping quality through our keyframe selection methods. Experimental results demonstrate the effectiveness of our approach, showing an incredibly fast speed up to 107 FPS (for the entire system) and superior quality of the reconstructed map.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15124v1">EndoGSLAM: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-22
    </div>
    <details class="paper-abstract">
      Precise camera tracking, high-fidelity 3D tissue reconstruction, and real-time online visualization are critical for intrabody medical imaging devices such as endoscopes and capsule robots. However, existing SLAM (Simultaneous Localization and Mapping) methods often struggle to achieve both complete high-quality surgical field reconstruction and efficient computation, restricting their intraoperative applications among endoscopic surgeries. In this paper, we introduce EndoGSLAM, an efficient SLAM approach for endoscopic surgeries, which integrates streamlined Gaussian representation and differentiable rasterization to facilitate over 100 fps rendering speed during online camera tracking and tissue reconstructing. Extensive experiments show that EndoGSLAM achieves a better trade-off between intraoperative availability and reconstruction quality than traditional or neural SLAM approaches, showing tremendous potential for endoscopic surgeries. The project page is at https://EndoGSLAM.loping151.com
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.14939v1">STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-03-22
    </div>
    <details class="paper-abstract">
      Recent progress in pre-trained diffusion models and 3D generation have spurred interest in 4D content creation. However, achieving high-fidelity 4D generation with spatial-temporal consistency remains a challenge. In this work, we propose STAG4D, a novel framework that combines pre-trained diffusion models with dynamic 3D Gaussian splatting for high-fidelity 4D generation. Drawing inspiration from 3D generation techniques, we utilize a multi-view diffusion model to initialize multi-view images anchoring on the input video frames, where the video can be either real-world captured or generated by a video diffusion model. To ensure the temporal consistency of the multi-view sequence initialization, we introduce a simple yet effective fusion strategy to leverage the first frame as a temporal anchor in the self-attention computation. With the almost consistent multi-view sequences, we then apply the score distillation sampling to optimize the 4D Gaussian point cloud. The 4D Gaussian spatting is specially crafted for the generation task, where an adaptive densification strategy is proposed to mitigate the unstable Gaussian gradient for robust optimization. Notably, the proposed pipeline does not require any pre-training or fine-tuning of diffusion networks, offering a more accessible and practical solution for the 4D generation task. Extensive experiments demonstrate that our method outperforms prior 4D generation works in rendering quality, spatial-temporal consistency, and generation robustness, setting a new state-of-the-art for 4D generation from diverse inputs, including text, image, and video.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.14554v1">Gaussian Frosting: Editable Complex Radiance Fields with Real-Time Rendering</a></div>
    <div class="paper-meta">
      📅 2024-03-21
      | 💬 Project Webpage: https://anttwo.github.io/frosting/
    </div>
    <details class="paper-abstract">
      We propose Gaussian Frosting, a novel mesh-based representation for high-quality rendering and editing of complex 3D effects in real-time. Our approach builds on the recent 3D Gaussian Splatting framework, which optimizes a set of 3D Gaussians to approximate a radiance field from images. We propose first extracting a base mesh from Gaussians during optimization, then building and refining an adaptive layer of Gaussians with a variable thickness around the mesh to better capture the fine details and volumetric effects near the surface, such as hair or grass. We call this layer Gaussian Frosting, as it resembles a coating of frosting on a cake. The fuzzier the material, the thicker the frosting. We also introduce a parameterization of the Gaussians to enforce them to stay inside the frosting layer and automatically adjust their parameters when deforming, rescaling, editing or animating the mesh. Our representation allows for efficient rendering using Gaussian splatting, as well as editing and animation by modifying the base mesh. We demonstrate the effectiveness of our method on various synthetic and real scenes, and show that it outperforms existing surface-based approaches. We will release our code and a web-based viewer as additional contributions. Our project page is the following: https://anttwo.github.io/frosting/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.14244v1">Isotropic Gaussian Splatting for Real-Time Radiance Field Rendering</a></div>
    <div class="paper-meta">
      📅 2024-03-21
    </div>
    <details class="paper-abstract">
      The 3D Gaussian splatting method has drawn a lot of attention, thanks to its high performance in training and high quality of the rendered image. However, it uses anisotropic Gaussian kernels to represent the scene. Although such anisotropic kernels have advantages in representing the geometry, they lead to difficulties in terms of computation, such as splitting or merging two kernels. In this paper, we propose to use isotropic Gaussian kernels to avoid such difficulties in the computation, leading to a higher performance method. The experiments confirm that the proposed method is about {\bf 100X} faster without losing the geometry representation accuracy. The proposed method can be applied in a large range applications where the radiance field is needed, such as 3D reconstruction, view synthesis, and dynamic object modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.13806v1">RadSplat: Radiance Field-Informed Gaussian Splatting for Robust Real-Time Rendering with 900+ FPS</a></div>
    <div class="paper-meta">
      📅 2024-03-20
      | 💬 Project page at https://m-niemeyer.github.io/radsplat/
    </div>
    <details class="paper-abstract">
      Recent advances in view synthesis and real-time rendering have achieved photorealistic quality at impressive rendering speeds. While Radiance Field-based methods achieve state-of-the-art quality in challenging scenarios such as in-the-wild captures and large-scale scenes, they often suffer from excessively high compute requirements linked to volumetric rendering. Gaussian Splatting-based methods, on the other hand, rely on rasterization and naturally achieve real-time rendering but suffer from brittle optimization heuristics that underperform on more challenging scenes. In this work, we present RadSplat, a lightweight method for robust real-time rendering of complex scenes. Our main contributions are threefold. First, we use radiance fields as a prior and supervision signal for optimizing point-based scene representations, leading to improved quality and more robust optimization. Next, we develop a novel pruning technique reducing the overall point count while maintaining high quality, leading to smaller and more compact scene representations with faster inference speeds. Finally, we propose a novel test-time filtering approach that further accelerates rendering and allows to scale to larger, house-sized scenes. We find that our method enables state-of-the-art synthesis of complex captures at 900+ FPS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.18561v2">Periodic Vibration Gaussian: Dynamic Urban Scene Reconstruction and Real-time Rendering</a></div>
    <div class="paper-meta">
      📅 2024-03-20
      | 💬 Project page: https://fudan-zvg.github.io/PVG/
    </div>
    <details class="paper-abstract">
      Modeling dynamic, large-scale urban scenes is challenging due to their highly intricate geometric structures and unconstrained dynamics in both space and time. Prior methods often employ high-level architectural priors, separating static and dynamic elements, resulting in suboptimal capture of their synergistic interactions. To address this challenge, we present a unified representation model, called Periodic Vibration Gaussian (PVG). PVG builds upon the efficient 3D Gaussian splatting technique, originally designed for static scene representation, by introducing periodic vibration-based temporal dynamics. This innovation enables PVG to elegantly and uniformly represent the characteristics of various objects and elements in dynamic urban scenes. To enhance temporally coherent and large scene representation learning with sparse training data, we introduce a novel temporal smoothing mechanism and a position-aware adaptive control strategy respectively. Extensive experiments on Waymo Open Dataset and KITTI benchmarks demonstrate that PVG surpasses state-of-the-art alternatives in both reconstruction and novel view synthesis for both dynamic and static scenes. Notably, PVG achieves this without relying on manually labeled object bounding boxes or expensive optical flow estimation. Moreover, PVG exhibits 900-fold acceleration in rendering over the best alternative.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09031v2">iComMa: Inverting 3D Gaussian Splatting for Camera Pose Estimation via Comparing and Matching</a></div>
    <div class="paper-meta">
      📅 2024-03-20
    </div>
    <details class="paper-abstract">
      We present a method named iComMa to address the 6D camera pose estimation problem in computer vision. Conventional pose estimation methods typically rely on the target's CAD model or necessitate specific network training tailored to particular object classes. Some existing methods have achieved promising results in mesh-free object and scene pose estimation by inverting the Neural Radiance Fields (NeRF). However, they still struggle with adverse initializations such as large rotations and translations. To address this issue, we propose an efficient method for accurate camera pose estimation by inverting 3D Gaussian Splatting (3DGS). Specifically, a gradient-based differentiable framework optimizes camera pose by minimizing the residual between the query image and the rendered image, requiring no training. An end-to-end matching module is designed to enhance the model's robustness against adverse initializations, while minimizing pixel-level comparing loss aids in precise pose estimation. Experimental results on synthetic and complex real-world data demonstrate the effectiveness of the proposed approach in challenging conditions and the accuracy of camera pose estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07920v3">DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2024-03-20
    </div>
    <details class="paper-abstract">
      We present DrivingGaussian, an efficient and effective framework for surrounding dynamic autonomous driving scenes. For complex scenes with moving objects, we first sequentially and progressively model the static background of the entire scene with incremental static 3D Gaussians. We then leverage a composite dynamic Gaussian graph to handle multiple moving objects, individually reconstructing each object and restoring their accurate positions and occlusion relationships within the scene. We further use a LiDAR prior for Gaussian Splatting to reconstruct scenes with greater details and maintain panoramic consistency. DrivingGaussian outperforms existing methods in dynamic driving scene reconstruction and enables photorealistic surround-view synthesis with high-fidelity and multi-camera consistency. Our project page is at: https://github.com/VDIGPKU/DrivingGaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11625v2">GaussNav: Gaussian Splatting for Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2024-03-20
      | 💬 conference
    </div>
    <details class="paper-abstract">
      In embodied vision, Instance ImageGoal Navigation (IIN) requires an agent to locate a specific object depicted in a goal image within an unexplored environment. The primary difficulty of IIN stems from the necessity of recognizing the target object across varying viewpoints and rejecting potential distractors. Existing map-based navigation methods largely adopt the representation form of Bird's Eye View (BEV) maps, which, however, lack the representation of detailed textures in a scene. To address the above issues, we propose a new Gaussian Splatting Navigation (abbreviated as GaussNav) framework for IIN task, which constructs a novel map representation based on 3D Gaussian Splatting (3DGS). The proposed framework enables the agent to not only memorize the geometry and semantic information of the scene, but also retain the textural features of objects. Our GaussNav framework demonstrates a significant leap in performance, evidenced by an increase in Success weighted by Path Length (SPL) from 0.252 to 0.578 on the challenging Habitat-Matterport 3D (HM3D) dataset. Our code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.11831v2">BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-20
      | 💬 Project Page and Source Code: https://lingzhezhao.github.io/BAD-Gaussians/
    </div>
    <details class="paper-abstract">
      While neural rendering has demonstrated impressive capabilities in 3D scene reconstruction and novel view synthesis, it heavily relies on high-quality sharp images and accurate camera poses. Numerous approaches have been proposed to train Neural Radiance Fields (NeRF) with motion-blurred images, commonly encountered in real-world scenarios such as low-light or long-exposure conditions. However, the implicit representation of NeRF struggles to accurately recover intricate details from severely motion-blurred images and cannot achieve real-time rendering. In contrast, recent advancements in 3D Gaussian Splatting achieve high-quality 3D scene reconstruction and real-time rendering by explicitly optimizing point clouds as Gaussian spheres. In this paper, we introduce a novel approach, named BAD-Gaussians (Bundle Adjusted Deblur Gaussian Splatting), which leverages explicit Gaussian representation and handles severe motion-blurred images with inaccurate camera poses to achieve high-quality scene reconstruction. Our method models the physical image formation process of motion-blurred images and jointly learns the parameters of Gaussians while recovering camera motion trajectories during exposure time. In our experiments, we demonstrate that BAD-Gaussians not only achieves superior rendering quality compared to previous state-of-the-art deblur neural rendering methods on both synthetic and real datasets but also enables real-time rendering capabilities. Our project page and source code is available at https://lingzhezhao.github.io/BAD-Gaussians/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12722v1">HUGS: Holistic Urban 3D Scene Understanding via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 Our project page is at https://xdimlab.github.io/hugs_website
    </div>
    <details class="paper-abstract">
      Holistic understanding of urban scenes based on RGB images is a challenging yet important problem. It encompasses understanding both the geometry and appearance to enable novel view synthesis, parsing semantic labels, and tracking moving objects. Despite considerable progress, existing approaches often focus on specific aspects of this task and require additional inputs such as LiDAR scans or manually annotated 3D bounding boxes. In this paper, we introduce a novel pipeline that utilizes 3D Gaussian Splatting for holistic urban scene understanding. Our main idea involves the joint optimization of geometry, appearance, semantics, and motion using a combination of static and dynamic 3D Gaussians, where moving object poses are regularized via physical constraints. Our approach offers the ability to render new viewpoints in real-time, yielding 2D and 3D semantic information with high accuracy, and reconstruct dynamic scenes, even in scenarios where 3D bounding box detection are highly noisy. Experimental results on KITTI, KITTI-360, and Virtual KITTI 2 demonstrate the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11831v2">BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 Project Page and Source Code: https://lingzhezhao.github.io/BAD-Gaussians/
    </div>
    <details class="paper-abstract">
      While neural rendering has demonstrated impressive capabilities in 3D scene reconstruction and novel view synthesis, it heavily relies on high-quality sharp images and accurate camera poses. Numerous approaches have been proposed to train Neural Radiance Fields (NeRF) with motion-blurred images, commonly encountered in real-world scenarios such as low-light or long-exposure conditions. However, the implicit representation of NeRF struggles to accurately recover intricate details from severely motion-blurred images and cannot achieve real-time rendering. In contrast, recent advancements in 3D Gaussian Splatting achieve high-quality 3D scene reconstruction and real-time rendering by explicitly optimizing point clouds as Gaussian spheres. In this paper, we introduce a novel approach, named BAD-Gaussians (Bundle Adjusted Deblur Gaussian Splatting), which leverages explicit Gaussian representation and handles severe motion-blurred images with inaccurate camera poses to achieve high-quality scene reconstruction. Our method models the physical image formation process of motion-blurred images and jointly learns the parameters of Gaussians while recovering camera motion trajectories during exposure time. In our experiments, we demonstrate that BAD-Gaussians not only achieves superior rendering quality compared to previous state-of-the-art deblur neural rendering methods on both synthetic and real datasets but also enables real-time rendering capabilities. Our project page and source code is available at https://lingzhezhao.github.io/BAD-Gaussians/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09981v2">Controllable Text-to-3D Generation via Surface-Aligned Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 Project page: https://lizhiqi49.github.io/MVControl/
    </div>
    <details class="paper-abstract">
      While text-to-3D and image-to-3D generation tasks have received considerable attention, one important but under-explored field between them is controllable text-to-3D generation, which we mainly focus on in this work. To address this task, 1) we introduce Multi-view ControlNet (MVControl), a novel neural network architecture designed to enhance existing pre-trained multi-view diffusion models by integrating additional input conditions, such as edge, depth, normal, and scribble maps. Our innovation lies in the introduction of a conditioning module that controls the base diffusion model using both local and global embeddings, which are computed from the input condition images and camera poses. Once trained, MVControl is able to offer 3D diffusion guidance for optimization-based 3D generation. And, 2) we propose an efficient multi-stage 3D generation pipeline that leverages the benefits of recent large reconstruction models and score distillation algorithm. Building upon our MVControl architecture, we employ a unique hybrid diffusion guidance method to direct the optimization process. In pursuit of efficiency, we adopt 3D Gaussians as our representation instead of the commonly used implicit representations. We also pioneer the use of SuGaR, a hybrid representation that binds Gaussians to mesh triangle faces. This approach alleviates the issue of poor geometry in 3D Gaussians and enables the direct sculpting of fine-grained geometry on the mesh. Extensive experiments demonstrate that our method achieves robust generalization and enables the controllable generation of high-quality 3D content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10147v2">GGRt: Towards Pose-free Generalizable 3D Gaussian Splatting in Real-time</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 Project page: https://3d-aigc.github.io/GGRt
    </div>
    <details class="paper-abstract">
      This paper presents GGRt, a novel approach to generalizable novel view synthesis that alleviates the need for real camera poses, complexity in processing high-resolution images, and lengthy optimization processes, thus facilitating stronger applicability of 3D Gaussian Splatting (3D-GS) in real-world scenarios. Specifically, we design a novel joint learning framework that consists of an Iterative Pose Optimization Network (IPO-Net) and a Generalizable 3D-Gaussians (G-3DG) model. With the joint learning mechanism, the proposed framework can inherently estimate robust relative pose information from the image observations and thus primarily alleviate the requirement of real camera poses. Moreover, we implement a deferred back-propagation mechanism that enables high-resolution training and inference, overcoming the resolution constraints of previous methods. To enhance the speed and efficiency, we further introduce a progressive Gaussian cache module that dynamically adjusts during training and inference. As the first pose-free generalizable 3D-GS framework, GGRt achieves inference at $\ge$ 5 FPS and real-time rendering at $\ge$ 100 FPS. Through extensive experimentation, we demonstrate that our method outperforms existing NeRF-based pose-free techniques in terms of inference speed and effectiveness. It can also approach the real pose-based 3D-GS methods. Our contributions provide a significant leap forward for the integration of computer vision and computer graphics into practical applications, offering state-of-the-art results on LLFF, KITTI, and Waymo Open datasets and enabling real-time rendering for immersive experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.11367v1">3DGS-ReLoc: 3D Gaussian Splatting for Map Representation and Visual ReLocalization</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      This paper presents a novel system designed for 3D mapping and visual relocalization using 3D Gaussian Splatting. Our proposed method uses LiDAR and camera data to create accurate and visually plausible representations of the environment. By leveraging LiDAR data to initiate the training of the 3D Gaussian Splatting map, our system constructs maps that are both detailed and geometrically accurate. To mitigate excessive GPU memory usage and facilitate rapid spatial queries, we employ a combination of a 2D voxel map and a KD-tree. This preparation makes our method well-suited for visual localization tasks, enabling efficient identification of correspondences between the query image and the rendered image from the Gaussian Splatting map via normalized cross-correlation (NCC). Additionally, we refine the camera pose of the query image using feature-based matching and the Perspective-n-Point (PnP) technique. The effectiveness, adaptability, and precision of our system are demonstrated through extensive evaluation on the KITTI360 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.11831v2">BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 Project Page and Source Code: https://lingzhezhao.github.io/BAD-Gaussians/
    </div>
    <details class="paper-abstract">
      While neural rendering has demonstrated impressive capabilities in 3D scene reconstruction and novel view synthesis, it heavily relies on high-quality sharp images and accurate camera poses. Numerous approaches have been proposed to train Neural Radiance Fields (NeRF) with motion-blurred images, commonly encountered in real-world scenarios such as low-light or long-exposure conditions. However, the implicit representation of NeRF struggles to accurately recover intricate details from severely motion-blurred images and cannot achieve real-time rendering. In contrast, recent advancements in 3D Gaussian Splatting achieve high-quality 3D scene reconstruction and real-time rendering by explicitly optimizing point clouds as Gaussian spheres. In this paper, we introduce a novel approach, named BAD-Gaussians (Bundle Adjusted Deblur Gaussian Splatting), which leverages explicit Gaussian representation and handles severe motion-blurred images with inaccurate camera poses to achieve high-quality scene reconstruction. Our method models the physical image formation process of motion-blurred images and jointly learns the parameters of Gaussians while recovering camera motion trajectories during exposure time. In our experiments, we demonstrate that BAD-Gaussians not only achieves superior rendering quality compared to previous state-of-the-art deblur neural rendering methods on both synthetic and real datasets but also enables real-time rendering capabilities. Our project page and source code is available at https://lingzhezhao.github.io/BAD-Gaussians/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11460v1">Fed3DGS: Scalable 3D Gaussian Splatting with Federated Learning</a></div>
    <div class="paper-meta">
      📅 2024-03-18
      | 💬 Code: https://github.com/DensoITLab/Fed3DGS
    </div>
    <details class="paper-abstract">
      In this work, we present Fed3DGS, a scalable 3D reconstruction framework based on 3D Gaussian splatting (3DGS) with federated learning. Existing city-scale reconstruction methods typically adopt a centralized approach, which gathers all data in a central server and reconstructs scenes. The approach hampers scalability because it places a heavy load on the server and demands extensive data storage when reconstructing scenes on a scale beyond city-scale. In pursuit of a more scalable 3D reconstruction, we propose a federated learning framework with 3DGS, which is a decentralized framework and can potentially use distributed computational resources across millions of clients. We tailor a distillation-based model update scheme for 3DGS and introduce appearance modeling for handling non-IID data in the scenario of 3D reconstruction with federated learning. We simulate our method on several large-scale benchmarks, and our method demonstrates rendered image quality comparable to centralized approaches. In addition, we also simulate our method with data collected in different seasons, demonstrating that our framework can reflect changes in the scenes and our appearance modeling captures changes due to seasonal variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11453v1">Bridging 3D Gaussian and Mesh for Freeview Video Rendering</a></div>
    <div class="paper-meta">
      📅 2024-03-18
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      This is only a preview version of GauMesh. Recently, primitive-based rendering has been proven to achieve convincing results in solving the problem of modeling and rendering the 3D dynamic scene from 2D images. Despite this, in the context of novel view synthesis, each type of primitive has its inherent defects in terms of representation ability. It is difficult to exploit the mesh to depict the fuzzy geometry. Meanwhile, the point-based splatting (e.g. the 3D Gaussian Splatting) method usually produces artifacts or blurry pixels in the area with smooth geometry and sharp textures. As a result, it is difficult, even not impossible, to represent the complex and dynamic scene with a single type of primitive. To this end, we propose a novel approach, GauMesh, to bridge the 3D Gaussian and Mesh for modeling and rendering the dynamic scenes. Given a sequence of tracked mesh as initialization, our goal is to simultaneously optimize the mesh geometry, color texture, opacity maps, a set of 3D Gaussians, and the deformation field. At a specific time, we perform $\alpha$-blending on the RGB and opacity values based on the merged and re-ordered z-buffers from mesh and 3D Gaussian rasterizations. This produces the final rendering, which is supervised by the ground-truth image. Experiments demonstrate that our approach adapts the appropriate type of primitives to represent the different parts of the dynamic scene and outperforms all the baseline methods in both quantitative and qualitative comparisons without losing render speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11447v1">Motion-aware 3D Gaussian Splatting for Efficient Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-03-18
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become an emerging tool for dynamic scene reconstruction. However, existing methods focus mainly on extending static 3DGS into a time-variant representation, while overlooking the rich motion information carried by 2D observations, thus suffering from performance degradation and model redundancy. To address the above problem, we propose a novel motion-aware enhancement framework for dynamic scene reconstruction, which mines useful motion cues from optical flow to improve different paradigms of dynamic 3DGS. Specifically, we first establish a correspondence between 3D Gaussian movements and pixel-level flow. Then a novel flow augmentation method is introduced with additional insights into uncertainty and loss collaboration. Moreover, for the prevalent deformation-based paradigm that presents a harder optimization problem, a transient-aware deformation auxiliary module is proposed. We conduct extensive experiments on both multi-view and monocular scenes to verify the merits of our work. Compared with the baselines, our method shows significant superiority in both rendering quality and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11427v1">BAGS: Building Animatable Gaussian Splatting from a Monocular Video with Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2024-03-18
      | 💬 https://talegqz.github.io/BAGS/
    </div>
    <details class="paper-abstract">
      Animatable 3D reconstruction has significant applications across various fields, primarily relying on artists' handcraft creation. Recently, some studies have successfully constructed animatable 3D models from monocular videos. However, these approaches require sufficient view coverage of the object within the input video and typically necessitate significant time and computational costs for training and rendering. This limitation restricts the practical applications. In this work, we propose a method to build animatable 3D Gaussian Splatting from monocular video with diffusion priors. The 3D Gaussian representations significantly accelerate the training and rendering process, and the diffusion priors allow the method to learn 3D models with limited viewpoints. We also present the rigid regularization to enhance the utilization of the priors. We perform an extensive evaluation across various real-world videos, demonstrating its superior performance compared to the current state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11367v1">3DGS-ReLoc: 3D Gaussian Splatting for Map Representation and Visual ReLocalization</a></div>
    <div class="paper-meta">
      📅 2024-03-17
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      This paper presents a novel system designed for 3D mapping and visual relocalization using 3D Gaussian Splatting. Our proposed method uses LiDAR and camera data to create accurate and visually plausible representations of the environment. By leveraging LiDAR data to initiate the training of the 3D Gaussian Splatting map, our system constructs maps that are both detailed and geometrically accurate. To mitigate excessive GPU memory usage and facilitate rapid spatial queries, we employ a combination of a 2D voxel map and a KD-tree. This preparation makes our method well-suited for visual localization tasks, enabling efficient identification of correspondences between the query image and the rendered image from the Gaussian Splatting map via normalized cross-correlation (NCC). Additionally, we refine the camera pose of the query image using feature-based matching and the Perspective-n-Point (PnP) technique. The effectiveness, adaptability, and precision of our system are demonstrated through extensive evaluation on the KITTI360 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11364v1">Creating Seamless 3D Maps Using Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2024-03-17
      | 💬 10 pages with figures
    </div>
    <details class="paper-abstract">
      It is desirable to create 3D object models and 3D maps from 2D input images for applications such as navigation, virtual tourism, and urban planning. The traditional methods of creating 3D maps, (such as photogrammetry), require a large number of images and odometry. Additionally, traditional methods have difficulty with reflective surfaces and specular reflections; windows and chrome in the scene can be problematic. Google Road View is a familiar application, which uses traditional methods to fuse a collection of 2D input images into the illusion of a 3D map. However, Google Road View does not create an actual 3D object model, only a collection of views. The objective of this work is to create an actual 3D object model using updated techniques. Neural Radiance Fields (NeRF[1]) has emerged as a potential solution, offering the capability to produce more precise and intricate 3D maps. Gaussian Splatting[4] is another contemporary technique. This investigation compares Neural Radiance Fields to Gaussian Splatting, and describes some of their inner workings. Our primary contribution is a method for improving the results of the 3D reconstructed models. Our results indicate that Gaussian Splatting was superior to the NeRF technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.11367v1">3DGS-ReLoc: 3D Gaussian Splatting for Map Representation and Visual ReLocalization</a></div>
    <div class="paper-meta">
      📅 2024-03-17
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      This paper presents a novel system designed for 3D mapping and visual relocalization using 3D Gaussian Splatting. Our proposed method uses LiDAR and camera data to create accurate and visually plausible representations of the environment. By leveraging LiDAR data to initiate the training of the 3D Gaussian Splatting map, our system constructs maps that are both detailed and geometrically accurate. To mitigate excessive GPU memory usage and facilitate rapid spatial queries, we employ a combination of a 2D voxel map and a KD-tree. This preparation makes our method well-suited for visual localization tasks, enabling efficient identification of correspondences between the query image and the rendered image from the Gaussian Splatting map via normalized cross-correlation (NCC). Additionally, we refine the camera pose of the query image using feature-based matching and the Perspective-n-Point (PnP) technique. The effectiveness, adaptability, and precision of our system are demonstrated through extensive evaluation on the KITTI360 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10050v1">Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing</a></div>
    <div class="paper-meta">
      📅 2024-03-15
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting, emerging as a groundbreaking approach, has drawn increasing attention for its capabilities of high-fidelity reconstruction and real-time rendering. However, it couples the appearance and geometry of the scene within the Gaussian attributes, which hinders the flexibility of editing operations, such as texture swapping. To address this issue, we propose a novel approach, namely Texture-GS, to disentangle the appearance from the geometry by representing it as a 2D texture mapped onto the 3D surface, thereby facilitating appearance editing. Technically, the disentanglement is achieved by our proposed texture mapping module, which consists of a UV mapping MLP to learn the UV coordinates for the 3D Gaussian centers, a local Taylor expansion of the MLP to efficiently approximate the UV coordinates for the ray-Gaussian intersections, and a learnable texture to capture the fine-grained appearance. Extensive experiments on the DTU dataset demonstrate that our method not only facilitates high-fidelity appearance editing but also achieves real-time rendering on consumer-level devices, e.g. a single RTX 2080 Ti GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.09143v1">A New Split Algorithm for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-15
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting models, as a novel explicit 3D representation, have been applied in many domains recently, such as explicit geometric editing and geometry generation. Progress has been rapid. However, due to their mixed scales and cluttered shapes, 3D Gaussian splatting models can produce a blurred or needle-like effect near the surface. At the same time, 3D Gaussian splatting models tend to flatten large untextured regions, yielding a very sparse point cloud. These problems are caused by the non-uniform nature of 3D Gaussian splatting models, so in this paper, we propose a new 3D Gaussian splitting algorithm, which can produce a more uniform and surface-bounded 3D Gaussian splatting model. Our algorithm splits an $N$-dimensional Gaussian into two N-dimensional Gaussians. It ensures consistency of mathematical characteristics and similarity of appearance, allowing resulting 3D Gaussian splatting models to be more uniform and a better fit to the underlying surface, and thus more suitable for explicit editing, point cloud extraction and other tasks. Meanwhile, our 3D Gaussian splitting approach has a very simple closed-form solution, making it readily applicable to any 3D Gaussian model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09637v1">GaussianGrasper: 3D Language Gaussian Splatting for Open-vocabulary Robotic Grasping</a></div>
    <div class="paper-meta">
      📅 2024-03-14
    </div>
    <details class="paper-abstract">
      Constructing a 3D scene capable of accommodating open-ended language queries, is a pivotal pursuit, particularly within the domain of robotics. Such technology facilitates robots in executing object manipulations based on human language directives. To tackle this challenge, some research efforts have been dedicated to the development of language-embedded implicit fields. However, implicit fields (e.g. NeRF) encounter limitations due to the necessity of processing a large number of input views for reconstruction, coupled with their inherent inefficiencies in inference. Thus, we present the GaussianGrasper, which utilizes 3D Gaussian Splatting to explicitly represent the scene as a collection of Gaussian primitives. Our approach takes a limited set of RGB-D views and employs a tile-based splatting technique to create a feature field. In particular, we propose an Efficient Feature Distillation (EFD) module that employs contrastive learning to efficiently and accurately distill language embeddings derived from foundational models. With the reconstructed geometry of the Gaussian field, our method enables the pre-trained grasping model to generate collision-free grasp pose candidates. Furthermore, we propose a normal-guided grasp module to select the best grasp pose. Through comprehensive real-world experiments, we demonstrate that GaussianGrasper enables robots to accurately query and grasp objects with language instructions, providing a new solution for language-guided manipulation tasks. Data and codes can be available at https://github.com/MrSecant/GaussianGrasper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17061v2">HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-14
      | 💬 Accepted by CVPR 2024, camera-ready version. Project Page: https://alvinliu0.github.io/projects/HumanGaussian
    </div>
    <details class="paper-abstract">
      Realistic 3D human generation from text prompts is a desirable yet challenging task. Existing methods optimize 3D representations like mesh or neural fields via score distillation sampling (SDS), which suffers from inadequate fine details or excessive training time. In this paper, we propose an efficient yet effective framework, HumanGaussian, that generates high-quality 3D humans with fine-grained geometry and realistic appearance. Our key insight is that 3D Gaussian Splatting is an efficient renderer with periodic Gaussian shrinkage or growing, where such adaptive density control can be naturally guided by intrinsic human structures. Specifically, 1) we first propose a Structure-Aware SDS that simultaneously optimizes human appearance and geometry. The multi-modal score function from both RGB and depth space is leveraged to distill the Gaussian densification and pruning process. 2) Moreover, we devise an Annealed Negative Prompt Guidance by decomposing SDS into a noisier generative score and a cleaner classifier score, which well addresses the over-saturation issue. The floating artifacts are further eliminated based on Gaussian size in a prune-only phase to enhance generation smoothness. Extensive experiments demonstrate the superior efficiency and competitive quality of our framework, rendering vivid 3D humans under diverse scenarios. Project Page: https://alvinliu0.github.io/projects/HumanGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09143v1">A New Split Algorithm for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-14
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting models, as a novel explicit 3D representation, have been applied in many domains recently, such as explicit geometric editing and geometry generation. Progress has been rapid. However, due to their mixed scales and cluttered shapes, 3D Gaussian splatting models can produce a blurred or needle-like effect near the surface. At the same time, 3D Gaussian splatting models tend to flatten large untextured regions, yielding a very sparse point cloud. These problems are caused by the non-uniform nature of 3D Gaussian splatting models, so in this paper, we propose a new 3D Gaussian splitting algorithm, which can produce a more uniform and surface-bounded 3D Gaussian splatting model. Our algorithm splits an $N$-dimensional Gaussian into two N-dimensional Gaussians. It ensures consistency of mathematical characteristics and similarity of appearance, allowing resulting 3D Gaussian splatting models to be more uniform and a better fit to the underlying surface, and thus more suitable for explicit editing, point cloud extraction and other tasks. Meanwhile, our 3D Gaussian splitting approach has a very simple closed-form solution, making it readily applicable to any 3D Gaussian model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.09143v1">A New Split Algorithm for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-14
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting models, as a novel explicit 3D representation, have been applied in many domains recently, such as explicit geometric editing and geometry generation. Progress has been rapid. However, due to their mixed scales and cluttered shapes, 3D Gaussian splatting models can produce a blurred or needle-like effect near the surface. At the same time, 3D Gaussian splatting models tend to flatten large untextured regions, yielding a very sparse point cloud. These problems are caused by the non-uniform nature of 3D Gaussian splatting models, so in this paper, we propose a new 3D Gaussian splitting algorithm, which can produce a more uniform and surface-bounded 3D Gaussian splatting model. Our algorithm splits an $N$-dimensional Gaussian into two N-dimensional Gaussians. It ensures consistency of mathematical characteristics and similarity of appearance, allowing resulting 3D Gaussian splatting models to be more uniform and a better fit to the underlying surface, and thus more suitable for explicit editing, point cloud extraction and other tasks. Meanwhile, our 3D Gaussian splitting approach has a very simple closed-form solution, making it readily applicable to any 3D Gaussian model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07807v1">StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-12
    </div>
    <details class="paper-abstract">
      We introduce StyleGaussian, a novel 3D style transfer technique that allows instant transfer of any image's style to a 3D scene at 10 frames per second (fps). Leveraging 3D Gaussian Splatting (3DGS), StyleGaussian achieves style transfer without compromising its real-time rendering ability and multi-view consistency. It achieves instant style transfer with three steps: embedding, transfer, and decoding. Initially, 2D VGG scene features are embedded into reconstructed 3D Gaussians. Next, the embedded features are transformed according to a reference style image. Finally, the transformed features are decoded into the stylized RGB. StyleGaussian has two novel designs. The first is an efficient feature rendering strategy that first renders low-dimensional features and then maps them into high-dimensional features while embedding VGG features. It cuts the memory consumption significantly and enables 3DGS to render the high-dimensional memory-intensive features. The second is a K-nearest-neighbor-based 3D CNN. Working as the decoder for the stylized features, it eliminates the 2D CNN operations that compromise strict multi-view consistency. Extensive experiments show that StyleGaussian achieves instant 3D stylization with superior stylization quality while preserving real-time rendering and strict multi-view consistency. Project page: https://kunhao-liu.github.io/StyleGaussian/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05087v1">SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-03-08
      | 💬 [CVPR 2024] Code and data are available at https://github.com/initialneil/SplattingAvatar
    </div>
    <details class="paper-abstract">
      We present SplattingAvatar, a hybrid 3D representation of photorealistic human avatars with Gaussian Splatting embedded on a triangle mesh, which renders over 300 FPS on a modern GPU and 30 FPS on a mobile device. We disentangle the motion and appearance of a virtual human with explicit mesh geometry and implicit appearance modeling with Gaussian Splatting. The Gaussians are defined by barycentric coordinates and displacement on a triangle mesh as Phong surfaces. We extend lifted optimization to simultaneously optimize the parameters of the Gaussians while walking on the triangle mesh. SplattingAvatar is a hybrid representation of virtual humans where the mesh represents low-frequency motion and surface deformation, while the Gaussians take over the high-frequency geometry and detailed appearance. Unlike existing deformation methods that rely on an MLP-based linear blend skinning (LBS) field for motion, we control the rotation and translation of the Gaussians directly by mesh, which empowers its compatibility with various animation techniques, e.g., skeletal animation, blend shapes, and mesh editing. Trainable from monocular videos for both full-body and head avatars, SplattingAvatar shows state-of-the-art rendering quality across multiple datasets.
    </details>
</div>
