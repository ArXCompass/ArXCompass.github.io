# gaussian splatting - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

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
