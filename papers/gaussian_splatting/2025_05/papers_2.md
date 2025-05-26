# gaussian splatting - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01235v1">Compensating Spatiotemporally Inconsistent Observations for Online Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 SIGGRAPH 2025, Project page: https://bbangsik13.github.io/OR2
    </div>
    <details class="paper-abstract">
      Online reconstruction of dynamic scenes is significant as it enables learning scenes from live-streaming video inputs, while existing offline dynamic reconstruction methods rely on recorded video inputs. However, previous online reconstruction approaches have primarily focused on efficiency and rendering quality, overlooking the temporal consistency of their results, which often contain noticeable artifacts in static regions. This paper identifies that errors such as noise in real-world recordings affect temporal inconsistency in online reconstruction. We propose a method that enhances temporal consistency in online reconstruction from observations with temporal inconsistency which is inevitable in cameras. We show that our method restores the ideal observation by subtracting the learned error. We demonstrate that applying our method to various baselines significantly enhances both temporal consistency and rendering quality across datasets. Code, video results, and checkpoints are available at https://bbangsik13.github.io/OR2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00362v2">UDGS-SLAM : UniDepth Assisted Gaussian Splatting for Monocular SLAM</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      Recent advancements in monocular neural depth estimation, particularly those achieved by the UniDepth network, have prompted the investigation of integrating UniDepth within a Gaussian splatting framework for monocular SLAM. This study presents UDGS-SLAM, a novel approach that eliminates the necessity of RGB-D sensors for depth estimation within Gaussian splatting framework. UDGS-SLAM employs statistical filtering to ensure local consistency of the estimated depth and jointly optimizes camera trajectory and Gaussian scene representation parameters. The proposed method achieves high-fidelity rendered images and low ATERMSE of the camera trajectory. The performance of UDGS-SLAM is rigorously evaluated using the TUM RGB-D dataset and benchmarked against several baseline methods, demonstrating superior performance across various scenarios. Additionally, an ablation study is conducted to validate design choices and investigate the impact of different network backbone encoders on system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15122v3">MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 The first two authors contributed equally to this work (equal contribution). The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/mobgs-site/
    </div>
    <details class="paper-abstract">
      We present MoBGS, a novel deblurring dynamic 3D Gaussian Splatting (3DGS) framework capable of reconstructing sharp and high-quality novel spatio-temporal views from blurry monocular videos in an end-to-end manner. Existing dynamic novel view synthesis (NVS) methods are highly sensitive to motion blur in casually captured videos, resulting in significant degradation of rendering quality. While recent approaches address motion-blurred inputs for NVS, they primarily focus on static scene reconstruction and lack dedicated motion modeling for dynamic objects. To overcome these limitations, our MoBGS introduces a novel Blur-adaptive Latent Camera Estimation (BLCE) method for effective latent camera trajectory estimation, improving global camera motion deblurring. In addition, we propose a physically-inspired Latent Camera-induced Exposure Estimation (LCEE) method to ensure consistent deblurring of both global camera and local object motion. Our MoBGS framework ensures the temporal consistency of unseen latent timestamps and robust motion decomposition of static and dynamic regions. Extensive experiments on the Stereo Blur dataset and real-world blurry videos show that our MoBGS significantly outperforms the very recent advanced methods (DyBluRF and Deblur4DGS), achieving state-of-the-art performance for dynamic NVS under motion blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00421v1">Real-Time Animatable 2DGS-Avatars with Detail Enhancement from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      High-quality, animatable 3D human avatar reconstruction from monocular videos offers significant potential for reducing reliance on complex hardware, making it highly practical for applications in game development, augmented reality, and social media. However, existing methods still face substantial challenges in capturing fine geometric details and maintaining animation stability, particularly under dynamic or complex poses. To address these issues, we propose a novel real-time framework for animatable human avatar reconstruction based on 2D Gaussian Splatting (2DGS). By leveraging 2DGS and global SMPL pose parameters, our framework not only aligns positional and rotational discrepancies but also enables robust and natural pose-driven animation of the reconstructed avatars. Furthermore, we introduce a Rotation Compensation Network (RCN) that learns rotation residuals by integrating local geometric features with global pose parameters. This network significantly improves the handling of non-rigid deformations and ensures smooth, artifact-free pose transitions during animation. Experimental results demonstrate that our method successfully reconstructs realistic and highly animatable human avatars from monocular videos, effectively preserving fine-grained details while ensuring stable and natural pose variation. Our approach surpasses current state-of-the-art methods in both reconstruction quality and animation robustness on public benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18768v2">TransparentGS: Fast Inverse Rendering of Transparent Objects with Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 accepted by SIGGRAPH 2025; https://letianhuang.github.io/transparentgs/
    </div>
    <details class="paper-abstract">
      The emergence of neural and Gaussian-based radiance field methods has led to considerable advancements in novel view synthesis and 3D object reconstruction. Nonetheless, specular reflection and refraction continue to pose significant challenges due to the instability and incorrect overfitting of radiance fields to high-frequency light variations. Currently, even 3D Gaussian Splatting (3D-GS), as a powerful and efficient tool, falls short in recovering transparent objects with nearby contents due to the existence of apparent secondary ray effects. To address this issue, we propose TransparentGS, a fast inverse rendering pipeline for transparent objects based on 3D-GS. The main contributions are three-fold. Firstly, an efficient representation of transparent objects, transparent Gaussian primitives, is designed to enable specular refraction through a deferred refraction strategy. Secondly, we leverage Gaussian light field probes (GaussProbe) to encode both ambient light and nearby contents in a unified framework. Thirdly, a depth-based iterative probes query (IterQuery) algorithm is proposed to reduce the parallax errors in our probe-based framework. Experiments demonstrate the speed and accuracy of our approach in recovering transparent objects from complex environments, as well as several applications in computer graphics and vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20379v2">GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      In this paper, we present a method for localizing a query image with respect to a precomputed 3D Gaussian Splatting (3DGS) scene representation. First, the method uses 3DGS to render a synthetic RGBD image at some initial pose estimate. Second, it establishes 2D-2D correspondences between the query image and this synthetic image. Third, it uses the depth map to lift the 2D-2D correspondences to 2D-3D correspondences and solves a perspective-n-point (PnP) problem to produce a final pose estimate. Results from evaluation across three existing datasets with 38 scenes and over 2,700 test images show that our method significantly reduces both inference time (by over two orders of magnitude, from more than 10 seconds to as fast as 0.1 seconds) and estimation error compared to baseline methods that use photometric loss minimization. Results also show that our method tolerates large errors in the initial pose estimate of up to 55{\deg} in rotation and 1.1 units in translation (normalized by scene scale), achieving final pose errors of less than 5{\deg} in rotation and 0.05 units in translation on 90% of images from the Synthetic NeRF and Mip-NeRF360 datasets and on 42% of images from the more challenging Tanks and Temples dataset.
    </details>
</div>
