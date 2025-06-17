# gaussian splatting - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17345v2">SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting and a Physically Grounded Image Formation Model</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ICRA 2025. Project page here: https://seasplat.github.io
    </div>
    <details class="paper-abstract">
      We introduce SeaSplat, a method to enable real-time rendering of underwater scenes leveraging recent advances in 3D radiance fields. Underwater scenes are challenging visual environments, as rendering through a medium such as water introduces both range and color dependent effects on image capture. We constrain 3D Gaussian Splatting (3DGS), a recent advance in radiance fields enabling rapid training and real-time rendering of full 3D scenes, with a physically grounded underwater image formation model. Applying SeaSplat to the real-world scenes from SeaThru-NeRF dataset, a scene collected by an underwater vehicle in the US Virgin Islands, and simulation-degraded real-world scenes, not only do we see increased quantitative performance on rendering novel viewpoints from the scene with the medium present, but are also able to recover the underlying true color of the scene and restore renders to be without the presence of the intervening medium. We show that the underwater image formation helps learn scene structure, with better depth maps, as well as show that our improvements maintain the significant computational improvements afforded by leveraging a 3D Gaussian representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01822v1">GSCodec Studio: A Modular Framework for Gaussian Splat Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Repository of the project: https://github.com/JasonLSC/GSCodec_Studio
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting and its extension to 4D dynamic scenes enable photorealistic, real-time rendering from real-world captures, positioning Gaussian Splats (GS) as a promising format for next-generation immersive media. However, their high storage requirements pose significant challenges for practical use in sharing, transmission, and storage. Despite various studies exploring GS compression from different perspectives, these efforts remain scattered across separate repositories, complicating benchmarking and the integration of best practices. To address this gap, we present GSCodec Studio, a unified and modular framework for GS reconstruction, compression, and rendering. The framework incorporates a diverse set of 3D/4D GS reconstruction methods and GS compression techniques as modular components, facilitating flexible combinations and comprehensive comparisons. By integrating best practices from community research and our own explorations, GSCodec Studio supports the development of compact representation and compression solutions for static and dynamic Gaussian Splats, namely our Static and Dynamic GSCodec, achieving competitive rate-distortion performance in static and dynamic GS compression. The code for our framework is publicly available at https://github.com/JasonLSC/GSCodec_Studio , to advance the research on Gaussian Splats compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01799v1">WorldExplorer: Towards Generating Fully Navigable 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 project page: see https://the-world-explorer.github.io/, video: see https://youtu.be/c1lBnwJWNmE
    </div>
    <details class="paper-abstract">
      Generating 3D worlds from text is a highly anticipated goal in computer vision. Existing works are limited by the degree of exploration they allow inside of a scene, i.e., produce streched-out and noisy artifacts when moving beyond central or panoramic perspectives. To this end, we propose WorldExplorer, a novel method based on autoregressive video trajectory generation, which builds fully navigable 3D scenes with consistent visual quality across a wide range of viewpoints. We initialize our scenes by creating multi-view consistent images corresponding to a 360 degree panorama. Then, we expand it by leveraging video diffusion models in an iterative scene generation pipeline. Concretely, we generate multiple videos along short, pre-defined trajectories, that explore the scene in depth, including motion around objects. Our novel scene memory conditions each video on the most relevant prior views, while a collision-detection mechanism prevents degenerate results, like moving into objects. Finally, we fuse all generated views into a unified 3D representation via 3D Gaussian Splatting optimization. Compared to prior approaches, WorldExplorer produces high-quality scenes that remain stable under large camera motion, enabling for the first time realistic and unrestricted exploration. We believe this marks a significant step toward generating immersive and truly explorable virtual 3D environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01600v1">WoMAP: World Models For Embodied Open-Vocabulary Object Localization</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Language-instructed active object localization is a critical challenge for robots, requiring efficient exploration of partially observable environments. However, state-of-the-art approaches either struggle to generalize beyond demonstration datasets (e.g., imitation learning methods) or fail to generate physically grounded actions (e.g., VLMs). To address these limitations, we introduce WoMAP (World Models for Active Perception): a recipe for training open-vocabulary object localization policies that: (i) uses a Gaussian Splatting-based real-to-sim-to-real pipeline for scalable data generation without the need for expert demonstrations, (ii) distills dense rewards signals from open-vocabulary object detectors, and (iii) leverages a latent world model for dynamics and rewards prediction to ground high-level action proposals at inference time. Rigorous simulation and hardware experiments demonstrate WoMAP's superior performance in a broad range of zero-shot object localization tasks, with more than 9x and 2x higher success rates compared to VLM and diffusion policy baselines, respectively. Further, we show that WoMAP achieves strong generalization and sim-to-real transfer on a TidyBot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01379v1">RadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      High-Fidelity 3D scene reconstruction plays a crucial role in autonomous driving by enabling novel data generation from existing datasets. This allows simulating safety-critical scenarios and augmenting training datasets without incurring further data collection costs. While recent advances in radiance fields have demonstrated promising results in 3D reconstruction and sensor data synthesis using cameras and LiDAR, their potential for radar remains largely unexplored. Radar is crucial for autonomous driving due to its robustness in adverse weather conditions like rain, fog, and snow, where optical sensors often struggle. Although the state-of-the-art radar-based neural representation shows promise for 3D driving scene reconstruction, it performs poorly in scenarios with significant radar noise, including receiver saturation and multipath reflection. Moreover, it is limited to synthesizing preprocessed, noise-excluded radar images, failing to address realistic radar data synthesis. To address these limitations, this paper proposes RadarSplat, which integrates Gaussian Splatting with novel radar noise modeling to enable realistic radar data synthesis and enhanced 3D reconstruction. Compared to the state-of-the-art, RadarSplat achieves superior radar image synthesis (+3.4 PSNR / 2.6x SSIM) and improved geometric reconstruction (-40% RMSE / 1.5x Accuracy), demonstrating its effectiveness in generating high-fidelity radar data and scene reconstruction. A project page is available at https://umautobots.github.io/radarsplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04343v2">Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Project page: https://www.robots.ox.ac.uk/~vgg/research/flash3d/
    </div>
    <details class="paper-abstract">
      We propose Flash3D, a method for scene reconstruction and novel view synthesis from a single image which is both very generalisable and efficient. For generalisability, we start from a "foundation" model for monocular depth estimation and extend it to a full 3D shape and appearance reconstructor. For efficiency, we base this extension on feed-forward Gaussian Splatting. Specifically, we predict a first layer of 3D Gaussians at the predicted depth, and then add additional layers of Gaussians that are offset in space, allowing the model to complete the reconstruction behind occlusions and truncations. Flash3D is very efficient, trainable on a single GPU in a day, and thus accessible to most researchers. It achieves state-of-the-art results when trained and tested on RealEstate10k. When transferred to unseen datasets like NYU it outperforms competitors by a large margin. More impressively, when transferred to KITTI, Flash3D achieves better PSNR than methods trained specifically on that dataset. In some instances, it even outperforms recent methods that use multiple views as input. Code, models, demo, and more results are available at https://www.robots.ox.ac.uk/~vgg/research/flash3d/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01109v1">CountingFruit: Real-Time 3D Fruit Counting with Language-Guided Semantic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Accurate fruit counting in real-world agricultural environments is a longstanding challenge due to visual occlusions, semantic ambiguity, and the high computational demands of 3D reconstruction. Existing methods based on neural radiance fields suffer from low inference speed, limited generalization, and lack support for open-set semantic control. This paper presents FruitLangGS, a real-time 3D fruit counting framework that addresses these limitations through spatial reconstruction, semantic embedding, and language-guided instance estimation. FruitLangGS first reconstructs orchard-scale scenes using an adaptive Gaussian splatting pipeline with radius-aware pruning and tile-based rasterization for efficient rendering. To enable semantic control, each Gaussian encodes a compressed CLIP-aligned language embedding, forming a compact and queryable 3D representation. At inference time, prompt-based semantic filtering is applied directly in 3D space, without relying on image-space segmentation or view-level fusion. The selected Gaussians are then converted into dense point clouds via distribution-aware sampling and clustered to estimate fruit counts. Experimental results on real orchard data demonstrate that FruitLangGS achieves higher rendering speed, semantic flexibility, and counting accuracy compared to prior approaches, offering a new perspective for language-driven, real-time neural rendering across open-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00970v1">Globally Consistent RGB-D SLAM with 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting-based RGB-D SLAM displays remarkable performance of high-fidelity 3D reconstruction. However, the lack of depth rendering consistency and efficient loop closure limits the quality of its geometric reconstructions and its ability to perform globally consistent mapping online. In this paper, we present 2DGS-SLAM, an RGB-D SLAM system using 2D Gaussian splatting as the map representation. By leveraging the depth-consistent rendering property of the 2D variant, we propose an accurate camera pose optimization method and achieve geometrically accurate 3D reconstruction. In addition, we implement efficient loop detection and camera relocalization by leveraging MASt3R, a 3D foundation model, and achieve efficient map updates by maintaining a local active map. Experiments show that our 2DGS-SLAM approach achieves superior tracking accuracy, higher surface reconstruction quality, and more consistent global map reconstruction compared to existing rendering-based SLAM methods, while maintaining high-fidelity image rendering and improved computational efficiency.
    </details>
</div>
