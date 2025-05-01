# gaussian splatting - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24270v2">Visual Acoustic Fields</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Objects produce different sounds when hit, and humans can intuitively infer how an object might sound based on its appearance and material properties. Inspired by this intuition, we propose Visual Acoustic Fields, a framework that bridges hitting sounds and visual signals within a 3D space using 3D Gaussian Splatting (3DGS). Our approach features two key modules: sound generation and sound localization. The sound generation module leverages a conditional diffusion model, which takes multiscale features rendered from a feature-augmented 3DGS to generate realistic hitting sounds. Meanwhile, the sound localization module enables querying the 3D scene, represented by the feature-augmented 3DGS, to localize hitting positions based on the sound sources. To support this framework, we introduce a novel pipeline for collecting scene-level visual-sound sample pairs, achieving alignment between captured images, impact locations, and corresponding sounds. To the best of our knowledge, this is the first dataset to connect visual and acoustic signals in a 3D context. Extensive experiments on our dataset demonstrate the effectiveness of Visual Acoustic Fields in generating plausible impact sounds and accurately localizing impact sources. Our project page is at https://yuelei0428.github.io/projects/Visual-Acoustic-Fields/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10437v2">4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 CVPR 2025. Project Page: https://4d-langsplat.github.io
    </div>
    <details class="paper-abstract">
      Learning 4D language fields to enable time-sensitive, open-ended language queries in dynamic scenes is essential for many real-world applications. While LangSplat successfully grounds CLIP features into 3D Gaussian representations, achieving precision and efficiency in 3D static scenes, it lacks the ability to handle dynamic 4D fields as CLIP, designed for static image-text tasks, cannot capture temporal dynamics in videos. Real-world environments are inherently dynamic, with object semantics evolving over time. Building a precise 4D language field necessitates obtaining pixel-aligned, object-wise video features, which current vision models struggle to achieve. To address these challenges, we propose 4D LangSplat, which learns 4D language fields to handle time-agnostic or time-sensitive open-vocabulary queries in dynamic scenes efficiently. 4D LangSplat bypasses learning the language field from vision features and instead learns directly from text generated from object-wise video captions via Multimodal Large Language Models (MLLMs). Specifically, we propose a multimodal object-wise video prompting method, consisting of visual and text prompts that guide MLLMs to generate detailed, temporally consistent, high-quality captions for objects throughout a video. These captions are encoded using a Large Language Model into high-quality sentence embeddings, which then serve as pixel-aligned, object-specific feature supervision, facilitating open-vocabulary text queries through shared embedding spaces. Recognizing that objects in 4D scenes exhibit smooth transitions across states, we further propose a status deformable network to model these continuous changes over time effectively. Our results across multiple benchmarks demonstrate that 4D LangSplat attains precise and efficient results for both time-sensitive and time-agnostic open-vocabulary queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00370v2">Scalable Real2Sim: Physics-Aware Asset Generation Via Robotic Pick-and-Place Setups</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Website: https://scalable-real2sim.github.io/
    </div>
    <details class="paper-abstract">
      Simulating object dynamics from real-world perception shows great promise for digital twins and robotic manipulation but often demands labor-intensive measurements and expertise. We present a fully automated Real2Sim pipeline that generates simulation-ready assets for real-world objects through robotic interaction. Using only a robot's joint torque sensors and an external camera, the pipeline identifies visual geometry, collision geometry, and physical properties such as inertial parameters. Our approach introduces a general method for extracting high-quality, object-centric meshes from photometric reconstruction techniques (e.g., NeRF, Gaussian Splatting) by employing alpha-transparent training while explicitly distinguishing foreground occlusions from background subtraction. We validate the full pipeline through extensive experiments, demonstrating its effectiveness across diverse objects. By eliminating the need for manual intervention or environment modifications, our pipeline can be integrated directly into existing pick-and-place setups, enabling scalable and efficient dataset creation. Project page (with code and data): https://scalable-real2sim.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22876v2">VizFlyt: Perception-centric Pedagogical Framework For Autonomous Aerial Robots</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted at ICRA 2025. Projected Page: https://pear.wpi.edu/research/vizflyt.html
    </div>
    <details class="paper-abstract">
      Autonomous aerial robots are becoming commonplace in our lives. Hands-on aerial robotics courses are pivotal in training the next-generation workforce to meet the growing market demands. Such an efficient and compelling course depends on a reliable testbed. In this paper, we present VizFlyt, an open-source perception-centric Hardware-In-The-Loop (HITL) photorealistic testing framework for aerial robotics courses. We utilize pose from an external localization system to hallucinate real-time and photorealistic visual sensors using 3D Gaussian Splatting. This enables stress-free testing of autonomy algorithms on aerial robots without the risk of crashing into obstacles. We achieve over 100Hz of system update rate. Lastly, we build upon our past experiences of offering hands-on aerial robotics courses and propose a new open-source and open-hardware curriculum based on VizFlyt for the future. We test our framework on various course projects in real-world HITL experiments and present the results showing the efficacy of such a system and its large potential use cases. Code, datasets, hardware guides and demo videos are available at https://pear.wpi.edu/research/vizflyt.html
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00773v1">DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting (3DGS) has gained considerable attentions in the field of novel view synthesis due to its fast performance while yielding the excellent image quality. However, 3DGS in sparse-view settings (e.g., three-view inputs) often faces with the problem of overfitting to training views, which significantly drops the visual quality of novel view images. Many existing approaches have tackled this issue by using strong priors, such as 2D generative contextual information and external depth signals. In contrast, this paper introduces a prior-free method, so-called DropGaussian, with simple changes in 3D Gaussian splatting. Specifically, we randomly remove Gaussians during the training process in a similar way of dropout, which allows non-excluded Gaussians to have larger gradients while improving their visibility. This makes the remaining Gaussians to contribute more to the optimization process for rendering with sparse input views. Such simple operation effectively alleviates the overfitting problem and enhances the quality of novel view synthesis. By simply applying DropGaussian to the original 3DGS framework, we can achieve the competitive performance with existing prior-based 3DGS methods in sparse-view settings of benchmark datasets without any additional complexity. The code and model are publicly available at: https://github.com/DCVL-3D/DropGaussian release.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00763v1">UnIRe: Unsupervised Instance Decomposition for Dynamic Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Reconstructing and decomposing dynamic urban scenes is crucial for autonomous driving, urban planning, and scene editing. However, existing methods fail to perform instance-aware decomposition without manual annotations, which is crucial for instance-level scene editing.We propose UnIRe, a 3D Gaussian Splatting (3DGS) based approach that decomposes a scene into a static background and individual dynamic instances using only RGB images and LiDAR point clouds. At its core, we introduce 4D superpoints, a novel representation that clusters multi-frame LiDAR points in 4D space, enabling unsupervised instance separation based on spatiotemporal correlations. These 4D superpoints serve as the foundation for our decomposed 4D initialization, i.e., providing spatial and temporal initialization to train a dynamic 3DGS for arbitrary dynamic classes without requiring bounding boxes or object templates.Furthermore, we introduce a smoothness regularization strategy in both 2D and 3D space, further improving the temporal stability.Experiments on benchmark datasets show that our method outperforms existing methods in decomposed dynamic scene reconstruction while enabling accurate and flexible instance-level editing, making it a practical solution for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00665v1">Monocular and Generalizable Gaussian Talking Head Animation</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      In this work, we introduce Monocular and Generalizable Gaussian Talking Head Animation (MGGTalk), which requires monocular datasets and generalizes to unseen identities without personalized re-training. Compared with previous 3D Gaussian Splatting (3DGS) methods that requires elusive multi-view datasets or tedious personalized learning/inference, MGGtalk enables more practical and broader applications. However, in the absence of multi-view and personalized training data, the incompleteness of geometric and appearance information poses a significant challenge. To address these challenges, MGGTalk explores depth information to enhance geometric and facial symmetry characteristics to supplement both geometric and appearance features. Initially, based on the pixel-wise geometric information obtained from depth estimation, we incorporate symmetry operations and point cloud filtering techniques to ensure a complete and precise position parameter for 3DGS. Subsequently, we adopt a two-stage strategy with symmetric priors for predicting the remaining 3DGS parameters. We begin by predicting Gaussian parameters for the visible facial regions of the source image. These parameters are subsequently utilized to improve the prediction of Gaussian parameters for the non-visible regions. Extensive experiments demonstrate that MGGTalk surpasses previous state-of-the-art methods, achieving superior performance across various metrics.
    </details>
</div>
