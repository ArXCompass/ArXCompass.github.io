# gaussian splatting - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23309v2">SurgTPGS: Semantic 3D Surgical Scene Understanding with Text Promptable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 MICCAI 2025. Project Page: https://lastbasket.github.io/MICCAI-2025-SurgTPGS/
    </div>
    <details class="paper-abstract">
      In contemporary surgical research and practice, accurately comprehending 3D surgical scenes with text-promptable capabilities is particularly crucial for surgical planning and real-time intra-operative guidance, where precisely identifying and interacting with surgical tools and anatomical structures is paramount. However, existing works focus on surgical vision-language model (VLM), 3D reconstruction, and segmentation separately, lacking support for real-time text-promptable 3D queries. In this paper, we present SurgTPGS, a novel text-promptable Gaussian Splatting method to fill this gap. We introduce a 3D semantics feature learning strategy incorporating the Segment Anything model and state-of-the-art vision-language models. We extract the segmented language features for 3D surgical scene reconstruction, enabling a more in-depth understanding of the complex surgical environment. We also propose semantic-aware deformation tracking to capture the seamless deformation of semantic features, providing a more precise reconstruction for both texture and semantic features. Furthermore, we present semantic region-aware optimization, which utilizes regional-based semantic information to supervise the training, particularly promoting the reconstruction quality and semantic smoothness. We conduct comprehensive experiments on two real-world surgical datasets to demonstrate the superiority of SurgTPGS over state-of-the-art methods, highlighting its potential to revolutionize surgical practices. SurgTPGS paves the way for developing next-generation intelligent surgical systems by enhancing surgical precision and safety. Our code is available at: https://github.com/lastbasket/SurgTPGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22099v2">BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 Accepted at ICCV 2025, Project Page: https://github.com/fudan-zvg/BezierGS
    </div>
    <details class="paper-abstract">
      The realistic reconstruction of street scenes is critical for developing real-world simulators in autonomous driving. Most existing methods rely on object pose annotations, using these poses to reconstruct dynamic objects and move them during the rendering process. This dependence on high-precision object annotations limits large-scale and extensive scene reconstruction. To address this challenge, we propose B\'ezier curve Gaussian splatting (B\'ezierGS), which represents the motion trajectories of dynamic objects using learnable B\'ezier curves. This approach fully leverages the temporal information of dynamic objects and, through learnable curve modeling, automatically corrects pose errors. By introducing additional supervision on dynamic object rendering and inter-curve consistency constraints, we achieve reasonable and accurate separation and reconstruction of scene elements. Extensive experiments on the Waymo Open Dataset and the nuPlan benchmark demonstrate that B\'ezierGS outperforms state-of-the-art alternatives in both dynamic and static scene components reconstruction and novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07007v3">Grounding Creativity in Physics: A Brief Survey of Physical Priors in AIGC</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 Accepted by IJCAI 2025 Survey Track
    </div>
    <details class="paper-abstract">
      Recent advancements in AI-generated content have significantly improved the realism of 3D and 4D generation. However, most existing methods prioritize appearance consistency while neglecting underlying physical principles, leading to artifacts such as unrealistic deformations, unstable dynamics, and implausible objects interactions. Incorporating physics priors into generative models has become a crucial research direction to enhance structural integrity and motion realism. This survey provides a review of physics-aware generative methods, systematically analyzing how physical constraints are integrated into 3D and 4D generation. First, we examine recent works in incorporating physical priors into static and dynamic 3D generation, categorizing methods based on representation types, including vision-based, NeRF-based, and Gaussian Splatting-based approaches. Second, we explore emerging techniques in 4D generation, focusing on methods that model temporal dynamics with physical simulations. Finally, we conduct a comparative analysis of major methods, highlighting their strengths, limitations, and suitability for different materials and motion dynamics. By presenting an in-depth analysis of physics-grounded AIGC, this survey aims to bridge the gap between generative models and physical realism, providing insights that inspire future research in physically consistent content generation.
    </details>
</div>
