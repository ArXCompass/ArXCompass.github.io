# gaussian splatting - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00498v1">GOR-IS: 3D Gaussian Object Removal in the Intrinsic Space</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      Recent advances in Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have made it standard practice to reconstruct 3D scenes from multi-view images. Removing objects from such 3D representations is a fundamental editing task that requires complete and seamless inpainting of occluded regions, ensuring consistency in geometry and appearance. Although existing methods have made notable progress in improving inpainting consistency, they often neglect global lighting effects, leading to physically implausible results. Moreover, these methods struggle with view-dependent non-Lambertian surfaces, where appearance varies across viewpoints, leading to unreliable inpainting. In this paper, we present 3D Gaussian Object Removal in the Intrinsic Space (GOR-IS), a novel framework for physically consistent and visually coherent 3D object removal. Our approach decomposes the scene into intrinsic components and explicitly models light transport to maintain global lighting effects consistency. Furthermore, we introduce an intrinsic-space inpainting module that operates directly in the material and lighting domains, effectively addressing the challenges posed by non-Lambertian surfaces. Extensive experiments on both synthetic and real-world datasets demonstrate that our framework substantially improves the physical consistency and visual coherence of object removal, outperforming existing methods by 13% in perceptual similarity (LPIPS) and 2dB in peak signal-to-noise ratio (PSNR). Code is publicly available at https://applezyh.github.io/GOR-IS-project-page/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04349v2">VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      3D editing has emerged as a critical research area to provide users with flexible control over 3D assets. While current editing approaches predominantly focus on 3D Gaussian Splatting or multi-view images, the direct editing of 3D meshes remains underexplored. Prior attempts, such as VoxHammer, rely on voxel-based representations that suffer from limited resolution and necessitate labor-intensive 3D mask. To address these limitations, we propose \textbf{VecSet-Edit}, the first pipeline that leverages the high-fidelity VecSet Large Reconstruction Model (LRM) as a backbone for mesh editing. Our approach is grounded on a analysis of the spatial properties in VecSet tokens, revealing that token subsets govern distinct geometric regions. Based on this insight, we introduce Mask-guided Token Seeding and Attention-aligned Token Gating strategies to precisely localize target regions using only 2D image conditions. Also, considering the difference between VecSet diffusion process versus voxel we design a Drift-aware Token Pruning to reject geometric outliers during the denoising process. Finally, our Detail-preserving Texture Baking module ensures that we not only preserve the geometric details of original mesh but also the textural information. More details can be found in our project page: https://github.com/BlueDyee/VecSet-Edit/tree/main
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26262v2">Semantic Foam: Unifying Spatial and Semantic Scene Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-05-01
      | 💬 15 pages, 10 figures, Accepted to CVPR 2026 (Highlight) , Project page: http://semanticfoam.github.io/
    </div>
    <details class="paper-abstract">
      Modern scene reconstruction methods, such as 3D Gaussian Splatting, deliver photo-realistic novel view synthesis at real-time speeds, yet their adoption in interactive graphics applications has been limited. A major bottleneck is the difficulty of interacting with these representations compared to traditional, human-authored 3D assets. While previous research has attempted to impose semantic decomposition on these models, significant challenges remain regarding segmentation quality and consistency. To address this, we introduce Semantic Foam, extending the recently proposed Radiant Foam representations to semantic decomposition tasks. Our approach integrates the natural spatial volumetric decomposition of Radiant Foam's Voronoi mesh with an explicit semantic feature field parameterized at the cell level. This explicit structure enables direct spatial regularization, which prevents artifacts caused by occlusion or inconsistent supervision across views - common pitfalls for other point-based representations. Experimental results show that our method achieves superior object-level segmentation performance compared to state-of-the-art methods like Gaussian Grouping and SAGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28111v2">GSDrive: Reinforcing Driving Policies by Multi-mode Trajectory Probing with 3D Gaussian Splatting Environment</a></div>
    <div class="paper-meta">
      📅 2026-05-01
      | 💬 initial version
    </div>
    <details class="paper-abstract">
      End-to-end (E2E) autonomous driving presents a promising approach for translating perceptual inputs directly into driving actions. However, prohibitive annotation costs and temporal data quality degradation hinder long-term real-world deployment. While combining imitation learning (IL) and reinforcement learning (RL) is a common strategy for policy improvement, conventional RL training relies on delayed, event-based rewards-policies learn only from catastrophic outcomes such as collisions, leading to premature convergence to suboptimal behaviors. To address these limitations, we introduce GSDrive, a framework that exploits 3D Gaussian Splatting (3DGS) for differentiable, physics-based reward shaping in E2E driving policy improvement. Our method incorporates a flow matching-based trajectory predictor within the 3DGS simulator, enabling multi-mode trajectory probing where candidate trajectories are rolled out to assess prospective rewards. This establishes a bidirectional knowledge exchange between IL and RL by grounding reward functions in physically simulated interaction signals, offering immediate dense feedback instead of sparse catastrophic events. Evaluated on the reconstructed nuScenes dataset, our method surpasses existing simulation-based RL driving approaches in closed-loop experiments. Code is available at https://github.com/ZionGo6/GSDrive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00408v1">Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has demonstrated impressive real-time rendering performance, its efficacy remains constrained by a reliance on heuristic density control. Despite numerous refinements to these handcrafted rules, such methods inherently lack the flexibility to adapt to diverse scenes with complex geometries. In this paper, we propose a paradigm shift for density control from rigid heuristics to fully learnable policies. Specifically, we introduce \textbf{LeGS}, a framework that reformulates density control as a parameterized policy network optimized via Reinforcement Learning (RL). Central to our approach is the tailored effective reward function grounded in sensitivity analysis, which precisely quantifies the marginal contribution of individual Gaussians to reconstruction quality. To maintain computational tractability, we derive a closed-form solution that reduces the complexity of reward calculation from $O(N^2)$ to $O(N)$. Extensive experiments on the Mip-NeRF 360, Tanks \& Temples, and Deep Blending datasets demonstrate that \textbf{LeGS} significantly outperforms state-of-the-art methods, striking a superior balance between reconstruction quality and efficiency. The code will be released at https://github.com/AaronNZH/LeGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04929v5">CryoSplat: Gaussian Splatting for Cryo-EM Homogeneous Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-05-01
      | 💬 Published at ICLR 2026 (Camera-ready). Code available at https://github.com/Chen-Suyi/cryosplat
    </div>
    <details class="paper-abstract">
      As a critical modality for structural biology, cryogenic electron microscopy (cryo-EM) facilitates the determination of macromolecular structures at near-atomic resolution. The core computational task in single-particle cryo-EM is to reconstruct the 3D electrostatic potential of a molecule from noisy 2D projections acquired at unknown orientations. Gaussian mixture models (GMMs) provide a continuous, compact, and physically interpretable representation for molecular density and have recently gained interest in cryo-EM reconstruction. However, existing methods rely on external consensus maps or atomic models for initialization, limiting their use in self-contained pipelines. In parallel, differentiable rendering techniques such as Gaussian splatting have demonstrated remarkable scalability and efficiency for volumetric representations, suggesting a natural fit for GMM-based cryo-EM reconstruction. However, off-the-shelf Gaussian splatting methods are designed for photorealistic view synthesis and remain incompatible with cryo-EM due to mismatches in the image formation physics, reconstruction objectives, and coordinate systems. Addressing these issues, we propose cryoSplat, a GMM-based method that integrates Gaussian splatting with the physics of cryo-EM image formation. In particular, we develop an orthogonal projection-aware Gaussian splatting, with adaptations such as a view-dependent normalization term and FFT-aligned coordinate system tailored for cryo-EM imaging. These innovations enable stable and efficient homogeneous reconstruction directly from raw cryo-EM particle images using random initialization. Experimental results on real datasets validate the effectiveness and robustness of cryoSplat over representative baselines. The code will be released at https://github.com/Chen-Suyi/cryosplat.
    </details>
</div>
