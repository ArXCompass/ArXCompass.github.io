# gaussian splatting - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2501.00342v1">SG-Splatting: Accelerating 3D Gaussian Splatting with Spherical Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting is emerging as a state-of-the-art technique in novel view synthesis, recognized for its impressive balance between visual quality, speed, and rendering efficiency. However, reliance on third-degree spherical harmonics for color representation introduces significant storage demands and computational overhead, resulting in a large memory footprint and slower rendering speed. We introduce SG-Splatting with Spherical Gaussians based color representation, a novel approach to enhance rendering speed and quality in novel view synthesis. Our method first represents view-dependent color using Spherical Gaussians, instead of three degree spherical harmonics, which largely reduces the number of parameters used for color representation, and significantly accelerates the rendering process. We then develop an efficient strategy for organizing multiple Spherical Gaussians, optimizing their arrangement to achieve a balanced and accurate scene representation. To further improve rendering quality, we propose a mixed representation that combines Spherical Gaussians with low-degree spherical harmonics, capturing both high- and low-frequency color information effectively. SG-Splatting also has plug-and-play capability, allowing it to be easily integrated into existing systems. This approach improves computational efficiency and overall visual fidelity, making it a practical solution for real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01101v1">Deformable Gaussian Splatting for Efficient and High-Fidelity Reconstruction of Surgical Scenes</a></div>
    <div class="paper-meta">
      📅 2025-01-02
      | 💬 7 pages, 4 figures, submitted to ICRA 2025
    </div>
    <details class="paper-abstract">
      Efficient and high-fidelity reconstruction of deformable surgical scenes is a critical yet challenging task. Building on recent advancements in 3D Gaussian splatting, current methods have seen significant improvements in both reconstruction quality and rendering speed. However, two major limitations remain: (1) difficulty in handling irreversible dynamic changes, such as tissue shearing, which are common in surgical scenes; and (2) the lack of hierarchical modeling for surgical scene deformation, which reduces rendering speed. To address these challenges, we introduce EH-SurGS, an efficient and high-fidelity reconstruction algorithm for deformable surgical scenes. We propose a deformation modeling approach that incorporates the life cycle of 3D Gaussians, effectively capturing both regular and irreversible deformations, thus enhancing reconstruction quality. Additionally, we present an adaptive motion hierarchy strategy that distinguishes between static and deformable regions within the surgical scene. This strategy reduces the number of 3D Gaussians passing through the deformation field, thereby improving rendering speed. Extensive experiments demonstrate that our method surpasses existing state-of-the-art approaches in both reconstruction quality and rendering speed. Ablation studies further validate the effectiveness and necessity of our proposed components. We will open-source our code upon acceptance of the paper.
    </details>
</div>
