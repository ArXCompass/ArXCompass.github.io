# embodied ai - 2022_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2203.17251v1">Continuous Scene Representations for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2022-03-31
      | 💬 CVPR 2022
    </div>
    <details class="paper-abstract">
      We propose Continuous Scene Representations (CSR), a scene representation constructed by an embodied agent navigating within a space, where objects and their relationships are modeled by continuous valued embeddings. Our method captures feature relationships between objects, composes them into a graph structure on-the-fly, and situates an embodied agent within the representation. Our key insight is to embed pair-wise relationships between objects in a latent space. This allows for a richer representation compared to discrete relations (e.g., [support], [next-to]) commonly used for building scene representations. CSR can track objects as the agent moves in a scene, update the representation accordingly, and detect changes in room configurations. Using CSR, we outperform state-of-the-art approaches for the challenging downstream task of visual room rearrangement, without any task specific training. Moreover, we show the learned embeddings capture salient spatial details of the scene and show applicability to real world data. A summery video and code is available at https://prior.allenai.org/projects/csr.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2203.08141v1">Object Manipulation via Visual Target Localization</a></div>
    <div class="paper-meta">
      📅 2022-03-15
    </div>
    <details class="paper-abstract">
      Object manipulation is a critical skill required for Embodied AI agents interacting with the world around them. Training agents to manipulate objects, poses many challenges. These include occlusion of the target object by the agent's arm, noisy object detection and localization, and the target frequently going out of view as the agent moves around in the scene. We propose Manipulation via Visual Object Location Estimation (m-VOLE), an approach that explores the environment in search for target objects, computes their 3D coordinates once they are located, and then continues to estimate their 3D locations even when the objects are not visible, thus robustly aiding the task of manipulating these objects throughout the episode. Our evaluations show a massive 3x improvement in success rate over a model that has access to the same sensory suite but is trained without the object location estimator, and our analysis shows that our agent is robust to noise in depth perception and agent localization. Importantly, our proposed approach relaxes several assumptions about idealized localization and perception that are commonly employed by recent works in embodied AI -- an important step towards training agents for object manipulation in the real world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2203.00600v1">Dual Embodied-Symbolic Concept Representations for Deep Learning</a></div>
    <div class="paper-meta">
      📅 2022-03-02
    </div>
    <details class="paper-abstract">
      Motivated by recent findings from cognitive neural science, we advocate the use of a dual-level model for concept representations: the embodied level consists of concept-oriented feature representations, and the symbolic level consists of concept graphs. Embodied concept representations are modality specific and exist in the form of feature vectors in a feature space. Symbolic concept representations, on the other hand, are amodal and language specific, and exist in the form of word / knowledge-graph embeddings in a concept / knowledge space. The human conceptual system comprises both embodied representations and symbolic representations, which typically interact to drive conceptual processing. As such, we further advocate the use of dual embodied-symbolic concept representations for deep learning. To demonstrate their usage and value, we discuss two important use cases: embodied-symbolic knowledge distillation for few-shot class incremental learning, and embodied-symbolic fused representation for image-text matching. Dual embodied-symbolic concept representations are the foundation for deep learning and symbolic AI integration. We discuss two important examples of such integration: scene graph generation with knowledge graph bridging, and multimodal knowledge graphs.
    </details>
</div>
