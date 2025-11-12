# embodied ai - 2022_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2202.08227v3">Ditto: Building Digital Twins of Articulated Objects from Interaction</a></div>
    <div class="paper-meta">
      📅 2022-04-29
      | 💬 CVPR 2022 Oral. Code and additional results are available at https://ut-austin-rpl.github.io/Ditto
    </div>
    <details class="paper-abstract">
      Digitizing physical objects into the virtual world has the potential to unlock new research and applications in embodied AI and mixed reality. This work focuses on recreating interactive digital twins of real-world articulated objects, which can be directly imported into virtual environments. We introduce Ditto to learn articulation model estimation and 3D geometry reconstruction of an articulated object through interactive perception. Given a pair of visual observations of an articulated object before and after interaction, Ditto reconstructs part-level geometry and estimates the articulation model of the object. We employ implicit neural representations for joint geometry and articulation modeling. Our experiments show that Ditto effectively builds digital twins of articulated objects in a category-agnostic way. We also apply Ditto to real-world objects and deploy the recreated digital twins in physical simulation. Code and additional results are available at https://ut-austin-rpl.github.io/Ditto
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2109.07703v3">ROS-X-Habitat: Bridging the ROS Ecosystem with Embodied AI</a></div>
    <div class="paper-meta">
      📅 2022-04-29
      | 💬 Camera-ready version submitted to Canadian Conference on Computer and Robot Vision (CRV) 2022
    </div>
    <details class="paper-abstract">
      We introduce ROS-X-Habitat, a software interface that bridges the AI Habitat platform for embodied learning-based agents with other robotics resources via ROS. This interface not only offers standardized communication protocols between embodied agents and simulators, but also enables physically and photorealistic simulation that benefits the training and/or testing of vision-based embodied agents. With this interface, roboticists can evaluate their own Habitat RL agents in another ROS-based simulator or use Habitat Sim v2 as the test bed for their own robotic algorithms. Through in silico experiments, we demonstrate that ROS-X-Habitat has minimal impact on the navigation performance and simulation speed of a Habitat RGBD agent; that a standard set of ROS mapping, planning and navigation tools can run in Habitat Sim v2; and that a Habitat agent can run in the standard ROS simulator Gazebo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2201.03413v2">Systems Challenges for Trustworthy Embodied Systems</a></div>
    <div class="paper-meta">
      📅 2022-04-27
      | 💬 57 pages, 7 figures, 3 tables. Public project deliverable, fortiss whitepaper
    </div>
    <details class="paper-abstract">
      A new generation of increasingly autonomous and self-learning embodied systems is about to be developed. When deploying embodied systems into a real-life context we face various engineering challenges, as it is crucial to coordinate the behavior of embodied systems in a beneficial manner, ensure their compatibility with our human-centered social values, and design verifiably safe and reliable human-machine interaction. We are arguing that traditional systems engineering is coming to a climacteric from embedded to embodied systems, and with assuring the trustworthiness of dynamic federations of situationally aware, intent-driven, explorative, ever-evolving, largely non-predictable, and increasingly autonomous embodied systems in uncertain, complex, and unpredictable real-world contexts. We are therefore identifying a number of urgent systems challenges for trustworthy embodied systems, including robust and human-centric AI, cognitive architectures, uncertainty quantification, trustworthy self-integration, and continual analysis and assurance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2204.08502v1">Spot the Difference: A Novel Task for Embodied Agents in Changing Environments</a></div>
    <div class="paper-meta">
      📅 2022-04-18
      | 💬 Accepted by 26TH International Conference on Pattern Recognition (ICPR 2022)
    </div>
    <details class="paper-abstract">
      Embodied AI is a recent research area that aims at creating intelligent agents that can move and operate inside an environment. Existing approaches in this field demand the agents to act in completely new and unexplored scenes. However, this setting is far from realistic use cases that instead require executing multiple tasks in the same environment. Even if the environment changes over time, the agent could still count on its global knowledge about the scene while trying to adapt its internal representation to the current state of the environment. To make a step towards this setting, we propose Spot the Difference: a novel task for Embodied AI where the agent has access to an outdated map of the environment and needs to recover the correct layout in a fixed time budget. To this end, we collect a new dataset of occupancy maps starting from existing datasets of 3D spaces and generating a number of possible layouts for a single environment. This dataset can be employed in the popular Habitat simulator and is fully compliant with existing methods that employ reconstructed occupancy maps during navigation. Furthermore, we propose an exploration policy that can take advantage of previous knowledge of the environment and identify changes in the scene faster and more effectively than existing agents. Experimental results show that the proposed architecture outperforms existing state-of-the-art models for exploration on this new setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2111.09888v2">Simple but Effective: CLIP Embeddings for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2022-04-18
      | 💬 Published in CVPR 2022
    </div>
    <details class="paper-abstract">
      Contrastive language image pretraining (CLIP) encoders have been shown to be beneficial for a range of visual tasks from classification and detection to captioning and image manipulation. We investigate the effectiveness of CLIP visual backbones for Embodied AI tasks. We build incredibly simple baselines, named EmbCLIP, with no task specific architectures, inductive biases (such as the use of semantic maps), auxiliary tasks during training, or depth maps -- yet we find that our improved baselines perform very well across a range of tasks and simulators. EmbCLIP tops the RoboTHOR ObjectNav leaderboard by a huge margin of 20 pts (Success Rate). It tops the iTHOR 1-Phase Rearrangement leaderboard, beating the next best submission, which employs Active Neural Mapping, and more than doubling the % Fixed Strict metric (0.08 to 0.17). It also beats the winners of the 2021 Habitat ObjectNav Challenge, which employ auxiliary tasks, depth maps, and human demonstrations, and those of the 2019 Habitat PointNav Challenge. We evaluate the ability of CLIP's visual representations at capturing semantic information about input observations -- primitives that are useful for navigation-heavy embodied tasks -- and find that CLIP's representations encode these primitives more effectively than ImageNet-pretrained backbones. Finally, we extend one of our baselines, producing an agent capable of zero-shot object navigation that can navigate to objects that were not used as targets during training. Our code and models are available at https://github.com/allenai/embodied-clip
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2111.09888v2">Simple but Effective: CLIP Embeddings for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2022-04-15
      | 💬 Published in CVPR 2022
    </div>
    <details class="paper-abstract">
      Contrastive language image pretraining (CLIP) encoders have been shown to be beneficial for a range of visual tasks from classification and detection to captioning and image manipulation. We investigate the effectiveness of CLIP visual backbones for Embodied AI tasks. We build incredibly simple baselines, named EmbCLIP, with no task specific architectures, inductive biases (such as the use of semantic maps), auxiliary tasks during training, or depth maps -- yet we find that our improved baselines perform very well across a range of tasks and simulators. EmbCLIP tops the RoboTHOR ObjectNav leaderboard by a huge margin of 20 pts (Success Rate). It tops the iTHOR 1-Phase Rearrangement leaderboard, beating the next best submission, which employs Active Neural Mapping, and more than doubling the % Fixed Strict metric (0.08 to 0.17). It also beats the winners of the 2021 Habitat ObjectNav Challenge, which employ auxiliary tasks, depth maps, and human demonstrations, and those of the 2019 Habitat PointNav Challenge. We evaluate the ability of CLIP's visual representations at capturing semantic information about input observations -- primitives that are useful for navigation-heavy embodied tasks -- and find that CLIP's representations encode these primitives more effectively than ImageNet-pretrained backbones. Finally, we extend one of our baselines, producing an agent capable of zero-shot object navigation that can navigate to objects that were not used as targets during training. Our code and models are available at https://github.com/allenai/embodied-clip
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2112.15459v3">Social Neuro AI: Social Interaction as the "dark matter" of AI</a></div>
    <div class="paper-meta">
      📅 2022-04-12
      | 💬 14 pages, 2 figures
    </div>
    <details class="paper-abstract">
      This article introduces a three-axis framework indicating how AI can be informed by biological examples of social learning mechanisms. We argue that the complex human cognitive architecture owes a large portion of its expressive power to its ability to engage in social and cultural learning. However, the field of AI has mostly embraced a solipsistic perspective on intelligence. We thus argue that social interactions not only are largely unexplored in this field but also are an essential element of advanced cognitive ability, and therefore constitute metaphorically the dark matter of AI. In the first section, we discuss how social learning plays a key role in the development of intelligence. We do so by discussing social and cultural learning theories and empirical findings from social neuroscience. Then, we discuss three lines of research that fall under the umbrella of Social NeuroAI and can contribute to developing socially intelligent embodied agents in complex environments. First, neuroscientific theories of cognitive architecture, such as the global workspace theory and the attention schema theory, can enhance biological plausibility and help us understand how we could bridge individual and social theories of intelligence. Second, intelligence occurs in time as opposed to over time, and this is naturally incorporated by dynamical systems. Third, embodiment has been demonstrated to provide more sophisticated array of communicative signals. To conclude, we discuss the example of active inference, which offers powerful insights for developing agents that possess biological realism, can self-organize in time, and are socially embodied.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2203.17251v1">Continuous Scene Representations for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2022-04-01
      | 💬 CVPR 2022
    </div>
    <details class="paper-abstract">
      We propose Continuous Scene Representations (CSR), a scene representation constructed by an embodied agent navigating within a space, where objects and their relationships are modeled by continuous valued embeddings. Our method captures feature relationships between objects, composes them into a graph structure on-the-fly, and situates an embodied agent within the representation. Our key insight is to embed pair-wise relationships between objects in a latent space. This allows for a richer representation compared to discrete relations (e.g., [support], [next-to]) commonly used for building scene representations. CSR can track objects as the agent moves in a scene, update the representation accordingly, and detect changes in room configurations. Using CSR, we outperform state-of-the-art approaches for the challenging downstream task of visual room rearrangement, without any task specific training. Moreover, we show the learned embeddings capture salient spatial details of the scene and show applicability to real world data. A summery video and code is available at https://prior.allenai.org/projects/csr.
    </details>
</div>
