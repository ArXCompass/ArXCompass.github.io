# embodied ai - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03629v1">SeGuE: Semantic Guided Exploration for Mobile Robots</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 6 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      The rise of embodied AI applications has enabled robots to perform complex tasks which require a sophisticated understanding of their environment. To enable successful robot operation in such settings, maps must be constructed so that they include semantic information, in addition to geometric information. In this paper, we address the novel problem of semantic exploration, whereby a mobile robot must autonomously explore an environment to fully map both its structure and the semantic appearance of features. We develop a method based on next-best-view exploration, where potential poses are scored based on the semantic features visible from that pose. We explore two alternative methods for sampling potential views and demonstrate the effectiveness of our framework in both simulation and physical experiments. Automatic creation of high-quality semantic maps can enable robots to better understand and interact with their environments and enable future embodied AI applications to be more easily deployed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17735v5">3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Constructing compact and informative 3D scene representations is essential for effective embodied exploration and reasoning, especially in complex environments over extended periods. Existing representations, such as object-centric 3D scene graphs, oversimplify spatial relationships by modeling scenes as isolated objects with restrictive textual relationships, making it difficult to address queries requiring nuanced spatial understanding. Moreover, these representations lack natural mechanisms for active exploration and memory management, hindering their application to lifelong autonomy. In this work, we propose 3D-Mem, a novel 3D scene memory framework for embodied agents. 3D-Mem employs informative multi-view images, termed Memory Snapshots, to represent the scene and capture rich visual information of explored regions. It further integrates frontier-based exploration by introducing Frontier Snapshots-glimpses of unexplored areas-enabling agents to make informed decisions by considering both known and potential new information. To support lifelong memory in active exploration settings, we present an incremental construction pipeline for 3D-Mem, as well as a memory retrieval technique for memory management. Experimental results on three benchmarks demonstrate that 3D-Mem significantly enhances agents' exploration and reasoning capabilities in 3D environments, highlighting its potential for advancing applications in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03153v1">MORAL: A Multimodal Reinforcement Learning Framework for Decision Making in Autonomous Laboratories</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 9 pages, 14 figures and 3 tables
    </div>
    <details class="paper-abstract">
      We propose MORAL (a multimodal reinforcement learning framework for decision making in autonomous laboratories) that enhances sequential decision-making in autonomous robotic laboratories through the integration of visual and textual inputs. Using the BridgeData V2 dataset, we generate fine-tuned image captions with a pretrained BLIP-2 vision-language model and combine them with visual features through an early fusion strategy. The fused representations are processed using Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents. Experimental results demonstrate that multimodal agents achieve a 20% improvement in task completion rates and significantly outperform visual-only and textual-only baselines after sufficient training. Compared to transformer-based and recurrent multimodal RL models, our approach achieves superior performance in cumulative reward and caption quality metrics (BLEU, METEOR, ROUGE-L). These results highlight the impact of semantically aligned language cues in enhancing agent learning efficiency and generalization. The proposed framework contributes to the advancement of multimodal reinforcement learning and embodied AI systems in dynamic, real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19901v2">TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Synthesizing diverse and physically plausible Human-Scene Interactions (HSI) is pivotal for both computer animation and embodied AI. Despite encouraging progress, current methods mainly focus on developing separate controllers, each specialized for a specific interaction task. This significantly hinders the ability to tackle a wide variety of challenging HSI tasks that require the integration of multiple skills, e.g., sitting down while carrying an object. To address this issue, we present TokenHSI, a single, unified transformer-based policy capable of multi-skill unification and flexible adaptation. The key insight is to model the humanoid proprioception as a separate shared token and combine it with distinct task tokens via a masking mechanism. Such a unified policy enables effective knowledge sharing across skills, thereby facilitating the multi-task training. Moreover, our policy architecture supports variable length inputs, enabling flexible adaptation of learned skills to new scenarios. By training additional task tokenizers, we can not only modify the geometries of interaction targets but also coordinate multiple skills to address complex tasks. The experiments demonstrate that our approach can significantly improve versatility, adaptability, and extensibility in various HSI tasks. Website: https://liangpan99.github.io/TokenHSI/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00954v1">IDMR: Towards Instance-Driven Precise Visual Correspondence in Multimodal Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Multimodal retrieval systems are becoming increasingly vital for cutting-edge AI technologies, such as embodied AI and AI-driven digital content industries. However, current multimodal retrieval tasks lack sufficient complexity and demonstrate limited practical application value. It spires us to design Instance-Driven Multimodal Image Retrieval (IDMR), a novel task that requires models to retrieve images containing the same instance as a query image while matching a text-described scenario. Unlike existing retrieval tasks focused on global image similarity or category-level matching, IDMR demands fine-grained instance-level consistency across diverse contexts. To benchmark this capability, we develop IDMR-bench using real-world object tracking and first-person video data. Addressing the scarcity of training data, we propose a cross-domain synthesis method that creates 557K training samples by cropping objects from standard detection datasets. Our Multimodal Large Language Model (MLLM) based retrieval model, trained on 1.2M samples, outperforms state-of-the-art approaches on both traditional benchmarks and our zero-shot IDMR-bench. Experimental results demonstrate previous models' limitations in instance-aware retrieval and highlight the potential of MLLM for advanced retrieval applications. The whole training dataset, codes and models, with wide ranges of sizes, are available at https://github.com/BwLiu01/IDMR.
    </details>
</div>
