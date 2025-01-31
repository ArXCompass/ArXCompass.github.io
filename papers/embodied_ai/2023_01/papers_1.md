# embodied ai - 2023_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2301.05223v1">NOPA: Neurally-guided Online Probabilistic Assistance for Building Socially Intelligent Home Assistants</a></div>
    <div class="paper-meta">
      📅 2023-01-12
      | 💬 Project website: https://www.tshu.io/online_watch_and_help. Code: https://github.com/xavierpuigf/online_watch_and_help
    </div>
    <details class="paper-abstract">
      In this work, we study how to build socially intelligent robots to assist people in their homes. In particular, we focus on assistance with online goal inference, where robots must simultaneously infer humans' goals and how to help them achieve those goals. Prior assistance methods either lack the adaptivity to adjust helping strategies (i.e., when and how to help) in response to uncertainty about goals or the scalability to conduct fast inference in a large goal space. Our NOPA (Neurally-guided Online Probabilistic Assistance) method addresses both of these challenges. NOPA consists of (1) an online goal inference module combining neural goal proposals with inverse planning and particle filtering for robust inference under uncertainty, and (2) a helping planner that discovers valuable subgoals to help with and is aware of the uncertainty in goal inference. We compare NOPA against multiple baselines in a new embodied AI assistance challenge: Online Watch-And-Help, in which a helper agent needs to simultaneously watch a main agent's action, infer its goal, and help perform a common household task faster in realistic virtual home environments. Experiments show that our helper agent robustly updates its goal inference and adapts its helping plans to the changing level of uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2301.02382v2">ReVoLT: Relational Reasoning and Voronoi Local Graph Planning for Target-driven Navigation</a></div>
    <div class="paper-meta">
      📅 2023-01-10
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Embodied AI is an inevitable trend that emphasizes the interaction between intelligent entities and the real world, with broad applications in Robotics, especially target-driven navigation. This task requires the robot to find an object of a certain category efficiently in an unknown domestic environment. Recent works focus on exploiting layout relationships by graph neural networks (GNNs). However, most of them obtain robot actions directly from observations in an end-to-end manner via an incomplete relation graph, which is not interpretable and reliable. We decouple this task and propose ReVoLT, a hierarchical framework: (a) an object detection visual front-end, (b) a high-level reasoner (infers semantic sub-goals), (c) an intermediate-level planner (computes geometrical positions), and (d) a low-level controller (executes actions). ReVoLT operates with a multi-layer semantic-spatial topological graph. The reasoner uses multiform structured relations as priors, which are obtained from combinatorial relation extraction networks composed of unsupervised GraphSAGE, GCN, and GraphRNN-based Region Rollout. The reasoner performs with Upper Confidence Bound for Tree (UCT) to infer semantic sub-goals, accounting for trade-offs between exploitation (depth-first searching) and exploration (regretting). The lightweight intermediate-level planner generates instantaneous spatial sub-goal locations via an online constructed Voronoi local graph. The simulation experiments demonstrate that our framework achieves better performance in the target-driven navigation tasks and generalizes well, which has an 80% improvement compared to the existing state-of-the-art method. The code and result video will be released at https://ventusff.github.io/ReVoLT-website/.
    </details>
</div>
