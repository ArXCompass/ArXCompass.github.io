# embodied ai - 2022_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2211.16309v1">A Contextual Bandit Approach for Learning to Plan in Environments with Probabilistic Goal Configurations</a></div>
    <div class="paper-meta">
      📅 2022-11-29
      | 💬 Shorter version accepted at NeurIPS 2022 Workshop on Robot Learning: Trustworthy Robotics
    </div>
    <details class="paper-abstract">
      Object-goal navigation (Object-nav) entails searching, recognizing and navigating to a target object. Object-nav has been extensively studied by the Embodied-AI community, but most solutions are often restricted to considering static objects (e.g., television, fridge, etc.). We propose a modular framework for object-nav that is able to efficiently search indoor environments for not just static objects but also movable objects (e.g. fruits, glasses, phones, etc.) that frequently change their positions due to human intervention. Our contextual-bandit agent efficiently explores the environment by showing optimism in the face of uncertainty and learns a model of the likelihood of spotting different objects from each navigable location. The likelihoods are used as rewards in a weighted minimum latency solver to deduce a trajectory for the robot. We evaluate our algorithms in two simulated environments and a real-world setting, to demonstrate high sample efficiency and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2211.09960v1">Ask4Help: Learning to Leverage an Expert for Embodied Tasks</a></div>
    <div class="paper-meta">
      📅 2022-11-18
      | 💬 Accepted at NeurIPS, 2022
    </div>
    <details class="paper-abstract">
      Embodied AI agents continue to become more capable every year with the advent of new models, environments, and benchmarks, but are still far away from being performant and reliable enough to be deployed in real, user-facing, applications. In this paper, we ask: can we bridge this gap by enabling agents to ask for assistance from an expert such as a human being? To this end, we propose the Ask4Help policy that augments agents with the ability to request, and then use expert assistance. Ask4Help policies can be efficiently trained without modifying the original agent's parameters and learn a desirable trade-off between task performance and the amount of requested help, thereby reducing the cost of querying the expert. We evaluate Ask4Help on two different tasks -- object goal navigation and room rearrangement and see substantial improvements in performance using minimal help. On object navigation, an agent that achieves a $52\%$ success rate is raised to $86\%$ with $13\%$ help and for rearrangement, the state-of-the-art model with a $7\%$ success rate is dramatically improved to $90.4\%$ using $39\%$ help. Human trials with Ask4Help demonstrate the efficacy of our approach in practical scenarios. We release the code for Ask4Help here: https://github.com/allenai/ask4help.
    </details>
</div>
