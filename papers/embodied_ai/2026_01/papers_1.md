# embodied ai - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03136v1">Limited Linguistic Diversity in Embodied AI Datasets</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Language plays a critical role in Vision-Language-Action (VLA) models, yet the linguistic characteristics of the datasets used to train and evaluate these systems remain poorly documented. In this work, we present a systematic dataset audit of several widely used VLA corpora, aiming to characterize what kinds of instructions these datasets actually contain and how much linguistic variety they provide. We quantify instruction language along complementary dimensions-including lexical variety, duplication and overlap, semantic similarity, and syntactic complexity. Our analysis shows that many datasets rely on highly repetitive, template-like commands with limited structural variation, yielding a narrow distribution of instruction forms. We position these findings as descriptive documentation of the language signal available in current VLA training and evaluation data, intended to support more detailed dataset reporting, more principled dataset selection, and targeted curation or augmentation strategies that broaden language coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16853v2">ISCS: Parameter-Guided Feature Pruning for Resource-Constrained Embodied Perception</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Significant revision: The focus has been pivoted from learned image compression to embodied perception tasks. Experimental results and downstream applications have been updated to demonstrate the method's efficiency in split computing
    </div>
    <details class="paper-abstract">
      Prior studies in embodied AI consistently show that robust perception is critical for human-robot interaction, yet deploying high-fidelity visual models on resource-constrained agents remains challenging due to limited on-device computation power and transmission latency. Exploiting the redundancy in latent representations could improve system efficiency, yet existing approaches often rely on costly dataset-specific ablation tests or heavy entropy models unsuitable for real-time edge-robot collaboration. We propose a generalizable, dataset-agnostic method to identify and selectively transmit structure-critical channels in pretrained encoders. Instead of brute-force empirical evaluations, our approach leverages intrinsic parameter statistics-weight variances and biases-to estimate channel importance. This analysis reveals a consistent organizational structure, termed the Invariant Salient Channel Space (ISCS), where Salient-Core channels capture dominant structures while Salient-Auxiliary channels encode fine visual details. Building on ISCS, we introduce a deterministic static pruning strategy that enables lightweight split-computing. Experiments across different datasets demonstrate that our method achieves a deterministic, ultra-low latency pipeline by bypassing heavy entropy modeling. Our method reduces end-to-end latency, providing a critical speed-accuracy trade-off for resource-constrained human-aware embodied systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01705v1">Explicit World Models for Reliable Human-Robot Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      This paper addresses the topic of robustness under sensing noise, ambiguous instructions, and human-robot interaction. We take a radically different tack to the issue of reliable embodied AI: instead of focusing on formal verification methods aimed at achieving model predictability and robustness, we emphasise the dynamic, ambiguous and subjective nature of human-robot interactions that requires embodied AI systems to perceive, interpret, and respond to human intentions in a manner that is consistent, comprehensible and aligned with human expectations. We argue that when embodied agents operate in human environments that are inherently social, multimodal, and fluid, reliability is contextually determined and only has meaning in relation to the goals and expectations of humans involved in the interaction. This calls for a fundamentally different approach to achieving reliable embodied AI that is centred on building and updating an accessible "explicit world model" representing the common ground between human and AI, that is used to align robot behaviours with human expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10071v3">Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Post-challenge bug fix
    </div>
    <details class="paper-abstract">
      The 2025 BEHAVIOR Challenge is designed to rigorously track progress toward solving long-horizon tasks by physical agents in simulated environments. BEHAVIOR-1K focuses on everyday household tasks that people most want robots to assist with and these tasks introduce long-horizon mobile manipulation challenges in realistic settings, bridging the gap between current research and real-world, human-centric applications. This report presents our solution to the 2025 BEHAVIOR Challenge in a very close 2nd place and substantially outperforms the rest of the submissions. Building on $π_{0.5}$, we focus on systematically building our solution by studying the effects of training techniques and data. Through careful ablation studies, we reveal the scaling benefits in both the pre-training and post-training phases, leading to a validation Q-score of 0.345, significantly surpassing previous state-of-the-art performance. We summarize our practical lessons and design recommendations that we hope will provide actionable insights for the broader embodied AI community when adapting powerful foundation models to complex embodied scenarios. Project page: https://github.com/mli0603/openpi-comet
    </details>
</div>
