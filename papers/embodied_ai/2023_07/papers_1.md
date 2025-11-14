# embodied ai - 2023_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2307.13850v1">MAEA: Multimodal Attribution for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-07-27
    </div>
    <details class="paper-abstract">
      Understanding multimodal perception for embodied AI is an open question because such inputs may contain highly complementary as well as redundant information for the task. A relevant direction for multimodal policies is understanding the global trends of each modality at the fusion layer. To this end, we disentangle the attributions for visual, language, and previous action inputs across different policies trained on the ALFRED dataset. Attribution analysis can be utilized to rank and group the failure scenarios, investigate modeling and dataset biases, and critically analyze multimodal EAI policies for robustness and user trust before deployment. We present MAEA, a framework to compute global attributions per modality of any differentiable policy. In addition, we show how attributions enable lower-level behavior analysis in EAI policies for language and visual attributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.13850v1">MAEA: Multimodal Attribution for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-07-25
    </div>
    <details class="paper-abstract">
      Understanding multimodal perception for embodied AI is an open question because such inputs may contain highly complementary as well as redundant information for the task. A relevant direction for multimodal policies is understanding the global trends of each modality at the fusion layer. To this end, we disentangle the attributions for visual, language, and previous action inputs across different policies trained on the ALFRED dataset. Attribution analysis can be utilized to rank and group the failure scenarios, investigate modeling and dataset biases, and critically analyze multimodal EAI policies for robustness and user trust before deployment. We present MAEA, a framework to compute global attributions per modality of any differentiable policy. In addition, we show how attributions enable lower-level behavior analysis in EAI policies for language and visual attributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2307.13850v1">MAEA: Multimodal Attribution for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-07-25
    </div>
    <details class="paper-abstract">
      Understanding multimodal perception for embodied AI is an open question because such inputs may contain highly complementary as well as redundant information for the task. A relevant direction for multimodal policies is understanding the global trends of each modality at the fusion layer. To this end, we disentangle the attributions for visual, language, and previous action inputs across different policies trained on the ALFRED dataset. Attribution analysis can be utilized to rank and group the failure scenarios, investigate modeling and dataset biases, and critically analyze multimodal EAI policies for robustness and user trust before deployment. We present MAEA, a framework to compute global attributions per modality of any differentiable policy. In addition, we show how attributions enable lower-level behavior analysis in EAI policies for language and visual attributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.11343v1">A Two-stage Fine-tuning Strategy for Generalizable Manipulation Skill of Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-07-21
      | 💬 5 pages, 2 figures, 5 tables, accept by Robotics: Science and Systems 2023 - Workshop Interdisciplinary Exploration of Generalizable Manipulation Policy Learning:Paradigms and Debates
    </div>
    <details class="paper-abstract">
      The advent of Chat-GPT has led to a surge of interest in Embodied AI. However, many existing Embodied AI models heavily rely on massive interactions with training environments, which may not be practical in real-world situations. To this end, the Maniskill2 has introduced a full-physics simulation benchmark for manipulating various 3D objects. This benchmark enables agents to be trained using diverse datasets of demonstrations and evaluates their ability to generalize to unseen scenarios in testing environments. In this paper, we propose a novel two-stage fine-tuning strategy that aims to further enhance the generalization capability of our model based on the Maniskill2 benchmark. Through extensive experiments, we demonstrate the effectiveness of our approach by achieving the 1st prize in all three tracks of the ManiSkill2 Challenge. Our findings highlight the potential of our method to improve the generalization abilities of Embodied AI models and pave the way for their ractical applications in real-world scenarios. All codes and models of our solution is available at https://github.com/xtli12/GXU-LIPE.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2307.11343v1">A Two-stage Fine-tuning Strategy for Generalizable Manipulation Skill of Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-07-21
      | 💬 5 pages, 2 figures, 5 tables, accept by Robotics: Science and Systems 2023 - Workshop Interdisciplinary Exploration of Generalizable Manipulation Policy Learning:Paradigms and Debates
    </div>
    <details class="paper-abstract">
      The advent of Chat-GPT has led to a surge of interest in Embodied AI. However, many existing Embodied AI models heavily rely on massive interactions with training environments, which may not be practical in real-world situations. To this end, the Maniskill2 has introduced a full-physics simulation benchmark for manipulating various 3D objects. This benchmark enables agents to be trained using diverse datasets of demonstrations and evaluates their ability to generalize to unseen scenarios in testing environments. In this paper, we propose a novel two-stage fine-tuning strategy that aims to further enhance the generalization capability of our model based on the Maniskill2 benchmark. Through extensive experiments, we demonstrate the effectiveness of our approach by achieving the 1st prize in all three tracks of the ManiSkill2 Challenge. Our findings highlight the potential of our method to improve the generalization abilities of Embodied AI models and pave the way for their ractical applications in real-world scenarios. All codes and models of our solution is available at https://github.com/xtli12/GXU-LIPE.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.07166v1">Switching Head-Tail Funnel UNITER for Dual Referring Expression Comprehension with Fetch-and-Carry Tasks</a></div>
    <div class="paper-meta">
      📅 2023-07-14
      | 💬 Accepted for presentation at IROS2023
    </div>
    <details class="paper-abstract">
      This paper describes a domestic service robot (DSR) that fetches everyday objects and carries them to specified destinations according to free-form natural language instructions. Given an instruction such as "Move the bottle on the left side of the plate to the empty chair," the DSR is expected to identify the bottle and the chair from multiple candidates in the environment and carry the target object to the destination. Most of the existing multimodal language understanding methods are impractical in terms of computational complexity because they require inferences for all combinations of target object candidates and destination candidates. We propose Switching Head-Tail Funnel UNITER, which solves the task by predicting the target object and the destination individually using a single model. Our method is validated on a newly-built dataset consisting of object manipulation instructions and semi photo-realistic images captured in a standard Embodied AI simulator. The results show that our method outperforms the baseline method in terms of language comprehension accuracy. Furthermore, we conduct physical experiments in which a DSR delivers standardized everyday objects in a standardized domestic environment as requested by instructions with referring expressions. The experimental results show that the object grasping and placing actions are achieved with success rates of more than 90%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.14824v3">Kosmos-2: Grounding Multimodal Large Language Models to the World</a></div>
    <div class="paper-meta">
      📅 2023-07-13
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      We introduce Kosmos-2, a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent refer expressions as links in Markdown, i.e., ``[text span](bounding boxes)'', where object descriptions are sequences of location tokens. Together with multimodal corpora, we construct large-scale data of grounded image-text pairs (called GrIT) to train the model. In addition to the existing capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning), Kosmos-2 integrates the grounding capability into downstream applications. We evaluate Kosmos-2 on a wide range of tasks, including (i) multimodal grounding, such as referring expression comprehension, and phrase grounding, (ii) multimodal referring, such as referring expression generation, (iii) perception-language tasks, and (iv) language understanding and generation. This work lays out the foundation for the development of Embodiment AI and sheds light on the big convergence of language, multimodal perception, action, and world modeling, which is a key step toward artificial general intelligence. Code and pretrained models are available at https://aka.ms/kosmos-2.
    </details>
</div>
