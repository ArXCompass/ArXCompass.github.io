# embodied ai - 2021_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2106.00596v2">Look Wide and Interpret Twice: Improving Performance on Interactive Instruction-following Tasks</a></div>
    <div class="paper-meta">
      📅 2021-06-06
      | 💬 To appear in IJCAI2021. 8-page main paper and Appendix following. Appendix E for details of entry submission to EAI 2021. Github: https://github.com/davidnvq/lwit-alfred
    </div>
    <details class="paper-abstract">
      There is a growing interest in the community in making an embodied AI agent perform a complicated task while interacting with an environment following natural language directives. Recent studies have tackled the problem using ALFRED, a well-designed dataset for the task, but achieved only very low accuracy. This paper proposes a new method, which outperforms the previous methods by a large margin. It is based on a combination of several new ideas. One is a two-stage interpretation of the provided instructions. The method first selects and interprets an instruction without using visual information, yielding a tentative action sequence prediction. It then integrates the prediction with the visual information etc., yielding the final prediction of an action and an object. As the object's class to interact is identified in the first stage, it can accurately select the correct object from the input image. Moreover, our method considers multiple egocentric views of the environment and extracts essential information by applying hierarchical attention conditioned on the current instruction. This contributes to the accurate prediction of actions for navigation. A preliminary version of the method won the ALFRED Challenge 2020. The current version achieves the unseen environment's success rate of 4.45% with a single view, which is further improved to 8.37% with multiple views.
    </details>
</div>
