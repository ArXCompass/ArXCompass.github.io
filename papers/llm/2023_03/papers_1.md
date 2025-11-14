# llm - 2023_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2303.12810v1">Are LLMs the Master of All Trades? : Exploring Domain-Agnostic Reasoning Skills of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-03-28
    </div>
    <details class="paper-abstract">
      The potential of large language models (LLMs) to reason like humans has been a highly contested topic in Machine Learning communities. However, the reasoning abilities of humans are multifaceted and can be seen in various forms, including analogical, spatial and moral reasoning, among others. This fact raises the question whether LLMs can perform equally well across all these different domains. This research work aims to investigate the performance of LLMs on different reasoning tasks by conducting experiments that directly use or draw inspirations from existing datasets on analogical and spatial reasoning. Additionally, to evaluate the ability of LLMs to reason like human, their performance is evaluted on more open-ended, natural language questions. My findings indicate that LLMs excel at analogical and moral reasoning, yet struggle to perform as proficiently on spatial reasoning tasks. I believe these experiments are crucial for informing the future development of LLMs, particularly in contexts that require diverse reasoning proficiencies. By shedding light on the reasoning abilities of LLMs, this study aims to push forward our understanding of how they can better emulate the cognitive abilities of humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2303.12810v1">Are LLMs the Master of All Trades? : Exploring Domain-Agnostic Reasoning Skills of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-03-22
    </div>
    <details class="paper-abstract">
      The potential of large language models (LLMs) to reason like humans has been a highly contested topic in Machine Learning communities. However, the reasoning abilities of humans are multifaceted and can be seen in various forms, including analogical, spatial and moral reasoning, among others. This fact raises the question whether LLMs can perform equally well across all these different domains. This research work aims to investigate the performance of LLMs on different reasoning tasks by conducting experiments that directly use or draw inspirations from existing datasets on analogical and spatial reasoning. Additionally, to evaluate the ability of LLMs to reason like human, their performance is evaluted on more open-ended, natural language questions. My findings indicate that LLMs excel at analogical and moral reasoning, yet struggle to perform as proficiently on spatial reasoning tasks. I believe these experiments are crucial for informing the future development of LLMs, particularly in contexts that require diverse reasoning proficiencies. By shedding light on the reasoning abilities of LLMs, this study aims to push forward our understanding of how they can better emulate the cognitive abilities of humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2303.12810v1">Are LLMs the Master of All Trades? : Exploring Domain-Agnostic Reasoning Skills of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-03-22
    </div>
    <details class="paper-abstract">
      The potential of large language models (LLMs) to reason like humans has been a highly contested topic in Machine Learning communities. However, the reasoning abilities of humans are multifaceted and can be seen in various forms, including analogical, spatial and moral reasoning, among others. This fact raises the question whether LLMs can perform equally well across all these different domains. This research work aims to investigate the performance of LLMs on different reasoning tasks by conducting experiments that directly use or draw inspirations from existing datasets on analogical and spatial reasoning. Additionally, to evaluate the ability of LLMs to reason like human, their performance is evaluted on more open-ended, natural language questions. My findings indicate that LLMs excel at analogical and moral reasoning, yet struggle to perform as proficiently on spatial reasoning tasks. I believe these experiments are crucial for informing the future development of LLMs, particularly in contexts that require diverse reasoning proficiencies. By shedding light on the reasoning abilities of LLMs, this study aims to push forward our understanding of how they can better emulate the cognitive abilities of humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2303.07263v1">InferFix: End-to-End Program Repair with LLMs</a></div>
    <div class="paper-meta">
      📅 2023-03-13
    </div>
    <details class="paper-abstract">
      Software development life cycle is profoundly influenced by bugs: their introduction, identification, and eventual resolution account for a significant portion of software cost. This has motivated software engineering researchers and practitioners to propose different approaches for automating the identification and repair of software defects. Large language models have been adapted to the program repair task through few-shot demonstration learning and instruction prompting, treating this as an infilling task. However, these models have only focused on learning general bug-fixing patterns for uncategorized bugs mined from public repositories. In this paper, we propose InferFix: a transformer-based program repair framework paired with a state-of-the-art static analyzer to fix critical security and performance bugs. InferFix combines a Retriever -- transformer encoder model pretrained via contrastive learning objective, which aims at searching for semantically equivalent bugs and corresponding fixes; and a Generator -- a large language model (Codex Cushman) finetuned on supervised bug-fix data with prompts augmented via bug type annotations and semantically similar fixes retrieved from an external non-parametric memory. To train and evaluate our approach, we curated InferredBugs, a novel, metadata-rich dataset of bugs extracted by executing the Infer static analyzer on the change histories of thousands of Java and C# repositories. Our evaluation demonstrates that InferFix outperforms strong LLM baselines, with a top-1 accuracy of 65.6% for generating fixes in C# and 76.8% in Java. We discuss the deployment of InferFix alongside Infer at Microsoft which offers an end-to-end solution for detection, classification, and localization of bugs, as well as fixing and validation of candidate patches, integrated in the continuous integration pipeline to automate the software development workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2303.06223v1">Who's Thinking? A Push for Human-Centered Evaluation of LLMs using the XAI Playbook</a></div>
    <div class="paper-meta">
      📅 2023-03-10
      | 💬 Accepted to CHI 2023 workshop on Generative AI and HCI
    </div>
    <details class="paper-abstract">
      Deployed artificial intelligence (AI) often impacts humans, and there is no one-size-fits-all metric to evaluate these tools. Human-centered evaluation of AI-based systems combines quantitative and qualitative analysis and human input. It has been explored to some depth in the explainable AI (XAI) and human-computer interaction (HCI) communities. Gaps remain, but the basic understanding that humans interact with AI and accompanying explanations, and that humans' needs -- complete with their cognitive biases and quirks -- should be held front and center, is accepted by the community. In this paper, we draw parallels between the relatively mature field of XAI and the rapidly evolving research boom around large language models (LLMs). Accepted evaluative metrics for LLMs are not human-centered. We argue that many of the same paths tread by the XAI community over the past decade will be retread when discussing LLMs. Specifically, we argue that humans' tendencies -- again, complete with their cognitive biases and quirks -- should rest front and center when evaluating deployed LLMs. We outline three developed focus areas of human-centered evaluation of XAI: mental models, use case utility, and cognitive engagement, and we highlight the importance of exploring each of these concepts for LLMs. Our goal is to jumpstart human-centered LLM evaluation.
    </details>
</div>
