# llm - 2023_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.01202v1">From Voices to Validity: Leveraging Large Language Models (LLMs) for Textual Analysis of Policy Stakeholder Interviews</a></div>
    <div class="paper-meta">
      📅 2023-12-02
    </div>
    <details class="paper-abstract">
      Obtaining stakeholders' diverse experiences and opinions about current policy in a timely manner is crucial for policymakers to identify strengths and gaps in resource allocation, thereby supporting effective policy design and implementation. However, manually coding even moderately sized interview texts or open-ended survey responses from stakeholders can often be labor-intensive and time-consuming. This study explores the integration of Large Language Models (LLMs)--like GPT-4--with human expertise to enhance text analysis of stakeholder interviews regarding K-12 education policy within one U.S. state. Employing a mixed-methods approach, human experts developed a codebook and coding processes as informed by domain knowledge and unsupervised topic modeling results. They then designed prompts to guide GPT-4 analysis and iteratively evaluate different prompts' performances. This combined human-computer method enabled nuanced thematic and sentiment analysis. Results reveal that while GPT-4 thematic coding aligned with human coding by 77.89% at specific themes, expanding to broader themes increased congruence to 96.02%, surpassing traditional Natural Language Processing (NLP) methods by over 25%. Additionally, GPT-4 is more closely matched to expert sentiment analysis than lexicon-based methods. Findings from quantitative measures and qualitative reviews underscore the complementary roles of human domain expertise and automated analysis as LLMs offer new perspectives and coding consistency. The human-computer interactive approach enhances efficiency, validity, and interpretability of educational policy research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.01143v1">Towards leveraging LLMs for Conditional QA</a></div>
    <div class="paper-meta">
      📅 2023-12-02
    </div>
    <details class="paper-abstract">
      This study delves into the capabilities and limitations of Large Language Models (LLMs) in the challenging domain of conditional question-answering. Utilizing the Conditional Question Answering (CQA) dataset and focusing on generative models like T5 and UL2, we assess the performance of LLMs across diverse question types. Our findings reveal that fine-tuned LLMs can surpass the state-of-the-art (SOTA) performance in some cases, even without fully encoding all input context, with an increase of 7-8 points in Exact Match (EM) and F1 scores for Yes/No questions. However, these models encounter challenges in extractive question answering, where they lag behind the SOTA by over 10 points, and in mitigating the risk of injecting false information. A study with oracle-retrievers emphasizes the critical role of effective evidence retrieval, underscoring the necessity for advanced solutions in this area. Furthermore, we highlight the significant influence of evaluation metrics on performance assessments and advocate for a more comprehensive evaluation framework. The complexity of the task, the observed performance discrepancies, and the need for effective evidence retrieval underline the ongoing challenges in this field and underscore the need for future work focusing on refining training tasks and exploring prompt-based techniques to enhance LLM performance in conditional question-answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.09219v5">"Kelly is a Warm Person, Joseph is a Role Model": Gender Biases in LLM-Generated Reference Letters</a></div>
    <div class="paper-meta">
      📅 2023-12-01
      | 💬 Accepted to EMNLP 2023 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently emerged as an effective tool to assist individuals in writing various types of content, including professional documents such as recommendation letters. Though bringing convenience, this application also introduces unprecedented fairness concerns. Model-generated reference letters might be directly used by users in professional scenarios. If underlying biases exist in these model-constructed letters, using them without scrutinization could lead to direct societal harms, such as sabotaging application success rates for female applicants. In light of this pressing issue, it is imminent and necessary to comprehensively study fairness issues and associated harms in this real-world use case. In this paper, we critically examine gender biases in LLM-generated reference letters. Drawing inspiration from social science findings, we design evaluation methods to manifest biases through 2 dimensions: (1) biases in language style and (2) biases in lexical content. We further investigate the extent of bias propagation by analyzing the hallucination bias of models, a term that we define to be bias exacerbation in model-hallucinated contents. Through benchmarking evaluation on 2 popular LLMs- ChatGPT and Alpaca, we reveal significant gender biases in LLM-generated recommendation letters. Our findings not only warn against using LLMs for this application without scrutinization, but also illuminate the importance of thoroughly studying hidden biases and harms in LLM-generated professional documents.
    </details>
</div>
