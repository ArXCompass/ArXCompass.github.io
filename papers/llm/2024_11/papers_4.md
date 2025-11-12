# llm - 2024_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08563v1">Leveraging LLMs for Predictive Insights in Food Policy and Behavioral Interventions</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Food consumption and production contribute significantly to global greenhouse gas emissions, making them crucial entry points for mitigating climate change and maintaining a liveable planet. Over the past two decades, food policy initiatives have explored interventions to reshape production and consumption patterns, focusing on reducing food waste and curbing ruminant meat consumption. While the evidence of "what works" improves, evaluating which policies are appropriate and effective in specific contexts remains difficult due to external validity challenges. This paper demonstrates that a fine-tuned large language model (LLM) can accurately predict the direction of outcomes in approximately 80\% of empirical studies measuring dietary-based impacts (e.g. food choices, sales, waste) resulting from behavioral interventions and policies. Approximately 75 prompts were required to achieve optimal results, with performance showing signs of catastrophic loss beyond this point. Our findings indicate that greater input detail enhances predictive accuracy, although the model still faces challenges with unseen studies, underscoring the importance of a representative training sample. As LLMs continue to improve and diversify, they hold promise for advancing data-driven, evidence-based policymaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08553v1">CorrSynth -- A Correlated Sampling Method for Diverse Dataset Generation from LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 Published as a main conference paper at EMNLP 2024; First two authors contributed equally
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance in diverse tasks using zero-shot and few-shot prompting. Even though their capabilities of data synthesis have been studied well in recent years, the generated data suffers from a lack of diversity, less adherence to the prompt, and potential biases that creep into the data from the generator model. In this work, we tackle the challenge of generating datasets with high diversity, upon which a student model is trained for downstream tasks. Taking the route of decoding-time guidance-based approaches, we propose CorrSynth, which generates data that is more diverse and faithful to the input prompt using a correlated sampling strategy. Further, our method overcomes the complexity drawbacks of some other guidance-based techniques like classifier-based guidance. With extensive experiments, we show the effectiveness of our approach and substantiate our claims. In particular, we perform intrinsic evaluation to show the improvements in diversity. Our experiments show that CorrSynth improves both student metrics and intrinsic metrics upon competitive baselines across four datasets, showing the innate advantage of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08516v1">Tree-of-Table: Unleashing the Power of LLMs for Enhanced Large-Scale Table Understanding</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      The ubiquity and value of tables as semi-structured data across various domains necessitate advanced methods for understanding their complexity and vast amounts of information. Despite the impressive capabilities of large language models (LLMs) in advancing the natural language understanding frontier, their application to large-scale tabular data presents significant challenges, specifically regarding table size and complex intricate relationships. Existing works have shown promise with small-scale tables but often flounder when tasked with the complex reasoning required by larger, interconnected tables found in real-world scenarios. To address this gap, we introduce "Tree-of-Table", a novel approach designed to enhance LLMs' reasoning capabilities over large and complex tables. Our method employs Table Condensation and Decomposition to distill and reorganize relevant data into a manageable format, followed by the construction of a hierarchical Table-Tree that facilitates tree-structured reasoning. Through a meticulous Table-Tree Execution process, we systematically unravel the tree-structured reasoning chain to derive the solutions. Experiments across diverse datasets, including WikiTQ, TableFact, FeTaQA, and BIRD, demonstrate that Tree-of-Table sets a new benchmark with superior performance, showcasing remarkable efficiency and generalization capabilities in large-scale table reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08510v1">CorrectBench: Automatic Testbench Generation with Functional Self-Correction using LLMs for HDL Design</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Functional simulation is an essential step in digital hardware design. Recently, there has been a growing interest in leveraging Large Language Models (LLMs) for hardware testbench generation tasks. However, the inherent instability associated with LLMs often leads to functional errors in the generated testbenches. Previous methods do not incorporate automatic functional correction mechanisms without human intervention and still suffer from low success rates, especially for sequential tasks. To address this issue, we propose CorrectBench, an automatic testbench generation framework with functional self-validation and self-correction. Utilizing only the RTL specification in natural language, the proposed approach can validate the correctness of the generated testbenches with a success rate of 88.85%. Furthermore, the proposed LLM-based corrector employs bug information obtained during the self-validation process to perform functional self-correction on the generated testbenches. The comparative analysis demonstrates that our method achieves a pass ratio of 70.13% across all evaluated tasks, compared with the previous LLM-based testbench generation framework's 52.18% and a direct LLM-based generation method's 33.33%. Specifically in sequential circuits, our work's performance is 62.18% higher than previous work in sequential tasks and almost 5 times the pass ratio of the direct method. The codes and experimental results are open-sourced at the link: https://github.com/AutoBench/CorrectBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08404v1">Quantifying Qualitative Insights: Leveraging LLMs to Market Predict</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have the potential to transform financial analytics by integrating numerical and textual data. However, challenges such as insufficient context when fusing multimodal information and the difficulty in measuring the utility of qualitative outputs, which LLMs generate as text, have limited their effectiveness in tasks such as financial forecasting. This study addresses these challenges by leveraging daily reports from securities firms to create high-quality contextual information. The reports are segmented into text-based key factors and combined with numerical data, such as price information, to form context sets. By dynamically updating few-shot examples based on the query time, the sets incorporate the latest information, forming a highly relevant set closely aligned with the query point. Additionally, a crafted prompt is designed to assign scores to the key factors, converting qualitative insights into quantitative results. The derived scores undergo a scaling process, transforming them into real-world values that are used for prediction. Our experiments demonstrate that LLMs outperform time-series models in market forecasting, though challenges such as imperfect reproducibility and limited explainability remain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08348v1">Refining Translations with LLMs: A Constraint-Aware Iterative Prompting Approach</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in machine translation (MT), even without specific training on the languages in question. However, translating rare words in low-resource or domain-specific contexts remains challenging for LLMs. To address this issue, we propose a multi-step prompt chain that enhances translation faithfulness by prioritizing key terms crucial for semantic accuracy. Our method first identifies these keywords and retrieves their translations from a bilingual dictionary, integrating them into the LLM's context using Retrieval-Augmented Generation (RAG). We further mitigate potential output hallucinations caused by long prompts through an iterative self-checking mechanism, where the LLM refines its translations based on lexical and semantic constraints. Experiments using Llama and Qwen as base models on the FLORES-200 and WMT datasets demonstrate significant improvements over baselines, highlighting the effectiveness of our approach in enhancing translation faithfulness and robustness, particularly in low-resource scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15553v2">Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in various tasks, including instruction following, which is crucial for aligning model outputs with user expectations. However, evaluating LLMs' ability to follow instructions remains challenging due to the complexity and subjectivity of human language. Current benchmarks primarily focus on single-turn, monolingual instructions, which do not adequately reflect the complexities of real-world applications that require handling multi-turn and multilingual interactions. To address this gap, we introduce Multi-IF, a new benchmark designed to assess LLMs' proficiency in following multi-turn and multilingual instructions. Multi-IF, which utilizes a hybrid framework combining LLM and human annotators, expands upon the IFEval by incorporating multi-turn sequences and translating the English prompts into another 7 languages, resulting in a dataset of 4,501 multilingual conversations, where each has three turns. Our evaluation of 14 state-of-the-art LLMs on Multi-IF reveals that it presents a significantly more challenging task than existing benchmarks. All the models tested showed a higher rate of failure in executing instructions correctly with each additional turn. For example, o1-preview drops from 0.877 at the first turn to 0.707 at the third turn in terms of average accuracy over all languages. Moreover, languages with non-Latin scripts (Hindi, Russian, and Chinese) generally exhibit higher error rates, suggesting potential limitations in the models' multilingual capabilities. We release Multi-IF prompts and the evaluation code base to encourage further research in this critical area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08324v1">Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Many existing evaluation benchmarks for Large Language Models (LLMs) quickly become outdated due to the emergence of new models and training data. These benchmarks also fall short in assessing how LLM performance changes over time, as they consist of static questions without a temporal dimension. To address these limitations, we propose using future event prediction as a continuous evaluation method to assess LLMs' temporal generalization and forecasting abilities. Our benchmark, Daily Oracle, automatically generates question-answer (QA) pairs from daily news, challenging LLMs to predict "future" event outcomes. Our findings reveal that as pre-training data becomes outdated, LLM performance degrades over time. While Retrieval Augmented Generation (RAG) has the potential to enhance prediction accuracy, the performance degradation pattern persists, highlighting the need for continuous model updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08244v1">NVCiM-PT: An NVCiM-assisted Prompt Tuning Framework for Edge LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 Accepted by DATE 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deployed on edge devices, known as edge LLMs, need to continuously fine-tune their model parameters from user-generated data under limited resource constraints. However, most existing learning methods are not applicable for edge LLMs because of their reliance on high resources and low learning capacity. Prompt tuning (PT) has recently emerged as an effective fine-tuning method for edge LLMs by only modifying a small portion of LLM parameters, but it suffers from user domain shifts, resulting in repetitive training and losing resource efficiency. Conventional techniques to address domain shift issues often involve complex neural networks and sophisticated training, which are incompatible for PT for edge LLMs. Therefore, an open research question is how to address domain shift issues for edge LLMs with limited resources. In this paper, we propose a prompt tuning framework for edge LLMs, exploiting the benefits offered by non-volatile computing-in-memory (NVCiM) architectures. We introduce a novel NVCiM-assisted PT framework, where we narrow down the core operations to matrix-matrix multiplication, which can then be accelerated by performing in-situ computation on NVCiM. To the best of our knowledge, this is the first work employing NVCiM to improve the edge LLM PT performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08010v1">ExpressivityArena: Can LLMs Express Information Implicitly?</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 8 pages, 22 figures
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated remarkable performance in certain dimensions, their ability to express implicit language cues that human use for effective communication remains unclear. This paper presents ExpressivityArena, a Python library for measuring the implicit communication abilities of LLMs. We provide a comprehensive framework to evaluate expressivity of arbitrary LLMs and explore its practical implications. To this end, we refine the definition and measurements of ``expressivity,'' and use our framework in a set of small experiments. These experiments test LLMs in creative and logical tasks such as poetry, coding, and emotion-based responses. They are then evaluated by an automated grader, through ExpressivityArena, which we verify to be the most pragmatic for testing expressivity. Building on these experiments, we deepen our understanding of the expressivity of LLMs by assessing their ability to remain expressive in conversations. Our findings indicate that LLMs are capable of generating and understanding expressive content, however, with some limitations. These insights will inform the future development and deployment of expressive LLMs. We provide the code for ExpressivityArena alongside our paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07942v1">Towards Low-bit Communication for Tensor Parallel LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-11-12
    </div>
    <details class="paper-abstract">
      Tensor parallelism provides an effective way to increase server large language model (LLM) inference efficiency despite adding an additional communication cost. However, as server LLMs continue to scale in size, they will need to be distributed across more devices, magnifying the communication cost. One way to approach this problem is with quantization, but current methods for LLMs tend to avoid quantizing the features that tensor parallelism needs to communicate. Taking advantage of consistent outliers in communicated features, we introduce a quantization method that reduces communicated values on average from 16 bits to 4.2 bits while preserving nearly all of the original performance. For instance, our method maintains around 98.0% and 99.5% of Gemma 2 27B's and Llama 2 13B's original performance, respectively, averaged across all tasks we evaluated on.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07917v1">CryptoLLM: Unleashing the Power of Prompted LLMs for SmartQnA and Classification of Crypto Posts</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 Accepted at FIRE 2024 (Track: Opinion Extraction and Question Answering from CryptoCurrency-Related Tweets and Reddit posts (CryptOQA))
    </div>
    <details class="paper-abstract">
      The rapid growth of social media has resulted in an large volume of user-generated content, particularly in niche domains such as cryptocurrency. This task focuses on developing robust classification models to accurately categorize cryptocurrency-related social media posts into predefined classes, including but not limited to objective, positive, negative, etc. Additionally, the task requires participants to identify the most relevant answers from a set of posts in response to specific questions. By leveraging advanced LLMs, this research aims to enhance the understanding and filtering of cryptocurrency discourse, thereby facilitating more informed decision-making in this volatile sector. We have used a prompt-based technique to solve the classification task for reddit posts and twitter posts. Also, we have used 64-shot technique along with prompts on GPT-4-Turbo model to determine whether a answer is relevant to a question or not.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06008v2">The Dark Patterns of Personalized Persuasion in Large Language Models: Exposing Persuasive Linguistic Features for Big Five Personality Traits in LLMs Responses</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      This study explores how the Large Language Models (LLMs) adjust linguistic features to create personalized persuasive outputs. While research showed that LLMs personalize outputs, a gap remains in understanding the linguistic features of their persuasive capabilities. We identified 13 linguistic features crucial for influencing personalities across different levels of the Big Five model of personality. We analyzed how prompts with personality trait information influenced the output of 19 LLMs across five model families. The findings show that models use more anxiety-related words for neuroticism, increase achievement-related words for conscientiousness, and employ fewer cognitive processes words for openness to experience. Some model families excel at adapting language for openness to experience, others for conscientiousness, while only one model adapts language for neuroticism. Our findings show how LLMs tailor responses based on personality cues in prompts, indicating their potential to create persuasive content affecting the mind and well-being of the recipients.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05894v3">Efficient LLM Comparative Assessment: a Product of Experts Framework for Pairwise Comparisons</a></div>
    <div class="paper-meta">
      📅 2024-11-12
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge approaches are a practical and effective way of assessing a range of text tasks. However, when using pairwise comparisons to rank a set of candidates, the computational cost scales quadratically with the number of candidates, which has practical limitations. This paper introduces a Product of Expert (PoE) framework for efficient LLM Comparative Assessment. Here individual comparisons are considered experts that provide information on a pair's score difference. The PoE framework combines the information from these experts to yield an expression that can be maximized with respect to the underlying set of candidates, and is highly flexible where any form of expert can be assumed. When Gaussian experts are used one can derive simple closed-form solutions for the optimal candidate ranking, and expressions for selecting which comparisons should be made to maximize the probability of this ranking. Our approach enables efficient comparative assessment, where by using only a small subset of the possible comparisons, one can generate score predictions that correlate well with human judgements. We evaluate the approach on multiple NLG tasks and demonstrate that our framework can yield considerable computational savings when performing pairwise comparative assessment. With many candidate texts, using as few as 2% of comparisons the PoE solution can achieve similar performance to when all comparisons are used.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04799v2">Kwai-STaR: Transform LLMs into State-Transition Reasoners</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Mathematical reasoning presents a significant challenge to the cognitive capabilities of LLMs. Various methods have been proposed to enhance the mathematical ability of LLMs. However, few recognize the value of state transition for LLM reasoning. In this work, we define mathematical problem-solving as a process of transiting from an initial unsolved state to the final resolved state, and propose Kwai-STaR framework, which transforms LLMs into State-Transition Reasoners to improve their intuitive reasoning capabilities. Our approach comprises three main steps: (1) Define the state space tailored to the mathematical reasoning. (2) Generate state-transition data based on the state space. (3) Convert original LLMs into State-Transition Reasoners via a curricular training strategy. Our experiments validate the effectiveness of Kwai-STaR in enhancing mathematical reasoning: After training on the small-scale Kwai-STaR dataset, general LLMs, including Mistral-7B and LLaMA-3, achieve considerable performance gain on the GSM8K and GSM-Hard dataset. Additionally, the state transition-based design endows Kwai-STaR with remarkable training and inference efficiency. Further experiments are underway to establish the generality of Kwai-STaR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00722v2">LLMs for Generating and Evaluating Counterfactuals: A Comprehensive Study</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 Accepted to EMNLP Findings 2024
    </div>
    <details class="paper-abstract">
      As NLP models become more complex, understanding their decisions becomes more crucial. Counterfactuals (CFs), where minimal changes to inputs flip a model's prediction, offer a way to explain these models. While Large Language Models (LLMs) have shown remarkable performance in NLP tasks, their efficacy in generating high-quality CFs remains uncertain. This work fills this gap by investigating how well LLMs generate CFs for two NLU tasks. We conduct a comprehensive comparison of several common LLMs, and evaluate their CFs, assessing both intrinsic metrics, and the impact of these CFs on data augmentation. Moreover, we analyze differences between human and LLM-generated CFs, providing insights for future research directions. Our results show that LLMs generate fluent CFs, but struggle to keep the induced changes minimal. Generating CFs for Sentiment Analysis (SA) is less challenging than NLI where LLMs show weaknesses in generating CFs that flip the original label. This also reflects on the data augmentation performance, where we observe a large gap between augmenting with human and LLMs CFs. Furthermore, we evaluate LLMs' ability to assess CFs in a mislabelled data setting, and show that they have a strong bias towards agreeing with the provided labels. GPT4 is more robust against this bias and its scores correlate well with automatic metrics. Our findings reveal several limitations and point to potential future work directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02487v2">LiCoEval: Evaluating LLMs on License Compliance in Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 The 47th International Conference on Software Engineering(ICSE 2025)
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have revolutionized code generation, leading to widespread adoption of AI coding tools by developers. However, LLMs can generate license-protected code without providing the necessary license information, leading to potential intellectual property violations during software production. This paper addresses the critical, yet underexplored, issue of license compliance in LLM-generated code by establishing a benchmark to evaluate the ability of LLMs to provide accurate license information for their generated code. To establish this benchmark, we conduct an empirical study to identify a reasonable standard for "striking similarity" that excludes the possibility of independent creation, indicating a copy relationship between the LLM output and certain open-source code. Based on this standard, we propose LiCoEval, to evaluate the license compliance capabilities of LLMs, i.e., the ability to provide accurate license or copyright information when they generate code with striking similarity to already existing copyrighted code. Using LiCoEval, we evaluate 14 popular LLMs, finding that even top-performing LLMs produce a non-negligible proportion (0.88% to 2.01%) of code strikingly similar to existing open-source implementations. Notably, most LLMs fail to provide accurate license information, particularly for code under copyleft licenses. These findings underscore the urgent need to enhance LLM compliance capabilities in code generation tasks. Our study provides a foundation for future research and development to improve license compliance in AI-assisted software development, contributing to both the protection of open-source software copyrights and the mitigation of legal risks for LLM users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06634v2">Harnessing Earnings Reports for Stock Predictions: A QLoRA-Enhanced LLM Approach</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 Accepted by 2024 6th International Conference on Data-driven Optimization of Complex Systems
    </div>
    <details class="paper-abstract">
      Accurate stock market predictions following earnings reports are crucial for investors. Traditional methods, particularly classical machine learning models, struggle with these predictions because they cannot effectively process and interpret extensive textual data contained in earnings reports and often overlook nuances that influence market movements. This paper introduces an advanced approach by employing Large Language Models (LLMs) instruction fine-tuned with a novel combination of instruction-based techniques and quantized low-rank adaptation (QLoRA) compression. Our methodology integrates 'base factors', such as financial metric growth and earnings transcripts, with 'external factors', including recent market indices performances and analyst grades, to create a rich, supervised dataset. This comprehensive dataset enables our models to achieve superior predictive performance in terms of accuracy, weighted F1, and Matthews correlation coefficient (MCC), especially evident in the comparison with benchmarks such as GPT-4. We specifically highlight the efficacy of the llama-3-8b-Instruct-4bit model, which showcases significant improvements over baseline models. The paper also discusses the potential of expanding the output capabilities to include a 'Hold' option and extending the prediction horizon, aiming to accommodate various investment styles and time frames. This study not only demonstrates the power of integrating cutting-edge AI with fine-tuned financial data but also paves the way for future research in enhancing AI-driven financial analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05990v2">Game-theoretic LLM: Agent Workflow for Negotiation Games</a></div>
    <div class="paper-meta">
      📅 2024-11-12
      | 💬 45 pages, 12 figures
    </div>
    <details class="paper-abstract">
      This paper investigates the rationality of large language models (LLMs) in strategic decision-making contexts, specifically within the framework of game theory. We evaluate several state-of-the-art LLMs across a spectrum of complete-information and incomplete-information games. Our findings reveal that LLMs frequently deviate from rational strategies, particularly as the complexity of the game increases with larger payoff matrices or deeper sequential trees. To address these limitations, we design multiple game-theoretic workflows that guide the reasoning and decision-making processes of LLMs. These workflows aim to enhance the models' ability to compute Nash Equilibria and make rational choices, even under conditions of uncertainty and incomplete information. Experimental results demonstrate that the adoption of these workflows significantly improves the rationality and robustness of LLMs in game-theoretic tasks. Specifically, with the workflow, LLMs exhibit marked improvements in identifying optimal strategies, achieving near-optimal allocations in negotiation scenarios, and reducing susceptibility to exploitation during negotiations. Furthermore, we explore the meta-strategic considerations of whether it is rational for agents to adopt such workflows, recognizing that the decision to use or forgo the workflow constitutes a game-theoretic issue in itself. Our research contributes to a deeper understanding of LLMs' decision-making capabilities in strategic contexts and provides insights into enhancing their rationality through structured workflows. The findings have implications for the development of more robust and strategically sound AI agents capable of navigating complex interactive environments. Code and data supporting this study are available at \url{https://github.com/Wenyueh/game_theory}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07518v1">LLM App Squatting and Cloning</a></div>
    <div class="paper-meta">
      📅 2024-11-12
    </div>
    <details class="paper-abstract">
      Impersonation tactics, such as app squatting and app cloning, have posed longstanding challenges in mobile app stores, where malicious actors exploit the names and reputations of popular apps to deceive users. With the rapid growth of Large Language Model (LLM) stores like GPT Store and FlowGPT, these issues have similarly surfaced, threatening the integrity of the LLM app ecosystem. In this study, we present the first large-scale analysis of LLM app squatting and cloning using our custom-built tool, LLMappCrazy. LLMappCrazy covers 14 squatting generation techniques and integrates Levenshtein distance and BERT-based semantic analysis to detect cloning by analyzing app functional similarities. Using this tool, we generated variations of the top 1000 app names and found over 5,000 squatting apps in the dataset. Additionally, we observed 3,509 squatting apps and 9,575 cloning cases across six major platforms. After sampling, we find that 18.7% of the squatting apps and 4.9% of the cloning apps exhibited malicious behavior, including phishing, malware distribution, fake content dissemination, and aggressive ad injection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06611v2">vTune: Verifiable Fine-Tuning for LLMs Through Backdooring</a></div>
    <div class="paper-meta">
      📅 2024-11-12
    </div>
    <details class="paper-abstract">
      As fine-tuning large language models (LLMs) becomes increasingly prevalent, users often rely on third-party services with limited visibility into their fine-tuning processes. This lack of transparency raises the question: how do consumers verify that fine-tuning services are performed correctly? For instance, a service provider could claim to fine-tune a model for each user, yet simply send all users back the same base model. To address this issue, we propose vTune, a simple method that uses a small number of backdoor data points added to the training data to provide a statistical test for verifying that a provider fine-tuned a custom model on a particular user's dataset. Unlike existing works, vTune is able to scale to verification of fine-tuning on state-of-the-art LLMs, and can be used both with open-source and closed-source models. We test our approach across several model families and sizes as well as across multiple instruction-tuning datasets, and find that the statistical test is satisfied with p-values on the order of $\sim 10^{-40}$, with no negative impact on downstream task performance. Further, we explore several attacks that attempt to subvert vTune and demonstrate the method's robustness to these attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07494v1">Rapid Response: Mitigating LLM Jailbreaks with a Few Examples</a></div>
    <div class="paper-meta">
      📅 2024-11-12
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow more powerful, ensuring their safety against misuse becomes crucial. While researchers have focused on developing robust defenses, no method has yet achieved complete invulnerability to attacks. We propose an alternative approach: instead of seeking perfect adversarial robustness, we develop rapid response techniques to look to block whole classes of jailbreaks after observing only a handful of attacks. To study this setting, we develop RapidResponseBench, a benchmark that measures a defense's robustness against various jailbreak strategies after adapting to a few observed examples. We evaluate five rapid response methods, all of which use jailbreak proliferation, where we automatically generate additional jailbreaks similar to the examples observed. Our strongest method, which fine-tunes an input classifier to block proliferated jailbreaks, reduces attack success rate by a factor greater than 240 on an in-distribution set of jailbreaks and a factor greater than 15 on an out-of-distribution set, having observed just one example of each jailbreaking strategy. Moreover, further studies suggest that the quality of proliferation model and number of proliferated examples play an key role in the effectiveness of this defense. Overall, our results highlight the potential of responding rapidly to novel jailbreaks to limit LLM misuse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.13691v2">Jailbreaking LLM-Controlled Robots</a></div>
    <div class="paper-meta">
      📅 2024-11-12
    </div>
    <details class="paper-abstract">
      The recent introduction of large language models (LLMs) has revolutionized the field of robotics by enabling contextual reasoning and intuitive human-robot interaction in domains as varied as manipulation, locomotion, and self-driving vehicles. When viewed as a stand-alone technology, LLMs are known to be vulnerable to jailbreaking attacks, wherein malicious prompters elicit harmful text by bypassing LLM safety guardrails. To assess the risks of deploying LLMs in robotics, in this paper, we introduce RoboPAIR, the first algorithm designed to jailbreak LLM-controlled robots. Unlike existing, textual attacks on LLM chatbots, RoboPAIR elicits harmful physical actions from LLM-controlled robots, a phenomenon we experimentally demonstrate in three scenarios: (i) a white-box setting, wherein the attacker has full access to the NVIDIA Dolphins self-driving LLM, (ii) a gray-box setting, wherein the attacker has partial access to a Clearpath Robotics Jackal UGV robot equipped with a GPT-4o planner, and (iii) a black-box setting, wherein the attacker has only query access to the GPT-3.5-integrated Unitree Robotics Go2 robot dog. In each scenario and across three new datasets of harmful robotic actions, we demonstrate that RoboPAIR, as well as several static baselines, finds jailbreaks quickly and effectively, often achieving 100% attack success rates. Our results reveal, for the first time, that the risks of jailbroken LLMs extend far beyond text generation, given the distinct possibility that jailbroken robots could cause physical damage in the real world. Indeed, our results on the Unitree Go2 represent the first successful jailbreak of a deployed commercial robotic system. Addressing this emerging vulnerability is critical for ensuring the safe deployment of LLMs in robotics. Additional media is available at: https://robopair.org
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16914v3">DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05059v2">FineTuneBench: How well do commercial fine-tuning APIs infuse knowledge into LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      There is great interest in fine-tuning frontier large language models (LLMs) to inject new information and update existing knowledge. While commercial LLM fine-tuning APIs from providers such as OpenAI and Google promise flexible adaptation for various applications, the efficacy of fine-tuning remains unclear. In this study, we introduce FineTuneBench, an evaluation framework and dataset for understanding how well commercial fine-tuning APIs can successfully learn new and updated knowledge. We analyze five frontier LLMs with commercially available fine-tuning APIs, including GPT-4o and Gemini 1.5 Pro, on their effectiveness in two settings: (1) ingesting novel information, such as recent news events and new people profiles, and (2) updating existing knowledge, such as updated medical guidelines and code frameworks. Our results reveal substantial shortcomings in all the models' abilities to effectively learn new information through fine-tuning, with an average generalization accuracy of 37% across all models. When updating existing knowledge, such as incorporating medical guideline updates, commercial fine-tuning APIs show even more limited capability (average generalization accuracy of 19%). Overall, fine-tuning GPT-4o mini is the most effective for infusing new knowledge and updating knowledge, followed by GPT-3.5 Turbo and GPT-4o. The fine-tuning APIs for Gemini 1.5 Flesh and Gemini 1.5 Pro are unable to learn new knowledge or update existing knowledge. These findings underscore a major shortcoming in using current commercial fine-tuning services to achieve reliable knowledge infusion in common scenarios. We open source the FineTuneBench dataset at https://github.com/kevinwu23/StanfordFineTuneBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05778v2">LLMs as Method Actors: A Model for Prompt Engineering and Architecture</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      We introduce "Method Actors" as a mental model for guiding LLM prompt engineering and prompt architecture. Under this mental model, LLMs should be thought of as actors; prompts as scripts and cues; and LLM responses as performances. We apply this mental model to the task of improving LLM performance at playing Connections, a New York Times word puzzle game that prior research identified as a challenging benchmark for evaluating LLM reasoning. Our experiments with GPT-4o show that a "Method Actors" approach can significantly improve LLM performance over both a vanilla and "Chain of Thoughts" approach. A vanilla approach solves 27% of Connections puzzles in our dataset and a "Chain of Thoughts" approach solves 41% of puzzles, whereas our strongest "Method Actor" approach solves 86% of puzzles. We also test OpenAI's newest model designed specifically for complex reasoning tasks, o1-preview. When asked to solve a puzzle all at once, o1-preview solves 79% of Connections puzzles in our dataset, and when allowed to build puzzle solutions one guess at a time over multiple API calls, o1-preview solves 100% of the puzzles. Incorporating a "Method Actor" prompt architecture increases the percentage of puzzles that o1-preview solves perfectly from 76% to 87%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17238v2">LLM-Assisted Static Analysis for Detecting Security Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      Software is prone to security vulnerabilities. Program analysis tools to detect them have limited effectiveness in practice due to their reliance on human labeled specifications. Large language models (or LLMs) have shown impressive code generation capabilities but they cannot do complex reasoning over code to detect such vulnerabilities especially since this task requires whole-repository analysis. We propose IRIS, a neuro-symbolic approach that systematically combines LLMs with static analysis to perform whole-repository reasoning for security vulnerability detection. Specifically, IRIS leverages LLMs to infer taint specifications and perform contextual analysis, alleviating needs for human specifications and inspection. For evaluation, we curate a new dataset, CWE-Bench-Java, comprising 120 manually validated security vulnerabilities in real-world Java projects. A state-of-the-art static analysis tool CodeQL detects only 27 of these vulnerabilities whereas IRIS with GPT-4 detects 55 (+28) and improves upon CodeQL's average false discovery rate by 5% points. Furthermore, IRIS identifies 6 previously unknown vulnerabilities which cannot be found by existing tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15146v3">Rethinking LLM Memorization through the Lens of Adversarial Compression</a></div>
    <div class="paper-meta">
      📅 2024-11-11
      | 💬 https://locuslab.github.io/acr-memorization
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained on web-scale datasets raise substantial concerns regarding permissible data usage. One major question is whether these models "memorize" all their training data or they integrate many data sources in some way more akin to how a human would learn and synthesize information. The answer hinges, to a large degree, on how we define memorization. In this work, we propose the Adversarial Compression Ratio (ACR) as a metric for assessing memorization in LLMs. A given string from the training data is considered memorized if it can be elicited by a prompt (much) shorter than the string itself -- in other words, if these strings can be "compressed" with the model by computing adversarial prompts of fewer tokens. The ACR overcomes the limitations of existing notions of memorization by (i) offering an adversarial view of measuring memorization, especially for monitoring unlearning and compliance; and (ii) allowing for the flexibility to measure memorization for arbitrary strings at a reasonably low compute. Our definition serves as a practical tool for determining when model owners may be violating terms around data usage, providing a potential legal tool and a critical lens through which to address such scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02481v2">CDR: Customizable Density Ratios of Strong-over-weak LLMs for Preference Annotation</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      Preference tuning of large language models (LLMs) relies on high-quality human preference data, which is often expensive and time-consuming to gather. While existing methods can use trained reward models or proprietary model as judges for preference annotation, they have notable drawbacks: training reward models remain dependent on initial human data, and using proprietary model imposes license restrictions that inhibits commercial usage. In this paper, we introduce customized density ratio (CDR), a training-free and highly effective method that leverages off-the-shelf LLMs for preference data annotation. Our approach uses the log-density ratio between a better-aligned LLM and a less aligned LLM as a reward signal. We explores 221 different LLMs pairs and empirically demonstrate that increasing the performance gap between paired LLMs correlates with better reward generalization. Furthermore, we show that tailoring the density ratio reward function with specific criteria and preference exemplars enhances performance across domains and within target areas. In our experiment using density ratio from a pair of Mistral-7B models, CDR achieves a RewardBench score of 82.6, outperforming the best trained reward functions from same model class and demonstrating competitive performance against SoTA models in Safety (91.0) and Reasoning (88.0) domains. We use CDR to annotate an on-policy preference dataset with which we preference tune Llama-3-8B-Instruct with SimPO. Using reward signals from two relatively weak models, our approach pushes Llama-3-8B to achieve a 37.4% (+15.1%) win rate on ArenaHard and a 40.7% (+17.8%) win rate on Length-Controlled AlpacaEval 2.0, along with a score of 8.0 on MT-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07127v1">Benchmarking LLMs' Judgments with No Gold Standard</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      We introduce the GEM (Generative Estimator for Mutual Information), an evaluation metric for assessing language generation by Large Language Models (LLMs), particularly in generating informative judgments, without the need for a gold standard reference. GEM broadens the scenarios where we can benchmark LLM generation performance-from traditional ones, like machine translation and summarization, where gold standard references are readily available, to subjective tasks without clear gold standards, such as academic peer review. GEM uses a generative model to estimate mutual information between candidate and reference responses, without requiring the reference to be a gold standard. In experiments on a human-annotated dataset, GEM demonstrates competitive correlations with human scores compared to the state-of-the-art GPT-4o Examiner, and outperforms all other baselines. Additionally, GEM is more robust against strategic manipulations, such as rephrasing or elongation, which can artificially inflate scores under a GPT-4o Examiner. We also present GRE-bench (Generating Review Evaluation Benchmark) which evaluates LLMs based on how well they can generate high-quality peer reviews for academic research papers. Because GRE-bench is based upon GEM, it inherits its robustness properties. Additionally, GRE-bench circumvents data contamination problems (or data leakage) by using the continuous influx of new open-access research papers and peer reviews each year. We show GRE-bench results of various popular LLMs on their peer review capabilities using the ICLR2023 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09971v2">Advancing Object Goal Navigation Through LLM-enhanced Object Affinities Transfer</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      In object goal navigation, agents navigate towards objects identified by category labels using visual and spatial information. Previously, solely network-based methods typically rely on historical data for object affinities estimation, lacking adaptability to new environments and unseen targets. Simultaneously, employing Large Language Models (LLMs) for navigation as either planners or agents, though offering a broad knowledge base, is cost-inefficient and lacks targeted historical experience. Addressing these challenges, we present the LLM-enhanced Object Affinities Transfer (LOAT) framework, integrating LLM-derived object semantics with network-based approaches to leverage experiential object affinities, thus improving adaptability in unfamiliar settings. LOAT employs a dual-module strategy: a generalized affinities module for accessing LLMs' vast knowledge and an experiential affinities module for applying learned object semantic relationships, complemented by a dynamic fusion module harmonizing these information sources based on temporal context. The resulting scores activate semantic maps before feeding into downstream policies, enhancing navigation systems with context-aware inputs. Our evaluations conducted in the AI2-THOR and Habitat simulators indicate significant improvements in both navigation success rates and overall efficiency. Furthermore, the system performs effectively when deployed on a real robot without requiring additional training, thereby validating the efficacy of LOAT in integrating LLM insights for enhanced object-goal navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07091v1">Impact of LLM-based Review Comment Generation in Practice: A Mixed Open-/Closed-source User Study</a></div>
    <div class="paper-meta">
      📅 2024-11-11
      | 💬 12pages
    </div>
    <details class="paper-abstract">
      We conduct a large-scale empirical user study in a live setup to evaluate the acceptance of LLM-generated comments and their impact on the review process. This user study was performed in two organizations, Mozilla (which has its codebase available as open source) and Ubisoft (fully closed-source). Inside their usual review environment, participants were given access to RevMate, an LLM-based assistive tool suggesting generated review comments using an off-the-shelf LLM with Retrieval Augmented Generation to provide extra code and review context, combined with LLM-as-a-Judge, to auto-evaluate the generated comments and discard irrelevant cases. Based on more than 587 patch reviews provided by RevMate, we observed that 8.1% and 7.2%, respectively, of LLM-generated comments were accepted by reviewers in each organization, while 14.6% and 20.5% other comments were still marked as valuable as review or development tips. Refactoring-related comments are more likely to be accepted than Functional comments (18.2% and 18.6% compared to 4.8% and 5.2%). The extra time spent by reviewers to inspect generated comments or edit accepted ones (36/119), yielding an overall median of 43s per patch, is reasonable. The accepted generated comments are as likely to yield future revisions of the revised patch as human-written comments (74% vs 73% at chunk-level).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07071v1">Universal Response and Emergence of Induction in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-11
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      While induction is considered a key mechanism for in-context learning in LLMs, understanding its precise circuit decomposition beyond toy models remains elusive. Here, we study the emergence of induction behavior within LLMs by probing their response to weak single-token perturbations of the residual stream. We find that LLMs exhibit a robust, universal regime in which their response remains scale-invariant under changes in perturbation strength, thereby allowing us to quantify the build-up of token correlations throughout the model. By applying our method, we observe signatures of induction behavior within the residual stream of Gemma-2-2B, Llama-3.2-3B, and GPT-2-XL. Across all models, we find that these induction signatures gradually emerge within intermediate layers and identify the relevant model sections composing this behavior. Our results provide insights into the collective interplay of components within LLMs and serve as a benchmark for large-scale circuit analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16209v2">LLMCount: Enhancing Stationary mmWave Detection with Multimodal-LLM</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      Millimeter wave sensing provides people with the capability of sensing the surrounding crowds in a non-invasive and privacy-preserving manner, which holds huge application potential. However, detecting stationary crowds remains challenging due to several factors such as minimal movements (like breathing or casual fidgets), which can be easily treated as noise clusters during data collection and consequently filtered in the following processing procedures. Additionally, the uneven distribution of signal power due to signal power attenuation and interferences resulting from external reflectors or absorbers further complicates accurate detection. To address these challenges and enable stationary crowd detection across various application scenarios requiring specialized domain adaption, we introduce LLMCount, the first system to harness the capabilities of large-language models (LLMs) to enhance crowd detection performance. By exploiting the decision-making capability of LLM, we can successfully compensate the signal power to acquire a uniform distribution and thereby achieve a detection with higher accuracy. To assess the system's performance, comprehensive evaluations are conducted under diversified scenarios like hall, meeting room, and cinema. The evaluation results show that our proposed approach reaches high detection accuracy with lower overall latency compared with previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06950v1">Sniff AI: Is My 'Spicy' Your 'Spicy'? Exploring LLM's Perceptual Alignment with Human Smell Experiences</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      Aligning AI with human intent is important, yet perceptual alignment-how AI interprets what we see, hear, or smell-remains underexplored. This work focuses on olfaction, human smell experiences. We conducted a user study with 40 participants to investigate how well AI can interpret human descriptions of scents. Participants performed "sniff and describe" interactive tasks, with our designed AI system attempting to guess what scent the participants were experiencing based on their descriptions. These tasks evaluated the Large Language Model's (LLMs) contextual understanding and representation of scent relationships within its internal states - high-dimensional embedding space. Both quantitative and qualitative methods were used to evaluate the AI system's performance. Results indicated limited perceptual alignment, with biases towards certain scents, like lemon and peppermint, and continued failing to identify others, like rosemary. We discuss these findings in light of human-AI alignment advancements, highlighting the limitations and opportunities for enhancing HCI systems with multisensory experience integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07599v3">CTIBench: A Benchmark for Evaluating LLMs in Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      Cyber threat intelligence (CTI) is crucial in today's cybersecurity landscape, providing essential insights to understand and mitigate the ever-evolving cyber threats. The recent rise of Large Language Models (LLMs) have shown potential in this domain, but concerns about their reliability, accuracy, and hallucinations persist. While existing benchmarks provide general evaluations of LLMs, there are no benchmarks that address the practical and applied aspects of CTI-specific tasks. To bridge this gap, we introduce CTIBench, a benchmark designed to assess LLMs' performance in CTI applications. CTIBench includes multiple datasets focused on evaluating knowledge acquired by LLMs in the cyber-threat landscape. Our evaluation of several state-of-the-art models on these tasks provides insights into their strengths and weaknesses in CTI contexts, contributing to a better understanding of LLM capabilities in CTI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06899v1">LongSafetyBench: Long-Context LLMs Struggle with Safety Issues</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      With the development of large language models (LLMs), the sequence length of these models continues to increase, drawing significant attention to long-context language models. However, the evaluation of these models has been primarily limited to their capabilities, with a lack of research focusing on their safety. Existing work, such as ManyShotJailbreak, has to some extent demonstrated that long-context language models can exhibit safety concerns. However, the methods used are limited and lack comprehensiveness. In response, we introduce \textbf{LongSafetyBench}, the first benchmark designed to objectively and comprehensively evaluate the safety of long-context models. LongSafetyBench consists of 10 task categories, with an average length of 41,889 words. After testing eight long-context language models on LongSafetyBench, we found that existing models generally exhibit insufficient safety capabilities. The proportion of safe responses from most mainstream long-context LLMs is below 50\%. Moreover, models' safety performance in long-context scenarios does not always align with that in short-context scenarios. Further investigation revealed that long-context models tend to overlook harmful content within lengthy texts. We also proposed a simple yet effective solution, allowing open-source models to achieve performance comparable to that of top-tier closed-source models. We believe that LongSafetyBench can serve as a valuable benchmark for evaluating the safety capabilities of long-context language models. We hope that our work will encourage the broader community to pay attention to the safety of long-context models and contribute to the development of solutions to improve the safety of long-context LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06877v1">LLM-Assisted Relevance Assessments: When Should We Ask LLMs for Help?</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      Test collections are information retrieval tools that allow researchers to quickly and easily evaluate ranking algorithms. While test collections have become an integral part of IR research, the process of data creation involves significant efforts in manual annotations, which often makes it very expensive and time-consuming. Thus, the test collections could become small when the budget is limited, which may lead to unstable evaluations. As an alternative, recent studies have proposed the use of large language models (LLMs) to completely replace human assessors. However, while LLMs seem to somewhat correlate with human judgments, they are not perfect and often show bias. Moreover, even if a well-performing LLM or prompt is found on one dataset, there is no guarantee that it will perform similarly in practice, due to difference in tasks and data. Thus a complete replacement with LLMs is argued to be too risky and not fully trustable. Thus, in this paper, we propose \textbf{L}LM-\textbf{A}ssisted \textbf{R}elevance \textbf{A}ssessments (\textbf{LARA}), an effective method to balance manual annotations with LLM annotations, which helps to make a rich and reliable test collection. We use the LLM's predicted relevance probabilities in order to select the most profitable documents to manually annotate under a budget constraint. While solely relying on LLM's predicted probabilities to manually annotate performs fairly well, with theoretical reasoning, LARA guides the human annotation process even more effectively via online calibration learning. Then, using the calibration model learned from the limited manual annotations, LARA debiases the LLM predictions to annotate the remaining non-assessed data. Empirical evaluations on TREC-COVID and TREC-8 Ad Hoc datasets show that LARA outperforms the alternative solutions under almost any budget constraint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02943v2">Capturing research literature attitude towards Sustainable Development Goals: an LLM-based topic modeling approach</a></div>
    <div class="paper-meta">
      📅 2024-11-11
      | 💬 27 pages, 8 figures, 5 tables
    </div>
    <details class="paper-abstract">
      The world is facing a multitude of challenges that hinder the development of human civilization and the well-being of humanity on the planet. The Sustainable Development Goals (SDGs) were formulated by the United Nations in 2015 to address these global challenges by 2030. Natural language processing techniques can help uncover discussions on SDGs within research literature. We propose a completely automated pipeline to 1) fetch content from the Scopus database and prepare datasets dedicated to five groups of SDGs; 2) perform topic modeling, a statistical technique used to identify topics in large collections of textual data; and 3) enable topic exploration through keywords-based search and topic frequency time series extraction. For topic modeling, we leverage the stack of BERTopic scaled up to be applied on large corpora of textual documents (we find hundreds of topics on hundreds of thousands of documents), introducing i) a novel LLM-based embeddings computation for representing scientific abstracts in the continuous space and ii) a hyperparameter optimizer to efficiently find the best configuration for any new big datasets. We additionally produce the visualization of results on interactive dashboards reporting topics' temporal evolution. Results are made inspectable and explorable, contributing to the interpretability of the topic modeling process. Our proposed LLM-based topic modeling pipeline for big-text datasets allows users to capture insights on the evolution of the attitude toward SDGs within scientific abstracts in the 2006-2023 time span. All the results are reproducible by using our system; the workflow can be generalized to be applied at any point in time to any big corpus of textual documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16040v5">EHRNoteQA: An LLM Benchmark for Real-World Clinical Practice Using Discharge Summaries</a></div>
    <div class="paper-meta">
      📅 2024-11-11
      | 💬 NeurIPS 2024 (Datasets and Benchmarks)
    </div>
    <details class="paper-abstract">
      Discharge summaries in Electronic Health Records (EHRs) are crucial for clinical decision-making, but their length and complexity make information extraction challenging, especially when dealing with accumulated summaries across multiple patient admissions. Large Language Models (LLMs) show promise in addressing this challenge by efficiently analyzing vast and complex data. Existing benchmarks, however, fall short in properly evaluating LLMs' capabilities in this context, as they typically focus on single-note information or limited topics, failing to reflect the real-world inquiries required by clinicians. To bridge this gap, we introduce EHRNoteQA, a novel benchmark built on the MIMIC-IV EHR, comprising 962 different QA pairs each linked to distinct patients' discharge summaries. Every QA pair is initially generated using GPT-4 and then manually reviewed and refined by three clinicians to ensure clinical relevance. EHRNoteQA includes questions that require information across multiple discharge summaries and covers eight diverse topics, mirroring the complexity and diversity of real clinical inquiries. We offer EHRNoteQA in two formats: open-ended and multi-choice question answering, and propose a reliable evaluation method for each. We evaluate 27 LLMs using EHRNoteQA and examine various factors affecting the model performance (e.g., the length and number of discharge summaries). Furthermore, to validate EHRNoteQA as a reliable proxy for expert evaluations in clinical practice, we measure the correlation between the LLM performance on EHRNoteQA, and the LLM performance manually evaluated by clinicians. Results show that LLM performance on EHRNoteQA have higher correlation with clinician-evaluated performance (Spearman: 0.78, Kendall: 0.62) compared to other benchmarks, demonstrating its practical relevance in evaluating LLMs in clinical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06824v1">Combining Domain and Alignment Vectors to Achieve Better Knowledge-Safety Trade-offs in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      There is a growing interest in training domain-expert LLMs that excel in specific technical fields compared to their general-purpose instruction-tuned counterparts. However, these expert models often experience a loss in their safety abilities in the process, making them capable of generating harmful content. As a solution, we introduce an efficient and effective merging-based alignment method called \textsc{MergeAlign} that interpolates the domain and alignment vectors, creating safer domain-specific models while preserving their utility. We apply \textsc{MergeAlign} on Llama3 variants that are experts in medicine and finance, obtaining substantial alignment improvements with minimal to no degradation on domain-specific benchmarks. We study the impact of model merging through model similarity metrics and contributions of individual models being merged. We hope our findings open new research avenues and inspire more efficient development of safe expert LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06796v1">Automatically Write Code Checker: An LLM-based Approach with Logic-guided API Retrieval and Case by Case Iteration</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      With the rising demand for code quality assurance, developers are not only utilizing existing static code checkers but also seeking custom checkers to satisfy their specific needs. Nowadays, various code-checking frameworks provide extensive checker customization interfaces to meet this need. However, both the abstract checking logic as well as the complex API usage of large-scale frameworks make this task challenging. To this end, automated code checker generation is anticipated to ease the burden of checker development. In this paper, we explore the feasibility of automated checker generation and propose AutoChecker, an innovative LLM-powered approach that can write code checkers automatically based on only a rule description and a test suite. Instead of generating the checker at once, AutoChecker incrementally updates the checker with the rule and one single test case each time, i.e., it iteratively generates the checker case by case. During each iteration, AutoChecker first decomposes the whole logic into a series of sub-operations and then uses the logic-guided API-context retrieval strategy to search related API-contexts from all the framework APIs. To evaluate the effectiveness of AutoChecker, we apply AutoChecker and two LLM-based baseline approaches to automatically generate checkers for 20 built-in PMD rules, including easy rules and hard rules. Experimental results demonstrate that AutoChecker significantly outperforms baseline approaches across all effectiveness metrics, where its average test pass rate improved over 4.2 times. Moreover, the checkers generated by AutoChecker are successfully applied to real-world projects, matching the performance of official checkers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06767v1">PDC & DM-SFT: A Road for LLM SQL Bug-Fix Enhancing</a></div>
    <div class="paper-meta">
      📅 2024-11-11
      | 💬 COLING-Industry 2025 accepted
    </div>
    <details class="paper-abstract">
      Code Large Language Models (Code LLMs), such as Code llama and DeepSeek-Coder, have demonstrated exceptional performance in the code generation tasks. However, most existing models focus on the abilities of generating correct code, but often struggle with bug repair. We introduce a suit of methods to enhance LLM's SQL bug-fixing abilities. The methods are mainly consisted of two parts: A Progressive Dataset Construction (PDC) from scratch and Dynamic Mask Supervised Fine-tuning (DM-SFT). PDC proposes two data expansion methods from the perspectives of breadth first and depth first respectively. DM-SFT introduces an efficient bug-fixing supervised learning approach, which effectively reduce the total training steps and mitigate the "disorientation" in SQL code bug-fixing training. In our evaluation, the code LLM models trained with two methods have exceeds all current best performing model which size is much larger.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16758v2">Towards Fast Multilingual LLM Inference: Speculative Decoding and Specialized Drafters</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized natural language processing and broadened their applicability across diverse commercial applications. However, the deployment of these models is constrained by high inference time in multilingual settings. To mitigate this challenge, this paper explores a training recipe of an assistant model in speculative decoding, which is leveraged to draft and-then its future tokens are verified by the target LLM. We show that language-specific draft models, optimized through a targeted pretrain-and-finetune strategy, substantially brings a speedup in inference time compared to the previous methods. We validate these models across various languages in inference time, out-of-domain speedup, and GPT-4o evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.09874v4">TF-DCon: Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-11-11
      | 💬 An updated version
    </div>
    <details class="paper-abstract">
      Modern techniques in Content-based Recommendation (CBR) leverage item content information to provide personalized services to users, but suffer from resource-intensive training on large datasets. To address this issue, we explore the dataset condensation for textual CBR in this paper. The goal of dataset condensation is to synthesize a small yet informative dataset, upon which models can achieve performance comparable to those trained on large datasets. While existing condensation approaches are tailored to classification tasks for continuous data like images or embeddings, direct application of them to CBR has limitations. To bridge this gap, we investigate efficient dataset condensation for content-based recommendation. Inspired by the remarkable abilities of large language models (LLMs) in text comprehension and generation, we leverage LLMs to empower the generation of textual content during condensation. To handle the interaction data involving both users and items, we devise a dual-level condensation method: content-level and user-level. At content-level, we utilize LLMs to condense all contents of an item into a new informative title. At user-level, we design a clustering-based synthesis module, where we first utilize LLMs to extract user interests. Then, the user interests and user embeddings are incorporated to condense users and generate interactions for condensed users. Notably, the condensation paradigm of this method is forward and free from iterative optimization on the synthesized dataset. Extensive empirical findings from our study, conducted on three authentic datasets, substantiate the efficacy of the proposed method. Particularly, we are able to approximate up to 97% of the original performance while reducing the dataset size by 95% (i.e., on dataset MIND).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06723v1">Script-Strategy Aligned Generation: Aligning LLMs with Expert-Crafted Dialogue Scripts and Therapeutic Strategies for Psychotherapy</a></div>
    <div class="paper-meta">
      📅 2024-11-11
    </div>
    <details class="paper-abstract">
      Chatbots or conversational agents (CAs) are increasingly used to improve access to digital psychotherapy. Many current systems rely on rigid, rule-based designs, heavily dependent on expert-crafted dialogue scripts for guiding therapeutic conversations. Although recent advances in large language models (LLMs) offer the potential for more flexible interactions, their lack of controllability and transparency poses significant challenges in sensitive areas like psychotherapy. In this work, we explored how aligning LLMs with expert-crafted scripts can enhance psychotherapeutic chatbot performance. Our comparative study showed that LLMs aligned with expert-crafted scripts through prompting and fine-tuning significantly outperformed both pure LLMs and rule-based chatbots, achieving a more effective balance between dialogue flexibility and adherence to therapeutic principles. Building on findings, we proposed ``Script-Strategy Aligned Generation (SSAG)'', a flexible alignment approach that reduces reliance on fully scripted content while enhancing LLMs' therapeutic adherence and controllability. In a 10-day field study, SSAG demonstrated performance comparable to full script alignment and outperformed rule-based chatbots, empirically supporting SSAG as an efficient approach for aligning LLMs with domain expertise. Our work advances LLM applications in psychotherapy by providing a controllable, adaptable, and scalable solution for digital interventions, reducing reliance on expert effort. It also provides a collaborative framework for domain experts and developers to efficiently build expertise-aligned chatbots, broadening access to psychotherapy and behavioral interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.24190v3">Hidden Persuaders: LLMs' Political Leaning and Their Influence on Voters</a></div>
    <div class="paper-meta">
      📅 2024-11-11
      | 💬 EMNLP 2024 Main
    </div>
    <details class="paper-abstract">
      How could LLMs influence our democracy? We investigate LLMs' political leanings and the potential influence of LLMs on voters by conducting multiple experiments in a U.S. presidential election context. Through a voting simulation, we first demonstrate 18 open- and closed-weight LLMs' political preference for a Democratic nominee over a Republican nominee. We show how this leaning towards the Democratic nominee becomes more pronounced in instruction-tuned models compared to their base versions by analyzing their responses to candidate-policy related questions. We further explore the potential impact of LLMs on voter choice by conducting an experiment with 935 U.S. registered voters. During the experiments, participants interacted with LLMs (Claude-3, Llama-3, and GPT-4) over five exchanges. The experiment results show a shift in voter choices towards the Democratic nominee following LLM interaction, widening the voting margin from 0.7% to 4.6%, even though LLMs were not asked to persuade users to support the Democratic nominee during the discourse. This effect is larger than many previous studies on the persuasiveness of political campaigns, which have shown minimal effects in presidential elections. Many users also expressed a desire for further political interaction with LLMs. Which aspects of LLM interactions drove these shifts in voter choice requires further study. Lastly, we explore how a safety method can make LLMs more politically neutral, while raising the question of whether such neutrality is truly the path forward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03494v3">Beyond Text: Utilizing Vocal Cues to Improve Decision Making in LLMs for Robot Navigation Tasks</a></div>
    <div class="paper-meta">
      📅 2024-11-11
      | 💬 30 pages, 7 figures
    </div>
    <details class="paper-abstract">
      While LLMs excel in processing text in these human conversations, they struggle with the nuances of verbal instructions in scenarios like social navigation, where ambiguity and uncertainty can erode trust in robotic and other AI systems. We can address this shortcoming by moving beyond text and additionally focusing on the paralinguistic features of these audio responses. These features are the aspects of spoken communication that do not involve the literal wording (lexical content) but convey meaning and nuance through how something is said. We present Beyond Text: an approach that improves LLM decision-making by integrating audio transcription along with a subsection of these features, which focus on the affect and more relevant in human-robot conversations.This approach not only achieves a 70.26% winning rate, outperforming existing LLMs by 22.16% to 48.30% (gemini-1.5-pro and gpt-3.5 respectively), but also enhances robustness against token manipulation adversarial attacks, highlighted by a 22.44% less decrease ratio than the text-only language model in winning rate. Beyond Text' marks an advancement in social robot navigation and broader Human-Robot interactions, seamlessly integrating text-based guidance with human-audio-informed language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06581v1">Federated LLMs Fine-tuned with Adaptive Importance-Aware LoRA</a></div>
    <div class="paper-meta">
      📅 2024-11-10
    </div>
    <details class="paper-abstract">
      Federated fine-tuning of pre-trained Large Language Models (LLMs) enables task-specific adaptation across diverse datasets while preserving data privacy. However, the large model size and heterogeneity in client resources pose significant computational and communication challenges. To address these issues, in this paper, we propose a novel Heterogeneous Adaptive Federated Low-Rank Adaptation (LoRA) fine-tuned LLM framework (HAFL). To accommodate client resource heterogeneity, we first introduce an importance-based parameter truncation scheme, which allows clients to have different LoRA ranks, and smoothed sensitivity scores are used as importance indicators. Despite its flexibility, the truncation process may cause performance degradation. To tackle this problem, we develop an importance-based parameter freezing scheme. In this approach, both the cloud server and clients maintain the same LoRA rank, while clients selectively update only the most important decomposed LoRA rank-1 matrices, keeping the rest frozen. To mitigate the information dilution caused by the zero-padding aggregation method, we propose an adaptive aggregation approach that operates at the decomposed rank-1 matrix level. Experiments on the 20 News Group classification task show that our method converges quickly with low communication size, and avoids performance degradation when distributing models to clients compared to truncation-based heterogeneous LoRA rank scheme. Additionally, our adaptive aggregation method achieves faster convergence compared to the zero-padding approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06559v1">Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents</a></div>
    <div class="paper-meta">
      📅 2024-11-10
      | 💬 18 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Language agents have demonstrated promising capabilities in automating web-based tasks, though their current reactive approaches still underperform largely compared to humans. While incorporating advanced planning algorithms, particularly tree search methods, could enhance these agents' performance, implementing tree search directly on live websites poses significant safety risks and practical constraints due to irreversible actions such as confirming a purchase. In this paper, we introduce a novel paradigm that augments language agents with model-based planning, pioneering the innovative use of large language models (LLMs) as world models in complex web environments. Our method, WebDreamer, builds on the key insight that LLMs inherently encode comprehensive knowledge about website structures and functionalities. Specifically, WebDreamer uses LLMs to simulate outcomes for each candidate action (e.g., "what would happen if I click this button?") using natural language descriptions, and then evaluates these imagined outcomes to determine the optimal action at each step. Empirical results on two representative web agent benchmarks with online interaction -- VisualWebArena and Mind2Web-live -- demonstrate that WebDreamer achieves substantial improvements over reactive baselines. By establishing the viability of LLMs as world models in web environments, this work lays the groundwork for a paradigm shift in automated web interaction. More broadly, our findings open exciting new avenues for future research into 1) optimizing LLMs specifically for world modeling in complex, dynamic environments, and 2) model-based speculative planning for language agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06535v1">Probabilistic Consensus through Ensemble Validation: A Framework for LLM Reliability</a></div>
    <div class="paper-meta">
      📅 2024-11-10
      | 💬 8 pages, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant advances in text generation but often lack the reliability needed for autonomous deployment in high-stakes domains like healthcare, law, and finance. Existing approaches rely on external knowledge or human oversight, limiting scalability. We introduce a novel framework that repurposes ensemble methods for content validation through model consensus. In tests across 78 complex cases requiring factual accuracy and causal consistency, our framework improved precision from 73.1% to 93.9% with two models (95% CI: 83.5%-97.9%) and to 95.6% with three models (95% CI: 85.2%-98.8%). Statistical analysis indicates strong inter-model agreement ($\kappa$ > 0.76) while preserving sufficient independence to catch errors through disagreement. We outline a clear pathway to further enhance precision with additional validators and refinements. Although the current approach is constrained by multiple-choice format requirements and processing latency, it offers immediate value for enabling reliable autonomous AI systems in critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08719v1">Balancing Speed and Stability: The Trade-offs of FP8 vs. BF16 Training in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-10
      | 💬 2 pages,extended abstract
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have attracted significant attention due to their human-like language understanding and generation capabilities, as well as their applicability across various domains. These models, characterized by their massive scale and extensive training data, continue to push the boundaries of what is possible in natural language processing. The Llama 3 series, for instance, exemplifies this trend with its flagship model boasting 405 billion parameters trained on 15.6 trillion tokens. The immense computational demands associated with training such models have spurred ongoing research into optimizing the efficiency of the training process, particularly through the use of lower-precision formats. NVIDIA's H100 GPU, which introduces support for FP8 in addition to the more conventional FP16 and BF16 formats, has emerged as a focal point in this optimization effort. Preliminary studies suggest that FP8 could offer substantial reductions in training time without sacrificing model performance when compared to BF16, making it a promising candidate for large-scale model training. However, the broader implications of adopting FP8, particularly in terms of training stability and downstream task performance, have yet to be fully understood. In this study, we delve into the practical trade-offs involved in adopting FP8 over BF16 for training LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06469v1">ClinicalBench: Can LLMs Beat Traditional ML Models in Clinical Prediction?</a></div>
    <div class="paper-meta">
      📅 2024-11-10
      | 💬 The first two authors contributed equally. 10 pages for main paper, 66 pages including appendix. Project website: https://clinicalbench.github.io
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold great promise to revolutionize current clinical systems for their superior capacities on medical text processing tasks and medical licensing exams. Meanwhile, traditional ML models such as SVM and XGBoost have still been mainly adopted in clinical prediction tasks. An emerging question is Can LLMs beat traditional ML models in clinical prediction? Thus, we build a new benchmark ClinicalBench to comprehensively study the clinical predictive modeling capacities of both general-purpose and medical LLMs, and compare them with traditional ML models. ClinicalBench embraces three common clinical prediction tasks, two databases, 14 general-purpose LLMs, 8 medical LLMs, and 11 traditional ML models. Through extensive empirical investigation, we discover that both general-purpose and medical LLMs, even with different model scales, diverse prompting or fine-tuning strategies, still cannot beat traditional ML models in clinical prediction yet, shedding light on their potential deficiency in clinical reasoning and decision-making. We call for caution when practitioners adopt LLMs in clinical applications. ClinicalBench can be utilized to bridge the gap between LLMs' development for healthcare and real-world clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16247v4">AutoManual: Constructing Instruction Manuals by LLM Agents via Interactive Environmental Learning</a></div>
    <div class="paper-meta">
      📅 2024-11-10
      | 💬 Accepted at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) based agents have shown promise in autonomously completing tasks across various domains, e.g., robotics, games, and web navigation. However, these agents typically require elaborate design and expert prompts to solve tasks in specific domains, which limits their adaptability. We introduce AutoManual, a framework enabling LLM agents to autonomously build their understanding through interaction and adapt to new environments. AutoManual categorizes environmental knowledge into diverse rules and optimizes them in an online fashion by two agents: 1) The Planner codes actionable plans based on current rules for interacting with the environment. 2) The Builder updates the rules through a well-structured rule system that facilitates online rule management and essential detail retention. To mitigate hallucinations in managing rules, we introduce a *case-conditioned prompting* strategy for the Builder. Finally, the Formulator agent compiles these rules into a comprehensive manual. The self-generated manual can not only improve the adaptability but also guide the planning of smaller LLMs while being human-readable. Given only one simple demonstration, AutoManual significantly improves task success rates, achieving 97.4\% with GPT-4-turbo and 86.2\% with GPT-3.5-turbo on ALFWorld benchmark tasks. The code is available at https://github.com/minghchen/automanual.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06437v1">CTC-Assisted LLM-Based Contextual ASR</a></div>
    <div class="paper-meta">
      📅 2024-11-10
      | 💬 SLT 2024
    </div>
    <details class="paper-abstract">
      Contextual ASR or hotword customization holds substantial practical value. Despite the impressive performance of current end-to-end (E2E) automatic speech recognition (ASR) systems, they often face challenges in accurately recognizing rare words. Typical E2E contextual ASR models commonly feature complex architectures and decoding mechanisms, limited in performance and susceptible to interference from distractor words. With large language model (LLM)-based ASR models emerging as the new mainstream, we propose a CTC-Assisted LLM-Based Contextual ASR model with an efficient filtering algorithm. By using coarse CTC decoding results to filter potential relevant hotwords and incorporating them into LLM prompt input, our model attains WER/B-WER of 1.27%/3.67% and 2.72%/8.02% on the Librispeech test-clean and test-other sets targeting on recognizing rare long-tail words, demonstrating significant improvements compared to the baseline LLM-based ASR model, and substantially surpassing other related work. More remarkably, with the help of the large language model and proposed filtering algorithm, our contextual ASR model still performs well with 2000 biasing words.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.17017v2">Dynamic Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling</a></div>
    <div class="paper-meta">
      📅 2024-11-10
    </div>
    <details class="paper-abstract">
      Self-Consistency (SC) is a widely used method to mitigate hallucinations in Large Language Models (LLMs) by sampling the LLM multiple times and outputting the most frequent solution. Despite its benefits, SC results in significant computational costs proportional to the number of samples generated. Previous early-stopping approaches, such as Early Stopping Self Consistency and Adaptive Consistency, have aimed to reduce these costs by considering output consistency, but they do not analyze the quality of the reasoning paths (RPs) themselves. To address this issue, we propose Reasoning-Aware Self-Consistency (RASC), an innovative early-stopping framework that dynamically adjusts the number of sample generations by considering both the output answer and the RPs from Chain of Thought (CoT) prompting. RASC assigns confidence scores sequentially to the generated samples, stops when certain criteria are met, and then employs weighted majority voting to optimize sample usage and enhance answer reliability. We comprehensively test RASC with multiple LLMs across varied QA datasets. RASC outperformed existing methods and significantly reduces sample usage by an average of 80% while maintaining or improving accuracy up to 5% compared to the original SC
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06364v1">EcoServe: Maximizing Multi-Resource Utilization with SLO Guarantees in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-11-10
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) continue to grow, reducing costs and alleviating GPU demands has become increasingly critical. However, existing schedulers primarily target either GPU compute or Key-Value Cache (KVC) utilization, failing to fully optimize both GPU compute and KVC usage during each iteration or guarantee timely KVC allocations when needed. To address these challenges, we conducted a trace-based experimental analysis and made insightful observations, leading to the design of a system called EcoServe. EcoServe maximizes multi-resource utilization while ensuring service-level objective (SLO) guarantees in LLM serving. To enable adding prompts to a batch to maximize GPU utilization in each iteration, EcoServe maintains separate waiting queues for prompt processing tasks (PTs) and generation tasks (GTs). It batches GTs with the same predicted response lengths (RL) to save scheduling time and allocates KVC space for the predicted RL to avoid KVC allocation failures. It further has a novel KVC pipelining method, allowing sharing allocated but unused KVC space to enhance KVC utilization. In addition, it prioritizes queued requests that occupy more KVC to release KVC earlier and satisfy request service-level-objective (SLO). Experimental results demonstrate that EcoServe increases throughput by up to 4$\times$ with the same level of latency, generates up to 91\% lower job completion time and up to 91\% higher SLO satisfaction ratio compared to vLLM. It also reduces the number of GPUs used in DistServe by up to 78\% while maintaining the same level of goodput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13691v2">Jailbreaking LLM-Controlled Robots</a></div>
    <div class="paper-meta">
      📅 2024-11-09
    </div>
    <details class="paper-abstract">
      The recent introduction of large language models (LLMs) has revolutionized the field of robotics by enabling contextual reasoning and intuitive human-robot interaction in domains as varied as manipulation, locomotion, and self-driving vehicles. When viewed as a stand-alone technology, LLMs are known to be vulnerable to jailbreaking attacks, wherein malicious prompters elicit harmful text by bypassing LLM safety guardrails. To assess the risks of deploying LLMs in robotics, in this paper, we introduce RoboPAIR, the first algorithm designed to jailbreak LLM-controlled robots. Unlike existing, textual attacks on LLM chatbots, RoboPAIR elicits harmful physical actions from LLM-controlled robots, a phenomenon we experimentally demonstrate in three scenarios: (i) a white-box setting, wherein the attacker has full access to the NVIDIA Dolphins self-driving LLM, (ii) a gray-box setting, wherein the attacker has partial access to a Clearpath Robotics Jackal UGV robot equipped with a GPT-4o planner, and (iii) a black-box setting, wherein the attacker has only query access to the GPT-3.5-integrated Unitree Robotics Go2 robot dog. In each scenario and across three new datasets of harmful robotic actions, we demonstrate that RoboPAIR, as well as several static baselines, finds jailbreaks quickly and effectively, often achieving 100% attack success rates. Our results reveal, for the first time, that the risks of jailbroken LLMs extend far beyond text generation, given the distinct possibility that jailbroken robots could cause physical damage in the real world. Indeed, our results on the Unitree Go2 represent the first successful jailbreak of a deployed commercial robotic system. Addressing this emerging vulnerability is critical for ensuring the safe deployment of LLMs in robotics. Additional media is available at: https://robopair.org
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06248v1">Robust Detection of LLM-Generated Text: A Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2024-11-09
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The ability of large language models to generate complex texts allows them to be widely integrated into many aspects of life, and their output can quickly fill all network resources. As the impact of LLMs grows, it becomes increasingly important to develop powerful detectors for the generated text. This detector is essential to prevent the potential misuse of these technologies and to protect areas such as social media from the negative effects of false content generated by LLMS. The main goal of LLM-generated text detection is to determine whether text is generated by an LLM, which is a basic binary classification task. In our work, we mainly use three different classification methods based on open source datasets: traditional machine learning techniques such as logistic regression, k-means clustering, Gaussian Naive Bayes, support vector machines, and methods based on converters such as BERT, and finally algorithms that use LLMs to detect LLM-generated text. We focus on model generalization, potential adversarial attacks, and accuracy of model evaluation. Finally, the possible research direction in the future is proposed, and the current experimental results are summarized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01493v2">Sample-Efficient Alignment for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-09
    </div>
    <details class="paper-abstract">
      We study methods for efficiently aligning large language models (LLMs) with human preferences given budgeted online feedback. We first formulate the LLM alignment problem in the frame of contextual dueling bandits. This formulation, subsuming recent paradigms such as online RLHF and online DPO, inherently quests for sample-efficient algorithms that incorporate online active exploration. Leveraging insights from bandit theory, we introduce a unified algorithm based on Thompson sampling and highlight its applications in two distinct LLM alignment scenarios. The practical agent that efficiently implements this algorithm, named SEA (Sample-Efficient Alignment), is empirically validated through extensive experiments across three model scales (1B, 2.8B, 6.9B) and three preference learning algorithms (DPO, IPO, SLiC). The results demonstrate that SEA achieves highly sample-efficient alignment with oracle's preferences, outperforming recent active exploration methods for LLMs. Additionally, we release the implementation of SEA together with an efficient codebase designed for online alignment of LLMs, aiming to accelerate future research in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16442v4">Fast and Efficient 2-bit LLM Inference on GPU: 2/4/16-bit in a Weight Matrix with Asynchronous Dequantization</a></div>
    <div class="paper-meta">
      📅 2024-11-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive abilities in various domains while the inference cost is expensive. Many previous studies exploit quantization methods to reduce LLM inference cost by reducing latency and memory consumption. Applying 2-bit single-precision weight quantization brings >3% accuracy loss, so the state-of-the-art methods use mixed-precision methods for LLMs (e.g. Llama2-7b, etc.) to improve the accuracy. However, challenges still exist: (1) Uneven distribution in weight matrix. (2) Large speed degradation by adding sparse outliers. (3) Time-consuming dequantization operations on GPUs. To tackle these challenges and enable fast and efficient LLM inference on GPUs, we propose the following techniques in this paper. (1) Intra-weight mixed-precision quantization. (2) Exclusive 2-bit sparse outlier with minimum speed degradation. (3) Asynchronous dequantization. We conduct extensive experiments on different model families (e.g. Llama3, etc.) and model sizes. We achieve 2.91-bit for each weight considering all scales/zeros for different models with negligible loss. As a result, with our 2/4/16 mixed-precision quantization for each weight matrix and asynchronous dequantization during inference, our design achieves an end-to-end speedup for Llama2-7b is 1.74x over the original model, and we reduce both runtime cost and total cost by up to 2.53x and 2.29x with less GPU requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05915v2">Give me a hint: Can LLMs take a hint to solve math problems?</a></div>
    <div class="paper-meta">
      📅 2024-11-09
    </div>
    <details class="paper-abstract">
      While state-of-the-art LLMs have shown poor logical and basic mathematical reasoning, recent works try to improve their problem-solving abilities using prompting techniques. We propose giving "hints" to improve the language model's performance on advanced mathematical problems, taking inspiration from how humans approach math pedagogically. We also test robustness to adversarial hints and demonstrate their sensitivity to them. We demonstrate the effectiveness of our approach by evaluating various diverse LLMs, presenting them with a broad set of problems of different difficulties and topics from the MATH dataset and comparing against techniques such as one-shot, few-shot, and chain of thought prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06099v1">CoPrompter: User-Centric Evaluation of LLM Instruction Alignment for Improved Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2024-11-09
    </div>
    <details class="paper-abstract">
      Ensuring large language models' (LLMs) responses align with prompt instructions is crucial for application development. Based on our formative study with industry professionals, the alignment requires heavy human involvement and tedious trial-and-error especially when there are many instructions in the prompt. To address these challenges, we introduce CoPrompter, a framework that identifies misalignment based on assessing multiple LLM responses with criteria. It proposes a method to generate evaluation criteria questions derived directly from prompt requirements and an interface to turn these questions into a user-editable checklist. Our user study with industry prompt engineers shows that CoPrompter improves the ability to identify and refine instruction alignment with prompt requirements over traditional methods, helps them understand where and how frequently models fail to follow user's prompt requirements, and helps in clarifying their own requirements, giving them greater control over the response evaluation process. We also present the design lessons to underscore our system's potential to streamline the prompt engineering process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23728v2">GigaCheck: Detecting LLM-generated Content</a></div>
    <div class="paper-meta">
      📅 2024-11-09
      | 💬 11 pages, 1 figure
    </div>
    <details class="paper-abstract">
      With the increasing quality and spread of LLM-based assistants, the amount of LLM-generated content is growing rapidly. In many cases and tasks, such texts are already indistinguishable from those written by humans, and the quality of generation tends to only increase. At the same time, detection methods are developing more slowly, making it challenging to prevent misuse of generative AI technologies. In this work, we investigate the task of generated text detection by proposing the GigaCheck. Our research explores two approaches: (i) distinguishing human-written texts from LLM-generated ones, and (ii) detecting LLM-generated intervals in Human-Machine collaborative texts. For the first task, our approach utilizes a general-purpose LLM, leveraging its extensive language abilities to fine-tune efficiently for the downstream task of LLM-generated text detection, achieving high performance even with limited data. For the second task, we propose a novel approach that combines computer vision and natural language processing techniques. Specifically, we use a fine-tuned general-purpose LLM in conjunction with a DETR-like detection model, adapted from computer vision, to localize AI-generated intervals within text. We evaluate the GigaCheck on five classification datasets with English texts and three datasets designed for Human-Machine collaborative text analysis. Our results demonstrate that GigaCheck outperforms previous methods, even in out-of-distribution settings, establishing a strong baseline across all datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06046v1">Personalized News Recommendation System via LLM Embedding and Co-Occurrence Patterns</a></div>
    <div class="paper-meta">
      📅 2024-11-09
    </div>
    <details class="paper-abstract">
      In the past two years, large language models (LLMs) have achieved rapid development and demonstrated remarkable emerging capabilities. Concurrently, with powerful semantic understanding and reasoning capabilities, LLMs have significantly empowered the rapid advancement of the recommendation system field. Specifically, in news recommendation (NR), systems must comprehend and process a vast amount of clicked news text to infer the probability of candidate news clicks. This requirement exceeds the capabilities of traditional NR models but aligns well with the strengths of LLMs. In this paper, we propose a novel NR algorithm to reshape the news model via LLM Embedding and Co-Occurrence Pattern (LECOP). On one hand, we fintuned LLM by contrastive learning using large-scale datasets to encode news, which can fully explore the semantic information of news to thoroughly identify user preferences. On the other hand, we explored multiple co-occurrence patterns to mine collaborative information. Those patterns include news ID co-occurrence, Item-Item keywords co-occurrence and Intra-Item keywords co-occurrence. The keywords mentioned above are all generated by LLM. As far as we know, this is the first time that constructing such detailed Co-Occurrence Patterns via LLM to capture collaboration. Extensive experiments demonstrate the superior performance of our proposed novel method
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06032v1">LLM-GLOBE: A Benchmark Evaluating the Cultural Values Embedded in LLM Output</a></div>
    <div class="paper-meta">
      📅 2024-11-09
    </div>
    <details class="paper-abstract">
      Immense effort has been dedicated to minimizing the presence of harmful or biased generative content and better aligning AI output to human intention; however, research investigating the cultural values of LLMs is still in very early stages. Cultural values underpin how societies operate, providing profound insights into the norms, priorities, and decision making of their members. In recognition of this need for further research, we draw upon cultural psychology theory and the empirically-validated GLOBE framework to propose the LLM-GLOBE benchmark for evaluating the cultural value systems of LLMs, and we then leverage the benchmark to compare the values of Chinese and US LLMs. Our methodology includes a novel "LLMs-as-a-Jury" pipeline which automates the evaluation of open-ended content to enable large-scale analysis at a conceptual level. Results clarify similarities and differences that exist between Eastern and Western cultural value systems and suggest that open-generation tasks represent a more promising direction for evaluation of cultural values. We interpret the implications of this research for subsequent model development, evaluation, and deployment efforts as they relate to LLMs, AI cultural alignment more broadly, and the influence of AI cultural value systems on human-AI collaboration outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06018v1">A Picture is Worth A Thousand Numbers: Enabling LLMs Reason about Time Series via Visualization</a></div>
    <div class="paper-meta">
      📅 2024-11-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), with demonstrated reasoning abilities across multiple domains, are largely underexplored for time-series reasoning (TsR), which is ubiquitous in the real world. In this work, we propose TimerBed, the first comprehensive testbed for evaluating LLMs' TsR performance. Specifically, TimerBed includes stratified reasoning patterns with real-world tasks, comprehensive combinations of LLMs and reasoning strategies, and various supervised models as comparison anchors. We perform extensive experiments with TimerBed, test multiple current beliefs, and verify the initial failures of LLMs in TsR, evidenced by the ineffectiveness of zero shot (ZST) and performance degradation of few shot in-context learning (ICL). Further, we identify one possible root cause: the numerical modeling of data. To address this, we propose a prompt-based solution VL-Time, using visualization-modeled data and language-guided reasoning. Experimental results demonstrate that Vl-Time enables multimodal LLMs to be non-trivial ZST and powerful ICL reasoners for time series, achieving about 140% average performance improvement and 99% average token costs reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03417v2">Usefulness of LLMs as an Author Checklist Assistant for Scientific Papers: NeurIPS'24 Experiment</a></div>
    <div class="paper-meta">
      📅 2024-11-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) represent a promising, but controversial, tool in aiding scientific peer review. This study evaluates the usefulness of LLMs in a conference setting as a tool for vetting paper submissions against submission standards. We conduct an experiment at the 2024 Neural Information Processing Systems (NeurIPS) conference, where 234 papers were voluntarily submitted to an "LLM-based Checklist Assistant." This assistant validates whether papers adhere to the author checklist used by NeurIPS, which includes questions to ensure compliance with research and manuscript preparation standards. Evaluation of the assistant by NeurIPS paper authors suggests that the LLM-based assistant was generally helpful in verifying checklist completion. In post-usage surveys, over 70% of authors found the assistant useful, and 70% indicate that they would revise their papers or checklist responses based on its feedback. While causal attribution to the assistant is not definitive, qualitative evidence suggests that the LLM contributed to improving some submissions. Survey responses and analysis of re-submissions indicate that authors made substantive revisions to their submissions in response to specific feedback from the LLM. The experiment also highlights common issues with LLMs: inaccuracy (20/52) and excessive strictness (14/52) were the most frequent issues flagged by authors. We also conduct experiments to understand potential gaming of the system, which reveal that the assistant could be manipulated to enhance scores through fabricated justifications, highlighting potential vulnerabilities of automated review tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05982v1">Unmasking the Shadows: Pinpoint the Implementations of Anti-Dynamic Analysis Techniques in Malware Using LLM</a></div>
    <div class="paper-meta">
      📅 2024-11-08
    </div>
    <details class="paper-abstract">
      Sandboxes and other dynamic analysis processes are prevalent in malware detection systems nowadays to enhance the capability of detecting 0-day malware. Therefore, techniques of anti-dynamic analysis (TADA) are prevalent in modern malware samples, and sandboxes can suffer from false negatives and analysis failures when analyzing the samples with TADAs. In such cases, human reverse engineers will get involved in conducting dynamic analysis manually (i.e., debugging, patching), which in turn also gets obstructed by TADAs. In this work, we propose a Large Language Model (LLM) based workflow that can pinpoint the location of the TADA implementation in the code, to help reverse engineers place breakpoints used in debugging. Our evaluation shows that we successfully identified the locations of 87.80% known TADA implementations adopted from public repositories. In addition, we successfully pinpoint the locations of TADAs in 4 well-known malware samples that are documented in online malware analysis blogs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07264v1">Multi-Document Financial Question Answering using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-08
    </div>
    <details class="paper-abstract">
      We propose two new methods for multi-document financial question answering. First, a method that uses semantic tagging, and then, queries the index to get the context (RAG_SEM). And second, a Knowledge Graph (KG_RAG) based method that uses semantic tagging, and, retrieves knowledge graph triples from a graph database, as context. KG_RAG uses knowledge graphs constructed using a small model that is fine-tuned using knowledge distillation using a large teacher model. The data consists of 18 10K reports of Apple, Microsoft, Alphabet, NVIDIA, Amazon and Tesla for the years 2021, 2022 and 2023. The list of questions in the data consists of 111 complex questions including many esoteric questions that are difficult to answer and the answers are not completely obvious. As evaluation metrics, we use overall scores as well as segmented scores for measurement including the faithfulness, relevance, correctness, similarity, an LLM based overall score and the rouge scores as well as a similarity of embeddings. We find that both methods outperform plain RAG significantly. KG_RAG outperforms RAG_SEM in four out of nine metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09539v3">Logits of API-Protected LLMs Leak Proprietary Information</a></div>
    <div class="paper-meta">
      📅 2024-11-08
    </div>
    <details class="paper-abstract">
      Large language model (LLM) providers often hide the architectural details and parameters of their proprietary models by restricting public access to a limited API. In this work we show that, with only a conservative assumption about the model architecture, it is possible to learn a surprisingly large amount of non-public information about an API-protected LLM from a relatively small number of API queries (e.g., costing under $1000 USD for OpenAI's gpt-3.5-turbo). Our findings are centered on one key observation: most modern LLMs suffer from a softmax bottleneck, which restricts the model outputs to a linear subspace of the full output space. We exploit this fact to unlock several capabilities, including (but not limited to) obtaining cheap full-vocabulary outputs, auditing for specific types of model updates, identifying the source LLM given a single full LLM output, and even efficiently discovering the LLM's hidden size. Our empirical investigations show the effectiveness of our methods, which allow us to estimate the embedding size of OpenAI's gpt-3.5-turbo to be about 4096. Lastly, we discuss ways that LLM providers can guard against these attacks, as well as how these capabilities can be viewed as a feature (rather than a bug) by allowing for greater transparency and accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05775v1">Fact or Fiction? Can LLMs be Reliable Annotators for Political Truths?</a></div>
    <div class="paper-meta">
      📅 2024-11-08
      | 💬 Accepted at Socially Responsible Language Modelling Research (SoLaR) Workshop at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Political misinformation poses significant challenges to democratic processes, shaping public opinion and trust in media. Manual fact-checking methods face issues of scalability and annotator bias, while machine learning models require large, costly labelled datasets. This study investigates the use of state-of-the-art large language models (LLMs) as reliable annotators for detecting political factuality in news articles. Using open-source LLMs, we create a politically diverse dataset, labelled for bias through LLM-generated annotations. These annotations are validated by human experts and further evaluated by LLM-based judges to assess the accuracy and reliability of the annotations. Our approach offers a scalable and robust alternative to traditional fact-checking, enhancing transparency and public trust in media.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05653v1">The influence of persona and conversational task on social interactions with a LLM-controlled embodied conversational agent</a></div>
    <div class="paper-meta">
      📅 2024-11-08
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in conversational tasks. Embodying an LLM as a virtual human allows users to engage in face-to-face social interactions in Virtual Reality. However, the influence of person- and task-related factors in social interactions with LLM-controlled agents remains unclear. In this study, forty-six participants interacted with a virtual agent whose persona was manipulated as extravert or introvert in three different conversational tasks (small talk, knowledge test, convincing). Social-evaluation, emotional experience, and realism were assessed using ratings. Interactive engagement was measured by quantifying participants' words and conversational turns. Finally, we measured participants' willingness to ask the agent for help during the knowledge test. Our findings show that the extraverted agent was more positively evaluated, elicited a more pleasant experience and greater engagement, and was assessed as more realistic compared to the introverted agent. Whereas persona did not affect the tendency to ask for help, participants were generally more confident in the answer when they had help of the LLM. Variation of personality traits of LLM-controlled embodied virtual agents, therefore, affects social-emotional processing and behavior in virtual interactions. Embodied virtual agents allow the presentation of naturalistic social encounters in a virtual environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05651v1">LightVA: Lightweight Visual Analytics with LLM Agent-Based Task Planning and Execution</a></div>
    <div class="paper-meta">
      📅 2024-11-08
    </div>
    <details class="paper-abstract">
      Visual analytics (VA) requires analysts to iteratively propose analysis tasks based on observations and execute tasks by creating visualizations and interactive exploration to gain insights. This process demands skills in programming, data processing, and visualization tools, highlighting the need for a more intelligent, streamlined VA approach. Large language models (LLMs) have recently been developed as agents to handle various tasks with dynamic planning and tool-using capabilities, offering the potential to enhance the efficiency and versatility of VA. We propose LightVA, a lightweight VA framework that supports task decomposition, data analysis, and interactive exploration through human-agent collaboration. Our method is designed to help users progressively translate high-level analytical goals into low-level tasks, producing visualizations and deriving insights. Specifically, we introduce an LLM agent-based task planning and execution strategy, employing a recursive process involving a planner, executor, and controller. The planner is responsible for recommending and decomposing tasks, the executor handles task execution, including data analysis, visualization generation and multi-view composition, and the controller coordinates the interaction between the planner and executor. Building on the framework, we develop a system with a hybrid user interface that includes a task flow diagram for monitoring and managing the task planning process, a visualization panel for interactive data exploration, and a chat view for guiding the model through natural language instructions. We examine the effectiveness of our method through a usage scenario and an expert study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05555v1">AcceLLM: Accelerating LLM Inference using Redundancy for Load Balancing and Data Locality</a></div>
    <div class="paper-meta">
      📅 2024-11-08
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference on large-scale systems is expected to dominate future cloud infrastructures. Efficient LLM inference in cloud environments with numerous AI accelerators is challenging, necessitating extensive optimizations for optimal performance. Current systems batch prefill and decoding to boost throughput but encounter latency issues, while others disaggregate these phases, leading to resource underutilization. We propose AcceLLM, a novel method addressing latency and load balancing, inspired by the cache data management. It strategically utilizes redundant data to enhance inference via load balancing and optimal hardware use. Simulated evaluations on Nvidia H100 GPU and Huawei Ascend 910B2 show AcceLLM surpasses state-of-the-art systems up to 30% in latency and efficiency, handling diverse workloads effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05423v1">VISTA: Visual Integrated System for Tailored Automation in Math Problem Generation Using LLM</a></div>
    <div class="paper-meta">
      📅 2024-11-08
      | 💬 Accepted at NeurIPS 2024 Workshop on Large Foundation Models for Educational Assessment (FM-Assess)
    </div>
    <details class="paper-abstract">
      Generating accurate and consistent visual aids is a critical challenge in mathematics education, where visual representations like geometric shapes and functions play a pivotal role in enhancing student comprehension. This paper introduces a novel multi-agent framework that leverages Large Language Models (LLMs) to automate the creation of complex mathematical visualizations alongside coherent problem text. Our approach not only simplifies the generation of precise visual aids but also aligns these aids with the problem's core mathematical concepts, improving both problem creation and assessment. By integrating multiple agents, each responsible for distinct tasks such as numeric calculation, geometry validation, and visualization, our system delivers mathematically accurate and contextually relevant problems with visual aids. Evaluation across Geometry and Function problem types shows that our method significantly outperforms basic LLMs in terms of text coherence, consistency, relevance and similarity, while maintaining the essential geometrical and functional integrity of the original problems. Although some challenges remain in ensuring consistent visual outputs, our framework demonstrates the immense potential of LLMs in transforming the way educators generate and utilize visual aids in math education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04358v2">Robust and Efficient Fine-tuning of LLMs with Bayesian Reparameterization of Low-Rank Adaptation</a></div>
    <div class="paper-meta">
      📅 2024-11-08
      | 💬 48 pages, 10 figures, 10 tables, Code: https://github.com/LCS2-IIITD/MonteCLoRA
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are highly resource-intensive to fine-tune due to their enormous size. While low-rank adaptation is a prominent parameter-efficient fine-tuning approach, it suffers from sensitivity to hyperparameter choices, leading to instability in model performance on fine-tuning downstream tasks. This paper highlights the importance of effective parameterization in low-rank fine-tuning to reduce estimator variance and enhance the stability of final model outputs. We propose MonteCLoRA, an efficient fine-tuning technique, employing Monte Carlo estimation to learn an unbiased posterior estimation of low-rank parameters with low expected variance, which stabilizes fine-tuned LLMs with only O(1) additional parameters. MonteCLoRA shows significant improvements in accuracy and robustness, achieving up to 3.8% higher accuracy and 8.6% greater robustness than existing efficient fine-tuning methods on natural language understanding tasks with pre-trained RoBERTa-base. Furthermore, in generative tasks with pre-trained LLaMA-1-7B, MonteCLoRA demonstrates robust zero-shot performance with 50% lower variance than the contemporary efficient fine-tuning methods. The theoretical and empirical results presented in the paper underscore how parameterization and hyperpriors balance exploration-exploitation in the low-rank parametric space, therefore leading to more optimal and robust parameter estimation during efficient fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05349v1">Enhancing Cluster Resilience: LLM-agent Based Autonomous Intelligent Cluster Diagnosis System and Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2024-11-08
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) and related technologies such as Retrieval-Augmented Generation (RAG) and Diagram of Thought (DoT) have enabled the creation of autonomous intelligent systems capable of performing cluster diagnostics and troubleshooting. By integrating these technologies with self-play methodologies, we have developed an LLM-agent system designed to autonomously diagnose and resolve issues within AI clusters. Our innovations include a knowledge base tailored for cluster diagnostics, enhanced LLM algorithms, practical deployment strategies for agents, and a benchmark specifically designed for evaluating LLM capabilities in this domain. Through extensive experimentation across multiple dimensions, we have demonstrated the superiority of our system in addressing the challenges faced in cluster diagnostics, particularly in detecting and rectifying performance issues more efficiently and accurately than traditional methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05345v1">Reasoning Robustness of LLMs to Adversarial Typographical Errors</a></div>
    <div class="paper-meta">
      📅 2024-11-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning using Chain-of-Thought (CoT) prompting. However, CoT can be biased by users' instruction. In this work, we study the reasoning robustness of LLMs to typographical errors, which can naturally occur in users' queries. We design an Adversarial Typo Attack ($\texttt{ATA}$) algorithm that iteratively samples typos for words that are important to the query and selects the edit that is most likely to succeed in attacking. It shows that LLMs are sensitive to minimal adversarial typographical changes. Notably, with 1 character edit, Mistral-7B-Instruct's accuracy drops from 43.7% to 38.6% on GSM8K, while with 8 character edits the performance further drops to 19.2%. To extend our evaluation to larger and closed-source LLMs, we develop the $\texttt{R$^2$ATA}$ benchmark, which assesses models' $\underline{R}$easoning $\underline{R}$obustness to $\underline{\texttt{ATA}}$. It includes adversarial typographical questions derived from three widely used reasoning datasets-GSM8K, BBH, and MMLU-by applying $\texttt{ATA}$ to open-source LLMs. $\texttt{R$^2$ATA}$ demonstrates remarkable transferability and causes notable performance drops across multiple super large and closed-source LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16964v2">Exploring the LLM Journey from Cognition to Expression with Linear Representations</a></div>
    <div class="paper-meta">
      📅 2024-11-08
      | 💬 Published in ICML 2024
    </div>
    <details class="paper-abstract">
      This paper presents an in-depth examination of the evolution and interplay of cognitive and expressive capabilities in large language models (LLMs), with a specific focus on Baichuan-7B and Baichuan-33B, an advanced bilingual (Chinese and English) LLM series. We define and explore the model's cognitive and expressive capabilities through linear representations across three critical phases: Pretraining, Supervised Fine-Tuning (SFT), and Reinforcement Learning from Human Feedback (RLHF). Cognitive capability is defined as the quantity and quality of information conveyed by the neuron output vectors within the network, similar to the neural signal processing in human cognition. Expressive capability is defined as the model's capability to produce word-level output. Our findings unveil a sequential development pattern, where cognitive abilities are largely established during Pretraining, whereas expressive abilities predominantly advance during SFT and RLHF. Statistical analyses confirm a significant correlation between the two capabilities, suggesting that cognitive capacity may limit expressive potential. The paper also explores the theoretical underpinnings of these divergent developmental trajectories and their connection to the LLMs' architectural design. Moreover, we evaluate various optimization-independent strategies, such as few-shot learning and repeated sampling, which bridge the gap between cognitive and expressive capabilities. This research reveals the potential connection between the hidden space and the output space, contributing valuable insights into the interpretability and controllability of their training processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05328v1">Content Quality vs. Attention Allocation: An LLM-Based Case Study in Peer-to-peer Mental Health Networks</a></div>
    <div class="paper-meta">
      📅 2024-11-08
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      With the rise of social media and peer-to-peer networks, users increasingly rely on crowdsourced responses for information and assistance. However, the mechanisms used to rank and promote responses often prioritize and end up biasing in favor of timeliness over quality, which may result in suboptimal support for help-seekers. We analyze millions of responses to mental health-related posts, utilizing large language models (LLMs) to assess the multi-dimensional quality of content, including relevance, empathy, and cultural alignment, among other aspects. Our findings reveal a mismatch between content quality and attention allocation: earlier responses - despite being relatively lower in quality - receive disproportionately high fractions of upvotes and visibility due to platform ranking algorithms. We demonstrate that the quality of the top-ranked responses could be improved by up to 39 percent, and even the simplest re-ranking strategy could significantly improve the quality of top responses, highlighting the need for more nuanced ranking mechanisms that prioritize both timeliness and content quality, especially emotional engagement in online mental health communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05316v1">Exploring the Alignment Landscape: LLMs and Geometric Deep Models in Protein Representation</a></div>
    <div class="paper-meta">
      📅 2024-11-08
      | 💬 24 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Latent representation alignment has become a foundational technique for constructing multimodal large language models (MLLM) by mapping embeddings from different modalities into a shared space, often aligned with the embedding space of large language models (LLMs) to enable effective cross-modal understanding. While preliminary protein-focused MLLMs have emerged, they have predominantly relied on heuristic approaches, lacking a fundamental understanding of optimal alignment practices across representations. In this study, we explore the alignment of multimodal representations between LLMs and Geometric Deep Models (GDMs) in the protein domain. We comprehensively evaluate three state-of-the-art LLMs (Gemma2-2B, LLaMa3.1-8B, and LLaMa3.1-70B) with four protein-specialized GDMs (GearNet, GVP, ScanNet, GAT). Our work examines alignment factors from both model and protein perspectives, identifying challenges in current alignment methodologies and proposing strategies to improve the alignment process. Our key findings reveal that GDMs incorporating both graph and 3D structural information align better with LLMs, larger LLMs demonstrate improved alignment capabilities, and protein rarity significantly impacts alignment performance. We also find that increasing GDM embedding dimensions, using two-layer projection heads, and fine-tuning LLMs on protein-specific data substantially enhance alignment quality. These strategies offer potential enhancements to the performance of protein-related multimodal models. Our code and data are available at https://github.com/Tizzzzy/LLM-GDM-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05232v1">Abstract2Appendix: Academic Reviews Enhance LLM Long-Context Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 We share our latest dataset on https://github.com/findalexli/Abstract2Appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance across various tasks, yet their ability to handle long-context reading remains challenging. This study explores the effectiveness of leveraging high-quality academic peer review data for fine-tuning LLMs to enhance their long-context capabilities. We compare the Direct Preference Optimization (DPO) method with the Supervised Fine-Tuning (SFT) method, demonstrating DPO's superiority and data efficiency. Our experiments show that the fine-tuned model achieves a 4.04-point improvement over phi-3 and a 2.6\% increase on the Qasper benchmark using only 2000 samples. Despite facing limitations in data scale and processing costs, this study underscores the potential of DPO and high-quality data in advancing LLM performance. Additionally, the zero-shot benchmark results indicate that aggregated high-quality human reviews are overwhelmingly preferred over LLM-generated responses, even for the most capable models like GPT-4o. This suggests that high-quality human reviews are extremely rich in information, reasoning, and long-context retrieval, capabilities that even the most advanced models have not fully captured. These findings highlight the high utility of leveraging human reviews to further advance the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05209v1">Alopex: A Computational Framework for Enabling On-Device Function Calls with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-07
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has led to their increased integration into mobile devices for personalized assistance, which enables LLMs to call external API functions to enhance their performance. However, challenges such as data scarcity, ineffective question formatting, and catastrophic forgetting hinder the development of on-device LLM agents. To tackle these issues, we propose Alopex, a framework that enables precise on-device function calls using the Fox LLM. Alopex introduces a logic-based method for generating high-quality training data and a novel ``description-question-output'' format for fine-tuning, reducing risks of function information leakage. Additionally, a data mixing strategy is used to mitigate catastrophic forgetting, combining function call data with textbook datasets to enhance performance in various tasks. Experimental results show that Alopex improves function call accuracy and significantly reduces catastrophic forgetting, providing a robust solution for integrating function call capabilities into LLMs without manual intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05200v1">Toward Cultural Interpretability: A Linguistic Anthropological Framework for Describing and Evaluating Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 Accepted for publication in Big Data & Society, November 2, 2024
    </div>
    <details class="paper-abstract">
      This article proposes a new integration of linguistic anthropology and machine learning (ML) around convergent interests in both the underpinnings of language and making language technologies more socially responsible. While linguistic anthropology focuses on interpreting the cultural basis for human language use, the ML field of interpretability is concerned with uncovering the patterns that Large Language Models (LLMs) learn from human verbal behavior. Through the analysis of a conversation between a human user and an LLM-powered chatbot, we demonstrate the theoretical feasibility of a new, conjoint field of inquiry, cultural interpretability (CI). By focusing attention on the communicative competence involved in the way human users and AI chatbots co-produce meaning in the articulatory interface of human-computer interaction, CI emphasizes how the dynamic relationship between language and culture makes contextually sensitive, open-ended conversation possible. We suggest that, by examining how LLMs internally "represent" relationships between language and culture, CI can: (1) provide insight into long-standing linguistic anthropological questions about the patterning of those relationships; and (2) aid model developers and interface designers in improving value alignment between language models and stylistically diverse speakers and culturally diverse speech communities. Our discussion proposes three critical research axes: relativity, variation, and indexicality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21337v2">Fine-tuned Large Language Models (LLMs): Improved Prompt Injection Attacks Detection</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 I am requesting the withdrawal of my paper due to critical issues identified in the methodology/results that may impact its accuracy and reliability. I also plan to make substantial revisions that go beyond minor corrections
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming a popular tool as they have significantly advanced in their capability to tackle a wide range of language-based tasks. However, LLMs applications are highly vulnerable to prompt injection attacks, which poses a critical problem. These attacks target LLMs applications through using carefully designed input prompts to divert the model from adhering to original instruction, thereby it could execute unintended actions. These manipulations pose serious security threats which potentially results in data leaks, biased outputs, or harmful responses. This project explores the security vulnerabilities in relation to prompt injection attacks. To detect whether a prompt is vulnerable or not, we follows two approaches: 1) a pre-trained LLM, and 2) a fine-tuned LLM. Then, we conduct a thorough analysis and comparison of the classification performance. Firstly, we use pre-trained XLM-RoBERTa model to detect prompt injections using test dataset without any fine-tuning and evaluate it by zero-shot classification. Then, this proposed work will apply supervised fine-tuning to this pre-trained LLM using a task-specific labeled dataset from deepset in huggingface, and this fine-tuned model achieves impressive results with 99.13\% accuracy, 100\% precision, 98.33\% recall and 99.15\% F1-score thorough rigorous experimentation and evaluation. We observe that our approach is highly efficient in detecting prompt injection attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05185v1">PentestAgent: Incorporating LLM Agents to Automated Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 14 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Penetration testing is a critical technique for identifying security vulnerabilities, traditionally performed manually by skilled security specialists. This complex process involves gathering information about the target system, identifying entry points, exploiting the system, and reporting findings. Despite its effectiveness, manual penetration testing is time-consuming and expensive, often requiring significant expertise and resources that many organizations cannot afford. While automated penetration testing methods have been proposed, they often fall short in real-world applications due to limitations in flexibility, adaptability, and implementation. Recent advancements in large language models (LLMs) offer new opportunities for enhancing penetration testing through increased intelligence and automation. However, current LLM-based approaches still face significant challenges, including limited penetration testing knowledge and a lack of comprehensive automation capabilities. To address these gaps, we propose PentestAgent, a novel LLM-based automated penetration testing framework that leverages the power of LLMs and various LLM-based techniques like Retrieval Augmented Generation (RAG) to enhance penetration testing knowledge and automate various tasks. Our framework leverages multi-agent collaboration to automate intelligence gathering, vulnerability analysis, and exploitation stages, reducing manual intervention. We evaluate PentestAgent using a comprehensive benchmark, demonstrating superior performance in task completion and overall efficiency. This work significantly advances the practical applicability of automated penetration testing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02378v2">On the Intrinsic Self-Correction Capability of LLMs: Uncertainty and Latent Concept</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 21 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are able to improve their responses when instructed to do so, a capability known as self-correction. When instructions provide only the task's goal without specific details about potential issues in the response, LLMs must rely on their internal knowledge to improve response quality, a process referred to as intrinsic self-correction. The empirical success of intrinsic self-correction is evident in various applications, but how and why it is effective remains unknown. In this paper, we unveil that intrinsic self-correction can be progressively improved, allowing it to approach a converged state. Our findings are verified in: (1) the scenario of multi-round question answering, by comprehensively demonstrating that intrinsic self-correction can progressively introduce performance gains through iterative interactions, ultimately converging to stable performance; and (2) the context of intrinsic self-correction for enhanced morality, in which we provide empirical evidence that iteratively applying instructions reduces model uncertainty towards convergence, which then leads to convergence of both the calibration error and self-correction performance, ultimately resulting in a stable state of intrinsic self-correction. Furthermore, we introduce a mathematical formulation and a simulation task indicating that the latent concepts activated by self-correction instructions drive the reduction of model uncertainty. Based on our experimental results and analysis of the convergence of intrinsic self-correction, we reveal its underlying mechanism: consistent injected instructions reduce model uncertainty which yields converged, improved performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16971v3">AIOS: LLM Agent Operating System</a></div>
    <div class="paper-meta">
      📅 2024-11-07
    </div>
    <details class="paper-abstract">
      LLM-based intelligent agents face significant deployment challenges, particularly related to resource management. Allowing unrestricted access to LLM or tool resources can lead to inefficient or even potentially harmful resource allocation and utilization for agents. Furthermore, the absence of proper scheduling and resource management mechanisms in current agent designs hinders concurrent processing and limits overall system efficiency. As the diversity and complexity of agents continue to grow, addressing these resource management issues becomes increasingly critical to LLM-based agent systems. To address these challenges, this paper proposes the architecture of AIOS (LLM-based AI Agent Operating System) under the context of managing LLM-based agents. It introduces a novel architecture for serving LLM-based agents by isolating resources and LLM-specific services from agent applications into an AIOS kernel. This AIOS kernel provides fundamental services (e.g., scheduling, context management, memory management, storage management, access control) and efficient management of resources (e.g., LLM and external tools) for runtime agents. To enhance usability, AIOS also includes an AIOS-Agent SDK, a comprehensive suite of APIs designed for utilizing functionalities provided by the AIOS kernel. Experimental results demonstrate that using AIOS can achieve up to 2.1x faster execution for serving agents built by various agent frameworks. The source code is available at https://github.com/agiresearch/AIOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00922v3">MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 29 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Users typically engage with LLMs interactively, yet most existing benchmarks evaluate them in a static, single-turn format, posing reliability concerns in interactive scenarios. We identify a key obstacle towards reliability: LLMs are trained to answer any question, even with incomplete context or insufficient knowledge. In this paper, we propose to change the static paradigm to an interactive one, develop systems that proactively ask questions to gather more information and respond reliably, and introduce an benchmark - MediQ - to evaluate question-asking ability in LLMs. MediQ simulates clinical interactions consisting of a Patient System and an adaptive Expert System; with potentially incomplete initial information, the Expert refrains from making diagnostic decisions when unconfident, and instead elicits missing details via follow-up questions. We provide a pipeline to convert single-turn medical benchmarks into an interactive format. Our results show that directly prompting state-of-the-art LLMs to ask questions degrades performance, indicating that adapting LLMs to proactive information-seeking settings is nontrivial. We experiment with abstention strategies to better estimate model confidence and decide when to ask questions, improving diagnostic accuracy by 22.3%; however, performance still lags compared to an (unrealistic in practice) upper bound with complete information upfront. Further analyses show improved interactive performance with filtering irrelevant contexts and reformatting conversations. Overall, we introduce a novel problem towards LLM reliability, an interactive MediQ benchmark and a novel question-asking system, and highlight directions to extend LLMs' information-seeking abilities in critical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05000v1">Needle Threading: Can LLMs Follow Threads through Near-Million-Scale Haystacks?</a></div>
    <div class="paper-meta">
      📅 2024-11-07
    </div>
    <details class="paper-abstract">
      As the context limits of Large Language Models (LLMs) increase, the range of possible applications and downstream functions broadens. In many real-world tasks, decisions depend on details scattered across collections of often disparate documents containing mostly irrelevant information. Long-context LLMs appear well-suited to this form of complex information retrieval and reasoning, which has traditionally proven costly and time-consuming. However, although the development of longer context models has seen rapid gains in recent years, our understanding of how effectively LLMs use their context has not kept pace. To address this, we conduct a set of retrieval experiments designed to evaluate the capabilities of 17 leading LLMs, such as their ability to follow threads of information through the context window. Strikingly, we find that many models are remarkably threadsafe: capable of simultaneously following multiple threads without significant loss in performance. Still, for many models, we find the effective context limit is significantly shorter than the supported context length, with accuracy decreasing as the context window grows. Our study also highlights the important point that token counts from different tokenizers should not be directly compared -- they often correspond to substantially different numbers of written characters. We release our code and long-context experimental data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04965v1">BitNet a4.8: 4-bit Activations for 1-bit LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Recent research on the 1-bit Large Language Models (LLMs), such as BitNet b1.58, presents a promising direction for reducing the inference cost of LLMs while maintaining their performance. In this work, we introduce BitNet a4.8, enabling 4-bit activations for 1-bit LLMs. BitNet a4.8 employs a hybrid quantization and sparsification strategy to mitigate the quantization errors introduced by the outlier channels. Specifically, we utilize 4-bit activations for inputs to the attention and feed-forward network layers, while sparsifying intermediate states followed with 8-bit quantization. Extensive experiments demonstrate that BitNet a4.8 achieves performance comparable to BitNet b1.58 with equivalent training costs, while being faster in inference with enabling 4-bit (INT4/FP4) kernels. Additionally, BitNet a4.8 activates only 55% of parameters and supports 3-bit KV cache, further enhancing the efficiency of large-scale LLM deployment and inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02472v3">Meta-Models: An Architecture for Decoding LLM Behaviors Through Interpreted Embeddings and Natural Language</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly integrated into our daily lives, the potential harms from deceptive behavior underlie the need for faithfully interpreting their decision-making. While traditional probing methods have shown some effectiveness, they remain best for narrowly scoped tasks while more comprehensive explanations are still necessary. To this end, we investigate meta-models-an architecture using a "meta-model" that takes activations from an "input-model" and answers natural language questions about the input-model's behaviors. We evaluate the meta-model's ability to generalize by training them on selected task types and assessing their out-of-distribution performance in deceptive scenarios. Our findings show that meta-models generalize well to out-of-distribution tasks and point towards opportunities for future research in this area. Our code is available at https://github.com/acostarelli/meta-models-public .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13835v2">Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-07
    </div>
    <details class="paper-abstract">
      Practitioners have consistently observed three puzzling phenomena in transformer-based large language models (LLMs): attention sinks, value-state drains, and residual-state peaks, collectively referred to as extreme-token phenomena. These phenomena are characterized by certain so-called "sink tokens" receiving disproportionately high attention weights, exhibiting significantly smaller value states, and having much larger residual-state norms than those of other tokens. These extreme tokens give rise to various challenges in LLM inference, quantization, and interpretability. We elucidate the mechanisms behind extreme-token phenomena. First, we show that these phenomena arise in very simple architectures -- transformers with one to three layers -- trained on a toy model, the Bigram-Backcopy (BB) task. In this setting, we identify an active-dormant mechanism, where attention heads become sinks for specific input domains while remaining non-sinks for others. Our theoretical analysis of the training dynamics reveals that these phenomena are driven by a mutual reinforcement mechanism. Building on these insights, we propose strategies to mitigate extreme-token phenomena during pretraining, including replacing softmax with ReLU and Adam with SGD. Next, we extend our analysis to pretrained LLMs, including Llama and OLMo, showing that many attention heads exhibit a similar active-dormant mechanism as in the BB task, and that the mutual reinforcement mechanism also governs the emergence of extreme-token phenomena during LLM pretraining. Our results reveal that many of the static and dynamic properties of extreme-token phenomena predicted by the BB task align with observations in pretrained LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11892v1">Green My LLM: Studying the key factors affecting the energy consumption of code assistants</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 Submitted to JSS
    </div>
    <details class="paper-abstract">
      In recent years,Large Language Models (LLMs) have significantly improved in generating high-quality code, enabling their integration into developers' Integrated Development Environments (IDEs) as code assistants. These assistants, such as GitHub Copilot, deliver real-time code suggestions and can greatly enhance developers' productivity. However, the environmental impact of these tools, in particular their energy consumption, remains a key concern. This paper investigates the energy consumption of LLM-based code assistants by simulating developer interactions with GitHub Copilot and analyzing various configuration factors. We collected a dataset of development traces from 20 developers and conducted extensive software project development simulations to measure energy usage under different scenarios. Our findings reveal that the energy consumption and performance of code assistants are influenced by various factors, such as the number of concurrent developers, model size, quantization methods, and the use of streaming. Notably, a substantial portion of generation requests made by GitHub Copilot is either canceled or rejected by developers, indicating a potential area for reducing wasted computations. Based on these findings, we share actionable insights into optimizing configurations for different use cases, demonstrating that careful adjustments can lead to significant energy savings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14331v2">ChartifyText: Automated Chart Generation from Data-Involved Texts via LLM</a></div>
    <div class="paper-meta">
      📅 2024-11-07
    </div>
    <details class="paper-abstract">
      Text documents with numerical values involved are widely used in various applications such as scientific research, economy, public health and journalism. However, it is difficult for readers to quickly interpret such data-involved texts and gain deep insights. To fill this research gap, this work aims to automatically generate charts to accurately convey the underlying data and ideas to readers, which is essentially a challenging task. The challenges originate from text ambiguities, intrinsic sparsity and uncertainty of data in text documents, and subjective sentiment differences. Specifically, we propose ChartifyText, a novel fully-automated approach that leverages Large Language Models (LLMs) to convert complex data-involved texts to expressive charts. It consists of two major modules: tabular data inference and expressive chart generation. The tabular data inference module employs systematic prompt engineering to guide the LLM (e.g., GPT-4) to infer table data, where data ranges, uncertainties, missing data values and corresponding subjective sentiments are explicitly considered. The expressive chart generation module augments standard charts with intuitive visual encodings and concise texts to accurately convey the underlying data and insights. We extensively evaluate the effectiveness of ChartifyText on real-world data-involved text documents through case studies, in-depth interviews with three visualization experts, and a carefully-designed user study with 15 participants. The results demonstrate the usefulness and effectiveness of ChartifyText in helping readers efficiently and effectively make sense of data-involved texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04708v1">Exploring Hierarchical Molecular Graph Representation in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-07
    </div>
    <details class="paper-abstract">
      Following the milestones in large language models (LLMs) and multimodal models, we have seen a surge in applying LLMs to biochemical tasks. Leveraging graph features and molecular text representations, LLMs can tackle various tasks, such as predicting chemical reaction outcomes and describing molecular properties. However, most current work overlooks the multi-level nature of graph features. The impact of different feature levels on LLMs and the importance of each level remain unexplored, and it is possible that different chemistry tasks require different feature levels. In this work, we first investigate the effect of feature granularity by fusing GNN-generated feature tokens, discovering that even reducing all tokens to a single token does not significantly impact performance. We then explore the effect of various feature levels on performance, finding that both the quality of LLM-generated molecules and performance on different tasks benefit from different feature levels. We conclude with two key insights: (1) current molecular Multimodal LLMs(MLLMs) lack a comprehensive understanding of graph features, and (2) static processing is not sufficient for hierarchical graph feature. Our code will be publicly available soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04704v1">Distinguishing LLM-generated from Human-written Code by Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 30 pages, 6 figures, Accepted by TOSEM'24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), such as ChatGPT released by OpenAI, have attracted significant attention from both industry and academia due to their demonstrated ability to generate high-quality content for various tasks. Despite the impressive capabilities of LLMs, there are growing concerns regarding their potential risks in various fields, such as news, education, and software engineering. Recently, several commercial and open-source LLM-generated content detectors have been proposed, which, however, are primarily designed for detecting natural language content without considering the specific characteristics of program code. This paper aims to fill this gap by proposing a novel ChatGPT-generated code detector, CodeGPTSensor, based on a contrastive learning framework and a semantic encoder built with UniXcoder. To assess the effectiveness of CodeGPTSensor on differentiating ChatGPT-generated code from human-written code, we first curate a large-scale Human and Machine comparison Corpus (HMCorp), which includes 550K pairs of human-written and ChatGPT-generated code (i.e., 288K Python code pairs and 222K Java code pairs). Based on the HMCorp dataset, our qualitative and quantitative analysis of the characteristics of ChatGPT-generated code reveals the challenge and opportunity of distinguishing ChatGPT-generated code from human-written code with their representative features. Our experimental results indicate that CodeGPTSensor can effectively identify ChatGPT-generated code, outperforming all selected baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04671v1">CUIfy the XR: An Open-Source Package to Embed LLM-powered Conversational Agents in XR</a></div>
    <div class="paper-meta">
      📅 2024-11-07
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Recent developments in computer graphics, machine learning, and sensor technologies enable numerous opportunities for extended reality (XR) setups for everyday life, from skills training to entertainment. With large corporations offering consumer-grade head-mounted displays (HMDs) in an affordable way, it is likely that XR will become pervasive, and HMDs will develop as personal devices like smartphones and tablets. However, having intelligent spaces and naturalistic interactions in XR is as important as technological advances so that users grow their engagement in virtual and augmented spaces. To this end, large language model (LLM)--powered non-player characters (NPCs) with speech-to-text (STT) and text-to-speech (TTS) models bring significant advantages over conventional or pre-scripted NPCs for facilitating more natural conversational user interfaces (CUIs) in XR. In this paper, we provide the community with an open-source, customizable, extensible, and privacy-aware Unity package, CUIfy, that facilitates speech-based NPC-user interaction with various LLMs, STT, and TTS models. Our package also supports multiple LLM-powered NPCs per environment and minimizes the latency between different computational models through streaming to achieve usable interactions between users and NPCs. We publish our source code in the following repository: https://gitlab.lrz.de/hctl/cuify
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14125v3">ALI-Agent: Assessing LLMs' Alignment with Human Values via Agent-based Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-11-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can elicit unintended and even harmful content when misaligned with human values, posing severe risks to users and society. To mitigate these risks, current evaluation benchmarks predominantly employ expert-designed contextual scenarios to assess how well LLMs align with human values. However, the labor-intensive nature of these benchmarks limits their test scope, hindering their ability to generalize to the extensive variety of open-world use cases and identify rare but crucial long-tail risks. Additionally, these static tests fail to adapt to the rapid evolution of LLMs, making it hard to evaluate timely alignment issues. To address these challenges, we propose ALI-Agent, an evaluation framework that leverages the autonomous abilities of LLM-powered agents to conduct in-depth and adaptive alignment assessments. ALI-Agent operates through two principal stages: Emulation and Refinement. During the Emulation stage, ALI-Agent automates the generation of realistic test scenarios. In the Refinement stage, it iteratively refines the scenarios to probe long-tail risks. Specifically, ALI-Agent incorporates a memory module to guide test scenario generation, a tool-using module to reduce human labor in tasks such as evaluating feedback from target LLMs, and an action module to refine tests. Extensive experiments across three aspects of human values--stereotypes, morality, and legality--demonstrate that ALI-Agent, as a general evaluation framework, effectively identifies model misalignment. Systematic analysis also validates that the generated test scenarios represent meaningful use cases, as well as integrate enhanced measures to probe long-tail risks. Our code is available at https://github.com/SophieZheng998/ALI-Agent.git
    </details>
</div>
