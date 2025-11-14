# llm - 2024_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05874v2">LLM-Based Robust Product Classification in Commerce and Compliance</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Camera-ready version for Customizable NLP Workshop at EMNLP 2024. 11 pages
    </div>
    <details class="paper-abstract">
      Product classification is a crucial task in international trade, as compliance regulations are verified and taxes and duties are applied based on product categories. Manual classification of products is time-consuming and error-prone, and the sheer volume of products imported and exported renders the manual process infeasible. Consequently, e-commerce platforms and enterprises involved in international trade have turned to automatic product classification using machine learning. However, current approaches do not consider the real-world challenges associated with product classification, such as very abbreviated and incomplete product descriptions. In addition, recent advancements in generative Large Language Models (LLMs) and their reasoning capabilities are mainly untapped in product classification and e-commerce. In this research, we explore the real-life challenges of industrial classification and we propose data perturbations that allow for realistic data simulation. Furthermore, we employ LLM-based product classification to improve the robustness of the prediction in presence of incomplete data. Our research shows that LLMs with in-context learning outperform the supervised approaches in the clean-data scenario. Additionally, we illustrate that LLMs are significantly more robust than the supervised approaches when data attacks are present.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.18679v4">Data Interpreter: An LLM Agent For Data Science</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have shown effectiveness across many applications. However, their use in data science scenarios requiring solving long-term interconnected tasks, dynamic data adjustments and domain expertise remains challenging. Previous approaches primarily focus on individual tasks, making it difficult to assess the complete data science workflow. Moreover, they struggle to handle real-time changes in intermediate data and fail to adapt dynamically to evolving task dependencies inherent to data science problems. In this paper, we present Data Interpreter, an LLM-based agent designed to automatically solve various data science problems end-to-end. Our Data Interpreter incorporates two key modules: 1) Hierarchical Graph Modeling, which breaks down complex problems into manageable subproblems, enabling dynamic node generation and graph optimization; and 2) Programmable Node Generation, a technique that refines and verifies each subproblem to iteratively improve code generation results and robustness. Extensive experiments consistently demonstrate the superiority of Data Interpreter. On InfiAgent-DABench, it achieves a 25% performance boost, raising accuracy from 75.9% to 94.9%. For machine learning and open-ended tasks, it improves performance from 88% to 95%, and from 60% to 97%, respectively. Moreover, on the MATH dataset, Data Interpreter achieves remarkable performance with a 26% improvement compared to state-of-the-art baselines. The code is available at https://github.com/geekan/MetaGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11674v1">LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Time series forecasting using LLMs
    </div>
    <details class="paper-abstract">
      Time series forecasting remains a challenging task, particularly in the context of complex multiscale temporal patterns. This study presents LLM-Mixer, a framework that improves forecasting accuracy through the combination of multiscale time-series decomposition with pre-trained LLMs (Large Language Models). LLM-Mixer captures both short-term fluctuations and long-term trends by decomposing the data into multiple temporal resolutions and processing them with a frozen LLM, guided by a textual prompt specifically designed for time-series data. Extensive experiments conducted on multivariate and univariate datasets demonstrate that LLM-Mixer achieves competitive performance, outperforming recent state-of-the-art models across various forecasting horizons. This work highlights the potential of combining multiscale analysis and LLMs for effective and scalable time-series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11672v1">Leaving the barn door open for Clever Hans: Simple features predict LLM benchmark answers</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      The integrity of AI benchmarks is fundamental to accurately assess the capabilities of AI systems. The internal validity of these benchmarks - i.e., making sure they are free from confounding factors - is crucial for ensuring that they are measuring what they are designed to measure. In this paper, we explore a key issue related to internal validity: the possibility that AI systems can solve benchmarks in unintended ways, bypassing the capability being tested. This phenomenon, widely known in human and animal experiments, is often referred to as the 'Clever Hans' effect, where tasks are solved using spurious cues, often involving much simpler processes than those putatively assessed. Previous research suggests that language models can exhibit this behaviour as well. In several older Natural Language Processing (NLP) benchmarks, individual $n$-grams like "not" have been found to be highly predictive of the correct labels, and supervised NLP models have been shown to exploit these patterns. In this work, we investigate the extent to which simple $n$-grams extracted from benchmark instances can be combined to predict labels in modern multiple-choice benchmarks designed for LLMs, and whether LLMs might be using such $n$-gram patterns to solve these benchmarks. We show how simple classifiers trained on these $n$-grams can achieve high scores on several benchmarks, despite lacking the capabilities being tested. Additionally, we provide evidence that modern LLMs might be using these superficial patterns to solve benchmarks. This suggests that the internal validity of these benchmarks may be compromised and caution should be exercised when interpreting LLM performance results on them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11594v1">Black-box Uncertainty Quantification Method for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge is a widely used method for evaluating the performance of Large Language Models (LLMs) across various tasks. We address the challenge of quantifying the uncertainty of LLM-as-a-Judge evaluations. While uncertainty quantification has been well-studied in other domains, applying it effectively to LLMs poses unique challenges due to their complex decision-making capabilities and computational demands. In this paper, we introduce a novel method for quantifying uncertainty designed to enhance the trustworthiness of LLM-as-a-Judge evaluations. The method quantifies uncertainty by analyzing the relationships between generated assessments and possible ratings. By cross-evaluating these relationships and constructing a confusion matrix based on token probabilities, the method derives labels of high or low uncertainty. We evaluate our method across multiple benchmarks, demonstrating a strong correlation between the accuracy of LLM evaluations and the derived uncertainty scores. Our findings suggest that this method can significantly improve the reliability and consistency of LLM-as-a-Judge evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11531v1">AGENTiGraph: An Interactive Knowledge Graph Platform for LLM-based Chatbots Utilizing Private Data</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 30 pages, 7 figures; Submitted to COLING 2025 System Demonstrations Track
    </div>
    <details class="paper-abstract">
      Large Language Models~(LLMs) have demonstrated capabilities across various applications but face challenges such as hallucination, limited reasoning abilities, and factual inconsistencies, especially when tackling complex, domain-specific tasks like question answering~(QA). While Knowledge Graphs~(KGs) have been shown to help mitigate these issues, research on the integration of LLMs with background KGs remains limited. In particular, user accessibility and the flexibility of the underlying KG have not been thoroughly explored. We introduce AGENTiGraph (Adaptive Generative ENgine for Task-based Interaction and Graphical Representation), a platform for knowledge management through natural language interaction. It integrates knowledge extraction, integration, and real-time visualization. AGENTiGraph employs a multi-agent architecture to dynamically interpret user intents, manage tasks, and integrate new knowledge, ensuring adaptability to evolving user requirements and data contexts. Our approach demonstrates superior performance in knowledge graph interactions, particularly for complex domain-specific tasks. Experimental results on a dataset of 3,500 test cases show AGENTiGraph significantly outperforms state-of-the-art zero-shot baselines, achieving 95.12\% accuracy in task classification and 90.45\% success rate in task execution. User studies corroborate its effectiveness in real-world scenarios. To showcase versatility, we extended AGENTiGraph to legislation and healthcare domains, constructing specialized KGs capable of answering complex queries in legal and medical contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11526v1">Human-LLM Collaborative Construction of a Cantonese Emotion Lexicon</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in language understanding and generation. Advanced utilization of the knowledge embedded in LLMs for automated annotation has consistently been explored. This study proposed to develop an emotion lexicon for Cantonese, a low-resource language, through collaborative efforts between LLM and human annotators. By integrating emotion labels provided by LLM and human annotators, the study leveraged existing linguistic resources including lexicons in other languages and local forums to construct a Cantonese emotion lexicon enriched with colloquial expressions. The consistency of the proposed emotion lexicon in emotion extraction was assessed through modification and utilization of three distinct emotion text datasets. This study not only validates the efficacy of the constructed lexicon but also emphasizes that collaborative annotation between human and artificial intelligence can significantly enhance the quality of emotion labels, highlighting the potential of such partnerships in facilitating natural language processing tasks for low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11410v1">PMMT: Preference Alignment in Multilingual Machine Translation via LLM Distillation</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Translation is important for cross-language communication, and many efforts have been made to improve its accuracy. However, less investment is conducted in aligning translations with human preferences, such as translation tones or styles. In this paper, a new method is proposed to effectively generate large-scale multilingual parallel corpora with specific translation preferences using Large Language Models (LLMs). Meanwhile, an automatic pipeline is designed to distill human preferences into smaller Machine Translation (MT) models for efficiently and economically supporting large-scale calls in online services. Experiments indicate that the proposed method takes the lead in translation tasks with aligned human preferences by a large margin. Meanwhile, on popular public benchmarks like WMT and Flores, on which our models were not trained, the proposed method also shows a competitive performance compared to SOTA works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11404v1">MoChat: Joints-Grouped Spatio-Temporal Grounding LLM for Multi-Turn Motion Comprehension and Description</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Despite continuous advancements in deep learning for understanding human motion, existing models often struggle to accurately identify action timing and specific body parts, typically supporting only single-round interaction. Such limitations in capturing fine-grained motion details reduce their effectiveness in motion understanding tasks. In this paper, we propose MoChat, a multimodal large language model capable of spatio-temporal grounding of human motion and understanding multi-turn dialogue context. To achieve these capabilities, we group the spatial information of each skeleton frame based on human anatomical structure and then apply them with Joints-Grouped Skeleton Encoder, whose outputs are combined with LLM embeddings to create spatio-aware and temporal-aware embeddings separately. Additionally, we develop a pipeline for extracting timestamps from skeleton sequences based on textual annotations, and construct multi-turn dialogues for spatially grounding. Finally, various task instructions are generated for jointly training. Experimental results demonstrate that MoChat achieves state-of-the-art performance across multiple metrics in motion understanding tasks, making it as the first model capable of fine-grained spatio-temporal grounding of human motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11385v1">Do LLMs Have the Generalization Ability in Conducting Causal Inference?</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      In causal inference, generalization capability refers to the ability to conduct causal inference methods on new data to estimate the causal-effect between unknown phenomenon, which is crucial for expanding the boundaries of knowledge. Studies have evaluated the causal inference capabilities of Large Language Models (LLMs) concerning known phenomena, yet the generalization capabilities of LLMs concerning unseen phenomena remain unexplored. In this paper, we selected four tasks: Causal Path Discovery (CP), Backdoor Adjustment (BA), Factual Inference (FI), and Counterfactual Inference (CI) as representatives of causal inference tasks. To generate evaluation questions about previously unseen phenomena in new data on the four tasks, we propose a benchmark generation framework, which employs randomly generated graphs and node names to formulate questions within hypothetical new causal scenarios. Based on this framework, we compile a benchmark dataset of varying levels of question complexity. We extensively tested the generalization capabilities of five leading LLMs across four tasks. Experiment results reveal that while LLMs exhibit good generalization performance in solving simple CP, FI, and complex CI questions, they encounter difficulties when tackling BA questions and face obvious performance fluctuations as the problem complexity changes. Furthermore, when the names of phenomena incorporate existing terms, even if these names are entirely novel, their generalization performance can still be hindered by interference from familiar terms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11381v1">Survey and Evaluation of Converging Architecture in LLMs based on Footsteps of Operations</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 13 pages and 16 figures
    </div>
    <details class="paper-abstract">
      The advent of the Attention mechanism and Transformer architecture enables contextually natural text generation and compresses the burden of processing entire source information into singular vectors. Based on these two main ideas, model sizes gradually increases to accommodate more precise and comprehensive information, leading to the current state-of-the-art LLMs being very large, with parameters around 70 billion. As the model sizes are growing, the demand for substantial storage and computational capacity increases. This leads to the development of high-bandwidth memory and accelerators, as well as a variety of model architectures designed to meet these requirements. We note that LLM architectures have increasingly converged. This paper analyzes how these converged architectures perform in terms of layer configurations, operational mechanisms, and model sizes, considering various hyperparameter settings. In this paper, we conduct a concise survey of the history of LLMs by tracing the evolution of their operational improvements. Furthermore, we summarize the performance trends of LLMs under various hyperparameter settings using the RTX 6000, which features the state-of-the-art Ada Lovelace architecture. We conclude that even the same model can exhibit different behaviors depending on the hyperparameters or whether it is deployed in server or edge environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14963v4">Achieving >97% on GSM8K: Deeply Understanding the Problems Makes LLMs Better Solvers for Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has enhanced the performance of Large Language Models (LLMs) across various reasoning tasks. However, CoT still falls short in dealing with complex math word problems, as it usually suffers from three pitfalls: semantic misunderstanding errors, calculation errors, and step-missing errors. Prior studies involve addressing the calculation errors and step-missing errors, but neglect the semantic misunderstanding errors, which is the major factor limiting the reasoning performance of LLMs. To this end, we propose a simple-yet-effective method, namely Deeply Understanding the Problems (DUP), to improve the LLMs' math problem-solving ability by addressing semantic misunderstanding errors. The core of our method is to encourage the LLMs to deeply understand the problems and extract the key problem-solving information used for better reasoning. Extensive experiments on 10 diverse reasoning benchmarks show that our DUP method consistently outperforms the other counterparts by a large margin. More encouragingly, DUP achieves a new SOTA result on the GSM8K benchmark, with an accuracy of 97.1% under the zero-shot setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12878v3">Do LLMs have Consistent Values?</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 16 pages, 4 figures, and there are more in the appendix
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) technology is constantly improving towards human-like dialogue. Values are a basic driving force underlying human behavior, but little research has been done to study the values exhibited in text generated by LLMs. Here we study this question by turning to the rich literature on value structure in psychology. We ask whether LLMs exhibit the same value structure that has been demonstrated in humans, including the ranking of values, and correlation between values. We show that the results of this analysis depend on how the LLM is prompted, and that under a particular prompting strategy (referred to as "Value Anchoring") the agreement with human data is quite compelling. Our results serve both to improve our understanding of values in LLMs, as well as introduce novel methods for assessing consistency in LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11327v1">Sequential LLM Framework for Fashion Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      The fashion industry is one of the leading domains in the global e-commerce sector, prompting major online retailers to employ recommendation systems for product suggestions and customer convenience. While recommendation systems have been widely studied, most are designed for general e-commerce problems and struggle with the unique challenges of the fashion domain. To address these issues, we propose a sequential fashion recommendation framework that leverages a pre-trained large language model (LLM) enhanced with recommendation-specific prompts. Our framework employs parameter-efficient fine-tuning with extensive fashion data and introduces a novel mix-up-based retrieval technique for translating text into relevant product suggestions. Extensive experiments show our proposed framework significantly enhances fashion recommendation performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09318v2">Impeding LLM-assisted Cheating in Introductory Programming Assignments via Adversarial Perturbation</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      While Large language model (LLM)-based programming assistants such as CoPilot and ChatGPT can help improve the productivity of professional software developers, they can also facilitate cheating in introductory computer programming courses. Assuming instructors have limited control over the industrial-strength models, this paper investigates the baseline performance of 5 widely used LLMs on a collection of introductory programming problems, examines adversarial perturbations to degrade their performance, and describes the results of a user study aimed at understanding the efficacy of such perturbations in hindering actual code generation for introductory programming assignments. The user study suggests that i) perturbations combinedly reduced the average correctness score by 77%, ii) the drop in correctness caused by these perturbations was affected based on their detectability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11283v1">AdvBDGen: Adversarially Fortified Prompt-Specific Fuzzy Backdoor Generator Against LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Published at the Neurips Safe Generative AI Workshop 2024
    </div>
    <details class="paper-abstract">
      With the growing adoption of reinforcement learning with human feedback (RLHF) for aligning large language models (LLMs), the risk of backdoor installation during alignment has increased, leading to unintended and harmful behaviors. Existing backdoor triggers are typically limited to fixed word patterns, making them detectable during data cleaning and easily removable post-poisoning. In this work, we explore the use of prompt-specific paraphrases as backdoor triggers, enhancing their stealth and resistance to removal during LLM alignment. We propose AdvBDGen, an adversarially fortified generative fine-tuning framework that automatically generates prompt-specific backdoors that are effective, stealthy, and transferable across models. AdvBDGen employs a generator-discriminator pair, fortified by an adversary, to ensure the installability and stealthiness of backdoors. It enables the crafting and successful installation of complex triggers using as little as 3% of the fine-tuning data. Once installed, these backdoors can jailbreak LLMs during inference, demonstrate improved stability against perturbations compared to traditional constant triggers, and are more challenging to remove. These findings underscore an urgent need for the research community to develop more robust defenses against adversarial backdoor threats in LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.12052v1">Skill-LLM: Repurposing General-Purpose LLMs for Skill Extraction</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Accurate skill extraction from job descriptions is crucial in the hiring process but remains challenging. Named Entity Recognition (NER) is a common approach used to address this issue. With the demonstrated success of large language models (LLMs) in various NLP tasks, including NER, we propose fine-tuning a specialized Skill-LLM and a light weight model to improve the precision and quality of skill extraction. In our study, we evaluated the fine-tuned Skill-LLM and the light weight model using a benchmark dataset and compared its performance against state-of-the-art (SOTA) methods. Our results show that this approach outperforms existing SOTA techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11143v1">LLM Unlearning via Loss Adjustment with Only Forget Data</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 Paper under review
    </div>
    <details class="paper-abstract">
      Unlearning in Large Language Models (LLMs) is essential for ensuring ethical and responsible AI use, especially in addressing privacy leak, bias, safety, and evolving regulations. Existing approaches to LLM unlearning often rely on retain data or a reference LLM, yet they struggle to adequately balance unlearning performance with overall model utility. This challenge arises because leveraging explicit retain data or implicit knowledge of retain data from a reference LLM to fine-tune the model tends to blur the boundaries between the forgotten and retain data, as different queries often elicit similar responses. In this work, we propose eliminating the need to retain data or the reference LLM for response calibration in LLM unlearning. Recognizing that directly applying gradient ascent on the forget data often leads to optimization instability and poor performance, our method guides the LLM on what not to respond to, and importantly, how to respond, based on the forget data. Hence, we introduce Forget data only Loss AjustmenT (FLAT), a "flat" loss adjustment approach which addresses these issues by maximizing f-divergence between the available template answer and the forget answer only w.r.t. the forget data. The variational form of the defined f-divergence theoretically provides a way of loss adjustment by assigning different importance weights for the learning w.r.t. template responses and the forgetting of responses subject to unlearning. Empirical results demonstrate that our approach not only achieves superior unlearning performance compared to existing methods but also minimizes the impact on the model's retained capabilities, ensuring high utility across diverse tasks, including copyrighted content unlearning on Harry Potter dataset and MUSE Benchmark, and entity unlearning on the TOFU dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05440v2">Can LLMs Understand Time Series Anomalies?</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained popularity in time series forecasting, but their potential for anomaly detection remains largely unexplored. Our study investigates whether LLMs can understand and detect anomalies in time series data, focusing on zero-shot and few-shot scenarios. Inspired by conjectures about LLMs' behavior from time series forecasting research, we formulate key hypotheses about LLMs' capabilities in time series anomaly detection. We design and conduct principled experiments to test each of these hypotheses. Our investigation reveals several surprising findings about LLMs for time series: 1. LLMs understand time series better as images rather than as text 2. LLMs did not demonstrate enhanced performance when prompted to engage in explicit reasoning about time series analysis 3. Contrary to common beliefs, LLM's understanding of time series do not stem from their repetition biases or arithmetic abilities 4. LLMs' behaviors and performance in time series analysis vary significantly across different model architectures This study provides the first comprehensive analysis of contemporary LLM capabilities in time series anomaly detection. Our results suggest that while LLMs can understand time series anomalies, many common conjectures based on their reasoning capabilities do not hold. Our code and data are available at `https://github.com/Rose-STL-Lab/AnomLLM/`.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11114v1">Active Learning for Robust and Representative LLM Generation in Safety-Critical Scenarios</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Ensuring robust safety measures across a wide range of scenarios is crucial for user-facing systems. While Large Language Models (LLMs) can generate valuable data for safety measures, they often exhibit distributional biases, focusing on common scenarios and neglecting rare but critical cases. This can undermine the effectiveness of safety protocols developed using such data. To address this, we propose a novel framework that integrates active learning with clustering to guide LLM generation, enhancing their representativeness and robustness in safety scenarios. We demonstrate the effectiveness of our approach by constructing a dataset of 5.4K potential safety violations through an iterative process involving LLM generation and an active learner model's feedback. Our results show that the proposed framework produces a more representative set of safety scenarios without requiring prior knowledge of the underlying data distribution. Additionally, data acquired through our method improves the accuracy and F1 score of both the active learner model as well models outside the scope of active learning process, highlighting its broad applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11079v1">Code-Mixer Ya Nahi: Novel Approaches to Measuring Multilingual LLMs' Code-Mixing Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 Manuscript submitted to COLING 2025
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) have demonstrated exceptional performance in Machine Translation (MT) tasks. However, their MT abilities in the context of code-switching (the practice of mixing two or more languages in an utterance) remain under-explored. In this paper, we introduce Rule-Based Prompting, a novel prompting technique to generate code-mixed sentences. We measure and compare the code-mixed MT abilities of 3 popular multilingual LLMs: GPT-3.5-turbo, GPT-4, and Gemini Pro across five language pairs: English-{Hindi, Bengali, Gujarati, French, Spanish} using $k$-shot prompting ($k\in\{0, 1, 10, 20\}$) and Rule-Based Prompting. Our findings suggest that though $k$-shot prompting often leads to the best results, Rule-Based prompting shows promise in generating unique code-mixed sentences that vary in their style of code-mixing. We also use $k$-shot prompting to gauge the code-mixed to English translation abilities of multilingual LLMs. For this purpose, we create a gold-standard code-mixed dataset spanning five language pairs: English-{Hindi, Bengali, Gujarati, French, Spanish}. As a real-world application of our work, we create a code-mixed chatbot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11059v1">Assessing Bias in Metric Models for LLM Open-Ended Generation Bias Benchmarks</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 NeurIPS 2024 EvalEval Workshop
    </div>
    <details class="paper-abstract">
      Open-generation bias benchmarks evaluate social biases in Large Language Models (LLMs) by analyzing their outputs. However, the classifiers used in analysis often have inherent biases, leading to unfair conclusions. This study examines such biases in open-generation benchmarks like BOLD and SAGED. Using the MGSD dataset, we conduct two experiments. The first uses counterfactuals to measure prediction variations across demographic groups by altering stereotype-related prefixes. The second applies explainability tools (SHAP) to validate that the observed biases stem from these counterfactuals. Results reveal unequal treatment of demographic descriptors, calling for more robust bias metric models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11055v1">Varying Shades of Wrong: Aligning LLMs with Wrong Answers Only</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      In the absence of abundant reliable annotations for challenging tasks and contexts, how can we expand the frontier of LLM capabilities with potentially wrong answers? We focus on two research questions: (1) Can LLMs generate reliable preferences among wrong options? And if so, (2) Would alignment with such wrong-over-wrong preferences be helpful? We employ methods based on self-consistency, token probabilities, and LLM-as-a-judge to elicit wrong-over-wrong preferences, and fine-tune language models with preference optimization approaches using these synthesized preferences. Extensive experiments with seven LLMs and eight datasets demonstrate that (1) LLMs do have preliminary capability in distinguishing various shades of wrong, achieving up to 20.9% higher performance than random guess; (2) Alignment with wrong-over-wrong preferences helps LLMs to produce less wrong and sometimes even outright correct answers, while overall improving model calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11006v1">Effective Self-Mining of In-Context Examples for Unsupervised Machine Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance on a wide range of natural language processing (NLP) tasks, primarily through in-context learning (ICL). In ICL, the LLM is provided with examples that represent a given task such that it learns to generate answers for test inputs. However, access to these in-context examples is not guaranteed especially for low-resource or massively multilingual tasks. In this work, we propose an unsupervised approach to mine in-context examples for machine translation (MT), enabling unsupervised MT (UMT) across different languages. Our approach begins with word-level mining to acquire word translations that are then used to perform sentence-level mining. As the quality of mined parallel pairs may not be optimal due to noise or mistakes, we introduce a filtering criterion to select the optimal in-context examples from a pool of unsupervised parallel sentences. We evaluate our approach using two multilingual LLMs on 288 directions from the FLORES-200 dataset and analyze the impact of various linguistic features on performance. Our findings demonstrate the effectiveness of our unsupervised approach in mining in-context examples for MT, leading to better or comparable translation performance as translation with regular in-context samples (extracted from human-annotated data), while also outperforming the other state-of-the-art UMT methods by an average of $7$ BLEU points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10998v1">WILT: A Multi-Turn, Memorization-Robust Inductive Logic Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 Submitted to ICLR 2025. Preprint version 1
    </div>
    <details class="paper-abstract">
      While large language models have shown impressive capabilities across a wide range of domains, they still encounter significant challenges in reasoning tasks that require gathering evidence over multiple turns and drawing logical conclusions. These challenges present significant obstacles for LLM chat user interfaces, which rely on multi-turn interactions to facilitate effective collaboration. This limitation leads to real-world issues; for example, service chatbots must gather necessary information from customers over multiple turns to diagnose and resolve problems effectively. Despite the multi-turn nature of many real-world LLM use cases, most existing benchmarks rely on carefully curated single-turn tests, which often blur the line between memorization and genuine reasoning. To address this, we introduce the Wason Inductive Logic Test (WILT), a simple yet challenging multi-turn reasoning benchmark designed to resist memorization. WILT is inspired by the Wason 2-4-6 task, where participants must infer a boolean function involving three variables (e.g., $x < y < z$) by proposing test cases (such as $(2, 4, 6)$). In WILT, each test starts from a clean slate, with only the initial instructions provided, preventing models from relying on pre-learned responses. Over several turns, models must interact with the environment by suggesting test cases to narrow the possible hypotheses and ultimately infer the hidden function based on the outcomes. Our findings reveal that LLMs struggle with this task, exhibiting distinct strengths and weaknesses: some are better at narrowing down the hypothesis space by proposing valuable test cases, while others are more adept at deducing the hidden function from observed cases. Despite these variations, the best-performing model achieves only 28% accuracy, highlighting a significant gap in LLM performance on complex multi-turn reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10991v1">Performance in a dialectal profiling task of LLMs for varieties of Brazilian Portuguese</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 8 pages, XI Jornada de Descri\c{c}\~ao do Portugu\^es
    </div>
    <details class="paper-abstract">
      Different of biases are reproduced in LLM-generated responses, including dialectal biases. A study based on prompt engineering was carried out to uncover how LLMs discriminate varieties of Brazilian Portuguese, specifically if sociolinguistic rules are taken into account in four LLMs: GPT 3.5, GPT-4o, Gemini, and Sabi.-2. The results offer sociolinguistic contributions for an equity fluent NLP technology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10819v1">DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Deploying long-context large language models (LLMs) is essential but poses significant computational and memory challenges. Caching all Key and Value (KV) states across all attention heads consumes substantial memory. Existing KV cache pruning methods either damage the long-context capabilities of LLMs or offer only limited efficiency improvements. In this paper, we identify that only a fraction of attention heads, a.k.a, Retrieval Heads, are critical for processing long contexts and require full attention across all tokens. In contrast, all other heads, which primarily focus on recent tokens and attention sinks--referred to as Streaming Heads--do not require full attention. Based on this insight, we introduce DuoAttention, a framework that only applies a full KV cache to retrieval heads while using a light-weight, constant-length KV cache for streaming heads, which reduces both LLM's decoding and pre-filling memory and latency without compromising its long-context abilities. DuoAttention uses a lightweight, optimization-based algorithm with synthetic data to identify retrieval heads accurately. Our method significantly reduces long-context inference memory by up to 2.55x for MHA and 1.67x for GQA models while speeding up decoding by up to 2.18x and 1.50x and accelerating pre-filling by up to 1.73x and 1.63x for MHA and GQA models, respectively, with minimal accuracy loss compared to full attention. Notably, combined with quantization, DuoAttention enables Llama-3-8B decoding with 3.3 million context length on a single A100 GPU. Code is provided in https://github.com/mit-han-lab/duo-attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10756v1">Use Random Selection for Now: Investigation of Few-Shot Selection Strategies in LLM-based Text Augmentation for Classification</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      The generative large language models (LLMs) are increasingly used for data augmentation tasks, where text samples are paraphrased (or generated anew) and then used for classifier fine-tuning. Existing works on augmentation leverage the few-shot scenarios, where samples are given to LLMs as part of prompts, leading to better augmentations. Yet, the samples are mostly selected randomly and a comprehensive overview of the effects of other (more ``informed'') sample selection strategies is lacking. In this work, we compare sample selection strategies existing in few-shot learning literature and investigate their effects in LLM-based textual augmentation. We evaluate this on in-distribution and out-of-distribution classifier performance. Results indicate, that while some ``informed'' selection strategies increase the performance of models, especially for out-of-distribution data, it happens only seldom and with marginal performance increases. Unless further advances are made, a default of random sample selection remains a good option for augmentation practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09024v2">AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      The robustness of LLMs to jailbreak attacks, where users design prompts to circumvent safety measures and misuse model capabilities, has been studied primarily for LLMs acting as simple chatbots. Meanwhile, LLM agents -- which use external tools and can execute multi-stage tasks -- may pose a greater risk if misused, but their robustness remains underexplored. To facilitate research on LLM agent misuse, we propose a new benchmark called AgentHarm. The benchmark includes a diverse set of 110 explicitly malicious agent tasks (440 with augmentations), covering 11 harm categories including fraud, cybercrime, and harassment. In addition to measuring whether models refuse harmful agentic requests, scoring well on AgentHarm requires jailbroken agents to maintain their capabilities following an attack to complete a multi-step task. We evaluate a range of leading LLMs, and find (1) leading LLMs are surprisingly compliant with malicious agent requests without jailbreaking, (2) simple universal jailbreak templates can be adapted to effectively jailbreak agents, and (3) these jailbreaks enable coherent and malicious multi-step agent behavior and retain model capabilities. To enable simple and reliable evaluation of attacks and defenses for LLM-based agents, we publicly release AgentHarm at https://huggingface.co/datasets/ai-safety-institute/AgentHarm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10739v1">Balancing Continuous Pre-Training and Instruction Fine-Tuning: Optimizing Instruction-Following in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) for public use require continuous pre-training to remain up-to-date with the latest data. The models also need to be fine-tuned with specific instructions to maintain their ability to follow instructions accurately. Typically, LLMs are released in two versions: the Base LLM, pre-trained on diverse data, and the instruction-refined LLM, additionally trained with specific instructions for better instruction following. The question arises as to which model should undergo continuous pre-training to maintain its instruction-following abilities while also staying current with the latest data. In this study, we delve into the intricate relationship between continuous pre-training and instruction fine-tuning of the LLMs and investigate the impact of continuous pre-training on the instruction following abilities of both the base and its instruction finetuned model. Further, the instruction fine-tuning process is computationally intense and requires a substantial number of hand-annotated examples for the model to learn effectively. This study aims to find the most compute-efficient strategy to gain up-to-date knowledge and instruction-following capabilities without requiring any instruction data and fine-tuning. We empirically prove our findings on the LLaMa 3, 3.1 and Qwen 2, 2.5 family of base and instruction models, providing a comprehensive exploration of our hypotheses across varying sizes of pre-training data corpus and different LLMs settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10728v1">Towards LLM-guided Efficient and Interpretable Multi-linear Tensor Network Rank Selection</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      We propose a novel framework that leverages large language models (LLMs) to guide the rank selection in tensor network models for higher-order data analysis. By utilising the intrinsic reasoning capabilities and domain knowledge of LLMs, our approach offers enhanced interpretability of the rank choices and can effectively optimise the objective function. This framework enables users without specialised domain expertise to utilise tensor network decompositions and understand the underlying rationale within the rank selection process. Experimental results validate our method on financial higher-order datasets, demonstrating interpretable reasoning, strong generalisation to unseen test data, and its potential for self-enhancement over successive iterations. This work is placed at the intersection of large language models and higher-order data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08516v5">Converging Paradigms: The Synergy of Symbolic and Connectionist AI in LLM-Empowered Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      This article explores the convergence of connectionist and symbolic artificial intelligence (AI), from historical debates to contemporary advancements. Traditionally considered distinct paradigms, connectionist AI focuses on neural networks, while symbolic AI emphasizes symbolic representation and logic. Recent advancements in large language models (LLMs), exemplified by ChatGPT and GPT-4, highlight the potential of connectionist architectures in handling human language as a form of symbols. The study argues that LLM-empowered Autonomous Agents (LAAs) embody this paradigm convergence. By utilizing LLMs for text-based knowledge modeling and representation, LAAs integrate neuro-symbolic AI principles, showcasing enhanced reasoning and decision-making capabilities. Comparing LAAs with Knowledge Graphs within the neuro-symbolic AI theme highlights the unique strengths of LAAs in mimicking human-like reasoning processes, scaling effectively with large datasets, and leveraging in-context samples without explicit re-training. The research underscores promising avenues in neuro-vector-symbolic integration, instructional encoding, and implicit reasoning, aimed at further enhancing LAA capabilities. By exploring the progression of neuro-symbolic AI and proposing future research trajectories, this work advances the understanding and development of AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10700v1">Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      This study exposes the safety vulnerabilities of Large Language Models (LLMs) in multi-turn interactions, where malicious users can obscure harmful intents across several queries. We introduce ActorAttack, a novel multi-turn attack method inspired by actor-network theory, which models a network of semantically linked actors as attack clues to generate diverse and effective attack paths toward harmful targets. ActorAttack addresses two main challenges in multi-turn attacks: (1) concealing harmful intents by creating an innocuous conversation topic about the actor, and (2) uncovering diverse attack paths towards the same harmful target by leveraging LLMs' knowledge to specify the correlated actors as various attack clues. In this way, ActorAttack outperforms existing single-turn and multi-turn attack methods across advanced aligned LLMs, even for GPT-o1. We will publish a dataset called SafeMTData, which includes multi-turn adversarial prompts and safety alignment data, generated by ActorAttack. We demonstrate that models safety-tuned using our safety dataset are more robust to multi-turn attacks. Code is available at https://github.com/renqibing/ActorAttack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10644v1">Functional Flexibility in Generative AI Interfaces: Text Editing with LLMs through Conversations, Toolbars, and Prompts</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 Just submitted. This is the author's version of the work
    </div>
    <details class="paper-abstract">
      Prompting-based user interfaces (UIs) shift the task of defining and accessing relevant functions from developers to users. However, how UIs shape this flexibility has not yet been investigated explicitly. We explored interaction with Large Language Models (LLMs) over four years, before and after the rise of general-purpose LLMs: (1) Our survey (N=121) elicited how users envision to delegate writing tasks to AI. This informed a conversational UI design. (2) A user study (N=10) revealed that people regressed to using short command-like prompts. (3) When providing these directly as shortcuts in a toolbar UI, in addition to prompting, users in our second study (N=12) dynamically switched between specified and flexible AI functions. We discuss functional flexibility as a new theoretical construct and thinking tool. Our work highlights the value of moving beyond conversational UIs, by considering how different UIs shape users' access to the functional space of generative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14504v2">Translating Across Cultures: LLMs for Intralingual Cultural Adaptation</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 Accepted to CoNLL 2024
    </div>
    <details class="paper-abstract">
      LLMs are increasingly being deployed for multilingual applications and have demonstrated impressive translation capabilities between several low and high-resource languages. An aspect of translation that often gets overlooked is that of cultural adaptation, or modifying source culture references to suit the target culture. While specialized translation models still outperform LLMs on the machine translation task when viewed from the lens of correctness, they are not sensitive to cultural differences often requiring manual correction. LLMs on the other hand have a rich reservoir of cultural knowledge embedded within its parameters that can be potentially exploited for such applications. In this paper, we define the task of cultural adaptation and create an evaluation framework to evaluate the performance of modern LLMs for cultural adaptation and analyze their cross-cultural knowledge while connecting related concepts across different cultures. We also analyze possible issues with automatic adaptation. We hope that this task will offer more insight into the cultural understanding of LLMs and their creativity in cross-cultural scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10630v1">Thinking LLMs: General Instruction Following with Thought Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      LLMs are typically trained to answer user questions or follow instructions similarly to how human experts respond. However, in the standard alignment framework they lack the basic ability of explicit thinking before answering. Thinking is important for complex questions that require reasoning and planning -- but can be applied to any task. We propose a training method for equipping existing LLMs with such thinking abilities for general instruction following without use of additional human data. We achieve this by an iterative search and optimization procedure that explores the space of possible thought generations, allowing the model to learn how to think without direct supervision. For each instruction, the thought candidates are scored using a judge model to evaluate their responses only, and then optimized via preference optimization. We show that this procedure leads to superior performance on AlpacaEval and Arena-Hard, and shows gains from thinking on non-reasoning categories such as marketing, health and general knowledge, in addition to more traditional reasoning & problem-solving tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10628v1">Test smells in LLM-Generated Unit Tests</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) in automated test generation is gaining popularity, with much of the research focusing on metrics like compilability rate, code coverage and bug detection. However, an equally important quality metric is the presence of test smells design flaws or anti patterns in test code that hinder maintainability and readability. In this study, we explore the diffusion of test smells in LLM generated unit test suites and compare them to those found in human written ones. We analyze a benchmark of 20,500 LLM-generated test suites produced by four models (GPT-3.5, GPT-4, Mistral 7B, and Mixtral 8x7B) across five prompt engineering techniques, alongside a dataset of 780,144 human written test suites from 34,637 projects. Leveraging TsDetect, a state of the art tool capable of detecting 21 different types of test smells, we identify and analyze the prevalence and co-occurrence of various test smells in both human written and LLM-generated test suites. Our findings reveal new insights into the strengths and limitations of LLMs in test generation. First, regarding prevalence, we observe that LLMs frequently generate tests with common test smells, such as Magic Number Test and Assertion Roulette. Second, in terms of co occurrence, certain smells, like Long Test and Useless Test, tend to co occur in LLM-generated suites, influenced by specific prompt techniques. Third, we find that project complexity and LLM specific factors, including model size and context length, significantly affect the prevalence of test smells. Finally, the patterns of test smells in LLM-generated tests often mirror those in human-written tests, suggesting potential data leakage from training datasets. These insights underscore the need to refine LLM-based test generation for cleaner code and suggest improvements in both LLM capabilities and software testing practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10626v1">Efficiently Democratizing Medical LLMs for 50 Languages via a Mixture of Language Family Experts</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Adapting medical Large Language Models to local languages can reduce barriers to accessing healthcare services, but data scarcity remains a significant challenge, particularly for low-resource languages. To address this, we first construct a high-quality medical dataset and conduct analysis to ensure its quality. In order to leverage the generalization capability of multilingual LLMs to efficiently scale to more resource-constrained languages, we explore the internal information flow of LLMs from a multilingual perspective using Mixture of Experts (MoE) modularity. Technically, we propose a novel MoE routing method that employs language-specific experts and cross-lingual routing. Inspired by circuit theory, our routing analysis revealed a Spread Out in the End information flow mechanism: while earlier layers concentrate cross-lingual information flow, the later layers exhibit language-specific divergence. This insight directly led to the development of the Post-MoE architecture, which applies sparse routing only in the later layers while maintaining dense others. Experimental results demonstrate that this approach enhances the generalization of multilingual models to other languages while preserving interpretability. Finally, to efficiently scale the model to 50 languages, we introduce the concept of language family experts, drawing on linguistic priors, which enables scaling the number of languages without adding additional parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08820v2">Which Demographics do LLMs Default to During Annotation?</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Demographics and cultural background of annotators influence the labels they assign in text annotation -- for instance, an elderly woman might find it offensive to read a message addressed to a "bro", but a male teenager might find it appropriate. It is therefore important to acknowledge label variations to not under-represent members of a society. Two research directions developed out of this observation in the context of using large language models (LLM) for data annotations, namely (1) studying biases and inherent knowledge of LLMs and (2) injecting diversity in the output by manipulating the prompt with demographic information. We combine these two strands of research and ask the question to which demographics an LLM resorts to when no demographics is given. To answer this question, we evaluate which attributes of human annotators LLMs inherently mimic. Furthermore, we compare non-demographic conditioned prompts and placebo-conditioned prompts (e.g., "you are an annotator who lives in house number 5") to demographics-conditioned prompts ("You are a 45 year old man and an expert on politeness annotation. How do you rate {instance}"). We study these questions for politeness and offensiveness annotations on the POPQUORN data set, a corpus created in a controlled manner to investigate human label variations based on demographics which has not been used for LLM-based analyses so far. We observe notable influences related to gender, race, and age in demographic prompting, which contrasts with previous studies that found no such effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08102v2">Multi-Agent Collaborative Data Selection for Efficient LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Efficient data selection is crucial to accelerate the pretraining of large language models (LLMs). While various methods have been proposed to enhance data efficiency, limited research has addressed the inherent conflicts between these approaches to achieve optimal data selection for LLM pretraining. To tackle this problem, we propose a novel multi-agent collaborative data selection mechanism. In this framework, each data selection method serves as an independent agent, and an agent console is designed to dynamically integrate the information from all agents throughout the LLM training process. We conduct extensive empirical studies to evaluate our multi-agent framework. The experimental results demonstrate that our approach significantly improves data efficiency, accelerates convergence in LLM training, and achieves an average performance gain up to 10.5% across multiple language model benchmarks compared to the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10526v1">Generalized Adversarial Code-Suggestions: Exploiting Contexts of LLM-based Code-Completion</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      While convenient, relying on LLM-powered code assistants in day-to-day work gives rise to severe attacks. For instance, the assistant might introduce subtle flaws and suggest vulnerable code to the user. These adversarial code-suggestions can be introduced via data poisoning and, thus, unknowingly by the model creators. In this paper, we provide a generalized formulation of such attacks, spawning and extending related work in this domain. This formulation is defined over two components: First, a trigger pattern occurring in the prompts of a specific user group, and, second, a learnable map in embedding space from the prompt to an adversarial bait. The latter gives rise to novel and more flexible targeted attack-strategies, allowing the adversary to choose the most suitable trigger pattern for a specific user-group arbitrarily, without restrictions on the pattern's tokens. Our directional-map attacks and prompt-indexing attacks increase the stealthiness decisively. We extensively evaluate the effectiveness of these attacks and carefully investigate defensive mechanisms to explore the limits of generalized adversarial code-suggestions. We find that most defenses unfortunately offer little protection only.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10479v1">TMGBench: A Systematic Game Benchmark for Evaluating Strategic Reasoning Abilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has accelerated their application in reasoning, with strategic reasoning drawing increasing attention. To evaluate LLMs' strategic reasoning capabilities, game theory, with its concise structure, has become a preferred approach. However, current research focuses on a limited selection of games, resulting in low coverage. Classic game scenarios risk data leakage, and existing benchmarks often lack extensibility, making them inadequate for evaluating state-of-the-art models. To address these challenges, we propose TMGBench, a benchmark with comprehensive game type coverage, novel scenarios, and flexible organization. Specifically, we incorporate all 144 game types summarized by the Robinson-Goforth topology of 2x2 games, constructed as classic games. We also employ synthetic data generation to create diverse, higher-quality scenarios through topic guidance and human inspection, referred to as story-based games. Lastly, we provide a sustainable framework for increasingly powerful LLMs by treating these games as atomic units and organizing them into more complex forms via sequential, parallel, and nested structures. Our comprehensive evaluation of mainstream LLMs covers tests on rational reasoning, robustness, Theory-of-Mind (ToM), and reasoning in complex forms. Results reveal flaws in accuracy, consistency, and varying mastery of ToM. Additionally, o1-mini, OpenAI's latest reasoning model, achieved accuracy rates of 66.6%, 60.0%, and 70.0% on sequential, parallel, and nested games, highlighting TMGBench's challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10455v1">Advancing Academic Knowledge Retrieval via LLM-enhanced Representation Similarity Fusion</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 The 2nd Place of KDD Cup 2024 OAG-Challenge AQA
    </div>
    <details class="paper-abstract">
      In an era marked by robust technological growth and swift information renewal, furnishing researchers and the populace with top-tier, avant-garde academic insights spanning various domains has become an urgent necessity. The KDD Cup 2024 AQA Challenge is geared towards advancing retrieval models to identify pertinent academic terminologies from suitable papers for scientific inquiries. This paper introduces the LLM-KnowSimFuser proposed by Robo Space, which wins the 2nd place in the competition. With inspirations drawed from the superior performance of LLMs on multiple tasks, after careful analysis of the provided datasets, we firstly perform fine-tuning and inference using LLM-enhanced pre-trained retrieval models to introduce the tremendous language understanding and open-domain knowledge of LLMs into this task, followed by a weighted fusion based on the similarity matrix derived from the inference results. Finally, experiments conducted on the competition datasets show the superiority of our proposal, which achieved a score of 0.20726 on the final leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14531v2">Should We Respect LLMs? A Cross-Lingual Study on the Influence of Prompt Politeness on LLM Performance</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 SICon 2024
    </div>
    <details class="paper-abstract">
      We investigate the impact of politeness levels in prompts on the performance of large language models (LLMs). Polite language in human communications often garners more compliance and effectiveness, while rudeness can cause aversion, impacting response quality. We consider that LLMs mirror human communication traits, suggesting they align with human cultural norms. We assess the impact of politeness in prompts on LLMs across English, Chinese, and Japanese tasks. We observed that impolite prompts often result in poor performance, but overly polite language does not guarantee better outcomes. The best politeness level is different according to the language. This phenomenon suggests that LLMs not only reflect human behavior but are also influenced by language, particularly in different cultural contexts. Our findings highlight the need to factor in politeness for cross-cultural natural language processing and LLM usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10414v1">On Calibration of LLM-based Guard Models for Reliable Content Moderation</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 19 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) pose significant risks due to the potential for generating harmful content or users attempting to evade guardrails. Existing studies have developed LLM-based guard models designed to moderate the input and output of threat LLMs, ensuring adherence to safety policies by blocking content that violates these protocols upon deployment. However, limited attention has been given to the reliability and calibration of such guard models. In this work, we empirically conduct comprehensive investigations of confidence calibration for 9 existing LLM-based guard models on 12 benchmarks in both user input and model output classification. Our findings reveal that current LLM-based guard models tend to 1) produce overconfident predictions, 2) exhibit significant miscalibration when subjected to jailbreak attacks, and 3) demonstrate limited robustness to the outputs generated by different types of response models. Additionally, we assess the effectiveness of post-hoc calibration methods to mitigate miscalibration. We demonstrate the efficacy of temperature scaling and, for the first time, highlight the benefits of contextual calibration for confidence calibration of guard models, particularly in the absence of validation sets. Our analysis and experiments underscore the limitations of current LLM-based guard models and provide valuable insights for the future development of well-calibrated guard models toward more reliable content moderation. We also advocate for incorporating reliability evaluation of confidence calibration when releasing future LLM-based guard models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10348v1">Augmenting In-Context-Learning in LLMs via Automatic Data Labeling and Refinement</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      It has been shown that Large Language Models' (LLMs) performance can be improved for many tasks using Chain of Thought (CoT) or In-Context Learning (ICL), which involve demonstrating the steps needed to solve a task using a few examples. However, while datasets with input-output pairs are relatively easy to produce, providing demonstrations which include intermediate steps requires cumbersome manual work. These steps may be executable programs, as in agentic flows, or step-by-step reasoning as in CoT. In this work, we propose Automatic Data Labeling and Refinement (ADLR), a method to automatically generate and filter demonstrations which include the above intermediate steps, starting from a small seed of manually crafted examples. We demonstrate the advantage of ADLR in code-based table QA and mathematical reasoning, achieving up to a 5.5% gain. The code implementing our method is provided in the Supplementary material and will be made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00696v2">Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Rating-based human evaluation has become an essential tool to accurately evaluate the impressive performance of large language models (LLMs). However, current rating systems suffer from several important limitations: first, they fail to account for biases that significantly influence evaluation results, second, they require large and expensive preference datasets to obtain accurate ratings, and third, they do not facilitate meaningful comparisons of model ratings across different tasks. To address these issues, we introduce Polyrating, an expressive and flexible rating system based on maximum a posteriori estimation that enables a more nuanced and thorough analysis of model performance at lower costs. Polyrating can detect and quantify biases affecting human preferences, ensuring fairer model comparisons. Further, Polyrating can reduce the cost of human evaluations by up to $41\%$ for new models and up to $77\%$ for new tasks by leveraging existing benchmark scores. Lastly, Polyrating enables direct comparisons of ratings across different tasks, providing a comprehensive understanding of an LLMs' strengths, weaknesses, and relative performance across different applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10347v1">A Unified Approach to Routing and Cascading for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      The widespread applicability of large language models (LLMs) has increased the availability of many fine-tuned models of various sizes targeting specific tasks. Given a set of such specialized models, to maximize overall performance, it is important to figure out the optimal strategy for selecting the right model for a given user query. An effective strategy could drastically increase overall performance and even offer improvements over a single large monolithic model. Existing approaches typically fall into two categories: routing, where a single model is selected for each query, and cascading, which runs a sequence of increasingly larger models until a satisfactory answer is obtained. However, both have notable limitations: routing commits to an initial model without flexibility, while cascading requires executing every model in sequence, which can be inefficient. Additionally, the conditions under which these strategies are provably optimal remain unclear. In this work, we derive optimal strategies for both routing and cascading. Building on this analysis, we propose a novel approach called cascade routing, which combines the adaptability of routing with the cost-efficiency of cascading. Our experiments demonstrate that cascade routing consistently outperforms both routing and cascading across a variety of settings, improving both output quality and lowering computational cost, thus offering a unified and efficient solution to the model selection problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08442v2">JurEE not Judges: safeguarding llm interactions with small, specialised Encoder Ensembles</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      We introduce JurEE, an ensemble of efficient, encoder-only transformer models designed to strengthen safeguards in AI-User interactions within LLM-based systems. Unlike existing LLM-as-Judge methods, which often struggle with generalization across risk taxonomies and only provide textual outputs, JurEE offers probabilistic risk estimates across a wide range of prevalent risks. Our approach leverages diverse data sources and employs progressive synthetic data generation techniques, including LLM-assisted augmentation, to enhance model robustness and performance. We create an in-house benchmark comprising of other reputable benchmarks such as the OpenAI Moderation Dataset and ToxicChat, where we find JurEE significantly outperforms baseline models, demonstrating superior accuracy, speed, and cost-efficiency. This makes it particularly suitable for applications requiring stringent content moderation, such as customer-facing chatbots. The encoder-ensemble's modular design allows users to set tailored risk thresholds, enhancing its versatility across various safety-related applications. JurEE's collective decision-making process, where each specialized encoder model contributes to the final output, not only improves predictive accuracy but also enhances interpretability. This approach provides a more efficient, performant, and economical alternative to traditional LLMs for large-scale implementations requiring robust content moderation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10343v1">Locking Down the Finetuned LLMs Safety</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on additional datasets is often necessary to optimize them for specific downstream tasks. However, existing safety alignment measures, which restrict harmful behavior during inference, are insufficient to mitigate safety risks during fine-tuning. Alarmingly, fine-tuning with just 10 toxic sentences can make models comply with harmful instructions. We introduce SafetyLock, a novel alignment intervention method that maintains robust safety post-fine-tuning through efficient and transferable mechanisms. SafetyLock leverages our discovery that fine-tuned models retain similar safety-related activation representations to their base models. This insight enables us to extract what we term the Meta-SafetyLock, a set of safety bias directions representing key activation patterns associated with safe responses in the original model. We can then apply these directions universally to fine-tuned models to enhance their safety. By searching for activation directions across multiple token dimensions, SafetyLock achieves enhanced robustness and transferability. SafetyLock re-aligns fine-tuned models in under 0.01 seconds without additional computational cost. Our experiments demonstrate that SafetyLock can reduce the harmful instruction response rate from 60% to below 1% in toxic fine-tuned models. It surpasses traditional methods in both performance and efficiency, offering a scalable, non-invasive solution for ensuring the safety of customized LLMs. Our analysis across various fine-tuning scenarios confirms SafetyLock's robustness, advocating its integration into safety protocols for aligned LLMs. The code is released at https://github.com/zhu-minjun/SafetyLock.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10215v1">SkillAggregation: Reference-free LLM-Dependent Aggregation</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to assess NLP tasks due to their ability to generate human-like judgments. Single LLMs were used initially, however, recent work suggests using multiple LLMs as judges yields improved performance. An important step in exploiting multiple judgements is the combination stage, aggregation. Existing methods in NLP either assign equal weight to all LLM judgments or are designed for specific tasks such as hallucination detection. This work focuses on aggregating predictions from multiple systems where no reference labels are available. A new method called SkillAggregation is proposed, which learns to combine estimates from LLM judges without needing additional data or ground truth. It extends the Crowdlayer aggregation method, developed for image classification, to exploit the judge estimates during inference. The approach is compared to a range of standard aggregation methods on HaluEval-Dialogue, TruthfulQA and Chatbot Arena tasks. SkillAggregation outperforms Crowdlayer on all tasks, and yields the best performance over all approaches on the majority of tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13893v1">Can LLMs be Scammed? A Baseline Measurement Study</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Despite the importance of developing generative AI models that can effectively resist scams, current literature lacks a structured framework for evaluating their vulnerability to such threats. In this work, we address this gap by constructing a benchmark based on the FINRA taxonomy and systematically assessing Large Language Models' (LLMs') vulnerability to a variety of scam tactics. First, we incorporate 37 well-defined base scam scenarios reflecting the diverse scam categories identified by FINRA taxonomy, providing a focused evaluation of LLMs' scam detection capabilities. Second, we utilize representative proprietary (GPT-3.5, GPT-4) and open-source (Llama) models to analyze their performance in scam detection. Third, our research provides critical insights into which scam tactics are most effective against LLMs and how varying persona traits and persuasive techniques influence these vulnerabilities. We reveal distinct susceptibility patterns across different models and scenarios, underscoring the need for targeted enhancements in LLM design and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10150v1">Jailbreak Instruction-Tuned LLMs via end-of-sentence MLP Re-weighting</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      In this paper, we investigate the safety mechanisms of instruction fine-tuned large language models (LLMs). We discover that re-weighting MLP neurons can significantly compromise a model's safety, especially for MLPs in end-of-sentence inferences. We hypothesize that LLMs evaluate the harmfulness of prompts during end-of-sentence inferences, and MLP layers plays a critical role in this process. Based on this hypothesis, we develop 2 novel white-box jailbreak methods: a prompt-specific method and a prompt-general method. The prompt-specific method targets individual prompts and optimizes the attack on the fly, while the prompt-general method is pre-trained offline and can generalize to unseen harmful prompts. Our methods demonstrate robust performance across 7 popular open-source LLMs, size ranging from 2B to 72B. Furthermore, our study provides insights into vulnerabilities of instruction-tuned LLM's safety and deepens the understanding of the internal mechanisms of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11012v7">Connecting the Dots: Evaluating Abstract Reasoning Capabilities of LLMs Using the New York Times Connections Word Game</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      The New York Times Connections game has emerged as a popular and challenging pursuit for word puzzle enthusiasts. We collect 438 Connections games to evaluate the performance of state-of-the-art large language models (LLMs) against expert and novice human players. Our results show that even the best performing LLM, Claude 3.5 Sonnet, which has otherwise shown impressive reasoning abilities on a wide variety of benchmarks, can only fully solve 18% of the games. Novice and expert players perform better than Claude 3.5 Sonnet, with expert human players significantly outperforming it. We create a taxonomy of the knowledge types required to successfully cluster and categorize words in the Connections game. We find that while LLMs perform relatively well on categorizing words based on semantic relations they struggle with other types of knowledge such as Encyclopedic Knowledge, Multiword Expressions or knowledge that combines both Word Form and Meaning. Our results establish the New York Times Connections game as a challenging benchmark for evaluating abstract reasoning capabilities in AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19292v2">From Artificial Needles to Real Haystacks: Improving Retrieval Capabilities in LLMs by Finetuning on Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2024-10-14
    </div>
    <details class="paper-abstract">
      Recent studies have shown that Large Language Models (LLMs) struggle to accurately retrieve information and maintain reasoning capabilities when processing long-context inputs. To address these limitations, we propose a finetuning approach utilizing a carefully designed synthetic dataset comprising numerical key-value retrieval tasks. Our experiments on models like GPT-3.5 Turbo and Mistral 7B demonstrate that finetuning LLMs on this dataset significantly improves LLMs' information retrieval and reasoning capabilities in longer-context settings. We present an analysis of the finetuned models, illustrating the transfer of skills from synthetic to real task evaluations (e.g., $10.5\%$ improvement on $20$ documents MDQA at position $10$ for GPT-3.5 Turbo). We also find that finetuned LLMs' performance on general benchmarks remains almost constant while LLMs finetuned on other baseline long-context augmentation data can encourage hallucination (e.g., on TriviaQA, Mistral 7B finetuned on our synthetic data cause no performance drop while other baseline data can cause a drop that ranges from $2.33\%$ to $6.19\%$). Our study highlights the potential of finetuning on synthetic data for improving the performance of LLMs on longer-context tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16524v2">Harnessing the power of LLMs for normative reasoning in MASs</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 15 pages, 1 figure, presented at the COINE 2024 workshop at AAMAS 2024 (https://coin-workshop.github.io/coine-2024-auckland/accepted_papers.html). This paper will appear in the post-proceedings of the COINE-2024 workshop
    </div>
    <details class="paper-abstract">
      Software agents, both human and computational, do not exist in isolation and often need to collaborate or coordinate with others to achieve their goals. In human society, social mechanisms such as norms ensure efficient functioning, and these techniques have been adopted by researchers in multi-agent systems (MAS) to create socially aware agents. However, traditional techniques have limitations, such as operating in limited environments often using brittle symbolic reasoning. The advent of Large Language Models (LLMs) offers a promising solution, providing a rich and expressive vocabulary for norms and enabling norm-capable agents that can perform a range of tasks such as norm discovery, normative reasoning and decision-making. This paper examines the potential of LLM-based agents to acquire normative capabilities, drawing on recent Natural Language Processing (NLP) and LLM research. We present our vision for creating normative LLM agents. In particular, we discuss how the recently proposed "LLM agent" approaches can be extended to implement such normative LLM agents. We also highlight challenges in this emerging field. This paper thus aims to foster collaboration between MAS, NLP and LLM researchers in order to advance the field of normative agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19775v1">Gender Bias of LLM in Economics: An Existentialism Perspective</a></div>
    <div class="paper-meta">
      📅 2024-10-14
      | 💬 Gender Bias, Large Language Models, Decision-Making
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as GPT-4 and BERT, have rapidly gained traction in natural language processing (NLP) and are now integral to financial decision-making. However, their deployment introduces critical challenges, particularly in perpetuating gender biases that can distort decision-making outcomes in high-stakes economic environments. This paper investigates gender bias in LLMs through both mathematical proofs and empirical experiments using the Word Embedding Association Test (WEAT), demonstrating that LLMs inherently reinforce gender stereotypes even without explicit gender markers. By comparing the decision-making processes of humans and LLMs, we reveal fundamental differences: while humans can override biases through ethical reasoning and individualized understanding, LLMs maintain bias as a rational outcome of their mathematical optimization on biased data. Our analysis proves that bias in LLMs is not an unintended flaw but a systematic result of their rational processing, which tends to preserve and amplify existing societal biases encoded in training data. Drawing on existentialist theory, we argue that LLM-generated bias reflects entrenched societal structures and highlights the limitations of purely technical debiasing methods. This research underscores the need for new theoretical frameworks and interdisciplinary methodologies that address the ethical implications of integrating LLMs into economic and financial decision-making. We advocate for a reconceptualization of how LLMs influence economic decisions, emphasizing the importance of incorporating human-like ethical considerations into AI governance to ensure fairness and equity in AI-driven financial systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10039v1">A Multi-LLM Orchestration Engine for Personalized, Context-Rich Assistance</a></div>
    <div class="paper-meta">
      📅 2024-10-13
    </div>
    <details class="paper-abstract">
      In recent years, large language models have demonstrated remarkable capabilities in natural language understanding and generation. However, these models often struggle with hallucinations and maintaining long term contextual relevance, particularly when dealing with private or local data. This paper presents a novel architecture that addresses these challenges by integrating an orchestration engine that utilizes multiple LLMs in conjunction with a temporal graph database and a vector database. The proposed system captures user interactions, builds a graph representation of conversations, and stores nodes and edges that map associations between key concepts, entities, and behaviors over time. This graph based structure allows the system to develop an evolving understanding of the user preferences, providing personalized and contextually relevant answers. In addition to this, a vector database encodes private data to supply detailed information when needed, allowing the LLM to access and synthesize complex responses. To further enhance reliability, the orchestration engine coordinates multiple LLMs to generate comprehensive answers and iteratively reflect on their accuracy. The result is an adaptive, privacy centric AI assistant capable of offering deeper, more relevant interactions while minimizing the risk of hallucinations. This paper outlines the architecture, methodology, and potential applications of this system, contributing a new direction in personalized, context aware AI assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.14985v4">LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 Accepted by EMNLP 2024
    </div>
    <details class="paper-abstract">
      This paper explores the open research problem of understanding the social behaviors of LLM-based agents. Using Avalon as a testbed, we employ system prompts to guide LLM agents in gameplay. While previous studies have touched on gameplay with LLM agents, research on their social behaviors is lacking. We propose a novel framework, tailored for Avalon, features a multi-agent system facilitating efficient communication and interaction. We evaluate its performance based on game success and analyze LLM agents' social behaviors. Results affirm the framework's effectiveness in creating adaptive agents and suggest LLM-based agents' potential in navigating dynamic social interactions. By examining collaboration and confrontation behaviors, we offer insights into this field's research and applications. Our code is publicly available at https://github.com/3DAgentWorld/LLM-Game-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09992v1">Evaluating Gender Bias of LLMs in Making Morality Judgements</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 Accepted by EMNLP Findings 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in a multitude of Natural Language Processing (NLP) tasks. However, these models are still not immune to limitations such as social biases, especially gender bias. This work investigates whether current closed and open-source LLMs possess gender bias, especially when asked to give moral opinions. To evaluate these models, we curate and introduce a new dataset GenMO (Gender-bias in Morality Opinions) comprising parallel short stories featuring male and female characters respectively. Specifically, we test models from the GPT family (GPT-3.5-turbo, GPT-3.5-turbo-instruct, GPT-4-turbo), Llama 3 and 3.1 families (8B/70B), Mistral-7B and Claude 3 families (Sonnet and Opus). Surprisingly, despite employing safety checks, all production-standard models we tested display significant gender bias with GPT-3.5-turbo giving biased opinions in 24% of the samples. Additionally, all models consistently favour female characters, with GPT showing bias in 68-85% of cases and Llama 3 in around 81-85% instances. Additionally, our study investigates the impact of model parameters on gender bias and explores real-world situations where LLMs reveal biases in moral decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09978v1">When Neutral Summaries are not that Neutral: Quantifying Political Neutrality in LLM-Generated News Summaries</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 12 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      In an era where societal narratives are increasingly shaped by algorithmic curation, investigating the political neutrality of LLMs is an important research question. This study presents a fresh perspective on quantifying the political neutrality of LLMs through the lens of abstractive text summarization of polarizing news articles. We consider five pressing issues in current US politics: abortion, gun control/rights, healthcare, immigration, and LGBTQ+ rights. Via a substantial corpus of 20,344 news articles, our study reveals a consistent trend towards pro-Democratic biases in several well-known LLMs, with gun control and healthcare exhibiting the most pronounced biases (max polarization differences of -9.49% and -6.14%, respectively). Further analysis uncovers a strong convergence in the vocabulary of the LLM outputs for these divisive topics (55% overlap for Democrat-leaning representations, 52% for Republican). Being months away from a US election of consequence, we consider our findings important.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.12070v3">Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 20 pages, code available at https://github.com/ahans30/Binoculars
    </div>
    <details class="paper-abstract">
      Detecting text generated by modern large language models is thought to be hard, as both LLMs and humans can exhibit a wide range of complex behaviors. However, we find that a score based on contrasting two closely related language models is highly accurate at separating human-generated and machine-generated text. Based on this mechanism, we propose a novel LLM detector that only requires simple calculations using a pair of pre-trained LLMs. The method, called Binoculars, achieves state-of-the-art accuracy without any training data. It is capable of spotting machine text from a range of modern LLMs without any model-specific modifications. We comprehensively evaluate Binoculars on a number of text sources and in varied situations. Over a wide range of document types, Binoculars detects over 90% of generated samples from ChatGPT (and other LLMs) at a false positive rate of 0.01%, despite not being trained on any ChatGPT data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03974v2">LLM Roleplay: Simulating Human-Chatbot Interaction</a></div>
    <div class="paper-meta">
      📅 2024-10-13
    </div>
    <details class="paper-abstract">
      The development of chatbots requires collecting a large number of human-chatbot dialogues to reflect the breadth of users' sociodemographic backgrounds and conversational goals. However, the resource requirements to conduct the respective user studies can be prohibitively high and often only allow for a narrow analysis of specific dialogue goals and participant demographics. In this paper, we propose LLM Roleplay: a goal-oriented, persona-based method to automatically generate diverse multi-turn dialogues simulating human-chatbot interaction. LLM Roleplay can be applied to generate dialogues with any type of chatbot and uses large language models (LLMs) to play the role of textually described personas. To validate our method, we collect natural human-chatbot dialogues from different sociodemographic groups and conduct a user study to compare these with our generated dialogues. We evaluate the capabilities of state-of-the-art LLMs in maintaining a conversation during their embodiment of a specific persona and find that our method can simulate human-chatbot dialogues with a high indistinguishability rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09904v1">Equitable Access to Justice: Logical LLMs Show Promise</a></div>
    <div class="paper-meta">
      📅 2024-10-13
    </div>
    <details class="paper-abstract">
      The costs and complexity of the American judicial system limit access to legal solutions for many Americans. Large language models (LLMs) hold great potential to improve access to justice. However, a major challenge in applying AI and LLMs in legal contexts, where consistency and reliability are crucial, is the need for System 2 reasoning. In this paper, we explore the integration of LLMs with logic programming to enhance their ability to reason, bringing their strategic capabilities closer to that of a skilled lawyer. Our objective is to translate laws and contracts into logic programs that can be applied to specific legal cases, with a focus on insurance contracts. We demonstrate that while GPT-4o fails to encode a simple health insurance contract into logical code, the recently released OpenAI o1-preview model succeeds, exemplifying how LLMs with advanced System 2 reasoning capabilities can expand access to justice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09893v1">RMB: Comprehensively Benchmarking Reward Models in LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-10-13
    </div>
    <details class="paper-abstract">
      Reward models (RMs) guide the alignment of large language models (LLMs), steering them toward behaviors preferred by humans. Evaluating RMs is the key to better aligning LLMs. However, the current evaluation of RMs may not directly correspond to their alignment performance due to the limited distribution of evaluation data and evaluation methods that are not closely related to alignment objectives. To address these limitations, we propose RMB, a comprehensive RM benchmark that covers over 49 real-world scenarios and includes both pairwise and Best-of-N (BoN) evaluations to better reflect the effectiveness of RMs in guiding alignment optimization. We demonstrate a positive correlation between our benchmark and the downstream alignment task performance. Based on our benchmark, we conduct extensive analysis on the state-of-the-art RMs, revealing their generalization defects that were not discovered by previous benchmarks, and highlighting the potential of generative RMs. Furthermore, we delve into open questions in reward models, specifically examining the effectiveness of majority voting for the evaluation of reward models and analyzing the impact factors of generative RMs, including the influence of evaluation criteria and instructing methods. Our evaluation code and datasets are available at https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03798v2">Self-Powered LLM Modality Expansion for Large Speech-Text Models</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 Accepted to EMNLP 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable performance across diverse tasks, indicating their potential for expansion into large speech-text models (LSMs) by integrating speech capabilities. Although unified speech-text pre-training and multimodal data instruction-tuning offer considerable benefits, these methods generally entail significant resource demands and tend to overfit specific tasks. This study aims to refine the use of speech datasets for LSM training by addressing the limitations of vanilla instruction tuning. We explore the instruction-following dynamics within LSMs, identifying a critical issue termed speech anchor bias-a tendency for LSMs to over-rely on speech inputs, mistakenly interpreting the entire speech modality as directives, thereby neglecting textual instructions. To counteract this bias, we introduce a self-powered LSM that leverages augmented automatic speech recognition data generated by the model itself for more effective instruction tuning. Our experiments across a range of speech-based tasks demonstrate that self-powered LSM mitigates speech anchor bias and improves the fusion of speech and text modalities in LSMs. Data, code and scripts are freely available at https://github.com/ytf-philp/Self-powered-LSM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09854v1">A Model Is Not Built By A Single Prompt: LLM-Based Domain Modeling With Question Decomposition</a></div>
    <div class="paper-meta">
      📅 2024-10-13
    </div>
    <details class="paper-abstract">
      Domain modeling, a crucial part of model-driven engineering, demands extensive domain knowledge and experience from engineers. When the system description is highly complicated, the modeling task can become particularly challenging and time-consuming. Large language Models(LLMs) can assist by automatically generating an initial object model from the system description. Although LLMs have demonstrated remarkable code-generation ability, they still struggle with model-generation using a single prompt. In real-world domain modeling, engineers usually decompose complex tasks into easily solvable sub-tasks, significantly controlling complexity and enhancing model quality. Inspired by this, we propose an LLM-based domain modeling approach via question decomposition, similar to developer's modeling process. Following conventional modeling guidelines, we divide the model generation task into several sub-tasks, i.e., class generation, association and aggregation generation, and inheritance generation. For each sub-task, we carefully design the prompt by choosing more efficient query words and providing essential modeling knowledge to unlock the modeling potential of LLMs. To sum up all the sub-tasks solutions, we implemente a proof-of-object tool integrated into the standard Ecore editor that asks LLMs to generate an object model from the system description. We evaluate our approach with 20 systems from different application domains. The preliminary results show that our approach outperforms the single-prompt-based prompt by improving recall values and F1 scores in most systems for modeling the classes, attributes, and relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01082v2">Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 20 Pages, revised from 8 pages initially. Main additions include: General full rewrite/reformatting, more comparisons with other sampling methods (eta, epsilon, top-k) on 7B parameter models, more benchmarks for >70B parameter models, human evaluation, theoretical explanations, ethics statement, reproducibility and acknowledgements
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. However, popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures, leading to incoherent or repetitive outputs. To address this challenge, we propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by scaling according to the top token's probability. We conduct extensive experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing, demonstrating that min-p sampling improves both the quality and diversity of generated text, particularly at high temperatures. Moreover, human evaluations reveal a clear preference for min-p sampling in terms of both text quality and diversity. Min-p sampling has been adopted by multiple open-source LLM implementations, highlighting its practical utility and potential impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09780v1">Expanding Search Space with Diverse Prompting Agents: An Efficient Sampling Approach for LLM Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable capabilities in many complex tasks including mathematical reasoning. However, traditional approaches heavily rely on ensuring self-consistency within single prompting method, which limits the exploration of diverse problem-solving strategies. This study addresses these limitations by performing an experimental analysis of distinct prompting methods within the domain of mathematical reasoning. Our findings demonstrate that each method explores a distinct search space, and this differentiation becomes more evident with increasing problem complexity. To leverage this phenomenon, we applied efficient sampling process that uniformly combines samples from these diverse methods, which not only expands the maximum search space but achieves higher performance with fewer runs compared to single methods. Especially, within the subset of difficult questions of MATH dataset named MATH-hard, The maximum search space was achieved while utilizing approximately 43% fewer runs than single methods on average. These findings highlight the importance of integrating diverse problem-solving strategies to enhance the reasoning abilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09775v1">EasyJudge: an Easy-to-use Tool for Comprehensive Response Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-13
    </div>
    <details class="paper-abstract">
      Recently, there has been a growing trend of employing large language models (LLMs) to judge the quality of other LLMs. Many studies have adopted closed-source models, mainly using GPT-4 as the evaluator. However, due to the closed-source nature of the GPT-4 model, employing it as an evaluator has resulted in issues including transparency, controllability, and cost-effectiveness. Some researchers have turned to using fine-tuned open-source LLMs as evaluators. However, existing open-source evaluation LLMs generally lack a user-friendly visualization tool, and they have not been optimized for accelerated model inference, which causes inconvenience for researchers with limited resources and those working across different fields. This paper presents EasyJudge, a model developed to evaluate significant language model responses. It is lightweight, precise, efficient, and user-friendly, featuring an intuitive visualization interface for ease of deployment and use. EasyJudge uses detailed datasets and refined prompts for model optimization, achieving strong consistency with human and proprietary model evaluations. The model optimized with quantitative methods enables EasyJudge to run efficiently on consumer-grade GPUs or even CPUs. We also provide detailed analysis and case studies to further reveal the potential of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09724v1">Taming Overconfidence in LLMs: Reward Calibration in RLHF</a></div>
    <div class="paper-meta">
      📅 2024-10-13
    </div>
    <details class="paper-abstract">
      Language model calibration refers to the alignment between the confidence of the model and the actual performance of its responses. While previous studies point out the overconfidence phenomenon in Large Language Models (LLMs) and show that LLMs trained with Reinforcement Learning from Human Feedback (RLHF) are overconfident with a more sharpened output probability, in this study, we reveal that RLHF tends to lead models to express verbalized overconfidence in their own responses. We investigate the underlying cause of this overconfidence and demonstrate that reward models used for Proximal Policy Optimization (PPO) exhibit inherent biases towards high-confidence scores regardless of the actual quality of responses. Building upon this insight, we propose two PPO variants: PPO-M: PPO with Calibrated Reward Modeling and PPO-C: PPO with Calibrated Reward Calculation. PPO-M integrates explicit confidence scores in reward model training, which calibrates reward models to better capture the alignment between response quality and verbalized confidence. PPO-C adjusts the reward score during PPO based on the difference between the current reward and the moving average of past rewards. Both PPO-M and PPO-C can be seamlessly integrated into the current PPO pipeline and do not require additional golden labels. We evaluate our methods on both Llama3-8B and Mistral-7B across six diverse datasets including multiple-choice and open-ended generation. Experiment results demonstrate that both of our methods can reduce calibration error and maintain performance comparable to standard PPO. We further show that they do not compromise model capabilities in open-ended conversation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12864v1">Investigating Implicit Bias in Large Language Models: A Large-Scale Study of Over 50 LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being adopted across a wide range of tasks, including decision-making processes in industries where bias in AI systems is a significant concern. Recent research indicates that LLMs can harbor implicit biases even when they pass explicit bias evaluations. Building upon the frameworks of the LLM Implicit Association Test (IAT) Bias and LLM Decision Bias, this study highlights that newer or larger language models do not automatically exhibit reduced bias; in some cases, they displayed higher bias scores than their predecessors, such as in Meta's Llama series and OpenAI's GPT models. This suggests that increasing model complexity without deliberate bias mitigation strategies can unintentionally amplify existing biases. The variability in bias scores within and across providers underscores the need for standardized evaluation metrics and benchmarks for bias assessment. The lack of consistency indicates that bias mitigation is not yet a universally prioritized goal in model development, which can lead to unfair or discriminatory outcomes. By broadening the detection of implicit bias, this research provides a more comprehensive understanding of the biases present in advanced models and underscores the critical importance of addressing these issues to ensure the development of fair and responsible AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12708v2">AgentReview: Exploring Peer Review Dynamics with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-13
      | 💬 Accepted at EMNLP 2024. https://agentreview.github.io/
    </div>
    <details class="paper-abstract">
      Peer review is fundamental to the integrity and advancement of scientific publication. Traditional methods of peer review analyses often rely on exploration and statistics of existing peer review data, which do not adequately address the multivariate nature of the process, account for the latent variables, and are further constrained by privacy concerns due to the sensitive nature of the data. We introduce AgentReview, the first large language model (LLM) based peer review simulation framework, which effectively disentangles the impacts of multiple latent factors and addresses the privacy issue. Our study reveals significant insights, including a notable 37.1% variation in paper decisions due to reviewers' biases, supported by sociological theories such as the social influence theory, altruism fatigue, and authority bias. We believe that this study could offer valuable insights to improve the design of peer review mechanisms. Our code is available at https://github.com/Ahren09/AgentReview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10901v1">3DS: Decomposed Difficulty Data Selection's Case Study on LLM Medical Domain Adaptation</a></div>
    <div class="paper-meta">
      📅 2024-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models(LLMs) excel in general tasks but struggle in specialized domains like healthcare due to limited domain-specific knowledge.Supervised Fine-Tuning(SFT) data construction for domain adaptation often relies on heuristic methods, such as GPT-4 annotation or manual data selection, with a data-centric focus on presumed diverse, high-quality datasets. However, these methods overlook the model's inherent knowledge distribution, introducing noise, redundancy, and irrelevant data, leading to a mismatch between the selected data and the model's learning task, resulting in suboptimal performance. To address this, we propose a two-stage model-centric data selection framework, Decomposed Difficulty Data Selection (3DS), which aligns data with the model's knowledge distribution for optimized adaptation. In Stage1, we apply Prompt-Driven Data Selection via Explicit Alignment, where the the model filters irrelevant or redundant data based on its internal knowledge. In Stage2, we perform Decomposed Difficulty Data Selection, where data selection is guided by our defined difficulty decomposition, using three metrics: Instruction Understanding, Response Confidence, and Response Correctness. Additionally, an attention-based importance weighting mechanism captures token importance for more accurate difficulty calibration. This two-stage approach ensures the selected data is not only aligned with the model's knowledge and preferences but also appropriately challenging for the model to learn, leading to more effective and targeted domain adaptation. In the case study of the medical domain, our extensive experiments on real-world healthcare datasets demonstrate the superiority of 3DS over exisiting methods in accuracy by over 5.29%. Our dataset and code will be open-sourced at https://anonymous.4open.science/r/3DS-E67F.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02110v2">Can LLMs Reliably Simulate Human Learner Actions? A Simulation Authoring Framework for Open-Ended Learning Environments</a></div>
    <div class="paper-meta">
      📅 2024-10-12
    </div>
    <details class="paper-abstract">
      Simulating learner actions helps stress-test open-ended interactive learning environments and prototype new adaptations before deployment. While recent studies show the promise of using large language models (LLMs) for simulating human behavior, such approaches have not gone beyond rudimentary proof-of-concept stages due to key limitations. First, LLMs are highly sensitive to minor prompt variations, raising doubts about their ability to generalize to new scenarios without extensive prompt engineering. Moreover, apparently successful outcomes can often be unreliable, either because domain experts unintentionally guide LLMs to produce expected results, leading to self-fulfilling prophecies; or because the LLM has encountered highly similar scenarios in its training data, meaning that models may not be simulating behavior so much as regurgitating memorized content. To address these challenges, we propose Hyp-Mix, a simulation authoring framework that allows experts to develop and evaluate simulations by combining testable hypotheses about learner behavior. Testing this framework in a physics learning environment, we found that GPT-4 Turbo maintains calibrated behavior even as the underlying learner model changes, providing the first evidence that LLMs can be used to simulate realistic behaviors in open-ended interactive learning environments, a necessary prerequisite for useful LLM behavioral simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09615v1">SLiM: One-shot Quantized Sparse Plus Low-rank Approximation of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language understanding and generation tasks but suffer from high memory consumption and slow inference times due to their large parameter sizes. Traditional model compression techniques, such as quantization and pruning, mitigate these issues but often require retraining to maintain accuracy, which is computationally expensive. This paper introduces SLiM, a novel approach for compressing LLMs using a one-shot Quantized Sparse Plus Low-rank Approximation. SLiM eliminates the need for costly retraining by combining a symmetric quantization method (SLiM-Quant) with a saliency-based low-rank approximation. Our method reduces quantization error while leveraging sparse representations compatible with accelerated hardware architectures. Additionally, we propose a parameter-efficient fine-tuning recipe that significantly reduces overhead compared to conventional quantization-aware training. SLiM achieves up to a 5.4% improvement in model accuracy for sparsity patterns like 2:4, and the fine-tuning step further enhances accuracy by up to 5.8%, demonstrating state-of-the-art performance. This work provides a pathway for efficiently deploying large models in memory-constrained environments without compromising accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03211v2">How Does Quantization Affect Multilingual LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 Findings of EMNLP 2024 Camera-Ready
    </div>
    <details class="paper-abstract">
      Quantization techniques are widely used to improve inference speed and deployment of large language models. While a wide body of work examines the impact of quantization on LLMs in English, none have evaluated across languages. We conduct a thorough analysis of quantized multilingual LLMs, focusing on performance across languages and at varying scales. We use automatic benchmarks, LLM-as-a-Judge, and human evaluation, finding that (1) harmful effects of quantization are apparent in human evaluation, which automatic metrics severely underestimate: a 1.7% average drop in Japanese across automatic tasks corresponds to a 16.0% drop reported by human evaluators on realistic prompts; (2) languages are disparately affected by quantization, with non-Latin script languages impacted worst; and (3) challenging tasks like mathematical reasoning degrade fastest. As the ability to serve low-compute models is critical for wide global adoption of NLP technologies, our results urge consideration of multilingual performance as a key evaluation criterion for efficient models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10099v2">Know the Unknown: An Uncertainty-Sensitive Method for LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2024-10-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities but still face challenges such as hallucinations. One potential reason for hallucinations is the lack of relevant knowledge or context. Thus, a promising solution involves instructing LLMs to respond with "I do not know" when a question falls outside their knowledge domain or the provided context. However, in this work, we observed that LLMs struggle to admit their lack of knowledge, primarily due to existing instruction datasets designed to encourage specific answers. To improve models' capability to recognize the boundaries of their knowledge, we propose a novel approach called uncertainty-sensitive tuning. This method involves two-stage training designed for uncertainty recognition and prompt-sensitive activation. In the first stage, we guide the LLM to reject unknown questions. In the second stage, we force the model to follow the instructions by incorporating designed causal instructions. The experimental results demonstrate that our proposed uncertainty-sensitive tuning method enhance the model's ability to identify areas of uncertainty. Specifically, it achieves a substantial improvement of up to 34.7% in handling questions involving knowledge gaps compared to the original model. Moreover, our finetuned models even outperform GPT-4, exhibiting an overall performance improvement of up to 4.2%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01720v2">Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspective</a></div>
    <div class="paper-meta">
      📅 2024-10-12
    </div>
    <details class="paper-abstract">
      Synthetic data has become a pivotal resource in post-training tasks for large language models (LLMs) due to the scarcity of high-quality, specific data. While various methods have been developed to generate synthetic data, there remains a discernible gap between the practical effects of synthetic data and our theoretical comprehension. To address this challenge, we commence by presenting a detailed modeling of the prevalent synthetic data generation process. Building upon this modeling, we demonstrate that the generalization capability of the post-trained model is critically determined by the information gain derived from the generative model, as analyzed from a novel reverse-bottleneck perspective. Moreover, we introduce the concept of Generalization Gain via Mutual Information (GGMI) and elucidate the relationship between generalization gain and information gain. This analysis serves as a theoretical foundation for synthetic data generation and further highlights its connection with the generalization capability of post-trained models, offering an understanding about the design of synthetic data generation techniques and the optimization of the post-training process. We open source our code at https://github.com/ZyGan1999/Towards-a-Theoretical-Understanding-of-Synthetic-Data-in-LLM-Post-Training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06565v2">MixEval: Deriving Wisdom of the Crowd from LLM Benchmark Mixtures</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 Accepted to NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) is challenging. Traditional ground-truth-based benchmarks fail to capture the comprehensiveness and nuance of real-world queries, while LLM-as-judge benchmarks suffer from grading biases and limited query quantity. Both of them may also become contaminated over time. User-facing evaluation, such as Chatbot Arena, provides reliable signals but is costly and slow. In this work, we propose MixEval, a new paradigm for establishing efficient, gold-standard LLM evaluation by strategically mixing off-the-shelf benchmarks. It bridges (1) comprehensive and well-distributed real-world user queries and (2) efficient and fairly-graded ground-truth-based benchmarks, by matching queries mined from the web with similar queries from existing benchmarks. Based on MixEval, we further build MixEval-Hard, which offers more room for model improvement. Our benchmarks' advantages lie in (1) a 0.96 model ranking correlation with Chatbot Arena arising from the highly impartial query distribution and grading mechanism, (2) fast, cheap, and reproducible execution (6% of the time and cost of MMLU), and (3) dynamic evaluation enabled by the rapid and stable data update pipeline. We provide extensive meta-evaluation and analysis for our and existing LLM benchmarks to deepen the community's understanding of LLM evaluation and guide future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03640v6">Apollo: A Lightweight Multilingual Medical LLM towards Democratizing Medical AI to 6B People</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Despite the vast repository of global medical knowledge predominantly being in English, local languages are crucial for delivering tailored healthcare services, particularly in areas with limited medical resources. To extend the reach of medical AI advancements to a broader population, we aim to develop medical LLMs across the six most widely spoken languages, encompassing a global population of 6.1 billion. This effort culminates in the creation of the ApolloCorpora multilingual medical dataset and the XMedBench benchmark. In the multilingual medical benchmark, the released Apollo models, at various relatively-small sizes (i.e., 0.5B, 1.8B, 2B, 6B, and 7B), achieve the best performance among models of equivalent size. Especially, Apollo-7B is the state-of-the-art multilingual medical LLMs up to 70B. Additionally, these lite models could be used to improve the multi-lingual medical capabilities of larger models without fine-tuning in a proxy-tuning fashion. We will open-source training corpora, code, model weights and evaluation benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19949v2">Calibrating LLMs with Preference Optimization on Thought Trees for Generating Rationale in Science Question Scoring</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 Accepted at EMNLP 2024
    </div>
    <details class="paper-abstract">
      Generating rationales that justify scoring decisions has been a promising way to facilitate explainability in automated scoring systems. However, existing methods do not match the accuracy of classifier-based methods. Plus, the generated rationales often contain hallucinated information. To address these issues, we propose a novel framework capable of generating more faithful rationales and, more importantly, matching performance with classifier-based black-box scoring systems. We first mimic the human assessment process by querying Large Language Models (LLMs) to generate a thought tree. We then summarise intermediate assessment decisions from each thought tree path for creating synthetic rationale data and rationale preference data. Finally, we utilise the generated synthetic data to calibrate LLMs through a two-step training process: supervised fine-tuning and preference optimization. Extensive experimental results demonstrate that our framework achieves a 38% assessment performance improvement in the QWK score compared to prior work while producing higher-quality rationales, as recognised by human evaluators and LLMs. Our work sheds light on the effectiveness of performing preference optimization using synthetic preference data obtained from thought tree paths. Data and code are available at https://github.com/lijiazheng99/thought_tree_assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09503v1">SLAM-AAC: Enhancing Audio Captioning with Paraphrasing Augmentation and CLAP-Refine through LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-12
    </div>
    <details class="paper-abstract">
      Automated Audio Captioning (AAC) aims to generate natural textual descriptions for input audio signals. Recent progress in audio pre-trained models and large language models (LLMs) has significantly enhanced audio understanding and textual reasoning capabilities, making improvements in AAC possible. In this paper, we propose SLAM-AAC to further enhance AAC with paraphrasing augmentation and CLAP-Refine through LLMs. Our approach uses the self-supervised EAT model to extract fine-grained audio representations, which are then aligned with textual embeddings via lightweight linear layers. The caption generation LLM is efficiently fine-tuned using the LoRA adapter. Drawing inspiration from the back-translation method in machine translation, we implement paraphrasing augmentation to expand the Clotho dataset during pre-training. This strategy helps alleviate the limitation of scarce audio-text pairs and generates more diverse captions from a small set of audio clips. During inference, we introduce the plug-and-play CLAP-Refine strategy to fully exploit multiple decoding outputs, akin to the n-best rescoring strategy in speech recognition. Using the CLAP model for audio-text similarity calculation, we could select the textual descriptions generated by multiple searching beams that best match the input audio. Experimental results show that SLAM-AAC achieves state-of-the-art performance on Clotho V2 and AudioCaps, surpassing previous mainstream models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09457v1">Power-Softmax: Towards Secure LLM Inference over Encrypted Data</a></div>
    <div class="paper-meta">
      📅 2024-10-12
    </div>
    <details class="paper-abstract">
      Modern cryptographic methods for implementing privacy-preserving LLMs such as Homomorphic Encryption (HE) require the LLMs to have a polynomial form. Forming such a representation is challenging because Transformers include non-polynomial components, such as Softmax and layer normalization. Previous approaches have either directly approximated pre-trained models with large-degree polynomials, which are less efficient over HE, or replaced non-polynomial components with easier-to-approximate primitives before training, e.g., Softmax with pointwise attention. The latter approach might introduce scalability challenges. We present a new HE-friendly variant of self-attention that offers a stable form for training and is easy to approximate with polynomials for secure inference. Our work introduces the first polynomial LLMs with 32 layers and over a billion parameters, exceeding the size of previous models by more than tenfold. The resulting models demonstrate reasoning and in-context learning (ICL) capabilities comparable to standard transformers of the same size, representing a breakthrough in the field. Finally, we provide a detailed latency breakdown for each computation over encrypted data, paving the way for further optimization, and explore the differences in inductive bias between transformers relying on our HE-friendly variant and standard transformers. Our code is attached as a supplement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09454v1">Skipping Computations in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 Accepted at NeurIPS 2024 Workshop RBFM. Code: https://github.com/mshukor/ima-lmms
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success in both textual and multimodal domains. However, this success often comes with substantial computational costs, particularly when handling lengthy sequences of multimodal inputs. This has sparked many efforts focusing on enhancing efficiency during training and inference. In this study, we investigate the computation redundancy in Multimodal Large Language Models (MLLMs) during inference. We propose different methods to skip computations, such as skipping entire blocks, FFN or self-attention (SA) layers. Additionally, we explore parallelizing certain layers, such as FFN and SA layers. Our findings validate that (1) significant amount of computations can be avoided at inference time, especially for tasks such as Visual Question Answering (VQA). (2) Skipping computations during training can recover 97% of the original performance, even when skipping half of the blocks or removing 70% of the weights. Alternatively, (3) properly training with smaller LLMs can yield comparable performance to LLMs 2 or 3 times larger. To conclude, we extend our investigation to recent MLLMs, such as LLaVA-1.5, showing similar observations. Our work show that there is redundant computations inside MLLMs and thus the potential for significantly improving inference costs without sacrificing performance. The code is available here: https://github.com/mshukor/ima-lmms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01623v2">Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 Add further analysis of the scaling factor, code is available at: https://github.com/xichen-fy/Fira
    </div>
    <details class="paper-abstract">
      Low-rank training has emerged as a promising approach for reducing memory usage in training Large Language Models (LLMs). Previous methods either rely on decomposing weight matrices (e.g., LoRA), or seek to decompose gradient matrices (e.g., GaLore) to ensure reduced memory consumption. However, both of them constrain the training in a low-rank subspace, thus inevitably leading to sub-optimal performance. This raises a question: whether it is possible to consistently preserve the low-rank constraint for memory efficiency, while achieving full-rank training (i.e., training with full-rank gradients of full-rank weights) to avoid inferior outcomes? In this paper, we propose a new plug-and-play training framework for LLMs called Fira, as the first attempt to achieve this goal. First, we observe an interesting phenomenon during LLM training: the scaling impact of adaptive optimizers (e.g., Adam) on the gradient norm remains similar from low-rank to full-rank training. Based on this observation, we propose a norm-based scaling method, which utilizes the scaling impact of low-rank optimizers as substitutes for that of original full-rank optimizers to enable full-rank training. In this way, we can preserve the low-rank constraint in the optimizer while achieving full-rank training for better performance. Moreover, we find that there are sudden gradient rises during the optimization process, potentially causing loss spikes. To address this, we further put forward a norm-growth limiter to smooth the gradient via regulating the relative increase of gradient norms. Extensive experiments on the pre-training and fine-tuning of LLMs show that Fira outperforms both LoRA and GaLore, achieving performance that is comparable to or even better than full-rank training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02721v3">Self-Control of LLM Behaviors by Compressing Suffix Gradient into Prefix Controller</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 Website: https://llm-self-control.github.io/
    </div>
    <details class="paper-abstract">
      We propose SelfControl, an inference-time model control method utilizing gradients to control the behavior of large language models (LLMs) without explicit human annotations. Given a desired behavior expressed in a natural language suffix string concatenated to the input prompt, SelfControl computes gradients of the LLM's self-evaluation of the suffix with respect to its latent representations. The gradients are used to directly control the auto-regressive generation process towards desired behaviors, which eliminates human supervision, achieves precise and transparent control, and offers on-the-fly adaptability. To further enhance efficiency, we introduce SelfControl_{Prefix}, a compact module that encapsulates the learned representations from gradients into a SelfControl_{Prefix}, facilitating efficient inference-time control with no latency compared to the original model and allowing control for multiple behaviors simultaneously. Our experiments demonstrate SelfControl's efficacy across multiple domains, where it improves over SOTA for 8.3% in detoxification, 3.1% in truthfulness enhancement, 4%~10% in controlling on emotion tones, and 48.2% in privacy protection, i.e., completely remove privacy leakage issue. Additionally, we demonstrate that SelfControl can be used for data synthesis and to improve reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09412v1">FB-Bench: A Fine-Grained Multi-Task Benchmark for Evaluating LLMs' Responsiveness to Human Feedback</a></div>
    <div class="paper-meta">
      📅 2024-10-12
    </div>
    <details class="paper-abstract">
      Human feedback is crucial in the interactions between humans and Large Language Models (LLMs). However, existing research primarily focuses on benchmarking LLMs in single-turn dialogues. Even in benchmarks designed for multi-turn dialogues, the user inputs are often independent, neglecting the nuanced and complex nature of human feedback within real-world usage scenarios. To fill this research gap, we introduce FB-Bench, a fine-grained, multi-task benchmark designed to evaluate LLMs' responsiveness to human feedback in real-world usage scenarios. Drawing from the two main interaction scenarios, FB-Bench comprises 734 meticulously curated samples, encompassing eight task types, five deficiency types of response, and nine feedback types. We extensively evaluate a broad array of popular LLMs, revealing significant variations in their performance across different interaction scenarios. Further analysis indicates that task, human feedback, and deficiencies of previous responses can also significantly impact LLMs' responsiveness. Our findings underscore both the strengths and limitations of current models, providing valuable insights and directions for future research. Both the toolkits and the dataset of FB-Bench are available at https://github.com/PKU-Baichuan-MLSystemLab/FB-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16369v2">Don't Say No: Jailbreaking LLM by Suppressing Refusal</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 Update results on Llama3, Llama3.1, Gemma2, Mistral, Qwen2 models and upon JailbreakBnech, MaliciousInstruct datasets
    </div>
    <details class="paper-abstract">
      Ensuring the safety alignment of Large Language Models (LLMs) is crucial to generating responses consistent with human values. Despite their ability to recognize and avoid harmful queries, LLMs are vulnerable to jailbreaking attacks, where carefully crafted prompts seduce them to produce toxic content. One category of jailbreak attacks is reformulating the task as an optimization by eliciting the LLM to generate affirmative responses. However, such optimization objective has its own limitations, such as the restriction on the predefined objectionable behaviors, leading to suboptimal attack performance. In this study, we first uncover the reason why vanilla target loss is not optimal, then we explore and enhance the loss objective and introduce the DSN (Don't Say No) attack, which achieves successful attack by suppressing refusal. Another challenge in studying jailbreak attacks is the evaluation, as it is difficult to directly and accurately assess the harmfulness of the responses. The existing evaluation such as refusal keyword matching reveals numerous false positive and false negative instances. To overcome this challenge, we propose an Ensemble Evaluation pipeline that novelly incorporates Natural Language Inference (NLI) contradiction assessment and two external LLM evaluators. Extensive experiments demonstrate the potential of the DSN and effectiveness of Ensemble Evaluation compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08035v2">IntrinsicVoice: Empowering LLMs with Intrinsic Real-time Voice Interaction Abilities</a></div>
    <div class="paper-meta">
      📅 2024-10-12
    </div>
    <details class="paper-abstract">
      Current methods of building LLMs with voice interaction capabilities rely heavily on explicit text autoregressive generation before or during speech response generation to maintain content quality, which unfortunately brings computational overhead and increases latency in multi-turn interactions. To address this, we introduce IntrinsicVoic,e an LLM designed with intrinsic real-time voice interaction capabilities. IntrinsicVoice aims to facilitate the transfer of textual capabilities of pre-trained LLMs to the speech modality by mitigating the modality gap between text and speech. Our novelty architecture, GroupFormer, can reduce speech sequences to lengths comparable to text sequences while generating high-quality audio, significantly reducing the length difference between speech and text, speeding up inference, and alleviating long-text modeling issues. Additionally, we construct a multi-turn speech-to-speech dialogue dataset named \method-500k which includes nearly 500k turns of speech-to-speech dialogues, and a cross-modality training strategy to enhance the semantic alignment between speech and text. Experimental results demonstrate that IntrinsicVoice can generate high-quality speech response with latency lower than 100ms in multi-turn dialogue scenarios. Demos are available at https://instrinsicvoice.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18380v2">OwLore: Outlier-weighed Layerwise Sampled Low-Rank Projection for Memory-Efficient LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2024-10-12
    </div>
    <details class="paper-abstract">
      The rapid advancements in Large Language Models (LLMs) have revolutionized various natural language processing tasks. However, the substantial size of LLMs presents significant challenges in training or fine-tuning. While parameter-efficient approaches such as low-rank adaptation (LoRA) have gained popularity, they often compromise performance compared to full-rank fine-tuning. In this paper, we propose Outlier-weighed Layerwise Sampled Low-Rank Projection (OwLore), a new memory-efficient fine-tuning approach, inspired by the layerwise outlier distribution of LLMs. Unlike LoRA, which adds extra adapters to all layers, OwLore strategically assigns higher sampling probabilities to layers with more outliers, selectively sampling only a few layers and fine-tuning their pre-trained weights. To further increase the number of fine-tuned layers without a proportional rise in memory costs, we incorporate gradient low-rank projection, further boosting the approach's performance. Our extensive experiments across various architectures, including LLaMa2, LLaMa3, and Mistral, demonstrate that OwLore consistently outperforms baseline approaches, including full fine-tuning. Specifically, it achieves up to a 1.1% average accuracy gain on the Commonsense Reasoning benchmark, a 3.0% improvement on MMLU, and a notable 10% boost on MT-Bench, while being more memory efficient. OwLore allows us to fine-tune LLaMa2-7B with only 21GB of memory. Code is available at https://github.com/pixeli99/OwLore.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05975v2">LLaMAX: Scaling Linguistic Horizons of LLM by Enhancing Translation Capabilities Beyond 100 Languages</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 EMNLP 2024 findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable translation capabilities in high-resource language tasks, yet their performance in low-resource languages is hindered by insufficient multilingual data during pre-training. To address this, we conduct extensive multilingual continual pre-training on the LLaMA series models, enabling translation support across more than 100 languages. Through a comprehensive analysis of training strategies, such as vocabulary expansion and data augmentation, we develop LLaMAX. Remarkably, without sacrificing its generalization ability, LLaMAX achieves significantly higher translation performance compared to existing open-source LLMs (by more than 10 spBLEU points) and performs on-par with specialized translation model (M2M-100-12B) on the Flores-101 benchmark. Extensive experiments indicate that LLaMAX can serve as a robust multilingual foundation model. The code \footnote{\url{https://github.com/CONE-MT/LLaMAX/.}} and the models \footnote{\url{https://huggingface.co/LLaMAX/.}} are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10635v2">Strategist: Learning Strategic Skills by LLMs via Bi-Level Tree Search</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 website: https://llm-strategist.github.io
    </div>
    <details class="paper-abstract">
      In this paper, we propose a new method STRATEGIST that utilizes LLMs to acquire new skills for playing multi-agent games through a self-improvement process. Our method gathers quality feedback through self-play simulations with Monte Carlo tree search and LLM-based reflection, which can then be used to learn high-level strategic skills such as how to evaluate states that guide the low-level execution. We showcase how our method can be used in both action planning and dialogue generation in the context of games, achieving good performance on both tasks. Specifically, we demonstrate that our method can help train agents with better performance than both traditional reinforcement learning-based approaches and other LLM-based skill learning approaches in games including the Game of Pure Strategy (GOPS) and The Resistance: Avalon. STRATEGIST helps bridge the gap between foundation models and symbolic decision-making methods through its bi-level approach, leading to more robust decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09342v1">LLM$\times$MapReduce: Simplified Long-Sequence Processing using Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-10-12
      | 💬 Work in Progress. Code: https://github.com/thunlp/LLMxMapReduce
    </div>
    <details class="paper-abstract">
      Enlarging the context window of large language models (LLMs) has become a crucial research area, particularly for applications involving extremely long texts. In this work, we propose a novel training-free framework for processing long texts, utilizing a divide-and-conquer strategy to achieve comprehensive document understanding. The proposed LLM$\times$MapReduce framework splits the entire document into several chunks for LLMs to read and then aggregates the intermediate answers to produce the final output. The main challenge for divide-and-conquer long text processing frameworks lies in the risk of losing essential long-range information when splitting the document, which can lead the model to produce incomplete or incorrect answers based on the segmented texts. Disrupted long-range information can be classified into two categories: inter-chunk dependency and inter-chunk conflict. We design a structured information protocol to better cope with inter-chunk dependency and an in-context confidence calibration mechanism to resolve inter-chunk conflicts. Experimental results demonstrate that LLM$\times$MapReduce can outperform representative open-source and commercial long-context LLMs, and is applicable to several different models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11242v2">Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      LLMs are an integral component of retrieval-augmented generation (RAG) systems. While many studies focus on evaluating the overall quality of end-to-end RAG systems, there is a gap in understanding the appropriateness of LLMs for the RAG task. To address this, we introduce Trust-Score, a holistic metric that evaluates the trustworthiness of LLMs within the RAG framework. Our results show that various prompting methods, such as in-context learning, fail to effectively adapt LLMs to the RAG task as measured by Trust-Score. Consequently, we propose Trust-Align, a method to align LLMs for improved Trust-Score performance. The LLaMA-3 family, aligned using our method, significantly outperforms open-source LLMs of similar sizes on ASQA (up 14.0), QAMPARI (up 28.9), and ELI5 (up 13.7). We also demonstrate the effectiveness of Trust-Align across different open-weight models, including the LLaMA series (1b to 8b), Qwen-2.5 series (0.5b to 7b), and Phi3.5 (3.8b). We release our code at \url{https://anonymous.4open.science/r/trust-align}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00159v2">LLMs hallucinate graphs too: a structural perspective</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      It is known that LLMs do hallucinate, that is, they return incorrect information as facts. In this paper, we introduce the possibility to study these hallucinations under a structured form: graphs. Hallucinations in this context are incorrect outputs when prompted for well known graphs from the literature (e.g. Karate club, Les Mis\'erables, graph atlas). These hallucinated graphs have the advantage of being much richer than the factual accuracy -- or not -- of a statement; this paper thus argues that such rich hallucinations can be used to characterize the outputs of LLMs. Our first contribution observes the diversity of topological hallucinations from major modern LLMs. Our second contribution is the proposal of a metric for the amplitude of such hallucinations: the Graph Atlas Distance, that is the average graph edit distance from several graphs in the graph atlas set. We compare this metric to the Hallucination Leaderboard, a hallucination rank that leverages 10,000 times more prompts to obtain its ranking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15992v2">Can LLM Graph Reasoning Generalize beyond Pattern Memorization?</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 17 pages, 6 figures. EMNLP 2024 Findings. Code and data is publicly available at https://github.com/MatthewYZhang/NLGift
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate great potential for problems with implicit graphical structures, while recent works seek to enhance the graph reasoning capabilities of LLMs through specialized instruction tuning. The resulting 'graph LLMs' are evaluated with in-distribution settings only, thus it remains underexplored whether LLMs are learning generalizable graph reasoning skills or merely memorizing patterns in the synthetic training data. To this end, we propose the NLGift benchmark, an evaluation suite of LLM graph reasoning generalization: whether LLMs could go beyond semantic, numeric, structural, reasoning patterns in the synthetic training data and improve utility on real-world graph-based tasks. Extensive experiments with two LLMs across four graph reasoning tasks demonstrate that while generalization on simple patterns (semantic, numeric) is somewhat satisfactory, LLMs struggle to generalize across reasoning and real-world patterns, casting doubt on the benefit of synthetic graph tuning for real-world tasks with underlying network structures. We explore three strategies to improve LLM graph reasoning generalization, and we find that while post-training alignment is most promising for real-world tasks, empowering LLM graph reasoning to go beyond pattern memorization remains an open research question.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.17003v3">Safety Layers in Aligned Large Language Models: The Key to LLM Security</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Aligned LLMs are secure, capable of recognizing and refusing to answer malicious questions. However, the role of internal parameters in maintaining such security is not well understood yet, further these models can be vulnerable to security degradation when fine-tuned with non-malicious backdoor or normal data. To address these challenges, our work uncovers the mechanism behind security in aligned LLMs at the parameter level, identifying a small set of contiguous layers in the middle of the model that are crucial for distinguishing malicious queries from normal ones, referred to as "safety layers". We first confirm the existence of these safety layers by analyzing variations in input vectors within the model's internal layers. Additionally, we leverage the over-rejection phenomenon and parameters scaling analysis to precisely locate the safety layers. Building on these findings, we propose a novel fine-tuning approach, Safely Partial-Parameter Fine-Tuning (SPPFT), that fixes the gradient of the safety layers during fine-tuning to address the security degradation. Our experiments demonstrate that the proposed approach can significantly preserve LLM security while maintaining performance and reducing computational resources compared to full fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08527v1">Scaling Laws for Predicting Downstream Performance in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Precise estimation of downstream performance in large language models (LLMs) prior to training is essential for guiding their development process. Scaling laws analysis utilizes the statistics of a series of significantly smaller sampling language models (LMs) to predict the performance of the target LLM. For downstream performance prediction, the critical challenge lies in the emergent abilities in LLMs that occur beyond task-specific computational thresholds. In this work, we focus on the pre-training loss as a more computation-efficient metric for performance estimation. Our two-stage approach consists of first estimating a function that maps computational resources (e.g., FLOPs) to the pre-training Loss using a series of sampling models, followed by mapping the pre-training loss to downstream task Performance after the critical "emergent phase". In preliminary experiments, this FLP solution accurately predicts the performance of LLMs with 7B and 13B parameters using a series of sampling LMs up to 3B, achieving error margins of 5% and 10%, respectively, and significantly outperforming the FLOPs-to-Performance approach. This motivates FLP-M, a fundamental approach for performance prediction that addresses the practical need to integrate datasets from multiple sources during pre-training, specifically blending general corpora with code data to accurately represent the common necessity. FLP-M extends the power law analytical function to predict domain-specific pre-training loss based on FLOPs across data sources, and employs a two-layer neural network to model the non-linear relationship between multiple domain-specific loss and downstream performance. By utilizing a 3B LLM trained on a specific ratio and a series of smaller sampling LMs, FLP-M can effectively forecast the performance of 3B and 7B LLMs across various data mixtures for most benchmarks within 10% error margins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16180v2">Benchmarking Japanese Speech Recognition on ASR-LLM Setups with Multi-Pass Augmented Generative Error Correction</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      With the strong representational power of large language models (LLMs), generative error correction (GER) for automatic speech recognition (ASR) aims to provide semantic and phonetic refinements to address ASR errors. This work explores how LLM-based GER can enhance and expand the capabilities of Japanese language processing, presenting the first GER benchmark for Japanese ASR with 0.9-2.6k text utterances. We also introduce a new multi-pass augmented generative error correction (MPA GER) by integrating multiple system hypotheses on the input side with corrections from multiple LLMs on the output side and then merging them. To the best of our knowledge, this is the first investigation of the use of LLMs for Japanese GER, which involves second-pass language modeling on the output transcriptions generated by the ASR system (e.g., N-best hypotheses). Our experiments demonstrated performance improvement in the proposed methods of ASR quality and generalization both in SPREDS-U1-ja and CSJ data.
    </details>
</div>
