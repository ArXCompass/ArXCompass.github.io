# llm - 2024_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15353v1">Contextualization of ASR with LLM using phonetic retrieval-based augmentation</a></div>
    <div class="paper-meta">
      📅 2024-09-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown superb capability of modeling multimodal signals including audio and text, allowing the model to generate spoken or textual response given a speech input. However, it remains a challenge for the model to recognize personal named entities, such as contacts in a phone book, when the input modality is speech. In this work, we start with a speech recognition task and propose a retrieval-based solution to contextualize the LLM: we first let the LLM detect named entities in speech without any context, then use this named entity as a query to retrieve phonetically similar named entities from a personal database and feed them to the LLM, and finally run context-aware LLM decoding. In a voice assistant task, our solution achieved up to 30.2% relative word error rate reduction and 73.6% relative named entity error rate reduction compared to a baseline system without contextualization. Notably, our solution by design avoids prompting the LLM with the full named entity database, making it highly efficient and applicable to large named entity databases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07355v1">Think Together and Work Better: Combining Humans' and LLMs' Think-Aloud Outcomes for Effective Text Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-09-11
    </div>
    <details class="paper-abstract">
      This study introduces \textbf{InteractEval}, a framework that integrates human expertise and Large Language Models (LLMs) using the Think-Aloud (TA) method to generate attributes for checklist-based text evaluation. By combining human flexibility and reasoning with LLM consistency, InteractEval outperforms traditional non-LLM-based and LLM-based baselines across four distinct dimensions, consisting of Coherence, Fluency, Consistency, and Relevance. The experiment also investigates the effectiveness of the TA method, showing that it promotes divergent thinking in both humans and LLMs, leading to the generation of a wider range of relevant attributes and enhance text evaluation performance. Comparative analysis reveals that humans excel at identifying attributes related to internal quality (Coherence and Fluency), but LLMs perform better at those attributes related to external alignment (Consistency and Relevance). Consequently, leveraging both humans and LLMs together produces the best evaluation outcomes. In other words, this study emphasizes the necessity of effectively combining humans and LLMs in an automated checklist-based text evaluation framework. The code is available at \textbf{\url{https://github.com/BBeeChu/InteractEval.git}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07314v1">MEDIC: Towards a Comprehensive Framework for Evaluating LLMs in Clinical Applications</a></div>
    <div class="paper-meta">
      📅 2024-09-11
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      The rapid development of Large Language Models (LLMs) for healthcare applications has spurred calls for holistic evaluation beyond frequently-cited benchmarks like USMLE, to better reflect real-world performance. While real-world assessments are valuable indicators of utility, they often lag behind the pace of LLM evolution, likely rendering findings obsolete upon deployment. This temporal disconnect necessitates a comprehensive upfront evaluation that can guide model selection for specific clinical applications. We introduce MEDIC, a framework assessing LLMs across five critical dimensions of clinical competence: medical reasoning, ethics and bias, data and language understanding, in-context learning, and clinical safety. MEDIC features a novel cross-examination framework quantifying LLM performance across areas like coverage and hallucination detection, without requiring reference outputs. We apply MEDIC to evaluate LLMs on medical question-answering, safety, summarization, note generation, and other tasks. Our results show performance disparities across model sizes, baseline vs medically finetuned models, and have implications on model selection for applications requiring specific model strengths, such as low hallucination or lower cost of inference. MEDIC's multifaceted evaluation reveals these performance trade-offs, bridging the gap between theoretical capabilities and practical implementation in healthcare settings, ensuring that the most promising models are identified and adapted for diverse healthcare applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.05993v2">AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts</a></div>
    <div class="paper-meta">
      📅 2024-09-11
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) and generative AI become more widespread, the content safety risks associated with their use also increase. We find a notable deficiency in high-quality content safety datasets and benchmarks that comprehensively cover a wide range of critical safety areas. To address this, we define a broad content safety risk taxonomy, comprising 13 critical risk and 9 sparse risk categories. Additionally, we curate AEGISSAFETYDATASET, a new dataset of approximately 26, 000 human-LLM interaction instances, complete with human annotations adhering to the taxonomy. We plan to release this dataset to the community to further research and to help benchmark LLM models for safety. To demonstrate the effectiveness of the dataset, we instruction-tune multiple LLM-based safety models. We show that our models (named AEGISSAFETYEXPERTS), not only surpass or perform competitively with the state-of-the-art LLM-based safety models and general purpose LLMs, but also exhibit robustness across multiple jail-break attack categories. We also show how using AEGISSAFETYDATASET during the LLM alignment phase does not negatively impact the performance of the aligned models on MT Bench scores. Furthermore, we propose AEGIS, a novel application of a no-regret online adaptation framework with strong theoretical guarantees, to perform content moderation with an ensemble of LLM content safety experts in deployment
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07507v1">Traceable LLM-based validation of statements in knowledge graphs</a></div>
    <div class="paper-meta">
      📅 2024-09-11
    </div>
    <details class="paper-abstract">
      This article presents a method for verifying RDF triples using LLMs, with an emphasis on providing traceable arguments. Because the LLMs cannot currently reliably identify the origin of the information used to construct the response to the user query, our approach is to avoid using internal LLM factual knowledge altogether. Instead, verified RDF statements are compared to chunks of external documents retrieved through a web search or Wikipedia. To assess the possible application of this workflow on biosciences content, we evaluated 1,719 positive statements from the BioRED dataset and the same number of newly generated negative statements. The resulting precision is 88%, and recall is 44%. This indicates that the method requires human oversight. We demonstrate the method on Wikidata, where a SPARQL query is used to automatically retrieve statements needing verification. Overall, the results suggest that LLMs could be used for large-scale verification of statements in KGs, a task previously unfeasible due to human annotation costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02616v5">Adaptive Layer Splitting for Wireless LLM Inference in Edge Computing: A Model-Based Reinforcement Learning Approach</a></div>
    <div class="paper-meta">
      📅 2024-09-11
    </div>
    <details class="paper-abstract">
      Optimizing the deployment of large language models (LLMs) in edge computing environments is critical for enhancing privacy and computational efficiency. Toward efficient wireless LLM inference in edge computing, this study comprehensively analyzes the impact of different splitting points in mainstream open-source LLMs. On this basis, this study introduces a framework taking inspiration from model-based reinforcement learning (MBRL) to determine the optimal splitting point across the edge and user equipment (UE). By incorporating a reward surrogate model, our approach significantly reduces the computational cost of frequent performance evaluations. Extensive simulations demonstrate that this method effectively balances inference performance and computational load under varying network conditions, providing a robust solution for LLM deployment in decentralized settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07132v1">LLM-based feature generation from text for interpretable machine learning</a></div>
    <div class="paper-meta">
      📅 2024-09-11
    </div>
    <details class="paper-abstract">
      Existing text representations such as embeddings and bag-of-words are not suitable for rule learning due to their high dimensionality and absent or questionable feature-level interpretability. This article explores whether large language models (LLMs) could address this by extracting a small number of interpretable features from text. We demonstrate this process on two datasets (CORD-19 and M17+) containing several thousand scientific articles from multiple disciplines and a target being a proxy for research impact. An evaluation based on testing for the statistically significant correlation with research impact has shown that LLama 2-generated features are semantically meaningful. We consequently used these generated features in text classification to predict the binary target variable representing the citation rate for the CORD-19 dataset and the ordinal 5-class target representing an expert-awarded grade in the M17+ dataset. Machine-learning models trained on the LLM-generated features provided similar predictive performance to the state-of-the-art embedding model SciBERT for scientific text. The LLM used only 62 features compared to 768 features in SciBERT embeddings, and these features were directly interpretable, corresponding to notions such as article methodological rigor, novelty, or grammatical correctness. As the final step, we extract a small number of well-interpretable action rules. Consistently competitive results obtained with the same LLM feature set across both thematically diverse datasets show that this approach generalizes across domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16352v2">MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-11
      | 💬 ACL 2024 camera ready
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited great potential in mathematical reasoning. However, there remains a performance gap in this area between existing open-source models and closed-source models such as GPT-4. In this paper, we introduce MathGenie, a novel method for generating diverse and reliable math problems from a small-scale problem-solution dataset (denoted as seed data). We augment the ground-truth solutions of our seed data and train a back-translation model to translate the augmented solutions back into new questions. Subsequently, we generate code-integrated solutions for the new questions. To ensure the correctness of the code-integrated solutions, we employ rationale-based strategy for solution verification. Various pretrained models, ranging from 7B to 70B, are trained on the newly curated data to test the effectiveness of the proposed augmentation technique, resulting in a family of models known as MathGenieLM. These models consistently outperform previous open-source models across five representative mathematical reasoning datasets, achieving state-of-the-art performance. In particular, MathGenieLM-InternLM2 achieves an accuracy of 87.7% on GSM8K and 55.7% on MATH, securing the best overall score among open-source language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07085v1">Understanding Knowledge Drift in LLMs through Misinformation</a></div>
    <div class="paper-meta">
      📅 2024-09-11
      | 💬 13 pages, 3 figures. Accepted at DELTA workshop at KDD 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized numerous applications, making them an integral part of our digital ecosystem. However, their reliability becomes critical, especially when these models are exposed to misinformation. We primarily analyze the susceptibility of state-of-the-art LLMs to factual inaccuracies when they encounter false information in a QnA scenario, an issue that can lead to a phenomenon we refer to as *knowledge drift*, which significantly undermines the trustworthiness of these models. We evaluate the factuality and the uncertainty of the models' responses relying on Entropy, Perplexity, and Token Probability metrics. Our experiments reveal that an LLM's uncertainty can increase up to 56.6% when the question is answered incorrectly due to the exposure to false information. At the same time, repeated exposure to the same false information can decrease the models uncertainty again (-52.8% w.r.t. the answers on the untainted prompts), potentially manipulating the underlying model's beliefs and introducing a drift from its original knowledge. These findings provide insights into LLMs' robustness and vulnerability to adversarial inputs, paving the way for developing more reliable LLM applications across various domains. The code is available at https://github.com/afastowski/knowledge_drift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00060v2">Understanding Literary Texts by LLMs: A Case Study of Ancient Chinese Poetry</a></div>
    <div class="paper-meta">
      📅 2024-09-11
    </div>
    <details class="paper-abstract">
      The birth and rapid development of large language models (LLMs) have caused quite a stir in the field of literature. Once considered unattainable, AI's role in literary creation is increasingly becoming a reality. In genres such as poetry, jokes, and short stories, numerous AI tools have emerged, offering refreshing new perspectives. However, it's difficult to further improve the quality of these works. This is primarily because understanding and appreciating a good literary work involves a considerable threshold, such as knowledge of literary theory, aesthetic sensibility, interdisciplinary knowledge. Therefore, authoritative data in this area is quite lacking. Additionally, evaluating literary works is often complex and hard to fully quantify, which directly hinders the further development of AI creation. To address this issue, this paper attempts to explore the mysteries of literary texts from the perspective of LLMs, using ancient Chinese poetry as an example for experimentation. First, we collected a variety of ancient poems from different sources and had experts annotate a small portion of them. Then, we designed a range of comprehension metrics based on LLMs to evaluate all these poems. Finally, we analyzed the correlations and differences between various poem collections to identify literary patterns. Through our experiments, we observed a series of enlightening phenomena that provide technical support for the future development of high-level literary creation based on LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07503v1">AdaPPA: Adaptive Position Pre-Fill Jailbreak Attack Approach Targeting LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-11
    </div>
    <details class="paper-abstract">
      Jailbreak vulnerabilities in Large Language Models (LLMs) refer to methods that extract malicious content from the model by carefully crafting prompts or suffixes, which has garnered significant attention from the research community. However, traditional attack methods, which primarily focus on the semantic level, are easily detected by the model. These methods overlook the difference in the model's alignment protection capabilities at different output stages. To address this issue, we propose an adaptive position pre-fill jailbreak attack approach for executing jailbreak attacks on LLMs. Our method leverages the model's instruction-following capabilities to first output pre-filled safe content, then exploits its narrative-shifting abilities to generate harmful content. Extensive black-box experiments demonstrate our method can improve the attack success rate by 47% on the widely recognized secure model (Llama2) compared to existing approaches. Our code can be found at: https://github.com/Yummy416/AdaPPA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00008v2">ScaleLLM: A Resource-Frugal LLM Serving Framework by Optimizing End-to-End Efficiency</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have surged in popularity and are extensively used in commercial applications, where the efficiency of model serving is crucial for the user experience. Most current research focuses on optimizing individual sub-procedures, e.g. local inference and communication, however, there is no comprehensive framework that provides a holistic system view for optimizing LLM serving in an end-to-end manner. In this work, we conduct a detailed analysis to identify major bottlenecks that impact end-to-end latency in LLM serving systems. Our analysis reveals that a comprehensive LLM serving endpoint must address a series of efficiency bottlenecks that extend beyond LLM inference. We then propose ScaleLLM, an optimized system for resource-efficient LLM serving. Our extensive experiments reveal that with 64 concurrent requests, ScaleLLM achieves a 4.3x speed up over vLLM and outperforms state-of-the-arts with 1.5x higher throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06883v1">A Dataset for Evaluating LLM-based Evaluation Functions for Research Question Extraction Task</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      The progress in text summarization techniques has been remarkable. However the task of accurately extracting and summarizing necessary information from highly specialized documents such as research papers has not been sufficiently investigated. We are focusing on the task of extracting research questions (RQ) from research papers and construct a new dataset consisting of machine learning papers, RQ extracted from these papers by GPT-4, and human evaluations of the extracted RQ from multiple perspectives. Using this dataset, we systematically compared recently proposed LLM-based evaluation functions for summarizations, and found that none of the functions showed sufficiently high correlations with human evaluations. We expect our dataset provides a foundation for further research on developing better evaluation functions tailored to the RQ extraction task, and contribute to enhance the performance of the task. The dataset is available at https://github.com/auto-res/PaperRQ-HumanAnno-Dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05735v2">A System and Benchmark for LLM-based Q&A on Heterogeneous Data</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      In many industrial settings, users wish to ask questions whose answers may be found in structured data sources such as a spreadsheets, databases, APIs, or combinations thereof. Often, the user doesn't know how to identify or access the right data source. This problem is compounded even further if multiple (and potentially siloed) data sources must be assembled to derive the answer. Recently, various Text-to-SQL applications that leverage Large Language Models (LLMs) have addressed some of these problems by enabling users to ask questions in natural language. However, these applications remain impractical in realistic industrial settings because they fail to cope with the data source heterogeneity that typifies such environments. In this paper, we address heterogeneity by introducing the siwarex platform, which enables seamless natural language access to both databases and APIs. To demonstrate the effectiveness of siwarex, we extend the popular Spider dataset and benchmark by replacing some of its tables by data retrieval APIs. We find that siwarex does a good job of coping with data source heterogeneity. Our modified Spider benchmark will soon be available to the research community
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13918v3">Geo-Llama: Leveraging LLMs for Human Mobility Trajectory Generation with Spatiotemporal Constraints</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      Simulating human mobility data is essential for various application domains, including transportation, urban planning, and epidemic control, since real data are often inaccessible to researchers due to expensive costs and privacy issues. Several existing deep generative solutions propose learning from real trajectories to generate synthetic ones. Despite the progress, most of them suffer from training stability issues and scale poorly with growing data size. More importantly, they generally lack control mechanisms to steer the generated trajectories based on spatiotemporal constraints such as fixing specific visits. To address such limitations, we formally define the controlled trajectory generation problem with spatiotemporal constraints and propose Geo-Llama. This novel LLM-inspired framework enforces explicit visit constraints in a contextually coherent way. It fine-tunes pre-trained LLMs on trajectories with a visit-wise permutation strategy where each visit corresponds to a time and location. This enables the model to capture the spatiotemporal patterns regardless of visit orders and allows flexible and in-context constraint integration through prompts during generation. Extensive experiments on real-world and synthetic datasets validate the effectiveness of Geo-Llama, demonstrating its versatility and robustness in handling a broad range of constraints to generate more realistic trajectories compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06653v1">Human Perception of LLM-generated Text Content in Social Media Environments</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      Emerging technologies, particularly artificial intelligence (AI), and more specifically Large Language Models (LLMs) have provided malicious actors with powerful tools for manipulating digital discourse. LLMs have the potential to affect traditional forms of democratic engagements, such as voter choice, government surveys, or even online communication with regulators; since bots are capable of producing large quantities of credible text. To investigate the human perception of LLM-generated content, we recruited over 1,000 participants who then tried to differentiate bot from human posts in social media discussion threads. We found that humans perform poorly at identifying the true nature of user posts on social media. We also found patterns in how humans identify LLM-generated text content in social media discourse. Finally, we observed the Uncanny Valley effect in text dialogue in both user perception and identification. This indicates that despite humans being poor at the identification process, they can still sense discomfort when reading LLM-generated content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06643v1">Strategic management analysis: from data to strategy diagram by LLM</a></div>
    <div class="paper-meta">
      📅 2024-09-10
      | 💬 NLVIZ Workshop at IEEE VIZ 2024. 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Strategy management analyses are created by business consultants with common analysis frameworks (i.e. comparative analyses) and associated diagrams. We show these can be largely constructed using LLMs, starting with the extraction of insights from data, organization of those insights according to a strategy management framework, and then depiction in the typical strategy management diagram for that framework (static textual visualizations). We discuss caveats and future directions to generalize for broader uses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06558v1">MAPS: Energy-Reliability Tradeoff Management in Autonomous Vehicles Through LLMs Penetrated Science</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      As autonomous vehicles become more prevalent, highly accurate and efficient systems are increasingly critical to improve safety, performance, and energy consumption. Efficient management of energy-reliability tradeoffs in these systems demands the ability to predict various conditions during vehicle operations. With the promising improvement of Large Language Models (LLMs) and the emergence of well-known models like ChatGPT, unique opportunities for autonomous vehicle-related predictions have been provided in recent years. This paper proposed MAPS using LLMs as map reader co-drivers to predict the vital parameters to set during the autonomous vehicle operation to balance the energy-reliability tradeoff. The MAPS method demonstrates a 20% improvement in navigation accuracy compared to the best baseline method. MAPS also shows 11% energy savings in computational units and up to 54% in both mechanical and computational units.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06540v1">Mapping News Narratives Using LLMs and Narrative-Structured Text Embeddings</a></div>
    <div class="paper-meta">
      📅 2024-09-10
      | 💬 19 pages, 13 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Given the profound impact of narratives across various societal levels, from personal identities to international politics, it is crucial to understand their distribution and development over time. This is particularly important in online spaces. On the Web, narratives can spread rapidly and intensify societal divides and conflicts. While many qualitative approaches exist, quantifying narratives remains a significant challenge. Computational narrative analysis lacks frameworks that are both comprehensive and generalizable. To address this gap, we introduce a numerical narrative representation grounded in structuralist linguistic theory. Chiefly, Greimas' Actantial Model represents a narrative through a constellation of six functional character roles. These so-called actants are genre-agnostic, making the model highly generalizable. We extract the actants using an open-source LLM and integrate them into a Narrative-Structured Text Embedding that captures both the semantics and narrative structure of a text. We demonstrate the analytical insights of the method on the example of 5000 full-text news articles from Al Jazeera and The Washington Post on the Israel-Palestine conflict. Our method successfully distinguishes articles that cover the same topics but differ in narrative structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09654v2">Do LLMs Understand Visual Anomalies? Uncovering LLM's Capabilities in Zero-shot Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2024-09-10
      | 💬 Accepted by MM'24 (Oral)
    </div>
    <details class="paper-abstract">
      Large vision-language models (LVLMs) are markedly proficient in deriving visual representations guided by natural language. Recent explorations have utilized LVLMs to tackle zero-shot visual anomaly detection (VAD) challenges by pairing images with textual descriptions indicative of normal and abnormal conditions, referred to as anomaly prompts. However, existing approaches depend on static anomaly prompts that are prone to cross-semantic ambiguity, and prioritize global image-level representations over crucial local pixel-level image-to-text alignment that is necessary for accurate anomaly localization. In this paper, we present ALFA, a training-free approach designed to address these challenges via a unified model. We propose a run-time prompt adaptation strategy, which first generates informative anomaly prompts to leverage the capabilities of a large language model (LLM). This strategy is enhanced by a contextual scoring mechanism for per-image anomaly prompt adaptation and cross-semantic ambiguity mitigation. We further introduce a novel fine-grained aligner to fuse local pixel-level semantics for precise anomaly localization, by projecting the image-text alignment from global to local semantic spaces. Extensive evaluations on MVTec and VisA datasets confirm ALFA's effectiveness in harnessing the language potential for zero-shot VAD, achieving significant PRO improvements of 12.1% on MVTec and 8.9% on VisA compared to state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04775v2">Leveraging LLMs, Graphs and Object Hierarchies for Task Planning in Large-Scale Environments</a></div>
    <div class="paper-meta">
      📅 2024-09-10
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Planning methods struggle with computational intractability in solving task-level problems in large-scale environments. This work explores leveraging the commonsense knowledge encoded in LLMs to empower planning techniques to deal with these complex scenarios. We achieve this by efficiently using LLMs to prune irrelevant components from the planning problem's state space, substantially simplifying its complexity. We demonstrate the efficacy of this system through extensive experiments within a household simulation environment, alongside real-world validation using a 7-DoF manipulator (video https://youtu.be/6ro2UOtOQS4).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06328v1">Extracting Paragraphs from LLM Token Activations</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) excel in natural language processing tasks, yet their inner workings remain underexplored beyond token-level predictions. This study investigates the degree to which these models decide the content of a paragraph at its onset, shedding light on their contextual understanding. By examining the information encoded in single-token activations, specifically the "\textbackslash n\textbackslash n" double newline token, we demonstrate that patching these activations can transfer significant information about the context of the following paragraph, providing further insights into the model's capacity to plan ahead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02897v3">LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-context QA</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      Though current long-context large language models (LLMs) have demonstrated impressive capacities in answering user questions based on extensive text, the lack of citations in their responses makes user verification difficult, leading to concerns about their trustworthiness due to their potential hallucinations. In this work, we aim to enable long-context LLMs to generate responses with fine-grained sentence-level citations, improving their faithfulness and verifiability. We first introduce LongBench-Cite, an automated benchmark for assessing current LLMs' performance in Long-Context Question Answering with Citations (LQAC), revealing considerable room for improvement. To this end, we propose CoF (Coarse to Fine), a novel pipeline that utilizes off-the-shelf LLMs to automatically generate long-context QA instances with precise sentence-level citations, and leverage this pipeline to construct LongCite-45k, a large-scale SFT dataset for LQAC. Finally, we train LongCite-8B and LongCite-9B using the LongCite-45k dataset, successfully enabling their generation of accurate responses and fine-grained sentence-level citations in a single output. The evaluation results on LongBench-Cite show that our trained models achieve state-of-the-art citation quality, surpassing advanced proprietary models including GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06289v1">Automate Strategy Finding with LLM in Quant investment</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      Despite significant progress in deep learning for financial trading, existing models often face instability and high uncertainty, hindering their practical application. Leveraging advancements in Large Language Models (LLMs) and multi-agent architectures, we propose a novel framework for quantitative stock investment in portfolio management and alpha mining. Our framework addresses these issues by integrating LLMs to generate diversified alphas and employing a multi-agent approach to dynamically evaluate market conditions. This paper proposes a framework where large language models (LLMs) mine alpha factors from multimodal financial data, ensuring a comprehensive understanding of market dynamics. The first module extracts predictive signals by integrating numerical data, research papers, and visual charts. The second module uses ensemble learning to construct a diverse pool of trading agents with varying risk preferences, enhancing strategy performance through a broader market analysis. In the third module, a dynamic weight-gating mechanism selects and assigns weights to the most relevant agents based on real-time market conditions, enabling the creation of an adaptive and context-aware composite alpha formula. Extensive experiments on the Chinese stock markets demonstrate that this framework significantly outperforms state-of-the-art baselines across multiple financial metrics. The results underscore the efficacy of combining LLM-generated alphas with a multi-agent architecture to achieve superior trading performance and stability. This work highlights the potential of AI-driven approaches in enhancing quantitative investment strategies and sets a new benchmark for integrating advanced machine learning techniques in financial trading can also be applied on diverse markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06205v1">SHAPE-IT: Exploring Text-to-Shape-Display for Generative Shape-Changing Behaviors with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-10
      | 💬 Accepted for ACM UIST 2024
    </div>
    <details class="paper-abstract">
      This paper introduces text-to-shape-display, a novel approach to generating dynamic shape changes in pin-based shape displays through natural language commands. By leveraging large language models (LLMs) and AI-chaining, our approach allows users to author shape-changing behaviors on demand through text prompts without programming. We describe the foundational aspects necessary for such a system, including the identification of key generative elements (primitive, animation, and interaction) and design requirements to enhance user interaction, based on formative exploration and iterative design processes. Based on these insights, we develop SHAPE-IT, an LLM-based authoring tool for a 24 x 24 shape display, which translates the user's textual command into executable code and allows for quick exploration through a web-based control interface. We evaluate the effectiveness of SHAPE-IT in two ways: 1) performance evaluation and 2) user evaluation (N= 10). The study conclusions highlight the ability to facilitate rapid ideation of a wide range of shape-changing behaviors with AI. However, the findings also expose accuracy-related challenges and limitations, prompting further exploration into refining the framework for leveraging AI to better suit the unique requirements of shape-changing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06192v1">NOVI : Chatbot System for University Novice with BERT and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      To mitigate the difficulties of university freshmen in adapting to university life, we developed NOVI, a chatbot system based on GPT-4o. This system utilizes post and comment data from SKKU 'Everytime', a university community site. Developed using LangChain, NOVI's performance has been evaluated with a BLEU score, Perplexity score, ROUGE-1 score, ROUGE-2 score, ROUGE-L score and METEOR score. This approach is not only limited to help university freshmen but is also expected to help various people adapting to new environments with different data. This research explores the development and potential application of new educational technology tools, contributing to easier social adaptation for beginners and settling a foundation for future advancement in LLM studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15343v1">Advertiser Content Understanding via LLMs for Google Ads Safety</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      Ads Content Safety at Google requires classifying billions of ads for Google Ads content policies. Consistent and accurate policy enforcement is important for advertiser experience and user safety and it is a challenging problem, so there is a lot of value for improving it for advertisers and users. Inconsistent policy enforcement causes increased policy friction and poor experience with good advertisers, and bad advertisers exploit the inconsistency by creating multiple similar ads in the hope that some will get through our defenses. This study proposes a method to understand advertiser's intent for content policy violations, using Large Language Models (LLMs). We focus on identifying good advertisers to reduce content over-flagging and improve advertiser experience, though the approach can easily be extended to classify bad advertisers too. We generate advertiser's content profile based on multiple signals from their ads, domains, targeting info, etc. We then use LLMs to classify the advertiser content profile, along with relying on any knowledge the LLM has of the advertiser, their products or brand, to understand whether they are likely to violate a certain policy or not. After minimal prompt tuning our method was able to reach 95\% accuracy on a small test set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18276v3">Rome was Not Built in a Single Step: Hierarchical Prompting for LLM-based Chip Design</a></div>
    <div class="paper-meta">
      📅 2024-09-09
      | 💬 Accepted at MLCAD '24. 10 pages, 7 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are effective in computer hardware synthesis via hardware description language (HDL) generation. However, LLM-assisted approaches for HDL generation struggle when handling complex tasks. We introduce a suite of hierarchical prompting techniques which facilitate efficient stepwise design methods, and develop a generalizable automation pipeline for the process. To evaluate these techniques, we present a benchmark set of hardware designs which have solutions with or without architectural hierarchy. Using these benchmarks, we compare various open-source and proprietary LLMs, including our own fine-tuned Code Llama-Verilog model. Our hierarchical methods automatically produce successful designs for complex hardware modules that standard flat prompting methods cannot achieve, allowing smaller open-source LLMs to compete with large proprietary models. Hierarchical prompting reduces HDL generation time and yields savings on LLM costs. Our experiments detail which LLMs are capable of which applications, and how to apply hierarchical methods in various modes. We explore case studies of generating complex cores using automatic scripted hierarchical prompts, including the first-ever LLM-designed processor with no human feedback. Tools for the Recurrent Optimization via Machine Editing (ROME) method can be found at https://github.com/ajn313/ROME-LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13598v2">User-LLM: Efficient LLM Contextualization with User Embeddings</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across various domains, but effectively incorporating complex and potentially noisy user timeline data into LLMs remains a challenge. Current approaches often involve translating user timelines into text descriptions before feeding them to LLMs, which can be inefficient and may not fully capture the nuances of user behavior. Inspired by how LLMs are effectively integrated with images through direct embeddings, we propose User-LLM, a novel framework that leverages user embeddings to directly contextualize LLMs with user history interactions. These embeddings, generated by a user encoder pretrained using self-supervised learning on diverse user interactions, capture latent user behaviors and interests as well as their evolution over time. We integrate these user embeddings with LLMs through cross-attention, enabling LLMs to dynamically adapt their responses based on the context of a user's past actions and preferences. Our approach achieves significant efficiency gains by representing user timelines directly as embeddings, leading to substantial inference speedups of up to 78.1X. Comprehensive experiments on MovieLens, Amazon Review, and Google Local Review datasets demonstrate that User-LLM outperforms text-prompt-based contextualization on tasks requiring deep user understanding, with improvements of up to 16.33%, particularly excelling on long sequences that capture subtle shifts in user behavior. Furthermore, the incorporation of Perceiver layers streamlines the integration between user encoders and LLMs, yielding additional computational savings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05746v1">LLMs Will Always Hallucinate, and We Need to Live With This</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      As Large Language Models become more ubiquitous across domains, it becomes important to examine their inherent limitations critically. This work argues that hallucinations in language models are not just occasional errors but an inevitable feature of these systems. We demonstrate that hallucinations stem from the fundamental mathematical and logical structure of LLMs. It is, therefore, impossible to eliminate them through architectural improvements, dataset enhancements, or fact-checking mechanisms. Our analysis draws on computational theory and Godel's First Incompleteness Theorem, which references the undecidability of problems like the Halting, Emptiness, and Acceptance Problems. We demonstrate that every stage of the LLM process-from training data compilation to fact retrieval, intent classification, and text generation-will have a non-zero probability of producing hallucinations. This work introduces the concept of Structural Hallucination as an intrinsic nature of these systems. By establishing the mathematical certainty of hallucinations, we challenge the prevailing notion that they can be fully mitigated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.18799v2">X-InstructBLIP: A Framework for aligning X-Modal instruction-aware representations to LLMs and Emergent Cross-modal Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      Recent research has achieved significant advancements in visual reasoning tasks through learning image-to-language projections and leveraging the impressive reasoning abilities of Large Language Models (LLMs). This paper introduces an efficient and effective framework that integrates multiple modalities (images, 3D, audio and video) to a frozen LLM and demonstrates an emergent ability for cross-modal reasoning (2+ modality inputs). Our approach explores two distinct projection mechanisms: Q-Formers and Linear Projections (LPs). Through extensive experimentation across all four modalities on 16 benchmarks, we explore both methods and assess their adaptability in integrated and separate cross-modal reasoning. The Q-Former projection demonstrates superior performance in single modality scenarios and adaptability in joint versus discriminative reasoning involving two or more modalities. However, it exhibits lower generalization capabilities than linear projection in contexts where task-modality data are limited. To enable this framework, we devise a scalable pipeline that automatically generates high-quality, instruction-tuning datasets from readily available captioning data across different modalities, and contribute 24K QA data for audio and 250K QA data for 3D. To facilitate further research in cross-modal reasoning, we introduce the DisCRn (Discriminative Cross-modal Reasoning) benchmark comprising 9K audio-video QA samples and 28K image-3D QA samples that require the model to reason discriminatively across disparate input modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05559v1">CauseJudger: Identifying the Cause with LLMs for Abductive Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been utilized in solving diverse reasoning tasks, encompassing common sense, arithmetic and deduction tasks. However, with difficulties of reversing thinking patterns and irrelevant premises, how to determine the authenticity of the cause in abductive logical reasoning remains underexplored. Inspired by hypothesis and verification method and identification of irrelevant information in human thinking process, we propose a new framework for LLMs abductive logical reasoning called CauseJudger (CJ), which identifies the authenticity of possible cause by transforming thinking from reverse to forward and removing irrelevant information. In addition, we construct an abductive logical reasoning dataset for decision task called CauseLogics, which contains 200,000 tasks of varying reasoning lengths. Our experiments show the efficiency of CJ with overall experiments and ablation experiments as well as case studies on our dataset and reconstructed public dataset. Notably, CJ's implementation is efficient, requiring only two calls to LLM. Its impact is profound: when using gpt-3.5, CJ achieves a maximum correctness improvement of 41% compared to Zero-Shot-CoT. Moreover, with gpt-4, CJ attains an accuracy exceeding 90% across all datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.07601v4">Online Advertisements with LLMs: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      This paper explores the potential for leveraging Large Language Models (LLM) in the realm of online advertising systems. We introduce a general framework for LLM advertisement, consisting of modification, bidding, prediction, and auction modules. Different design considerations for each module are presented. These design choices are evaluated and discussed based on essential desiderata required to maintain a sustainable system. Further fundamental questions regarding practicality, efficiency, and implementation challenges are raised for future research. Finally, we exposit how recent approaches on mechanism design for LLM can be framed in our unified perspective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00523v2">Jailbreaking Text-to-Image Models with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      Recent advancements have significantly improved automated task-solving capabilities using autonomous agents powered by large language models (LLMs). However, most LLM-based agents focus on dialogue, programming, or specialized domains, leaving their potential for addressing generative AI safety tasks largely unexplored. In this paper, we propose Atlas, an advanced LLM-based multi-agent framework targeting generative AI models, specifically focusing on jailbreak attacks against text-to-image (T2I) models with built-in safety filters. Atlas consists of two agents, namely the mutation agent and the selection agent, each comprising four key modules: a vision-language model (VLM) or LLM brain, planning, memory, and tool usage. The mutation agent uses its VLM brain to determine whether a prompt triggers the T2I model's safety filter. It then collaborates iteratively with the LLM brain of the selection agent to generate new candidate jailbreak prompts with the highest potential to bypass the filter. In addition to multi-agent communication, we leverage in-context learning (ICL) memory mechanisms and the chain-of-thought (COT) approach to learn from past successes and failures, thereby enhancing Atlas's performance. Our evaluation demonstrates that Atlas successfully jailbreaks several state-of-the-art T2I models equipped with multi-modal safety filters in a black-box setting. Additionally, Atlas outperforms existing methods in both query efficiency and the quality of generated images. This work convincingly demonstrates the successful application of LLM-based agents in studying the safety vulnerabilities of popular text-to-image generation models. We urge the community to consider advanced techniques like ours in response to the rapidly evolving text-to-image generation field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04363v2">AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-09-09
      | 💬 Code for this work is avaliable at https://github.com/AIRI-Institute/AriGraph
    </div>
    <details class="paper-abstract">
      Advancements in the capabilities of Large Language Models (LLMs) have created a promising foundation for developing autonomous agents. With the right tools, these agents could learn to solve tasks in new environments by accumulating and updating their knowledge. Current LLM-based agents process past experiences using a full history of observations, summarization, retrieval augmentation. However, these unstructured memory representations do not facilitate the reasoning and planning essential for complex decision-making. In our study, we introduce AriGraph, a novel method wherein the agent constructs and updates a memory graph that integrates semantic and episodic memories while exploring the environment. We demonstrate that our Ariadne LLM agent, consisting of the proposed memory architecture augmented with planning and decision-making, effectively handles complex tasks within interactive text game environments difficult even for human players. Results show that our approach markedly outperforms other established memory methods and strong RL baselines in a range of problems of varying complexity. Additionally, AriGraph demonstrates competitive performance compared to dedicated knowledge graph-based methods in static multi-hop question-answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04392v3">Fine-Tuning, Quantization, and LLMs: Navigating Unintended Outcomes</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained widespread adoption across various domains, including chatbots and auto-task completion agents. However, these models are susceptible to safety vulnerabilities such as jailbreaking, prompt injection, and privacy leakage attacks. These vulnerabilities can lead to the generation of malicious content, unauthorized actions, or the disclosure of confidential information. While foundational LLMs undergo alignment training and incorporate safety measures, they are often subject to fine-tuning, or doing quantization resource-constrained environments. This study investigates the impact of these modifications on LLM safety, a critical consideration for building reliable and secure AI systems. We evaluate foundational models including Mistral, Llama series, Qwen, and MosaicML, along with their fine-tuned variants. Our comprehensive analysis reveals that fine-tuning generally increases the success rates of jailbreak attacks, while quantization has variable effects on attack success rates. Importantly, we find that properly implemented guardrails significantly enhance resistance to jailbreak attempts. These findings contribute to our understanding of LLM vulnerabilities and provide insights for developing more robust safety strategies in the deployment of language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14482v2">ChatQA 2: Bridging the Gap to Proprietary LLMs in Long Context and RAG Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-09-09
      | 💬 v2: major update with significantly improved results
    </div>
    <details class="paper-abstract">
      In this work, we introduce ChatQA 2, an Llama 3.0-based model with a 128K context window, designed to bridge the gap between open-source LLMs and leading proprietary models (e.g., GPT-4-Turbo) in long-context understanding and retrieval-augmented generation (RAG) capabilities. These two capabilities are essential for LLMs to process large volumes of information that cannot fit into a single prompt and are complementary to each other, depending on the downstream tasks and computational budgets. We present a detailed continued training recipe to extend the context window of Llama3-70B-base from 8K to 128K tokens, along with a three-stage instruction tuning process to enhance the model's instruction-following, RAG performance, and long-context understanding capabilities. Our results demonstrate that the Llama3-ChatQA-2-70B model outperforms most existing state-of-the-art models, including GPT-4-Turbo-2024-04-09, Qwen2-72B-Instruct, and Llama3.1-70B-Instruct, on ultra-long tasks beyond 100K tokens, as well as on the RAG benchmark using only a 4K context window, showing the strong long context capability across varying sequence lengths. We further provide extensive comparisons between direct long-context and RAG solutions using the same state-of-the-art long-context LLMs. Interestingly, we find that the performance of strong long-context LLMs using RAG improves when retrieving a larger number of chunks. With a large set of top-k chunks, RAG consistently outperforms direct long-context solution using the same state-of-the-art long-context models (e.g., Llama3-ChatQA-2-70B and Qwen2-72B-Instruct) on both 32K benchmarks and real-world 128K tasks. To advance research in this field, we open-sourced the model weights, training data, and the evaluation setup for the for the community: https://chatqa2-project.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11416v1">The Unseen AI Disruptions for Power Grids: LLM-Induced Transients</a></div>
    <div class="paper-meta">
      📅 2024-09-09
      | 💬 21 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Recent breakthroughs of large language models (LLMs) have exhibited superior capability across major industries and stimulated multi-hundred-billion-dollar investment in AI-centric data centers in the next 3-5 years. This, in turn, bring the increasing concerns on sustainability and AI-related energy usage. However, there is a largely overlooked issue as challenging and critical as AI model and infrastructure efficiency: the disruptive dynamic power consumption behaviour. With fast, transient dynamics, AI infrastructure features ultra-low inertia, sharp power surge and dip, and a significant peak-idle power ratio. The power scale covers from several hundred watts to megawatts, even to gigawatts. These never-seen-before characteristics make AI a very unique load and pose threats to the power grid reliability and resilience. To reveal this hidden problem, this paper examines the scale of AI power consumption, analyzes AI transient behaviour in various scenarios, develops high-level mathematical models to depict AI workload behaviour and discusses the multifaceted challenges and opportunities they potentially bring to existing power grids. Observing the rapidly evolving machine learning (ML) and AI technologies, this work emphasizes the critical need for interdisciplinary approaches to ensure reliable and sustainable AI infrastructure development, and provides a starting point for researchers and practitioners to tackle such challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05923v1">$\mathbb{USCD}$: Improving Code Generation of LLMs by Uncertainty-Aware Selective Contrastive Decoding</a></div>
    <div class="paper-meta">
      📅 2024-09-09
      | 💬 13pages,8 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in code generation. However, the effects of hallucinations (e.g., output noise) make it particularly challenging for LLMs to generate high-quality code in one pass. In this work, we propose a simple and effective \textbf{u}ncertainty-aware \textbf{s}elective \textbf{c}ontrastive \textbf{d}ecoding ($\mathbb{USCD}$) mechanism to improve the quality of one-pass code generation in LLMs and reduce the impact of output noise. To be specific, we first elaborately designed a negative prompt (namely lame prompt) to output noise by removing input-output examples from the standard few-shot prompt. Our preliminary study shows that the Jensen-Shannon divergence (JS divergence) between token distribution uncertainty and the output noise is relatively low (approximately $0.25$), indicating their high relevance. Then, we selectively eliminate output noise induced by lame prompts based on the uncertainty of the prediction distribution from the standard prompt. Notably, our proposed plug-and-play mechanism is an inference-only method, enjoying appealing flexibility. Extensive experiments on widely used benchmarks, e.g., HumanEval, MBPP, and MultiPL-E, upon several LLMs (i.e., Inocder-6b, CodeLlama-7b, WizardCoder-15b, StarCoder, and Llama2-7b), demonstrate that our proposed USCD significantly improves one-pass code generation, with an average \textit{pass@$1$} scores increase of 16.59\%. We will release code and data on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03515v2">A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) like GPT-4o into robotic systems represents a significant advancement in embodied artificial intelligence. These models can process multi-modal prompts, enabling them to generate more context-aware responses. However, this integration is not without challenges. One of the primary concerns is the potential security risks associated with using LLMs in robotic navigation tasks. These tasks require precise and reliable responses to ensure safe and effective operation. Multi-modal prompts, while enhancing the robot's understanding, also introduce complexities that can be exploited maliciously. For instance, adversarial inputs designed to mislead the model can lead to incorrect or dangerous navigational decisions. This study investigates the impact of prompt injections on mobile robot performance in LLM-integrated systems and explores secure prompt strategies to mitigate these risks. Our findings demonstrate a substantial overall improvement of approximately 30.8% in both attack detection and system performance with the implementation of robust defence mechanisms, highlighting their critical role in enhancing security and reliability in mission-oriented tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01527v2">Using LLMs to Establish Implicit User Sentiment of Software Desirability</a></div>
    <div class="paper-meta">
      📅 2024-09-08
      | 💬 6 pages, 2 figures, 2 tables, updated to incorporate feedback
    </div>
    <details class="paper-abstract">
      This study explores the use of LLMs for providing quantitative zero-shot sentiment analysis of implicit software desirability, addressing a critical challenge in product evaluation where traditional review scores, though convenient, fail to capture the richness of qualitative user feedback. Innovations include establishing a method that 1) works with qualitative user experience data without the need for explicit review scores, 2) focuses on implicit user satisfaction, and 3) provides scaled numerical sentiment analysis, offering a more nuanced understanding of user sentiment, instead of simply classifying sentiment as positive, neutral, or negative. Data is collected using the Microsoft Product Desirability Toolkit (PDT), a well-known qualitative user experience analysis tool. For initial exploration, the PDT metric was given to users of two software systems. PDT data was fed through several LLMs (Claude Sonnet 3 and 3.5, GPT4, and GPT4o) and through a leading transfer learning technique, Twitter-Roberta-Base-Sentiment, and Vader, a leading sentiment analysis tool. Each system was asked to evaluate the data in two ways, by looking at the sentiment expressed in the PDT word/explanation pairs; and by looking at the sentiment expressed by the users in their grouped selection of five words and explanations, as a whole. Each LLM provided a sentiment score, its confidence (low, medium, high) in the score, and an explanation of the score. All LLMs tested were able to statistically detect user sentiment from the users' grouped data, whereas TRBS and Vader were not. The confidence and explanation of confidence provided by the LLMs assisted in understanding user sentiment. This study adds deeper understanding of evaluating user experiences, toward the goal of creating a universal tool that quantifies implicit sentiment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09384v2">Tasks People Prompt: A Taxonomy of LLM Downstream Tasks in Software Verification and Falsification Approaches</a></div>
    <div class="paper-meta">
      📅 2024-09-08
    </div>
    <details class="paper-abstract">
      Prompting has become one of the main approaches to leverage emergent capabilities of Large Language Models [Brown et al. NeurIPS 2020, Wei et al. TMLR 2022, Wei et al. NeurIPS 2022]. Recently, researchers and practitioners have been "playing" with prompts (e.g., In-Context Learning) to see how to make the most of pre-trained Language Models. By homogeneously dissecting more than a hundred articles, we investigate how software testing and verification research communities have leveraged LLMs capabilities. First, we validate that downstream tasks are adequate to convey a nontrivial modular blueprint of prompt-based proposals in scope. Moreover, we name and classify the concrete downstream tasks we recover in both validation research papers and solution proposals. In order to perform classification, mapping, and analysis, we also develop a novel downstream-task taxonomy. The main taxonomy requirement is to highlight commonalities while exhibiting variation points of task types that enable pinpointing emerging patterns in a varied spectrum of Software Engineering problems that encompasses testing, fuzzing, fault localization, vulnerability detection, static analysis, and program verification approaches. Avenues for future research are also discussed based on conceptual clusters induced by the taxonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05028v1">LLM-based Abstraction and Concretization for GUI Test Migration</a></div>
    <div class="paper-meta">
      📅 2024-09-08
    </div>
    <details class="paper-abstract">
      GUI test migration aims to produce test cases with events and assertions to test specific functionalities of a target app. Existing migration approaches typically focus on the widget-mapping paradigm that maps widgets from source apps to target apps. However, since different apps may implement the same functionality in different ways, direct mapping may result in incomplete or buggy test cases, thus significantly impacting the effectiveness of testing target functionality and the practical applicability. In this paper, we propose a new migration paradigm (i.e., abstraction-concretization paradigm) that first abstracts the test logic for the target functionality and then utilizes this logic to generate the concrete GUI test case. Furthermore, we introduce MACdroid, the first approach that migrates GUI test cases based on this paradigm. Specifically, we propose an abstraction technique that utilizes source test cases from source apps targeting the same functionality to extract a general test logic for that functionality. Then, we propose a concretization technique that utilizes the general test logic to guide an LLM in generating the corresponding GUI test case (including events and assertions) for the target app. We evaluate MACdroid on two widely-used datasets (including 31 apps, 34 functionalities, and 123 test cases). On the FrUITeR dataset, the test cases generated by MACdroid successfully test 64% of the target functionalities, improving the baselines by 191%. On the Lin dataset, MACdroid successfully tests 75% of the target functionalities, outperforming the baselines by 42%. These results underscore the effectiveness of MACdroid in GUI test migration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11461v7">Hint of Thought prompting: an explainable and zero-shot approach to reasoning tasks with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-08
    </div>
    <details class="paper-abstract">
      Prompting becomes an increasingly important research topic for better utilization of LLMs. Although simple prompting performs well on single-step questions, it cannot permanently activate the correct knowledge path for multi-step reasoning tasks. The chain of thought (CoT), which often contains zero-shot CoT and few-shot CoT, is a recently developed prompting method that can explain the reasoning process to the LLM and outperforms simple prompting in three challenging reasoning tasks, including arithmetic, symbolic, and commonsense reasoning. Inspired by zero-shot CoT, and further extending the zero-shot ability, this paper proposes a novel hint of thought (HoT) prompting with explain-ability and zero-shot generalization. It is decomposed into three steps: explainable sub-questions, logical reasoning, and answering. Such three steps are sequentially ordered in step-by-step hints, which can be easily adjusted and explained to different tasks. Finally, experimental results demonstrate that our HoT prompting has a significant advantage on the zero-shot reasoning task compared to existing zero-shot CoT. We did zero-shot experiments on math tasks like GSM8K, ADDSUB, AQUA, SVAMP, and commonsense tasks such as StrategyQA. In particular, the accuracy of the proposed HoT prompting is improved with GSM8K from 40.50% to 70.65%, with AQUA from 31.9% to 46.4%, with SVAMP from 63.7% to 76.9%, and with ADDSUB from 74.7% to 87.34%, respectively, which even defeats the competitive PoT approach on GSM8k, AQUA, and SVAMP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04992v1">InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-09-08
    </div>
    <details class="paper-abstract">
      The widespread of Large Language Models (LLMs) marks a significant milestone in generative AI. Nevertheless, the increasing context length and batch size in offline LLM inference escalate the memory requirement of the key-value (KV) cache, which imposes a huge burden on the GPU VRAM, especially for resource-constraint scenarios (e.g., edge computing and personal devices). Several cost-effective solutions leverage host memory or SSDs to reduce storage costs for offline inference scenarios and improve the throughput. Nevertheless, they suffer from significant performance penalties imposed by intensive KV cache accesses due to limited PCIe bandwidth. To address these issues, we propose InstInfer, a novel LLM inference system that offloads the most performance-critical computation (i.e., attention in decoding phase) and data (i.e., KV cache) parts to Computational Storage Drives (CSDs), which minimize the enormous KV transfer overheads. InstInfer designs a dedicated flash-aware in-storage attention engine with KV cache management mechanisms to exploit the high internal bandwidths of CSDs instead of being limited by the PCIe bandwidth. The optimized P2P transmission between GPU and CSDs further reduces data migration overheads. Experimental results demonstrate that for a 13B model using an NVIDIA A6000 GPU, InstInfer improves throughput for long-sequence inference by up to 11.1$\times$, compared to existing SSD-based solutions such as FlexGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04987v1">How to Align Large Language Models for Teaching English? Designing and Developing LLM based-Chatbot for Teaching English Conversation in EFL, Findings and Limitations</a></div>
    <div class="paper-meta">
      📅 2024-09-08
      | 💬 56 pages
    </div>
    <details class="paper-abstract">
      This study investigates the design, development, and evaluation of a Large Language Model (LLM)-based chatbot for teaching English conversations in an English as a Foreign Language (EFL) context. Employing the Design and Development Research (DDR), we analyzed needs, established design principles, and iteratively refined a chatbot through experimenting various LLMs and alignment methods. Through both quantitative and qualitative evaluations, we identified the most effective LLM and its prompt combination to generate high-quality, contextually appropriate responses. Interviews with teachers provided insights into desirable system features, potential educational applications, and ethical considerations in the development and deployment of the chatbots. The design iterations yielded the importance of feedback mechanisms and customizable AI personas. Future research should explore adaptive feedback strategies, collaborative approaches with various stakeholders, and the integration of insights from human-computer interaction (HCI) and user experience (UX) design. This study contributes to the growing body of research on applying LLMs in language education, providing insights and recommendations for the design, development, and evaluation of LLM-based chatbots for EFL conversation practice. As the field evolves, ongoing research and collaboration among educators, AI engineers, and other stakeholders will be essential to harness the potential of these technologies to enhance language learning experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04827v1">Incorporate LLMs with Influential Recommender System</a></div>
    <div class="paper-meta">
      📅 2024-09-07
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Recommender systems have achieved increasing accuracy over the years. However, this precision often leads users to narrow their interests, resulting in issues such as limited diversity and the creation of echo chambers. Current research addresses these challenges through proactive recommender systems by recommending a sequence of items (called influence path) to guide user interest in the target item. However, existing methods struggle to construct a coherent influence path that builds up with items the user is likely to enjoy. In this paper, we leverage the Large Language Model's (LLMs) exceptional ability for path planning and instruction following, introducing a novel approach named LLM-based Influence Path Planning (LLM-IPP). Our approach maintains coherence between consecutive recommendations and enhances user acceptability of the recommended items. To evaluate LLM-IPP, we implement various user simulators and metrics to measure user acceptability and path coherence. Experimental results demonstrate that LLM-IPP significantly outperforms traditional proactive recommender systems. This study pioneers the integration of LLMs into proactive recommender systems, offering a reliable and user-engaging methodology for future recommendation technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04808v1">HULLMI: Human vs LLM identification with explainability</a></div>
    <div class="paper-meta">
      📅 2024-09-07
      | 💬 17 pages, 8 figures
    </div>
    <details class="paper-abstract">
      As LLMs become increasingly proficient at producing human-like responses, there has been a rise of academic and industrial pursuits dedicated to flagging a given piece of text as "human" or "AI". Most of these pursuits involve modern NLP detectors like T5-Sentinel and RoBERTa-Sentinel, without paying too much attention to issues of interpretability and explainability of these models. In our study, we provide a comprehensive analysis that shows that traditional ML models (Naive-Bayes,MLP, Random Forests, XGBoost) perform as well as modern NLP detectors, in human vs AI text detection. We achieve this by implementing a robust testing procedure on diverse datasets, including curated corpora and real-world samples. Subsequently, by employing the explainable AI technique LIME, we uncover parts of the input that contribute most to the prediction of each model, providing insights into the detection process. Our study contributes to the growing need for developing production-level LLM detection tools, which can leverage a wide range of traditional as well as modern NLP detectors we propose. Finally, the LIME techniques we demonstrate also have the potential to equip these detection tools with interpretability analysis features, making them more reliable and trustworthy in various domains like education, healthcare, and media.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.04900v2">HowToCaption: Prompting LLMs to Transform Video Annotations at Scale</a></div>
    <div class="paper-meta">
      📅 2024-09-07
      | 💬 https://github.com/ninatu/howtocaption
    </div>
    <details class="paper-abstract">
      Instructional videos are a common source for learning text-video or even multimodal representations by leveraging subtitles extracted with automatic speech recognition systems (ASR) from the audio signal in the videos. However, in contrast to human-annotated captions, both speech and subtitles naturally differ from the visual content of the videos and thus provide only noisy supervision. As a result, large-scale annotation-free web video training data remains sub-optimal for training text-video models. In this work, we propose to leverage the capabilities of large language models (LLMs) to obtain high-quality video descriptions aligned with videos at scale. Specifically, we prompt an LLM to create plausible video captions based on ASR subtitles of instructional videos. To this end, we introduce a prompting method that is able to take into account a longer text of subtitles, allowing us to capture the contextual information beyond one single sentence. We further prompt the LLM to generate timestamps for each produced caption based on the timestamps of the subtitles and finally align the generated captions to the video temporally. In this way, we obtain human-style video captions at scale without human supervision. We apply our method to the subtitles of the HowTo100M dataset, creating a new large-scale dataset, HowToCaption. Our evaluation shows that the resulting captions not only significantly improve the performance over many different benchmark datasets for zero-shot text-video retrieval and video captioning, but also lead to a disentangling of textual narration from the audio, boosting the performance in text-video-audio tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09296v2">Cross-Data Knowledge Graph Construction for LLM-enabled Educational Question-Answering System: A Case Study at HCMUT</a></div>
    <div class="paper-meta">
      📅 2024-09-07
      | 💬 8 pages, 7 figures, Accepted at AIQAM '24: Proceedings of the 1st ACM Workshop on AI-Powered Q&A Systems for Multimedia
    </div>
    <details class="paper-abstract">
      In today's rapidly evolving landscape of Artificial Intelligence, large language models (LLMs) have emerged as a vibrant research topic. LLMs find applications in various fields and contribute significantly. Despite their powerful language capabilities, similar to pre-trained language models (PLMs), LLMs still face challenges in remembering events, incorporating new information, and addressing domain-specific issues or hallucinations. To overcome these limitations, researchers have proposed Retrieval-Augmented Generation (RAG) techniques, some others have proposed the integration of LLMs with Knowledge Graphs (KGs) to provide factual context, thereby improving performance and delivering more accurate feedback to user queries. Education plays a crucial role in human development and progress. With the technology transformation, traditional education is being replaced by digital or blended education. Therefore, educational data in the digital environment is increasing day by day. Data in higher education institutions are diverse, comprising various sources such as unstructured/structured text, relational databases, web/app-based API access, etc. Constructing a Knowledge Graph from these cross-data sources is not a simple task. This article proposes a method for automatically constructing a Knowledge Graph from multiple data sources and discusses some initial applications (experimental trials) of KG in conjunction with LLMs for question-answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07832v3">DeliGrasp: Inferring Object Properties with LLMs for Adaptive Grasp Policies</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can provide rich physical descriptions of most worldly objects, allowing robots to achieve more informed and capable grasping. We leverage LLMs' common sense physical reasoning and code-writing abilities to infer an object's physical characteristics$\unicode{x2013}$mass $m$, friction coefficient $\mu$, and spring constant $k$$\unicode{x2013}$from a semantic description, and then translate those characteristics into an executable adaptive grasp policy. Using a two-finger gripper with a built-in depth camera that can control its torque by limiting motor current, we demonstrate that LLM-parameterized but first-principles grasp policies outperform both traditional adaptive grasp policies and direct LLM-as-code policies on a custom benchmark of 12 delicate and deformable items including food, produce, toys, and other everyday items, spanning two orders of magnitude in mass and required pick-up force. We then improve property estimation and grasp performance on variable size objects with model finetuning on property-based comparisons and eliciting such comparisons via chain-of-thought prompting. We also demonstrate how compliance feedback from DeliGrasp policies can aid in downstream tasks such as measuring produce ripeness. Our code and videos are available at: https://deligrasp.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04600v1">The emergence of Large Language Models (LLM) as a tool in literature reviews: an LLM automated systematic review</a></div>
    <div class="paper-meta">
      📅 2024-09-06
      | 💬 18 main pages with 5 figures and 1 table, references, followed by supplementary materials
    </div>
    <details class="paper-abstract">
      Objective: This study aims to summarize the usage of Large Language Models (LLMs) in the process of creating a scientific review. We look at the range of stages in a review that can be automated and assess the current state-of-the-art research projects in the field. Materials and Methods: The search was conducted in June 2024 in PubMed, Scopus, Dimensions, and Google Scholar databases by human reviewers. Screening and extraction process took place in Covidence with the help of LLM add-on which uses OpenAI gpt-4o model. ChatGPT was used to clean extracted data and generate code for figures in this manuscript, ChatGPT and Scite.ai were used in drafting all components of the manuscript, except the methods and discussion sections. Results: 3,788 articles were retrieved, and 172 studies were deemed eligible for the final review. ChatGPT and GPT-based LLM emerged as the most dominant architecture for review automation (n=126, 73.2%). A significant number of review automation projects were found, but only a limited number of papers (n=26, 15.1%) were actual reviews that used LLM during their creation. Most citations focused on automation of a particular stage of review, such as Searching for publications (n=60, 34.9%), and Data extraction (n=54, 31.4%). When comparing pooled performance of GPT-based and BERT-based models, the former were better in data extraction with mean precision 83.0% (SD=10.4), and recall 86.0% (SD=9.8), while being slightly less accurate in title and abstract screening stage (Maccuracy=77.3%, SD=13.0). Discussion/Conclusion: Our LLM-assisted systematic review revealed a significant number of research projects related to review automation using LLMs. The results looked promising, and we anticipate that LLMs will change in the near future the way the scientific reviews are conducted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04593v1">Paper Copilot: A Self-Evolving and Efficient LLM System for Personalized Academic Assistance</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      As scientific research proliferates, researchers face the daunting task of navigating and reading vast amounts of literature. Existing solutions, such as document QA, fail to provide personalized and up-to-date information efficiently. We present Paper Copilot, a self-evolving, efficient LLM system designed to assist researchers, based on thought-retrieval, user profile and high performance optimization. Specifically, Paper Copilot can offer personalized research services, maintaining a real-time updated database. Quantitative evaluation demonstrates that Paper Copilot saves 69.92\% of time after efficient deployment. This paper details the design and implementation of Paper Copilot, highlighting its contributions to personalized academic support and its potential to streamline the research process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15326v1">Evaluating the Impact of a Specialized LLM on Physician Experience in Clinical Decision Support: A Comparison of Ask Avo and ChatGPT-4</a></div>
    <div class="paper-meta">
      📅 2024-09-06
      | 💬 8 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The use of Large language models (LLMs) to augment clinical decision support systems is a topic with rapidly growing interest, but current shortcomings such as hallucinations and lack of clear source citations make them unreliable for use in the clinical environment. This study evaluates Ask Avo, an LLM-derived software by AvoMD that incorporates a proprietary Language Model Augmented Retrieval (LMAR) system, in-built visual citation cues, and prompt engineering designed for interactions with physicians, against ChatGPT-4 in end-user experience for physicians in a simulated clinical scenario environment. Eight clinical questions derived from medical guideline documents in various specialties were prompted to both models by 62 study participants, with each response rated on trustworthiness, actionability, relevancy, comprehensiveness, and friendly format from 1 to 5. Ask Avo significantly outperformed ChatGPT-4 in all criteria: trustworthiness (4.52 vs. 3.34, p<0.001), actionability (4.41 vs. 3.19, p<0.001), relevancy (4.55 vs. 3.49, p<0.001), comprehensiveness (4.50 vs. 3.37, p<0.001), and friendly format (4.52 vs. 3.60, p<0.001). Our findings suggest that specialized LLMs designed with the needs of clinicians in mind can offer substantial improvements in user experience over general-purpose LLMs. Ask Avo's evidence-based approach tailored to clinician needs shows promise in the adoption of LLM-augmented clinical decision support software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04508v1">Toward LLM-Powered Social Robots for Supporting Sensitive Disclosures of Stigmatized Health Conditions</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      Disclosing sensitive health conditions offers significant benefits at both individual and societal levels. However, patients often face challenges due to concerns about stigma. The use of social robots and chatbots to support sensitive disclosures is gaining traction, especially with the emergence of LLM models. Yet, numerous technical, ethical, privacy, safety, efficacy, and reporting concerns must be carefully addressed in this context. In this position paper, we focus on the example of HIV status disclosure, examining key opportunities, technical considerations, and risks associated with LLM-backed social robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00077v2">Are LLM-based methods good enough for detecting unfair terms of service?</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      Countless terms of service (ToS) are being signed everyday by users all over the world while interacting with all kinds of apps and websites. More often than not, these online contracts spanning double-digit pages are signed blindly by users who simply want immediate access to the desired service. What would normally require a consultation with a legal team, has now become a mundane activity consisting of a few clicks where users potentially sign away their rights, for instance in terms of their data privacy, to countless online entities/companies. Large language models (LLMs) are good at parsing long text-based documents, and could potentially be adopted to help users when dealing with dubious clauses in ToS and their underlying privacy policies. To investigate the utility of existing models for this task, we first build a dataset consisting of 12 questions applied individually to a set of privacy policies crawled from popular websites. Thereafter, a series of open-source as well as commercial chatbots such as ChatGPT, are queried over each question, with the answers being compared to a given ground truth. Our results show that some open-source models are able to provide a higher accuracy compared to some commercial models. However, the best performance is recorded from a commercial chatbot (ChatGPT4). Overall, all models perform only slightly better than random at this task. Consequently, their performance needs to be significantly improved before they can be adopted at large for this purpose.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04340v1">AGR: Age Group fairness Reward for Bias Mitigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-06
      | 💬 The first two authors contributed equally to this work. Corresponding to Zhiqiang Wang. ACKNOWLEDGMENT: we would like to thank the computing resources support from the State Key Laboratory of New Computer Software Technologies at Nanjing University
    </div>
    <details class="paper-abstract">
      LLMs can exhibit age biases, resulting in unequal treatment of individuals across age groups. While much research has addressed racial and gender biases, age bias remains little explored. The scarcity of instruction-tuning and preference datasets for age bias hampers its detection and measurement, and existing fine-tuning methods seldom address age-related fairness. In this paper, we construct age bias preference datasets and instruction-tuning datasets for RLHF. We introduce ARG, an age fairness reward to reduce differences in the response quality of LLMs across different age groups. Extensive experiments demonstrate that this reward significantly improves response accuracy and reduces performance disparities across age groups. Our source code and datasets are available at the anonymous \href{https://anonymous.4open.science/r/FairRLHF-D445/readme.md}{link}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00884v3">Zero-Shot Topic Classification of Column Headers: Leveraging LLMs for Metadata Enrichment</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      Traditional dataset retrieval systems rely on metadata for indexing, rather than on the underlying data values. However, high-quality metadata creation and enrichment often require manual annotations, which is a labour-intensive and challenging process to automate. In this study, we propose a method to support metadata enrichment using topic annotations generated by three Large Language Models (LLMs): ChatGPT-3.5, GoogleBard, and GoogleGemini. Our analysis focuses on classifying column headers based on domain-specific topics from the Consortium of European Social Science Data Archives (CESSDA), a Linked Data controlled vocabulary. Our approach operates in a zero-shot setting, integrating the controlled topic vocabulary directly within the input prompt. This integration serves as a Large Context Windows approach, with the aim of improving the results of the topic classification task. We evaluated the performance of the LLMs in terms of internal consistency, inter-machine alignment, and agreement with human classification. Additionally, we investigate the impact of contextual information (i.e., dataset description) on the classification outcomes. Our findings suggest that ChatGPT and GoogleGemini outperform GoogleBard in terms of internal consistency as well as LLM-human-agreement. Interestingly, we found that contextual information had no significant impact on LLM performance. This work proposes a novel approach that leverages LLMs for topic classification of column headers using a controlled vocabulary, presenting a practical application of LLMs and Large Context Windows within the Semantic Web domain. This approach has the potential to facilitate automated metadata enrichment, thereby enhancing dataset retrieval and the Findability, Accessibility, Interoperability, and Reusability (FAIR) of research data on the Web.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04318v1">Learning vs Retrieval: The Role of In-Context Examples in Regression with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs) are capable of being in-context learners. However, the underlying mechanism of in-context learning (ICL) is still a major research question, and experimental research results about how models exploit ICL are not always consistent. In this work, we propose a framework for evaluating in-context learning mechanisms, which we claim are a combination of retrieving internal knowledge and learning from in-context examples by focusing on regression tasks. First, we show that LLMs can perform regression on real-world datasets and then design experiments to measure the extent to which the LLM retrieves its internal knowledge versus learning from in-context examples. We argue that this process lies on a spectrum between these two extremes. We provide an in-depth analysis of the degrees to which these mechanisms are triggered depending on various factors, such as prior knowledge about the tasks and the type and richness of the information provided by the in-context examples. We employ three LLMs and utilize multiple datasets to corroborate the robustness of our findings. Our results shed light on how to engineer prompts to leverage meta-learning from in-context examples and foster knowledge retrieval depending on the problem being addressed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15324v1">Cognitive phantoms in LLMs through the lens of latent variables</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly reach real-world applications, necessitating a better understanding of their behaviour. Their size and complexity complicate traditional assessment methods, causing the emergence of alternative approaches inspired by the field of psychology. Recent studies administering psychometric questionnaires to LLMs report human-like traits in LLMs, potentially influencing LLM behaviour. However, this approach suffers from a validity problem: it presupposes that these traits exist in LLMs and that they are measurable with tools designed for humans. Typical procedures rarely acknowledge the validity problem in LLMs, comparing and interpreting average LLM scores. This study investigates this problem by comparing latent structures of personality between humans and three LLMs using two validated personality questionnaires. Findings suggest that questionnaires designed for humans do not validly measure similar constructs in LLMs, and that these constructs may not exist in LLMs at all, highlighting the need for psychometric analyses of LLM responses to avoid chasing cognitive phantoms. Keywords: large language models, psychometrics, machine behaviour, latent variable modeling, validity
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04168v1">From Calculation to Adjudication: Examining LLM judges on Mathematical Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      To reduce the need for human annotations, large language models (LLMs) have been proposed as judges of the quality of other candidate models. LLM judges are typically evaluated by measuring the correlation with human judgments on generation tasks such as summarization or machine translation. In contrast, we study LLM judges on mathematical reasoning tasks. These tasks require multi-step reasoning, and the correctness of their solutions is verifiable, enabling a more objective evaluation. We perform a detailed performance analysis and find that the used judges are mostly unable to improve task performance but are able to pick the better model. Our analysis uncovers a strong correlation between judgment performance and the candidate model task performance. We observe that judges tend to choose the model of higher quality even if its answer is incorrect. Further, we show that it is possible to use statistics, such as the task performances of the individual models, to predict judgment performance. In an ablation, we either swap or mask the candidate answers and observe that judges often keep the original judgment, providing evidence that judges incorporate writing style in their judgments. In summary, we find that regularities in the judgments are quantifiable using statistical measures and provide various angles on exploiting them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03637v4">QET: Enhancing Quantized LLM Parameters and KV cache Compression through Element Substitution and Residual Clustering</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      The matrix quantization entails representing matrix elements in a more space-efficient form to reduce storage usage, with dequantization restoring the original matrix for use. We formulate the Quantization Error Minimization (QEM) problem as minimizing the distance between a matrix before and after quantization, under the condition that the quantized matrix occupies the same memory space. Matrix quantization is crucial in various applications, including Large Language Models (LLMs) weight quantization, vector databases, KV cache quantization, graph compression, and image compression. Recent advancements in LLMs, such as GPT-4 and BERT, have highlighted the importance of matrix compression due to the large size of parameters and KV cache, which are stored as matrices. We propose Quantum Entanglement Trees (QET) to address the QEM problem by leveraging the local orderliness of matrix elements, involving iterative element swapping to form a locally ordered matrix. This matrix is then grouped and quantized by columns. To enhance QET, we introduce two optimizations: further quantizing residuals to reduce MSE, and using masking and batch processing to accelerate the algorithm. Experimental results demonstrate that QET can effectively reduce MSE to 5.05%, 13.33%, and 11.89% of the current best method on the LLM dataset, K cache, and V cache, respectively. Our contributions include the abstraction of the QEM problem, the design of the QET algorithm, and the proposal of two optimizations to improve accuracy and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04109v1">Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers</a></div>
    <div class="paper-meta">
      📅 2024-09-06
      | 💬 main paper is 20 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have sparked optimism about their potential to accelerate scientific discovery, with a growing number of works proposing research agents that autonomously generate and validate new ideas. Despite this, no evaluations have shown that LLM systems can take the very first step of producing novel, expert-level ideas, let alone perform the entire research process. We address this by establishing an experimental design that evaluates research idea generation while controlling for confounders and performs the first head-to-head comparison between expert NLP researchers and an LLM ideation agent. By recruiting over 100 NLP researchers to write novel ideas and blind reviews of both LLM and human ideas, we obtain the first statistically significant conclusion on current LLM capabilities for research ideation: we find LLM-generated ideas are judged as more novel (p < 0.05) than human expert ideas while being judged slightly weaker on feasibility. Studying our agent baselines closely, we identify open problems in building and evaluating research agents, including failures of LLM self-evaluation and their lack of diversity in generation. Finally, we acknowledge that human judgements of novelty can be difficult, even by experts, and propose an end-to-end study design which recruits researchers to execute these ideas into full projects, enabling us to study whether these novelty and feasibility judgements result in meaningful differences in research outcome.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03659v2">LLM-based multi-agent poetry generation in non-cooperative environments</a></div>
    <div class="paper-meta">
      📅 2024-09-06
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Despite substantial progress of large language models (LLMs) for automatic poetry generation, the generated poetry lacks diversity while the training process differs greatly from human learning. Under the rationale that the learning process of the poetry generation systems should be more human-like and their output more diverse and novel, we introduce a framework based on social learning where we emphasize non-cooperative interactions besides cooperative interactions to encourage diversity. Our experiments are the first attempt at LLM-based multi-agent systems in non-cooperative environments for poetry generation employing both TRAINING-BASED agents (GPT-2) and PROMPTING-BASED agents (GPT-3 and GPT-4). Our evaluation based on 96k generated poems shows that our framework benefits the poetry generation process for TRAINING-BASED agents resulting in 1) a 3.0-3.7 percentage point (pp) increase in diversity and a 5.6-11.3 pp increase in novelty according to distinct and novel n-grams. The generated poetry from TRAINING-BASED agents also exhibits group divergence in terms of lexicons, styles and semantics. PROMPTING-BASED agents in our framework also benefit from non-cooperative environments and a more diverse ensemble of models with non-homogeneous agents has the potential to further enhance diversity, with an increase of 7.0-17.5 pp according to our experiments. However, PROMPTING-BASED agents show a decrease in lexical diversity over time and do not exhibit the group-based divergence intended in the social network. Our paper argues for a paradigm shift in creative tasks such as automatic poetry generation to include social learning processes (via LLM-based agent modeling) similar to human interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04040v1">A First Look At Efficient And Secure On-Device LLM Inference Against KV Leakage</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      Running LLMs on end devices has garnered significant attention recently due to their advantages in privacy preservation. With the advent of lightweight LLM models and specially designed GPUs, on-device LLM inference has achieved the necessary accuracy and performance metrics. However, we have identified that LLM inference on GPUs can leak privacy-sensitive intermediate information, specifically the KV pairs. An attacker could exploit these KV pairs to reconstruct the entire user conversation, leading to significant vulnerabilities. Existing solutions, such as Fully Homomorphic Encryption (FHE) and Trusted Execution Environments (TEE), are either too computation-intensive or resource-limited. To address these issues, we designed KV-Shield, which operates in two phases. In the initialization phase, it permutes the weight matrices so that all KV pairs are correspondingly permuted. During the runtime phase, the attention vector is inversely permuted to ensure the correctness of the layer output. All permutation-related operations are executed within the TEE, ensuring that insecure GPUs cannot access the original KV pairs, thus preventing conversation reconstruction. Finally, we theoretically analyze the correctness of KV-Shield, along with its advantages and overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05572v2">Trust the PRoC3S: Solving Long-Horizon Robotics Problems with LLMs and Constraint Satisfaction</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      Recent developments in pretrained large language models (LLMs) applied to robotics have demonstrated their capacity for sequencing a set of discrete skills to achieve open-ended goals in simple robotic tasks. In this paper, we examine the topic of LLM planning for a set of continuously parameterized skills whose execution must avoid violations of a set of kinematic, geometric, and physical constraints. We prompt the LLM to output code for a function with open parameters, which, together with environmental constraints, can be viewed as a Continuous Constraint Satisfaction Problem (CCSP). This CCSP can be solved through sampling or optimization to find a skill sequence and continuous parameter settings that achieve the goal while avoiding constraint violations. Additionally, we consider cases where the LLM proposes unsatisfiable CCSPs, such as those that are kinematically infeasible, dynamically unstable, or lead to collisions, and re-prompt the LLM to form a new CCSP accordingly. Experiments across three different simulated 3D domains demonstrate that our proposed strategy, PRoC3S, is capable of solving a wide range of complex manipulation tasks with realistic constraints on continuous parameters much more efficiently and effectively than existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04298v3">SELF-[IN]CORRECT: LLMs Struggle with Discriminating Self-Generated Responses</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      Can LLMs consistently improve their previous outputs for better results? For this to be true, LLMs would need to be better at discriminating among previously-generated alternatives, than generating initial responses. We explore the validity of this hypothesis in practice. We first formulate a unified framework that allows us to compare the generative and discriminative capability of any model on any task. In our resulting experimental analysis of several open-source and industrial LLMs, we observe that models are not reliably better at discriminating among previously-generated alternatives than generating initial responses. This finding challenges the notion that LLMs may be able to enhance their performance only through their own judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03946v1">On The Role of Prompt Construction In Enhancing Efficacy and Efficiency of LLM-Based Tabular Data Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-06
    </div>
    <details class="paper-abstract">
      LLM-based data generation for real-world tabular data can be challenged by the lack of sufficient semantic context in feature names used to describe columns. We hypothesize that enriching prompts with domain-specific insights can improve both the quality and efficiency of data generation. To test this hypothesis, we explore three prompt construction protocols: Expert-guided, LLM-guided, and Novel-Mapping. Through empirical studies with the recently proposed GReaT framework, we find that context-enriched prompts lead to significantly improved data generation quality and training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03937v1">Harnessing LLMs for Cross-City OD Flow Prediction</a></div>
    <div class="paper-meta">
      📅 2024-09-05
      | 💬 12 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Understanding and predicting Origin-Destination (OD) flows is crucial for urban planning and transportation management. Traditional OD prediction models, while effective within single cities, often face limitations when applied across different cities due to varied traffic conditions, urban layouts, and socio-economic factors. In this paper, by employing Large Language Models (LLMs), we introduce a new method for cross-city OD flow prediction. Our approach leverages the advanced semantic understanding and contextual learning capabilities of LLMs to bridge the gap between cities with different characteristics, providing a robust and adaptable solution for accurate OD flow prediction that can be transferred from one city to another. Our novel framework involves four major components: collecting OD training datasets from a source city, instruction-tuning the LLMs, predicting destination POIs in a target city, and identifying the locations that best match the predicted destination POIs. We introduce a new loss function that integrates POI semantics and trip distance during training. By extracting high-quality semantic features from human mobility and POI data, the model understands spatial and functional relationships within urban spaces and captures interactions between individuals and various POIs. Extensive experimental results demonstrate the superiority of our approach over the state-of-the-art learning-based methods in cross-city OD flow prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03928v1">RETAIN: Interactive Tool for Regression Testing Guided LLM Migration</a></div>
    <div class="paper-meta">
      📅 2024-09-05
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into diverse applications. The rapid evolution of LLMs presents opportunities for developers to enhance applications continuously. However, this constant adaptation can also lead to performance regressions during model migrations. While several interactive tools have been proposed to streamline the complexity of prompt engineering, few address the specific requirements of regression testing for LLM Migrations. To bridge this gap, we introduce RETAIN (REgression Testing guided LLM migrAtIoN), a tool designed explicitly for regression testing in LLM Migrations. RETAIN comprises two key components: an interactive interface tailored to regression testing needs during LLM migrations, and an error discovery module that facilitates understanding of differences in model behaviors. The error discovery module generates textual descriptions of various errors or differences between model outputs, providing actionable insights for prompt refinement. Our automatic evaluation and empirical user studies demonstrate that RETAIN, when compared to manual evaluation, enabled participants to identify twice as many errors, facilitated experimentation with 75% more prompts, and achieves 12% higher metric scores in a given time frame.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03856v1">Sirius: Contextual Sparsity with Correction for Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      With the blossom of large language models (LLMs), inference efficiency becomes increasingly important. Various approximation methods are proposed to reduce the cost at inference time. Contextual Sparsity (CS) is appealing for its training-free nature and its ability to reach a higher compression ratio seemingly without quality degradation. However, after a comprehensive evaluation of contextual sparsity methods on various complex generation tasks, we find that although CS succeeds in prompt-understanding tasks, CS significantly degrades the model performance for reasoning, deduction, and knowledge-based tasks. Despite the gap in end-to-end accuracy, we observed that sparse models often share general problem-solving logic and require only a few token corrections to recover the original model performance. This paper introduces Sirius, an efficient correction mechanism, which significantly recovers CS models quality on reasoning tasks while maintaining its efficiency gain. Sirius is evaluated on 6 models with 8 difficult generation tasks in reasoning, math, and coding and shows consistent effectiveness and efficiency. Also, we carefully develop a system implementation for Sirius and show that Sirius achieves roughly 20% reduction in latency for 8B model on-chip and 35% reduction for 70B model offloading. We open-source our implementation of Sirius at https://github.com/Infini-AI-Lab/Sirius.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03810v1">How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with High-Quality Data</a></div>
    <div class="paper-meta">
      📅 2024-09-05
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      Recently, there has been a growing interest in studying how to construct better code instruction tuning data. However, we observe Code models trained with these datasets exhibit high performance on HumanEval but perform worse on other benchmarks such as LiveCodeBench. Upon further investigation, we find that many datasets suffer from severe data leakage. After cleaning up most of the leaked data, some well-known high-quality datasets perform poorly. This discovery reveals a new challenge: identifying which dataset genuinely qualify as high-quality code instruction data. To address this, we propose an efficient code data pruning strategy for selecting good samples. Our approach is based on three dimensions: instruction complexity, response quality, and instruction diversity. Based on our selected data, we present XCoder, a family of models finetuned from LLaMA3. Our experiments show XCoder achieves new state-of-the-art performance using fewer training data, which verify the effectiveness of our data strategy. Moreover, we perform a comprehensive analysis on the data composition and find existing code datasets have different characteristics according to their construction methods, which provide new insights for future code LLMs. Our models and dataset are released in https://github.com/banksy23/XCoder
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05498v2">SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner</a></div>
    <div class="paper-meta">
      📅 2024-09-05
      | 💬 This paper completes its earlier vision paper, available at arXiv:2402.15727. Updated to the latest analysis and results
    </div>
    <details class="paper-abstract">
      Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance to concurrently protect the target LLM instance in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs (both target and defense LLMs) have the capability to identify harmful prompts or intentions in user queries, which we empirically validate using the commonly used GPT-3.5/4 models across all major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. These models outperform six state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. We also empirically show that the tuned models are robust to adaptive jailbreaks and prompt injections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03563v1">100 instances is all you need: predicting the success of a new LLM on unseen data by testing on a few instances</a></div>
    <div class="paper-meta">
      📅 2024-09-05
      | 💬 Presented at the 2024 KDD workshop on Evaluation and Trustworthiness of Generative AI Models
    </div>
    <details class="paper-abstract">
      Predicting the performance of LLMs on individual task instances is essential to ensure their reliability in high-stakes applications. To do so, a possibility is to evaluate the considered LLM on a set of task instances and train an assessor to predict its performance based on features of the instances. However, this approach requires evaluating each new LLM on a sufficiently large set of task instances to train an assessor specific to it. In this work, we leverage the evaluation results of previously tested LLMs to reduce the number of evaluations required to predict the performance of a new LLM. In practice, we propose to test the new LLM on a small set of reference instances and train a generic assessor which predicts the performance of the LLM on an instance based on the performance of the former on the reference set and features of the instance of interest. We conduct empirical studies on HELM-Lite and KindsOfReasoning, a collection of existing reasoning datasets that we introduce, where we evaluate all instruction-fine-tuned OpenAI models until the January 2024 version of GPT4. When predicting performance on instances with the same distribution as those used to train the generic assessor, we find this achieves performance comparable to the LLM-specific assessors trained on the full set of instances. Additionally, we find that randomly selecting the reference instances performs as well as some advanced selection methods we tested. For out of distribution, however, no clear winner emerges and the overall performance is worse, suggesting that the inherent predictability of LLMs is low.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03512v1">From MOOC to MAIC: Reshaping Online Teaching and Learning through LLM-driven Agents</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      Since the first instances of online education, where courses were uploaded to accessible and shared online platforms, this form of scaling the dissemination of human knowledge to reach a broader audience has sparked extensive discussion and widespread adoption. Recognizing that personalized learning still holds significant potential for improvement, new AI technologies have been continuously integrated into this learning format, resulting in a variety of educational AI applications such as educational recommendation and intelligent tutoring. The emergence of intelligence in large language models (LLMs) has allowed for these educational enhancements to be built upon a unified foundational model, enabling deeper integration. In this context, we propose MAIC (Massive AI-empowered Course), a new form of online education that leverages LLM-driven multi-agent systems to construct an AI-augmented classroom, balancing scalability with adaptivity. Beyond exploring the conceptual framework and technical innovations, we conduct preliminary experiments at Tsinghua University, one of China's leading universities. Drawing from over 100,000 learning records of more than 500 students, we obtain a series of valuable observations and initial analyses. This project will continue to evolve, ultimately aiming to establish a comprehensive open platform that supports and unifies research, technology, and applications in exploring the possibilities of online education in the era of large model AI. We envision this platform as a collaborative hub, bringing together educators, researchers, and innovators to collectively explore the future of AI-driven online education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03478v1">LLM-based event abstraction and integration for IoT-sourced logs</a></div>
    <div class="paper-meta">
      📅 2024-09-05
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      The continuous flow of data collected by Internet of Things (IoT) devices, has revolutionised our ability to understand and interact with the world across various applications. However, this data must be prepared and transformed into event data before analysis can begin. In this paper, we shed light on the potential of leveraging Large Language Models (LLMs) in event abstraction and integration. Our approach aims to create event records from raw sensor readings and merge the logs from multiple IoT sources into a single event log suitable for further Process Mining applications. We demonstrate the capabilities of LLMs in event abstraction considering a case study for IoT application in elderly care and longitudinal health monitoring. The results, showing on average an accuracy of 90% in detecting high-level activities. These results highlight LLMs' promising potential in addressing event abstraction and integration challenges, effectively bridging the existing gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03440v1">Rx Strategist: Prescription Verification using LLM Agents System</a></div>
    <div class="paper-meta">
      📅 2024-09-05
      | 💬 17 Pages, 6 Figures, Under Review
    </div>
    <details class="paper-abstract">
      To protect patient safety, modern pharmaceutical complexity demands strict prescription verification. We offer a new approach - Rx Strategist - that makes use of knowledge graphs and different search strategies to enhance the power of Large Language Models (LLMs) inside an agentic framework. This multifaceted technique allows for a multi-stage LLM pipeline and reliable information retrieval from a custom-built active ingredient database. Different facets of prescription verification, such as indication, dose, and possible drug interactions, are covered in each stage of the pipeline. We alleviate the drawbacks of monolithic LLM techniques by spreading reasoning over these stages, improving correctness and reliability while reducing memory demands. Our findings demonstrate that Rx Strategist surpasses many current LLMs, achieving performance comparable to that of a highly experienced clinical pharmacist. In the complicated world of modern medications, this combination of LLMs with organized knowledge and sophisticated search methods presents a viable avenue for reducing prescription errors and enhancing patient outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03384v1">Hardware Acceleration of LLMs: A comprehensive survey and comparison</a></div>
    <div class="paper-meta">
      📅 2024-09-05
      | 💬 https://airtable.com/appC2VwR6X4EeZ50s/shrKwchys0iktvDwk
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for natural language processing tasks, revolutionizing the field with their ability to understand and generate human-like text. In this paper, we present a comprehensive survey of the several research efforts that have been presented for the acceleration of transformer networks for Large Language Models using hardware accelerators. The survey presents the frameworks that have been proposed and then performs a qualitative and quantitative comparison regarding the technology, the processing platform (FPGA, ASIC, In-Memory, GPU), the speedup, the energy efficiency, the performance (GOPs), and the energy efficiency (GOPs/W) of each framework. The main challenge in comparison is that every proposed scheme is implemented on a different process technology making hard a fair comparison. The main contribution of this paper is that we extrapolate the results of the performance and the energy efficiency on the same technology to make a fair comparison; one theoretical and one more practical. We implement part of the LLMs on several FPGA chips to extrapolate the results to the same process technology and then we make a fair comparison of the performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03346v1">Sketch: A Toolkit for Streamlining LLM Operations</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) represented by GPT family have achieved remarkable success. The characteristics of LLMs lie in their ability to accommodate a wide range of tasks through a generative approach. However, the flexibility of their output format poses challenges in controlling and harnessing the model's outputs, thereby constraining the application of LLMs in various domains. In this work, we present Sketch, an innovative toolkit designed to streamline LLM operations across diverse fields. Sketch comprises the following components: (1) a suite of task description schemas and prompt templates encompassing various NLP tasks; (2) a user-friendly, interactive process for building structured output LLM services tailored to various NLP tasks; (3) an open-source dataset for output format control, along with tools for dataset construction; and (4) an open-source model based on LLaMA3-8B-Instruct that adeptly comprehends and adheres to output formatting instructions. We anticipate this initiative to bring considerable convenience to LLM users, achieving the goal of ''plug-and-play'' for various applications. The components of Sketch will be progressively open-sourced at https://github.com/cofe-ai/Sketch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02727v2">Pooling And Attention: What Are Effective Designs For LLM-Based Embedding Models?</a></div>
    <div class="paper-meta">
      📅 2024-09-05
      | 💬 https://github.com/yixuantt/PoolingAndAttn
    </div>
    <details class="paper-abstract">
      The significant advancements of Large Language Models (LLMs) in generative tasks have led to a growing body of work exploring LLM-based embedding models. While these models, employing different pooling and attention strategies, have achieved state-of-the-art performance on public embedding benchmarks, questions still arise about what constitutes an effective design for LLM-based embedding models. However, these models are often trained on different datasets, using different LLM base models or training settings. Moreover, evaluations on public embedding benchmarks often fail to report statistical significance, making it difficult to determine which designs truly contribute to final performance. This complicates the process for practitioners seeking optimal training recipes for LLM-based embedding models. In this study, we conduct a large-scale experiment by training a series of LLM-based embedding models using the same training data and base model but differing in their pooling and attention strategies. The results show that there is no one-size-fits-all solution: while bidirectional attention and an additional trainable pooling layer outperform in text similarity and information retrieval tasks, they do not significantly surpass simpler designs like EOS-last token pooling and default causal attention in clustering and classification tasks. Furthermore, we propose a new pooling strategy, Multi-Layers Trainable Pooling, which transforms the outputs of all hidden layers, rather than just the last layer, using a cross-attention network. This method proves to be statistically superior in text similarity and retrieval tasks compared to existing pooling methods. Overall, this paper sheds light on effective training strategies for LLM-based embedding models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13702v1">Shaping the Future of Endangered and Low-Resource Languages -- Our Role in the Age of LLMs: A Keynote at ECIR 2024</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      Isidore of Seville is credited with the adage that it is language that gives birth to a people, and not the other way around , underlining the profound role played by language in the formation of cultural and social identity. Today, of the more than 7100 languages listed, a significant number are endangered. Since the 1970s, linguists, information seekers and enthusiasts have helped develop digital resources and automatic tools to support a wide range of languages, including endangered ones. The advent of Large Language Model (LLM) technologies holds both promise and peril. They offer unprecedented possibilities for the translation and generation of content and resources, key elements in the preservation and revitalisation of languages. They also present threat of homogenisation, cultural oversimplification and the further marginalisation of already vulnerable languages. The talk this paper is based on has proposed an initiatory journey, exploring the potential paths and partnerships between technology and tradition, with a particular focus on the Occitan language. Occitan is a language from Southern France, parts of Spain and Italy that played a major cultural and economic role, particularly in the Middle Ages. It is now endangered according to UNESCO. The talk critically has examined how human expertise and artificial intelligence can work together to offer hope for preserving the linguistic diversity that forms the foundation of our global and especially our European heritage while addressing some of the ethical and practical challenges that accompany the use of these powerful technologies. This paper is based on the keynote I gave at the 46th European Conference on Information Retrieval (ECIR 2024). As an alternative to reading this paper, a video talk is available online. 1 Date: 26 March 2024.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03271v1">Strategic Chain-of-Thought: Guiding Accurate Reasoning in LLMs through Strategy Elicitation</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      The Chain-of-Thought (CoT) paradigm has emerged as a critical approach for enhancing the reasoning capabilities of large language models (LLMs). However, despite their widespread adoption and success, CoT methods often exhibit instability due to their inability to consistently ensure the quality of generated reasoning paths, leading to sub-optimal reasoning performance. To address this challenge, we propose the \textbf{Strategic Chain-of-Thought} (SCoT), a novel methodology designed to refine LLM performance by integrating strategic knowledge prior to generating intermediate reasoning steps. SCoT employs a two-stage approach within a single prompt: first eliciting an effective problem-solving strategy, which is then used to guide the generation of high-quality CoT paths and final answers. Our experiments across eight challenging reasoning datasets demonstrate significant improvements, including a 21.05\% increase on the GSM8K dataset and 24.13\% on the Tracking\_Objects dataset, respectively, using the Llama3-8b model. Additionally, we extend the SCoT framework to develop a few-shot method with automatically matched demonstrations, yielding even stronger results. These findings underscore the efficacy of SCoT, highlighting its potential to substantially enhance LLM performance in complex reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03257v1">Understanding LLM Development Through Longitudinal Study: Insights from the Open Ko-LLM Leaderboard</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      This paper conducts a longitudinal study over eleven months to address the limitations of prior research on the Open Ko-LLM Leaderboard, which have relied on empirical studies with restricted observation periods of only five months. By extending the analysis duration, we aim to provide a more comprehensive understanding of the progression in developing Korean large language models (LLMs). Our study is guided by three primary research questions: (1) What are the specific challenges in improving LLM performance across diverse tasks on the Open Ko-LLM Leaderboard over time? (2) How does model size impact task performance correlations across various benchmarks? (3) How have the patterns in leaderboard rankings shifted over time on the Open Ko-LLM Leaderboard?. By analyzing 1,769 models over this period, our research offers a comprehensive examination of the ongoing advancements in LLMs and the evolving nature of evaluation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03247v1">End User Authoring of Personalized Content Classifiers: Comparing Example Labeling, Rule Writing, and LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      Existing tools for laypeople to create personal classifiers often assume a motivated user working uninterrupted in a single, lengthy session. However, users tend to engage with social media casually, with many short sessions on an ongoing, daily basis. To make creating personal classifiers for content curation easier for such users, tools should support rapid initialization and iterative refinement. In this work, we compare three strategies -- (1) example labeling, (2) rule writing, and (3) large language model (LLM) prompting -- for end users to build personal content classifiers. From an experiment with 37 non-programmers tasked with creating personalized comment moderation filters, we found that with LLM prompting, participants reached 95\% of peak performance in 5 minutes, beating other strategies due to higher recall, but all strategies struggled with iterative refinement. Despite LLM prompting's better performance, participants preferred different strategies in different contexts and, even when prompting, provided examples or wrote rule-like prompts, suggesting hybrid approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03225v1">Enhancing Healthcare LLM Trust with Atypical Presentations Recalibration</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      Black-box large language models (LLMs) are increasingly deployed in various environments, making it essential for these models to effectively convey their confidence and uncertainty, especially in high-stakes settings. However, these models often exhibit overconfidence, leading to potential risks and misjudgments. Existing techniques for eliciting and calibrating LLM confidence have primarily focused on general reasoning datasets, yielding only modest improvements. Accurate calibration is crucial for informed decision-making and preventing adverse outcomes but remains challenging due to the complexity and variability of tasks these models perform. In this work, we investigate the miscalibration behavior of black-box LLMs within the healthcare setting. We propose a novel method, \textit{Atypical Presentations Recalibration}, which leverages atypical presentations to adjust the model's confidence estimates. Our approach significantly improves calibration, reducing calibration errors by approximately 60\% on three medical question answering datasets and outperforming existing methods such as vanilla verbalized confidence, CoT verbalized confidence and others. Additionally, we provide an in-depth analysis of the role of atypicality within the recalibration framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03219v1">Content Moderation by LLM: From Accuracy to Legitimacy</a></div>
    <div class="paper-meta">
      📅 2024-09-05
    </div>
    <details class="paper-abstract">
      One trending application of LLM (large language model) is to use it for content moderation in online platforms. Most current studies on this application have focused on the metric of accuracy - the extent to which LLM makes correct decisions about content. This article argues that accuracy is insufficient and misleading, because it fails to grasp the distinction between easy cases and hard cases as well as the inevitable trade-offs in achieving higher accuracy. Closer examination reveals that content moderation is a constitutive part of platform governance, the key of which is to gain and enhance legitimacy. Instead of making moderation decisions correct, the chief goal of LLM is to make them legitimate. In this regard, this article proposes a paradigm shift from the single benchmark of accuracy towards a legitimacy-based framework of evaluating the performance of LLM moderators. The framework suggests that for easy cases, the key is to ensure accuracy, speed and transparency, while for hard cases, what matters is reasoned justification and user participation. Examined under this framework, LLM's real potential in moderation is not accuracy improvement. Rather, LLM can better contribute in four other aspects: to conduct screening of hard cases from easy cases, to provide quality explanations for moderation decisions, to assist human reviewers in getting more contextual information, and to facilitate user participation in a more interactive way. Using normative theories from law and social sciences to critically assess the new technological application, this article seeks to redefine LLM's role in content moderation and redirect relevant research in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03067v1">A Comparative Study of Offline Models and Online LLMs in Fake News Detection</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      Fake news detection remains a critical challenge in today's rapidly evolving digital landscape, where misinformation can spread faster than ever before. Traditional fake news detection models often rely on static datasets and auxiliary information, such as metadata or social media interactions, which limits their adaptability to real-time scenarios. Recent advancements in Large Language Models (LLMs) have demonstrated significant potential in addressing these challenges due to their extensive pre-trained knowledge and ability to analyze textual content without relying on auxiliary data. However, many of these LLM-based approaches are still rooted in static datasets, with limited exploration into their real-time processing capabilities. This paper presents a systematic evaluation of both traditional offline models and state-of-the-art LLMs for real-time fake news detection. We demonstrate the limitations of existing offline models, including their inability to adapt to dynamic misinformation patterns. Furthermore, we show that newer LLM models with online capabilities, such as GPT-4, Claude, and Gemini, are better suited for detecting emerging fake news in real-time contexts. Our findings emphasize the importance of transitioning from offline to online LLM models for real-time fake news detection. Additionally, the public accessibility of LLMs enhances their scalability and democratizes the tools needed to combat misinformation. By leveraging real-time data, our work marks a significant step toward more adaptive, effective, and scalable fake news detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00557v2">Learning to Ask: When LLMs Meet Unclear Instruction</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      Equipped with the capability to call functions, modern large language models (LLMs) can leverage external tools for addressing a range of tasks unattainable through language skills alone. However, the effective execution of these tools relies heavily not just on the advanced capabilities of LLMs but also on precise user instructions, which often cannot be ensured in the real world. To evaluate the performance of LLMs tool-use under imperfect instructions, we meticulously examine the real-world instructions queried from users, analyze the error patterns, and build a challenging tool-use benchmark called Noisy ToolBench (NoisyToolBench). We find that due to the next-token prediction training objective, LLMs tend to arbitrarily generate the missed argument, which may lead to hallucinations and risks. To address this issue, we propose a novel framework, Ask-when-Needed (AwN), which prompts LLMs to ask questions to users whenever they encounter obstacles due to unclear instructions. Moreover, to reduce the manual labor involved in user-LLM interaction and assess LLMs performance in tool utilization from both accuracy and efficiency perspectives, we design an automated evaluation tool named ToolEvaluator. Our experiments demonstrate that the AwN significantly outperforms existing frameworks for tool learning in the NoisyToolBench. We will release all related code and datasets to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00494v2">GenAI-powered Multi-Agent Paradigm for Smart Urban Mobility: Opportunities and Challenges for Integrating Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) with Intelligent Transportation Systems</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      Leveraging recent advances in generative AI, multi-agent systems are increasingly being developed to enhance the functionality and efficiency of smart city applications. This paper explores the transformative potential of large language models (LLMs) and emerging Retrieval-Augmented Generation (RAG) technologies in Intelligent Transportation Systems (ITS), paving the way for innovative solutions to address critical challenges in urban mobility. We begin by providing a comprehensive overview of the current state-of-the-art in mobility data, ITS, and Connected Vehicles (CV) applications. Building on this review, we discuss the rationale behind RAG and examine the opportunities for integrating these Generative AI (GenAI) technologies into the smart mobility sector. We propose a conceptual framework aimed at developing multi-agent systems capable of intelligently and conversationally delivering smart mobility services to urban commuters, transportation operators, and decision-makers. Our approach seeks to foster an autonomous and intelligent approach that (a) promotes science-based advisory to reduce traffic congestion, accidents, and carbon emissions at multiple scales, (b) facilitates public education and engagement in participatory mobility management, and (c) automates specialized transportation management tasks and the development of critical ITS platforms, such as data analytics and interpretation, knowledge representation, and traffic simulations. By integrating LLM and RAG, our approach seeks to overcome the limitations of traditional rule-based multi-agent systems, which rely on fixed knowledge bases and limited reasoning capabilities. This integration paves the way for a more scalable, intuitive, and automated multi-agent paradigm, driving advancements in ITS and urban mobility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02877v1">Configurable Foundation Models: Building LLMs from a Modular Perspective</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      Advancements in LLMs have recently unveiled challenges tied to computational efficiency and continual scalability due to their requirements of huge parameters, making the applications and evolution of these models on devices with limited computation resources and scenarios requiring various abilities increasingly cumbersome. Inspired by modularity within the human brain, there is a growing tendency to decompose LLMs into numerous functional modules, allowing for inference with part of modules and dynamic assembly of modules to tackle complex tasks, such as mixture-of-experts. To highlight the inherent efficiency and composability of the modular approach, we coin the term brick to represent each functional module, designating the modularized structure as configurable foundation models. In this paper, we offer a comprehensive overview and investigation of the construction, utilization, and limitation of configurable foundation models. We first formalize modules into emergent bricks - functional neuron partitions that emerge during the pre-training phase, and customized bricks - bricks constructed via additional post-training to improve the capabilities and knowledge of LLMs. Based on diverse functional bricks, we further present four brick-oriented operations: retrieval and routing, merging, updating, and growing. These operations allow for dynamic configuration of LLMs based on instructions to handle complex tasks. To verify our perspective, we conduct an empirical analysis on widely-used LLMs. We find that the FFN layers follow modular patterns with functional specialization of neurons and functional neuron partitions. Finally, we highlight several open issues and directions for future research. Overall, this paper aims to offer a fresh modular perspective on existing LLM research and inspire the future creation of more efficient and scalable foundational models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02691v1">LLM-Assisted Visual Analytics: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2024-09-04
      | 💬 Accepted at EG UK Computer Graphics & Visual Computing 2024
    </div>
    <details class="paper-abstract">
      We explore the integration of large language models (LLMs) into visual analytics (VA) systems to transform their capabilities through intuitive natural language interactions. We survey current research directions in this emerging field, examining how LLMs are integrated into data management, language interaction, visualisation generation, and language generation processes. We highlight the new possibilities that LLMs bring to VA, especially how they can change VA processes beyond the usual use cases. We especially highlight building new visualisation-language models, allowing access of a breadth of domain knowledge, multimodal interaction, and opportunities with guidance. Finally, we carefully consider the prominent challenges of using current LLMs in VA tasks. Our discussions in this paper aim to guide future researchers working on LLM-assisted VA systems and help them navigate common obstacles when developing these systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02604v1">Hypothesizing Missing Causal Variables with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-04
      | 💬 Code - https://github.com/ivaxi0s/hypothesizing-causal-variable-llm
    </div>
    <details class="paper-abstract">
      Scientific discovery is a catalyst for human intellectual advances, driven by the cycle of hypothesis generation, experimental design, data evaluation, and iterative assumption refinement. This process, while crucial, is expensive and heavily dependent on the domain knowledge of scientists to generate hypotheses and navigate the scientific cycle. Central to this is causality, the ability to establish the relationship between the cause and the effect. Motivated by the scientific discovery process, in this work, we formulate a novel task where the input is a partial causal graph with missing variables, and the output is a hypothesis about the missing variables to complete the partial graph. We design a benchmark with varying difficulty levels and knowledge assumptions about the causal graph. With the growing interest in using Large Language Models (LLMs) to assist in scientific discovery, we benchmark open-source and closed models on our testbed. We show the strong ability of LLMs to hypothesize the mediation variables between a cause and its effect. In contrast, they underperform in hypothesizing the cause and effect variables themselves. We also observe surprising results where some of the open-source models outperform the closed GPT-4 model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.04985v6">SparQ Attention: Bandwidth-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      The computational difficulties of large language model (LLM) inference remain a significant obstacle to their widespread deployment. The need for many applications to support long input sequences and process them in large batches typically causes token-generation to be bottlenecked by data transfer. For this reason, we introduce SparQ Attention, a technique for increasing the inference throughput of LLMs by utilising memory bandwidth more efficiently within the attention layers, through selective fetching of the cached history. Our proposed technique can be applied directly to off-the-shelf LLMs during inference, without requiring any modification to the pre-training setup or additional fine-tuning. We show that SparQ Attention brings up to 8x savings in attention data transfers without substantial drops in accuracy, by evaluating Llama 2 and 3, Mistral, Gemma and Pythia models on a wide range of downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11155v1">ISO: Overlap of Computation and Communication within Seqenence For LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      In the realm of Large Language Model (LLM) inference, the inherent structure of transformer models coupled with the multi-GPU tensor parallelism strategy leads to a sequential execution of computation and communication. This results in substantial underutilization of computing resources during the communication phase. To mitigate this inefficiency, various techniques have been developed to optimize the use of computational power throughout the communication process. These strategies primarily involve overlapping matrix computations and communications, as well as interleaving micro-batches across different requests. Nonetheless, these approaches either fall short of achieving ideal overlap or impose certain limitations on their application. To overcome these challenges, this paper introduces a novel strategy for computation-communication overlap that operates at the sequence level. This method not only enhances the degree of overlap but also minimizes the constraints on its applicability. Experimental evaluations conducted using 30b/70b models have demonstrated significant improvements in efficiency. Specifically, the proposed technique has been shown to reduce time consumption by approximately 35% on 4090 GPU and by roughly 15% on A800 GPU during the prefill stage of LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15412v5">Towards Measuring and Modeling "Culture" in LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      We present a survey of more than 90 recent papers that aim to study cultural representation and inclusion in large language models (LLMs). We observe that none of the studies explicitly define "culture, which is a complex, multifaceted concept; instead, they probe the models on some specially designed datasets which represent certain aspects of "culture". We call these aspects the proxies of culture, and organize them across two dimensions of demographic and semantic proxies. We also categorize the probing methods employed. Our analysis indicates that only certain aspects of ``culture,'' such as values and objectives, have been studied, leaving several other interesting and important facets, especially the multitude of semantic domains (Thompson et al., 2020) and aboutness (Hershcovich et al., 2022), unexplored. Two other crucial gaps are the lack of robustness of probing techniques and situated studies on the impact of cultural mis- and under-representation in LLM-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00128v2">Can AI Replace Human Subjects? A Large-Scale Replication of Psychological Experiments with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-04
      | 💬 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) is increasingly being integrated into scientific research, particularly in the social sciences, where understanding human behavior is critical. Large Language Models (LLMs) like GPT-4 have shown promise in replicating human-like responses in various psychological experiments. However, the extent to which LLMs can effectively replace human subjects across diverse experimental contexts remains unclear. Here, we conduct a large-scale study replicating 154 psychological experiments from top social science journals with 618 main effects and 138 interaction effects using GPT-4 as a simulated participant. We find that GPT-4 successfully replicates 76.0 percent of main effects and 47.0 percent of interaction effects observed in the original studies, closely mirroring human responses in both direction and significance. However, only 19.44 percent of GPT-4's replicated confidence intervals contain the original effect sizes, with the majority of replicated effect sizes exceeding the 95 percent confidence interval of the original studies. Additionally, there is a 71.6 percent rate of unexpected significant results where the original studies reported null findings, suggesting potential overestimation or false positives. Our results demonstrate the potential of LLMs as powerful tools in psychological research but also emphasize the need for caution in interpreting AI-driven findings. While LLMs can complement human studies, they cannot yet fully replace the nuanced insights provided by human subjects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15221v2">LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      Recent large language model (LLM) defenses have greatly improved models' ability to refuse harmful queries, even when adversarially attacked. However, LLM defenses are primarily evaluated against automated adversarial attacks in a single turn of conversation, an insufficient threat model for real-world malicious use. We demonstrate that multi-turn human jailbreaks uncover significant vulnerabilities, exceeding 70% attack success rate (ASR) on HarmBench against defenses that report single-digit ASRs with automated single-turn attacks. Human jailbreaks also reveal vulnerabilities in machine unlearning defenses, successfully recovering dual-use biosecurity knowledge from unlearned models. We compile these results into Multi-Turn Human Jailbreaks (MHJ), a dataset of 2,912 prompts across 537 multi-turn jailbreaks. We publicly release MHJ alongside a compendium of jailbreak tactics developed across dozens of commercial red teaming engagements, supporting research towards stronger LLM defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02244v1">Therapy as an NLP Task: Psychologists' Comparison of LLMs and Human Peers in CBT</a></div>
    <div class="paper-meta">
      📅 2024-09-03
    </div>
    <details class="paper-abstract">
      Wider access to therapeutic care is one of the biggest challenges in mental health treatment. Due to institutional barriers, some people seeking mental health support have turned to large language models (LLMs) for personalized therapy, even though these models are largely unsanctioned and untested. We investigate the potential and limitations of using LLMs as providers of evidence-based therapy by using mixed methods clinical metrics. Using HELPERT, a prompt run on a large language model using the same process and training as a comparative group of peer counselors, we replicated publicly accessible mental health conversations rooted in Cognitive Behavioral Therapy (CBT) to compare session dynamics and counselor's CBT-based behaviors between original peer support sessions and their reconstructed HELPERT sessions. Two licensed, CBT-trained clinical psychologists evaluated the sessions using the Cognitive Therapy Rating Scale and provided qualitative feedback. Our findings show that the peer sessions are characterized by empathy, small talk, therapeutic alliance, and shared experiences but often exhibit therapist drift. Conversely, HELPERT reconstructed sessions exhibit minimal therapist drift and higher adherence to CBT methods but display a lack of collaboration, empathy, and cultural understanding. Through CTRS ratings and psychologists' feedback, we highlight the importance of human-AI collaboration for scalable mental health. Our work outlines the ethical implication of imparting human-like subjective qualities to LLMs in therapeutic settings, particularly the risk of deceptive empathy, which may lead to unrealistic patient expectations and potential harm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02074v1">RACONTEUR: A Knowledgeable, Insightful, and Portable LLM-Powered Shell Command Explainer</a></div>
    <div class="paper-meta">
      📅 2024-09-03
      | 💬 Accepted by NDSS Symposium 2025. Please cite this paper as "Jiangyi Deng, Xinfeng Li, Yanjiao Chen, Yijie Bai, Haiqin Weng, Yan Liu, Tao Wei, Wenyuan Xu. RACONTEUR: A Knowledgeable, Insightful, and Portable LLM-Powered Shell Command Explainer. In the 32nd Annual Network and Distributed System Security Symposium (NDSS 2025)."
    </div>
    <details class="paper-abstract">
      Malicious shell commands are linchpins to many cyber-attacks, but may not be easy to understand by security analysts due to complicated and often disguised code structures. Advances in large language models (LLMs) have unlocked the possibility of generating understandable explanations for shell commands. However, existing general-purpose LLMs suffer from a lack of expert knowledge and a tendency to hallucinate in the task of shell command explanation. In this paper, we present Raconteur, a knowledgeable, expressive and portable shell command explainer powered by LLM. Raconteur is infused with professional knowledge to provide comprehensive explanations on shell commands, including not only what the command does (i.e., behavior) but also why the command does it (i.e., purpose). To shed light on the high-level intent of the command, we also translate the natural-language-based explanation into standard technique & tactic defined by MITRE ATT&CK, the worldwide knowledge base of cybersecurity. To enable Raconteur to explain unseen private commands, we further develop a documentation retriever to obtain relevant information from complementary documentations to assist the explanation process. We have created a large-scale dataset for training and conducted extensive experiments to evaluate the capability of Raconteur in shell command explanation. The experiments verify that Raconteur is able to provide high-quality explanations and in-depth insight of the intent of the command.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06385v3">Low-Rank Quantization-Aware Training for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are omnipresent, however their practical deployment is challenging due to their ever increasing computational and memory demands. Quantization is one of the most effective ways to make them more compute and memory efficient. Quantization-aware training (QAT) methods, generally produce the best quantized performance, however it comes at the cost of potentially long training time and excessive memory usage, making it impractical when applying for LLMs. Inspired by parameter-efficient fine-tuning (PEFT) and low-rank adaptation (LoRA) literature, we propose LR-QAT -- a lightweight and memory-efficient QAT algorithm for LLMs. LR-QAT employs several components to save memory without sacrificing predictive performance: (a) low-rank auxiliary weights that are aware of the quantization grid; (b) a downcasting operator using fixed-point or double-packed integers and (c) checkpointing. Unlike most related work, our method (i) is inference-efficient, leading to no additional overhead compared to traditional PTQ; (ii) can be seen as a general extended pretraining framework, meaning that the resulting model can still be utilized for any downstream task afterwards; (iii) can be applied across a wide range of quantization settings, such as different choices quantization granularity, activation quantization, and seamlessly combined with many PTQ techniques. We apply LR-QAT to LLaMA-1/2/3 and Mistral model families and validate its effectiveness on several downstream tasks. Our method outperforms common post-training quantization (PTQ) approaches and reaches the same model performance as full-model QAT at the fraction of its memory usage. Specifically, we can train a 7B LLM on a single consumer grade GPU with 24GB of memory. Our source code is available at https://github.com/qualcomm-ai-research/LR-QAT
    </details>
</div>
