# llm - 2023_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.00426v1">keqing: knowledge-based question answering is a nature chain-of-thought mentor of LLM</a></div>
    <div class="paper-meta">
      📅 2023-12-31
      | 💬 12 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited remarkable performance on various natural language processing (NLP) tasks, especially for question answering. However, in the face of problems beyond the scope of knowledge, these LLMs tend to talk nonsense with a straight face, where the potential solution could be incorporating an Information Retrieval (IR) module and generating response based on these retrieved knowledge. In this paper, we present a novel framework to assist LLMs, such as ChatGPT, to retrieve question-related structured information on the knowledge graph, and demonstrate that Knowledge-based question answering (Keqing) could be a nature Chain-of-Thought (CoT) mentor to guide the LLM to sequentially find the answer entities of a complex question through interpretable logical chains. Specifically, the workflow of Keqing will execute decomposing a complex question according to predefined templates, retrieving candidate entities on knowledge graph, reasoning answers of sub-questions, and finally generating response with reasoning paths, which greatly improves the reliability of LLM's response. The experimental results on KBQA datasets show that Keqing can achieve competitive performance and illustrate the logic of answering each question.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.00287v1">The Art of Defending: A Systematic Evaluation and Analysis of LLM Defense Strategies on Safety and Over-Defensiveness</a></div>
    <div class="paper-meta">
      📅 2023-12-30
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) play an increasingly pivotal role in natural language processing applications, their safety concerns become critical areas of NLP research. This paper presents Safety and Over-Defensiveness Evaluation (SODE) benchmark: a collection of diverse safe and unsafe prompts with carefully designed evaluation methods that facilitate systematic evaluation, comparison, and analysis over 'safety' and 'over-defensiveness.' With SODE, we study a variety of LLM defense strategies over multiple state-of-the-art LLMs, which reveals several interesting and important findings, such as (a) the widely popular 'self-checking' techniques indeed improve the safety against unsafe inputs, but this comes at the cost of extreme over-defensiveness on the safe inputs, (b) providing a safety instruction along with in-context exemplars (of both safe and unsafe inputs) consistently improves safety and also mitigates undue over-defensiveness of the models, (c) providing contextual knowledge easily breaks the safety guardrails and makes the models more vulnerable to generating unsafe responses. Overall, our work reveals numerous such critical findings that we believe will pave the way and facilitate further research in improving the safety of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.17601v1">The Tyranny of Possibilities in the Design of Task-Oriented LLM Systems: A Scoping Survey</a></div>
    <div class="paper-meta">
      📅 2023-12-29
      | 💬 18 pages, 6 figures. Work-in-progress draft published to gather feedback. Please reach out with comments, if any
    </div>
    <details class="paper-abstract">
      This scoping survey focuses on our current understanding of the design space for task-oriented LLM systems and elaborates on definitions and relationships among the available design parameters. The paper begins by defining a minimal task-oriented LLM system and exploring the design space of such systems through a thought experiment contemplating the performance of diverse LLM system configurations (involving single LLMs, single LLM-based agents, and multiple LLM-based agent systems) on a complex software development task and hypothesizes the results. We discuss a pattern in our results and formulate them into three conjectures. While these conjectures may be partly based on faulty assumptions, they provide a starting point for future research. The paper then surveys a select few design parameters: covering and organizing research in LLM augmentation, prompting techniques, and uncertainty estimation, and discussing their significance. The paper notes the lack of focus on computational and energy efficiency in evaluating research in these areas. Our survey findings provide a basis for developing the concept of linear and non-linear contexts, which we define and use to enable an agent-centric projection of prompting techniques providing a lens through which prompting techniques can be viewed as multi-agent systems. The paper discusses the implications of this lens, for the cross-pollination of research between LLM prompting and LLM-based multi-agent systems; and also, for the generation of synthetic training data based on existing prompting techniques in research. In all, the scoping survey presents seven conjectures that can help guide future research efforts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.14769v3">Large Language Model (LLM) Bias Index -- LLMBI</a></div>
    <div class="paper-meta">
      📅 2023-12-29
    </div>
    <details class="paper-abstract">
      The Large Language Model Bias Index (LLMBI) is a pioneering approach designed to quantify and address biases inherent in large language models (LLMs), such as GPT-4. We recognise the increasing prevalence and impact of LLMs across diverse sectors. This research introduces a novel metric, LLMBI, to systematically measure and mitigate biases potentially skewing model responses. We formulated LLMBI using a composite scoring system incorporating multiple dimensions of bias, including but not limited to age, gender, and racial biases. To operationalise this metric, we engaged in a multi-step process involving collecting and annotating LLM responses, applying sophisticated Natural Language Processing (NLP) techniques for bias detection, and computing the LLMBI score through a specially crafted mathematical formula. The formula integrates weighted averages of various bias dimensions, a penalty for dataset diversity deficiencies, and a correction for sentiment biases. Our empirical analysis, conducted using responses from OpenAI's API, employs advanced sentiment analysis as a representative method for bias detection. The research reveals LLMs, whilst demonstrating impressive capabilities in text generation, exhibit varying degrees of bias across different dimensions. LLMBI provides a quantifiable measure to compare biases across models and over time, offering a vital tool for systems engineers, researchers and regulators in enhancing the fairness and reliability of LLMs. It highlights the potential of LLMs in mimicking unbiased human-like responses. Additionally, it underscores the necessity of continuously monitoring and recalibrating such models to align with evolving societal norms and ethical standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.17535v1">Olapa-MCoT: Enhancing the Chinese Mathematical Reasoning Capability of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-29
      | 💬 10 pages, 1 figures
    </div>
    <details class="paper-abstract">
      CoT (Chain-of-Thought) is a way to solve reasoning problems for LLMs . Recently, many researches appear for improving the CoT capability of LLMs. In this work, we also proposed Olapa-MCoT, which is a LLMs based on llama2-13B PLM for finetuning and alignment learning. During the alignment training, we proposed the SimRRHF algorithm and Incorrect Data Relearning and mainly focused on optimizing the Chinese mathematical reasoning ability of Olapa-MCoT. The experiment achieved significant results, with the accuracy of Chinese mathematical reasoning up to 50%, 36% rise compared to llama2-13B. In addition, the accuracy of English reasoning ability also increased by nearly 4%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.17476v1">Exploring the Sensitivity of LLMs' Decision-Making Capabilities: Insights from Prompt Variation and Hyperparameters</a></div>
    <div class="paper-meta">
      📅 2023-12-29
      | 💬 EMNLP 2023
    </div>
    <details class="paper-abstract">
      The advancement of Large Language Models (LLMs) has led to their widespread use across a broad spectrum of tasks including decision making. Prior studies have compared the decision making abilities of LLMs with those of humans from a psychological perspective. However, these studies have not always properly accounted for the sensitivity of LLMs' behavior to hyperparameters and variations in the prompt. In this study, we examine LLMs' performance on the Horizon decision making task studied by Binz and Schulz (2023) analyzing how LLMs respond to variations in prompts and hyperparameters. By experimenting on three OpenAI language models possessing different capabilities, we observe that the decision making abilities fluctuate based on the input prompts and temperature settings. Contrary to previous findings language models display a human-like exploration exploitation tradeoff after simple adjustments to the prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.17117v1">Grounding-Prompter: Prompting LLM with Multimodal Information for Temporal Sentence Grounding in Long Videos</a></div>
    <div class="paper-meta">
      📅 2023-12-28
    </div>
    <details class="paper-abstract">
      Temporal Sentence Grounding (TSG), which aims to localize moments from videos based on the given natural language queries, has attracted widespread attention. Existing works are mainly designed for short videos, failing to handle TSG in long videos, which poses two challenges: i) complicated contexts in long videos require temporal reasoning over longer moment sequences, and ii) multiple modalities including textual speech with rich information require special designs for content understanding in long videos. To tackle these challenges, in this work we propose a Grounding-Prompter method, which is capable of conducting TSG in long videos through prompting LLM with multimodal information. In detail, we first transform the TSG task and its multimodal inputs including speech and visual, into compressed task textualization. Furthermore, to enhance temporal reasoning under complicated contexts, a Boundary-Perceptive Prompting strategy is proposed, which contains three folds: i) we design a novel Multiscale Denoising Chain-of-Thought (CoT) to combine global and local semantics with noise filtering step by step, ii) we set up validity principles capable of constraining LLM to generate reasonable predictions following specific formats, and iii) we introduce one-shot In-Context-Learning (ICL) to boost reasoning through imitation, enhancing LLM in TSG task understanding. Experiments demonstrate the state-of-the-art performance of our Grounding-Prompter method, revealing the benefits of prompting LLM with multimodal information for TSG in long videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.10384v2">Retrieval Augmented Generation of Symbolic Music with LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-28
      | 💬 LBD @ ISMIR 2023
    </div>
    <details class="paper-abstract">
      We explore the use of large language models (LLMs) for music generation using a retrieval system to select relevant examples. We find promising initial results for music generation in a dialogue with the user, especially considering the ease with which such a system can be implemented. The code is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.16549v1">How Robust are LLMs to In-Context Majority Label Bias?</a></div>
    <div class="paper-meta">
      📅 2023-12-27
      | 💬 6 pages, 3 figures, 2 table. Accepted at Workshop on Responsible Language Modeling, AAAI 2024, (www.aaai.org)
    </div>
    <details class="paper-abstract">
      In the In-Context Learning (ICL) setup, various forms of label biases can manifest. One such manifestation is majority label bias, which arises when the distribution of labeled examples in the in-context samples is skewed towards one or more specific classes making Large Language Models (LLMs) more prone to predict those labels. Such discrepancies can arise from various factors, including logistical constraints, inherent biases in data collection methods, limited access to diverse data sources, etc. which are unavoidable in a real-world industry setup. In this work, we study the robustness of in-context learning in LLMs to shifts that occur due to majority label bias within the purview of text classification tasks. Prior works have shown that in-context learning with LLMs is susceptible to such biases. In our study, we go one level deeper and show that the robustness boundary varies widely for different models and tasks, with certain LLMs being highly robust (~90%) to majority label bias. Additionally, our findings also highlight the impact of model size and the richness of instructional prompts contributing towards model robustness. We restrict our study to only publicly available open-source models to ensure transparency and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.16378v1">Automating Knowledge Acquisition for Content-Centric Cognitive Agents Using LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-27
      | 💬 7 pages, 7 figures, AAAI Fall Symposium Series 2023 on Integrating Cognitive Architecture and Generative Models
    </div>
    <details class="paper-abstract">
      The paper describes a system that uses large language model (LLM) technology to support the automatic learning of new entries in an intelligent agent's semantic lexicon. The process is bootstrapped by an existing non-toy lexicon and a natural language generator that converts formal, ontologically-grounded representations of meaning into natural language sentences. The learning method involves a sequence of LLM requests and includes an automatic quality control step. To date, this learning method has been applied to learning multiword expressions whose meanings are equivalent to those of transitive verbs in the agent's lexicon. The experiment demonstrates the benefits of a hybrid learning architecture that integrates knowledge-based methods and resources with both traditional data analytics and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.16351v1">LLMs with User-defined Prompts as Generic Data Operators for Reliable Data Processing</a></div>
    <div class="paper-meta">
      📅 2023-12-26
      | 💬 5 pages, 8 figures, 1st IEEE International Workshop on Data Engineering and Modeling for AI (DEMAI), IEEE BigData 2023
    </div>
    <details class="paper-abstract">
      Data processing is one of the fundamental steps in machine learning pipelines to ensure data quality. Majority of the applications consider the user-defined function (UDF) design pattern for data processing in databases. Although the UDF design pattern introduces flexibility, reusability and scalability, the increasing demand on machine learning pipelines brings three new challenges to this design pattern -- not low-code, not dependency-free and not knowledge-aware. To address these challenges, we propose a new design pattern that large language models (LLMs) could work as a generic data operator (LLM-GDO) for reliable data cleansing, transformation and modeling with their human-compatible performance. In the LLM-GDO design pattern, user-defined prompts (UDPs) are used to represent the data processing logic rather than implementations with a specific programming language. LLMs can be centrally maintained so users don't have to manage the dependencies at the run-time. Fine-tuning LLMs with domain-specific data could enhance the performance on the domain-specific tasks which makes data processing knowledge-aware. We illustrate these advantages with examples in different data processing tasks. Furthermore, we summarize the challenges and opportunities introduced by LLMs to provide a complete view of this design pattern for more discussions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.14135v2">V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-26
      | 💬 Project page with code: https://vstar-seal.github.io/
    </div>
    <details class="paper-abstract">
      When we look around and perform complex tasks, how we see and selectively process what we see is crucial. However, the lack of this visual search mechanism in current multimodal LLMs (MLLMs) hinders their ability to focus on important visual details, especially when handling high-resolution and visually crowded images. To address this, we introduce V*, an LLM-guided visual search mechanism that employs the world knowledge in LLMs for efficient visual querying. When combined with an MLLM, this mechanism enhances collaborative reasoning, contextual understanding, and precise targeting of specific visual elements. This integration results in a new MLLM meta-architecture, named Show, sEArch, and TelL (SEAL). We further create V*Bench, a benchmark specifically designed to evaluate MLLMs in their ability to process high-resolution images and focus on visual details. Our study highlights the necessity of incorporating visual search capabilities into multimodal systems. The code is available https://github.com/penghao-wu/vstar.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15784v1">AHAM: Adapt, Help, Ask, Model -- Harvesting LLMs for literature mining</a></div>
    <div class="paper-meta">
      📅 2023-12-25
      | 💬 Submitted to IDA 2024
    </div>
    <details class="paper-abstract">
      In an era marked by a rapid increase in scientific publications, researchers grapple with the challenge of keeping pace with field-specific advances. We present the `AHAM' methodology and a metric that guides the domain-specific \textbf{adapt}ation of the BERTopic topic modeling framework to improve scientific text analysis. By utilizing the LLaMa2 generative language model, we generate topic definitions via one-shot learning by crafting prompts with the \textbf{help} of domain experts to guide the LLM for literature mining by \textbf{asking} it to model the topic names. For inter-topic similarity evaluation, we leverage metrics from language generation and translation processes to assess lexical and semantic similarity of the generated topics. Our system aims to reduce both the ratio of outlier topics to the total number of topics and the similarity between topic definitions. The methodology has been assessed on a newly gathered corpus of scientific papers on literature-based discovery. Through rigorous evaluation by domain experts, AHAM has been validated as effective in uncovering intriguing and novel insights within broad research areas. We explore the impact of domain adaptation of sentence-transformers for the task of topic \textbf{model}ing using two datasets, each specialized to specific scientific domains within arXiv and medarxiv. We evaluate the impact of data size, the niche of adaptation, and the importance of domain adaptation. Our results suggest a strong interaction between domain adaptation and topic modeling precision in terms of outliers and topic definitions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.17264v1">ESGReveal: An LLM-based approach for extracting structured data from ESG reports</a></div>
    <div class="paper-meta">
      📅 2023-12-25
    </div>
    <details class="paper-abstract">
      ESGReveal is an innovative method proposed for efficiently extracting and analyzing Environmental, Social, and Governance (ESG) data from corporate reports, catering to the critical need for reliable ESG information retrieval. This approach utilizes Large Language Models (LLM) enhanced with Retrieval Augmented Generation (RAG) techniques. The ESGReveal system includes an ESG metadata module for targeted queries, a preprocessing module for assembling databases, and an LLM agent for data extraction. Its efficacy was appraised using ESG reports from 166 companies across various sectors listed on the Hong Kong Stock Exchange in 2022, ensuring comprehensive industry and market capitalization representation. Utilizing ESGReveal unearthed significant insights into ESG reporting with GPT-4, demonstrating an accuracy of 76.9% in data extraction and 83.7% in disclosure analysis, which is an improvement over baseline models. This highlights the framework's capacity to refine ESG data analysis precision. Moreover, it revealed a demand for reinforced ESG disclosures, with environmental and social data disclosures standing at 69.5% and 57.2%, respectively, suggesting a pursuit for more corporate transparency. While current iterations of ESGReveal do not process pictorial information, a functionality intended for future enhancement, the study calls for continued research to further develop and compare the analytical capabilities of various LLMs. In summary, ESGReveal is a stride forward in ESG data processing, offering stakeholders a sophisticated tool to better evaluate and advance corporate sustainability efforts. Its evolution is promising in promoting transparency in corporate reporting and aligning with broader sustainable development aims.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15450v1">Agent4Ranking: Semantic Robust Ranking via Personalized Query Rewriting Using Multi-agent LLM</a></div>
    <div class="paper-meta">
      📅 2023-12-24
    </div>
    <details class="paper-abstract">
      Search engines are crucial as they provide an efficient and easy way to access vast amounts of information on the internet for diverse information needs. User queries, even with a specific need, can differ significantly. Prior research has explored the resilience of ranking models against typical query variations like paraphrasing, misspellings, and order changes. Yet, these works overlook how diverse demographics uniquely formulate identical queries. For instance, older individuals tend to construct queries more naturally and in varied order compared to other groups. This demographic diversity necessitates enhancing the adaptability of ranking models to diverse query formulations. To this end, in this paper, we propose a framework that integrates a novel rewriting pipeline that rewrites queries from various demographic perspectives and a novel framework to enhance ranking robustness. To be specific, we use Chain of Thought (CoT) technology to utilize Large Language Models (LLMs) as agents to emulate various demographic profiles, then use them for efficient query rewriting, and we innovate a robust Multi-gate Mixture of Experts (MMoE) architecture coupled with a hybrid loss function, collectively strengthening the ranking models' robustness. Our extensive experimentation on both public and industrial datasets assesses the efficacy of our query rewriting approach and the enhanced accuracy and robustness of the ranking model. The findings highlight the sophistication and effectiveness of our proposed model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.05685v4">Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena</a></div>
    <div class="paper-meta">
      📅 2023-12-24
      | 💬 NeurIPS 2023 Datasets and Benchmarks Track
    </div>
    <details class="paper-abstract">
      Evaluating large language model (LLM) based chat assistants is challenging due to their broad capabilities and the inadequacy of existing benchmarks in measuring human preferences. To address this, we explore using strong LLMs as judges to evaluate these models on more open-ended questions. We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases, as well as limited reasoning ability, and propose solutions to mitigate some of them. We then verify the agreement between LLM judges and human preferences by introducing two benchmarks: MT-bench, a multi-turn question set; and Chatbot Arena, a crowdsourced battle platform. Our results reveal that strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80% agreement, the same level of agreement between humans. Hence, LLM-as-a-judge is a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain. Additionally, we show our benchmark and traditional benchmarks complement each other by evaluating several variants of LLaMA and Vicuna. The MT-bench questions, 3K expert votes, and 30K conversations with human preferences are publicly available at https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.12893v2">A Safer Vision-based Autonomous Planning System for Quadrotor UAVs with Dynamic Obstacle Trajectory Prediction and Its Application with LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-23
    </div>
    <details class="paper-abstract">
      For intelligent quadcopter UAVs, a robust and reliable autonomous planning system is crucial. Most current trajectory planning methods for UAVs are suitable for static environments but struggle to handle dynamic obstacles, which can pose challenges and even dangers to flight. To address this issue, this paper proposes a vision-based planning system that combines tracking and trajectory prediction of dynamic obstacles to achieve efficient and reliable autonomous flight. We use a lightweight object detection algorithm to identify dynamic obstacles and then use Kalman Filtering to track and estimate their motion states. During the planning phase, we not only consider static obstacles but also account for the potential movements of dynamic obstacles. For trajectory generation, we use a B-spline-based trajectory search algorithm, which is further optimized with various constraints to enhance safety and alignment with the UAV's motion characteristics. We conduct experiments in both simulation and real-world environments, and the results indicate that our approach can successfully detect and avoid obstacles in dynamic environments in real-time, offering greater reliability compared to existing approaches. Furthermore, with the advancements in Natural Language Processing (NLP) technology demonstrating exceptional zero-shot generalization capabilities, more user-friendly human-machine interactions have become feasible, and this study also explores the integration of autonomous planning systems with Large Language Models (LLMs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15296v2">DeTiME: Diffusion-Enhanced Topic Modeling using Encoder-decoder based LLM</a></div>
    <div class="paper-meta">
      📅 2023-12-23
      | 💬 19 pages, 4 figures, EMNLP 2023
    </div>
    <details class="paper-abstract">
      In the burgeoning field of natural language processing (NLP), Neural Topic Models (NTMs) , Large Language Models (LLMs) and Diffusion model have emerged as areas of significant research interest. Despite this, NTMs primarily utilize contextual embeddings from LLMs, which are not optimal for clustering or capable for topic based text generation. NTMs have never been combined with diffusion model for text generation. Our study addresses these gaps by introducing a novel framework named Diffusion-Enhanced Topic Modeling using Encoder-Decoder-based LLMs (DeTiME). DeTiME leverages Encoder-Decoder-based LLMs to produce highly clusterable embeddings that could generate topics that exhibit both superior clusterability and enhanced semantic coherence compared to existing methods. Additionally, by exploiting the power of diffusion model, our framework also provides the capability to do topic based text generation. This dual functionality allows users to efficiently produce highly clustered topics and topic based text generation simultaneously. DeTiME's potential extends to generating clustered embeddings as well. Notably, our proposed framework(both encoder-decoder based LLM and diffusion model) proves to be efficient to train and exhibits high adaptability to other LLMs and diffusion model, demonstrating its potential for a wide array of applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15033v1">Sparsity-Guided Holistic Explanation for LLMs with Interpretable Inference-Time Intervention</a></div>
    <div class="paper-meta">
      📅 2023-12-22
      | 💬 Accepted to AAAI 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved unprecedented breakthroughs in various natural language processing domains. However, the enigmatic ``black-box'' nature of LLMs remains a significant challenge for interpretability, hampering transparent and accountable applications. While past approaches, such as attention visualization, pivotal subnetwork extraction, and concept-based analyses, offer some insight, they often focus on either local or global explanations within a single dimension, occasionally falling short in providing comprehensive clarity. In response, we propose a novel methodology anchored in sparsity-guided techniques, aiming to provide a holistic interpretation of LLMs. Our framework, termed SparseCBM, innovatively integrates sparsity to elucidate three intertwined layers of interpretation: input, subnetwork, and concept levels. In addition, the newly introduced dimension of interpretable inference-time intervention facilitates dynamic adjustments to the model during deployment. Through rigorous empirical evaluations on real-world datasets, we demonstrate that SparseCBM delivers a profound understanding of LLM behaviors, setting it apart in both interpreting and ameliorating model inaccuracies. Codes are provided in supplements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.14670v1">Zero-shot Causal Graph Extrapolation from Text via LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-22
      | 💬 XAI4Sci Workshop @ AAAI24
    </div>
    <details class="paper-abstract">
      We evaluate the ability of large language models (LLMs) to infer causal relations from natural language. Compared to traditional natural language processing and deep learning techniques, LLMs show competitive performance in a benchmark of pairwise relations without needing (explicit) training samples. This motivates us to extend our approach to extrapolating causal graphs through iterated pairwise queries. We perform a preliminary analysis on a benchmark of biomedical abstracts with ground-truth causal graphs validated by experts. The results are promising and support the adoption of LLMs for such a crucial step in causal inference, especially in medical domains, where the amount of scientific text to analyse might be huge, and the causal statements are often implicit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.14327v1">Parameter Efficient Tuning Allows Scalable Personalization of LLMs for Text Entry: A Case Study on Abbreviation Expansion</a></div>
    <div class="paper-meta">
      📅 2023-12-21
    </div>
    <details class="paper-abstract">
      Abbreviation expansion is a strategy used to speed up communication by limiting the amount of typing and using a language model to suggest expansions. Here we look at personalizing a Large Language Model's (LLM) suggestions based on prior conversations to enhance the relevance of predictions, particularly when the user data is small (~1000 samples). Specifically, we compare fine-tuning, prompt-tuning, and retrieval augmented generation of expanded text suggestions for abbreviated inputs. Our case study with a deployed 8B parameter LLM on a real user living with ALS, and experiments on movie character personalization indicates that (1) customization may be necessary in some scenarios and prompt-tuning generalizes well to those, (2) fine-tuning on in-domain data (with as few as 600 samples) still shows some gains, however (3) retrieval augmented few-shot selection also outperforms fine-tuning. (4) Parameter efficient tuning allows for efficient and scalable personalization. For prompt-tuning, we also find that initializing the learned "soft-prompts" to user relevant concept tokens leads to higher accuracy than random initialization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.13961v1">ChatGPT as a commenter to the news: can LLMs generate human-like opinions?</a></div>
    <div class="paper-meta">
      📅 2023-12-21
      | 💬 Published as Tseng, R., Verberne, S., van der Putten, P. (2023). ChatGPT as a Commenter to the News: Can LLMs Generate Human-Like Opinions?. In: Ceolin, D., Caselli, T., Tulin, M. (eds) Disinformation in Open Online Media. MISDOOM 2023. Lecture Notes in Computer Science, vol 14397. Springer, Cham
    </div>
    <details class="paper-abstract">
      ChatGPT, GPT-3.5, and other large language models (LLMs) have drawn significant attention since their release, and the abilities of these models have been investigated for a wide variety of tasks. In this research we investigate to what extent GPT-3.5 can generate human-like comments on Dutch news articles. We define human likeness as `not distinguishable from human comments', approximated by the difficulty of automatic classification between human and GPT comments. We analyze human likeness across multiple prompting techniques. In particular, we utilize zero-shot, few-shot and context prompts, for two generated personas. We found that our fine-tuned BERT models can easily distinguish human-written comments from GPT-3.5 generated comments, with none of the used prompting methods performing noticeably better. We further analyzed that human comments consistently showed higher lexical diversity than GPT-generated comments. This indicates that although generative LLMs can generate fluent text, their capability to create human-like opinionated comments is still limited.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.13925v1">AsyncMLD: Asynchronous Multi-LLM Framework for Dialogue Recommendation System</a></div>
    <div class="paper-meta">
      📅 2023-12-21
      | 💬 This paper is part of the proceedings of the Dialogue Robot Competition 2023
    </div>
    <details class="paper-abstract">
      We have reached a practical and realistic phase in human-support dialogue agents by developing a large language model (LLM). However, when requiring expert knowledge or anticipating the utterance content using the massive size of the dialogue database, we still need help with the utterance content's effectiveness and the efficiency of its output speed, even if using LLM. Therefore, we propose a framework that uses LLM asynchronously in the part of the system that returns an appropriate response and in the part that understands the user's intention and searches the database. In particular, noting that it takes time for the robot to speak, threading related to database searches is performed while the robot is speaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.13126v1">Generative agents in the streets: Exploring the use of Large Language Models (LLMs) in collecting urban perceptions</a></div>
    <div class="paper-meta">
      📅 2023-12-20
      | 💬 30 Pages, 15 Figures, Submitted in a Journal for Peer review
    </div>
    <details class="paper-abstract">
      Evaluating the surroundings to gain understanding, frame perspectives, and anticipate behavioral reactions is an inherent human trait. However, these continuous encounters are diverse and complex, posing challenges to their study and experimentation. Researchers have been able to isolate environmental features and study their effect on human perception and behavior. However, the research attempts to replicate and study human behaviors with proxies, such as by integrating virtual mediums and interviews, have been inconsistent. Large language models (LLMs) have recently been unveiled as capable of contextual understanding and semantic reasoning. These models have been trained on large amounts of text and have evolved to mimic believable human behavior. This study explores the current advancements in Generative agents powered by LLMs with the help of perceptual experiments. The experiment employs Generative agents to interact with the urban environments using street view images to plan their journey toward specific goals. The agents are given virtual personalities, which make them distinguishable. They are also provided a memory database to store their thoughts and essential visual information and retrieve it when needed to plan their movement. Since LLMs do not possess embodiment, nor have access to the visual realm, and lack a sense of motion or direction, we designed movement and visual modules that help agents gain an overall understanding of surroundings. The agents are further employed to rate the surroundings they encounter based on their perceived sense of safety and liveliness. As these agents store details in their memory, we query the findings to get details regarding their thought processes. Overall, this study experiments with current AI developments and their potential in simulated human behavior in urban environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.12832v1">Turning Dust into Gold: Distilling Complex Reasoning Capabilities from LLMs by Leveraging Negative Data</a></div>
    <div class="paper-meta">
      📅 2023-12-20
      | 💬 AAAI 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have performed well on various reasoning tasks, but their inaccessibility and numerous parameters hinder wide application in practice. One promising way is distilling the reasoning ability from LLMs to small models by the generated chain-of-thought reasoning paths. In some cases, however, LLMs may produce incorrect reasoning chains, especially when facing complex mathematical problems. Previous studies only transfer knowledge from positive samples and drop the synthesized data with wrong answers. In this work, we illustrate the merit of negative data and propose a model specialization framework to distill LLMs with negative samples besides positive ones. The framework consists of three progressive steps, covering from training to inference stages, to absorb knowledge from negative data. We conduct extensive experiments across arithmetic reasoning tasks to demonstrate the role of negative data in distillation from LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.12028v2">Prompt Sapper: A LLM-Empowered Production Tool for Building AI Chains</a></div>
    <div class="paper-meta">
      📅 2023-12-20
      | 💬 23 pages, 5 figures, accepted to TOSEM 2023
    </div>
    <details class="paper-abstract">
      The emergence of foundation models, such as large language models (LLMs) GPT-4 and text-to-image models DALL-E, has opened up numerous possibilities across various domains. People can now use natural language (i.e. prompts) to communicate with AI to perform tasks. While people can use foundation models through chatbots (e.g., ChatGPT), chat, regardless of the capabilities of the underlying models, is not a production tool for building reusable AI services. APIs like LangChain allow for LLM-based application development but require substantial programming knowledge, thus posing a barrier. To mitigate this, we propose the concept of AI chain and introduce the best principles and practices that have been accumulated in software engineering for decades into AI chain engineering, to systematise AI chain engineering methodology. We also develop a no-code integrated development environment, Prompt Sapper, which embodies these AI chain engineering principles and patterns naturally in the process of building AI chains, thereby improving the performance and quality of AI chains. With Prompt Sapper, AI chain engineers can compose prompt-based AI services on top of foundation models through chat-based requirement analysis and visual programming. Our user study evaluated and demonstrated the efficiency and correctness of Prompt Sapper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08206v2">Human-Centric Autonomous Systems With LLMs for User Command Reasoning</a></div>
    <div class="paper-meta">
      📅 2023-12-19
      | 💬 In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops, 2024
    </div>
    <details class="paper-abstract">
      The evolution of autonomous driving has made remarkable advancements in recent years, evolving into a tangible reality. However, a human-centric large-scale adoption hinges on meeting a variety of multifaceted requirements. To ensure that the autonomous system meets the user's intent, it is essential to accurately discern and interpret user commands, especially in complex or emergency situations. To this end, we propose to leverage the reasoning capabilities of Large Language Models (LLMs) to infer system requirements from in-cabin users' commands. Through a series of experiments that include different LLM models and prompt designs, we explore the few-shot multivariate binary classification accuracy of system requirements from natural language textual commands. We confirm the general ability of LLMs to understand and reason about prompts but underline that their effectiveness is conditioned on the quality of both the LLM model and the design of appropriate sequential prompts. Code and models are public with the link \url{https://github.com/KTH-RPL/DriveCmd_LLM}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.12624v1">Building a Llama2-finetuned LLM for Odia Language Utilizing Domain Knowledge Instruction Set</a></div>
    <div class="paper-meta">
      📅 2023-12-19
    </div>
    <details class="paper-abstract">
      Building LLMs for languages other than English is in great demand due to the unavailability and performance of multilingual LLMs, such as understanding the local context. The problem is critical for low-resource languages due to the need for instruction sets. In a multilingual country like India, there is a need for LLMs supporting Indic languages to provide generative AI and LLM-based technologies and services to its citizens. This paper presents our approach of i) generating a large Odia instruction set, including domain knowledge data suitable for LLM fine-tuning, and ii) building a Llama2-finetuned model tailored for enhanced performance in the Odia domain. The proposed work will help researchers build an instruction set and LLM, particularly for Indic languages. We will release the model and instruction set for the public for research and noncommercial purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.05399v1">Automated Assessment of Students' Code Comprehension using LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-19
    </div>
    <details class="paper-abstract">
      Assessing student's answers and in particular natural language answers is a crucial challenge in the field of education. Advances in machine learning, including transformer-based models such as Large Language Models(LLMs), have led to significant progress in various natural language tasks. Nevertheless, amidst the growing trend of evaluating LLMs across diverse tasks, evaluating LLMs in the realm of automated answer assesment has not received much attention. To address this gap, we explore the potential of using LLMs for automated assessment of student's short and open-ended answer. Particularly, we use LLMs to compare students' explanations with expert explanations in the context of line-by-line explanations of computer programs. For comparison purposes, we assess both Large Language Models (LLMs) and encoder-based Semantic Textual Similarity (STS) models in the context of assessing the correctness of students' explanation of computer code. Our findings indicate that LLMs, when prompted in few-shot and chain-of-thought setting perform comparable to fine-tuned encoder-based models in evaluating students' short answers in programming domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.02775v3">AI-TA: Towards an Intelligent Question-Answer Teaching Assistant using Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-18
      | 💬 Updates for camera-ready submission
    </div>
    <details class="paper-abstract">
      Responding to the thousands of student questions on online QA platforms each semester has a considerable human cost, particularly in computing courses with rapidly growing enrollments. To address the challenges of scalable and intelligent question-answering (QA), we introduce an innovative solution that leverages open-source Large Language Models (LLMs) from the LLaMA-2 family to ensure data privacy. Our approach combines augmentation techniques such as retrieval augmented generation (RAG), supervised fine-tuning (SFT), and learning from human preferences data using Direct Preference Optimization (DPO). Through extensive experimentation on a Piazza dataset from an introductory CS course, comprising 10,000 QA pairs and 1,500 pairs of preference data, we demonstrate a significant 30% improvement in the quality of answers, with RAG being a particularly impactful addition. Our contributions include the development of a novel architecture for educational QA, extensive evaluations of LLM performance utilizing both human assessments and LLM-based metrics, and insights into the challenges and future directions of educational data processing. This work paves the way for the development of AI-TA, an intelligent QA assistant customizable for courses with an online QA platform
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11420v1">Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2023-12-18
      | 💬 The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      This paper introduces an efficient strategy to transform Large Language Models (LLMs) into Multi-Modal Large Language Models (MLLMs). By conceptualizing this transformation as a domain adaptation process, i.e., transitioning from text understanding to embracing multiple modalities, we intriguingly note that, within each attention block, tuning LayerNorm suffices to yield strong performance. Moreover, when benchmarked against other tuning approaches like full parameter finetuning or LoRA, its benefits on efficiency are substantial. For example, when compared to LoRA on a 13B model scale, performance can be enhanced by an average of over 20% across five multi-modal tasks, and meanwhile, results in a significant reduction of trainable parameters by 41.9% and a decrease in GPU memory usage by 17.6%. On top of this LayerNorm strategy, we showcase that selectively tuning only with conversational data can improve efficiency further. Beyond these empirical outcomes, we provide a comprehensive analysis to explore the role of LayerNorm in adapting LLMs to the multi-modal domain and improving the expressive power of the model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.10029v2">Challenges with unsupervised LLM knowledge discovery</a></div>
    <div class="paper-meta">
      📅 2023-12-18
      | 💬 12 pages (38 including references and appendices). First three authors equal contribution, randomised order
    </div>
    <details class="paper-abstract">
      We show that existing unsupervised methods on large language model (LLM) activations do not discover knowledge -- instead they seem to discover whatever feature of the activations is most prominent. The idea behind unsupervised knowledge elicitation is that knowledge satisfies a consistency structure, which can be used to discover knowledge. We first prove theoretically that arbitrary features (not just knowledge) satisfy the consistency structure of a particular leading unsupervised knowledge-elicitation method, contrast-consistent search (Burns et al. - arXiv:2212.03827). We then present a series of experiments showing settings in which unsupervised methods result in classifiers that do not predict knowledge, but instead predict a different prominent feature. We conclude that existing unsupervised methods for discovering latent knowledge are insufficient, and we contribute sanity checks to apply to evaluating future knowledge elicitation methods. Conceptually, we hypothesise that the identification issues explored here, e.g. distinguishing a model's knowledge from that of a simulated character's, will persist for future unsupervised methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11336v1">DRDT: Dynamic Reflection with Divergent Thinking for LLM-based Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2023-12-18
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has sparked interest in their application to sequential recommendation tasks as they can provide supportive item information. However, due to the inherent complexities of sequential recommendation, such as sequential patterns across datasets, noise within sequences, and the temporal evolution of user preferences, existing LLM reasoning strategies, such as in-context learning and chain-of-thought are not fully effective. To address these challenges, we introduce a novel reasoning principle: Dynamic Reflection with Divergent Thinking within a retriever-reranker framework. Our approach starts with a collaborative in-context demonstration retriever, which collects sequences exhibiting collaborative behaviors as in-context examples. Following this, we abstract high-level user preferences across multiple aspects, providing a more nuanced understanding of user interests and circumventing the noise within the raw sequences. The cornerstone of our methodology is dynamic reflection, a process that emulates human learning through probing, critiquing, and reflecting, using user feedback to tailor the analysis more effectively to the target user in a temporal manner. We evaluate our approach on three datasets using six pre-trained LLMs. The superior performance observed across these models demonstrates the efficacy of our reasoning strategy, notably achieved without the need to fine-tune the LLMs. With our principle, we managed to outperform GPT-Turbo-3.5 on three datasets using 7b models e.g., Vicuna-7b and Openchat-7b on NDCG@10. This research not only highlights the potential of LLMs in enhancing sequential recommendation systems but also underscores the importance of developing tailored reasoning strategies to fully harness their capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08583v2">ZeroQuant(4+2): Redefining LLMs Quantization with a New FP6-Centric Strategy for Diverse Generative Tasks</a></div>
    <div class="paper-meta">
      📅 2023-12-18
    </div>
    <details class="paper-abstract">
      This study examines 4-bit quantization methods like GPTQ in large language models (LLMs), highlighting GPTQ's overfitting and limited enhancement in Zero-Shot tasks. While prior works merely focusing on zero-shot measurement, we extend task scope to more generative categories such as code generation and abstractive summarization, in which we found that INT4 quantization can significantly underperform. However, simply shifting to higher precision formats like FP6 has been particularly challenging, thus overlooked, due to poor performance caused by the lack of sophisticated integration and system acceleration strategies on current AI hardware. Our results show that FP6, even with a coarse-grain quantization scheme, performs robustly across various algorithms and tasks, demonstrating its superiority in accuracy and versatility. Notably, with the FP6 quantization, \codestar-15B model performs comparably to its FP16 counterpart in code generation, and for smaller models like the 406M it closely matches their baselines in summarization. Neither can be achieved by INT4. To better accommodate various AI hardware and achieve the best system performance, we propose a novel 4+2 design for FP6 to achieve similar latency to the state-of-the-art INT4 fine-grain quantization. With our design, FP6 can become a promising solution to the current 4-bit quantization methods used in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.09936v3">BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions</a></div>
    <div class="paper-meta">
      📅 2023-12-18
      | 💬 Accepted at AAAI Conference on Artificial Intelligence (AAAI-24)
    </div>
    <details class="paper-abstract">
      Vision Language Models (VLMs), which extend Large Language Models (LLM) by incorporating visual understanding capability, have demonstrated significant advancements in addressing open-ended visual question-answering (VQA) tasks. However, these models cannot accurately interpret images infused with text, a common occurrence in real-world scenarios. Standard procedures for extracting information from images often involve learning a fixed set of query embeddings. These embeddings are designed to encapsulate image contexts and are later used as soft prompt inputs in LLMs. Yet, this process is limited to the token count, potentially curtailing the recognition of scenes with text-rich context. To improve upon them, the present study introduces BLIVA: an augmented version of InstructBLIP with Visual Assistant. BLIVA incorporates the query embeddings from InstructBLIP and also directly projects encoded patch embeddings into the LLM, a technique inspired by LLaVA. This approach assists the model to capture intricate details potentially missed during the query decoding process. Empirical evidence demonstrates that our model, BLIVA, significantly enhances performance in processing text-rich VQA benchmarks (up to 17.76% in OCR-VQA benchmark) and in undertaking general (not particularly text-rich) VQA benchmarks (up to 7.9% in Visual Spatial Reasoning benchmark), and achieved 17.72% overall improvement in a comprehensive multimodal LLM benchmark (MME), comparing to our baseline InstructBLIP. BLIVA demonstrates significant capability in decoding real-world images, irrespective of text presence. To demonstrate the broad industry applications enabled by BLIVA, we evaluate the model using a new dataset comprising YouTube thumbnails paired with question-answer sets across 11 diverse categories. Our code and models are freely accessible at https://github.com/mlpc-ucsd/BLIVA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.10610v1">Do LLMs Work on Charts? Designing Few-Shot Prompts for Chart Question Answering and Summarization</a></div>
    <div class="paper-meta">
      📅 2023-12-17
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      A number of tasks have been proposed recently to facilitate easy access to charts such as chart QA and summarization. The dominant paradigm to solve these tasks has been to fine-tune a pretrained model on the task data. However, this approach is not only expensive but also not generalizable to unseen tasks. On the other hand, large language models (LLMs) have shown impressive generalization capabilities to unseen tasks with zero- or few-shot prompting. However, their application to chart-related tasks is not trivial as these tasks typically involve considering not only the underlying data but also the visual features in the chart image. We propose PromptChart, a multimodal few-shot prompting framework with LLMs for chart-related applications. By analyzing the tasks carefully, we have come up with a set of prompting guidelines for each task to elicit the best few-shot performance from LLMs. We further propose a strategy to inject visual information into the prompts. Our experiments on three different chart-related information consumption tasks show that with properly designed prompts LLMs can excel on the benchmarks, achieving state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11541v1">CLIPSyntel: CLIP and LLM Synergy for Multimodal Question Summarization in Healthcare</a></div>
    <div class="paper-meta">
      📅 2023-12-16
      | 💬 AAAI 2024
    </div>
    <details class="paper-abstract">
      In the era of modern healthcare, swiftly generating medical question summaries is crucial for informed and timely patient care. Despite the increasing complexity and volume of medical data, existing studies have focused solely on text-based summarization, neglecting the integration of visual information. Recognizing the untapped potential of combining textual queries with visual representations of medical conditions, we introduce the Multimodal Medical Question Summarization (MMQS) Dataset. This dataset, a major contribution to our work, pairs medical queries with visual aids, facilitating a richer and more nuanced understanding of patient needs. We also propose a framework, utilizing the power of Contrastive Language Image Pretraining(CLIP) and Large Language Models(LLMs), consisting of four modules that identify medical disorders, generate relevant context, filter medical concepts, and craft visually aware summaries. Our comprehensive framework harnesses the power of CLIP, a multimodal foundation model, and various general-purpose LLMs, comprising four main modules: the medical disorder identification module, the relevant context generation module, the context filtration module for distilling relevant medical concepts and knowledge, and finally, a general-purpose LLM to generate visually aware medical question summaries. Leveraging our MMQS dataset, we showcase how visual cues from images enhance the generation of medically nuanced summaries. This multimodal approach not only enhances the decision-making process in healthcare but also fosters a more nuanced understanding of patient queries, laying the groundwork for future research in personalized and responsive medical care
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.10003v1">ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent</a></div>
    <div class="paper-meta">
      📅 2023-12-15
      | 💬 19 pages, 4 figures, 4 tables, 8 listings
    </div>
    <details class="paper-abstract">
      Answering complex natural language questions often necessitates multi-step reasoning and integrating external information. Several systems have combined knowledge retrieval with a large language model (LLM) to answer such questions. These systems, however, suffer from various failure cases, and we cannot directly train them end-to-end to fix such failures, as interaction with external knowledge is non-differentiable. To address these deficiencies, we define a ReAct-style LLM agent with the ability to reason and act upon external knowledge. We further refine the agent through a ReST-like method that iteratively trains on previous trajectories, employing growing-batch reinforcement learning with AI feedback for continuous self-improvement and self-distillation. Starting from a prompted large model and after just two iterations of the algorithm, we can produce a fine-tuned small model that achieves comparable performance on challenging compositional question-answering benchmarks with two orders of magnitude fewer parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09740v1">VITA: A Multi-modal LLM-based System for Longitudinal, Autonomous, and Adaptive Robotic Mental Well-being Coaching</a></div>
    <div class="paper-meta">
      📅 2023-12-15
    </div>
    <details class="paper-abstract">
      Recently, several works have explored if and how robotic coaches can promote and maintain mental well-being in different settings. However, findings from these studies revealed that these robotic coaches are not ready to be used and deployed in real-world settings due to several limitations that span from technological challenges to coaching success. To overcome these challenges, this paper presents VITA, a novel multi-modal LLM-based system that allows robotic coaches to autonomously adapt to the coachee's multi-modal behaviours (facial valence and speech duration) and deliver coaching exercises in order to promote mental well-being in adults. We identified five objectives that correspond to the challenges in the recent literature, and we show how the VITA system addresses these via experimental validations that include one in-lab pilot study (N=4) that enabled us to test different robotic coach configurations (pre-scripted, generic, and adaptive models) and inform its design for using it in the real world, and one real-world study (N=17) conducted in a workplace over 4 weeks. Our results show that: (i) coachees perceived the VITA adaptive and generic configurations more positively than the pre-scripted one, and they felt understood and heard by the adaptive robotic coach, (ii) the VITA adaptive robotic coach kept learning successfully by personalising to each coachee over time and did not detect any interaction ruptures during the coaching, (iii) coachees had significant mental well-being improvements via the VITA-based robotic coach practice. The code for the VITA system is openly available via: https://github.com/Cambridge-AFAR/VITA-system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09731v1">Uncovering the Causes of Emotions in Software Developer Communication Using Zero-shot LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-15
    </div>
    <details class="paper-abstract">
      Understanding and identifying the causes behind developers' emotions (e.g., Frustration caused by `delays in merging pull requests') can be crucial towards finding solutions to problems and fostering collaboration in open-source communities. Effectively identifying such information in the high volume of communications across the different project channels, such as chats, emails, and issue comments, requires automated recognition of emotions and their causes. To enable this automation, large-scale software engineering-specific datasets that can be used to train accurate machine learning models are required. However, such datasets are expensive to create with the variety and informal nature of software projects' communication channels. In this paper, we explore zero-shot LLMs that are pre-trained on massive datasets but without being fine-tuned specifically for the task of detecting emotion causes in software engineering: ChatGPT, GPT-4, and flan-alpaca. Our evaluation indicates that these recently available models can identify emotion categories when given detailed emotions, although they perform worse than the top-rated models. For emotion cause identification, our results indicate that zero-shot LLMs are effective at recognizing the correct emotion cause with a BLEU-2 score of 0.598. To highlight the potential use of these techniques, we conduct a case study of the causes of Frustration in the last year of development of a popular open-source project, revealing several interesting insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15539v2">SteloCoder: a Decoder-Only LLM for Multi-Language to Python Code Translation</a></div>
    <div class="paper-meta">
      📅 2023-12-15
    </div>
    <details class="paper-abstract">
      With the recent focus on Large Language Models (LLMs), both StarCoder (Li et al., 2023) and Code Llama (Rozi\`ere et al., 2023) have demonstrated remarkable performance in code generation. However, there is still a need for improvement in code translation functionality with efficient training techniques. In response to this, we introduce SteloCoder, a decoder-only StarCoder-based LLM designed specifically for multi-programming language-to-Python code translation. In particular, SteloCoder achieves C++, C#, JavaScript, Java, or PHP-to-Python code translation without specifying the input programming language. We modified StarCoder model architecture by incorporating a Mixture-of-Experts (MoE) technique featuring five experts and a gating network for multi-task handling. Experts are obtained by StarCoder fine-tuning. Specifically, we use a Low-Rank Adaptive Method (LoRA) technique, limiting each expert size as only 0.06% of number of StarCoder's parameters. At the same time, to enhance training efficiency in terms of time, we adopt curriculum learning strategy and use self-instruct data for efficient fine-tuning. As a result, each expert takes only 6 hours to train on one single 80Gb A100 HBM. With experiments on XLCoST datasets, SteloCoder achieves an average of 73.76 CodeBLEU score in multi-programming language-to-Python translation, surpassing the top performance from the leaderboard by at least 3.5. This accomplishment is attributed to only 45M extra parameters with StarCoder as the backbone and 32 hours of valid training on one 80GB A100 HBM. The source code is release here: https://github.com/sade-adrien/SteloCoder.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.18396v3">LLMs Can Understand Encrypted Prompt: Towards Privacy-Computing Friendly Transformers</a></div>
    <div class="paper-meta">
      📅 2023-12-15
    </div>
    <details class="paper-abstract">
      The community explored to build private inference frameworks for transformer-based large language models (LLMs) in a server-client setting, where the server holds the model parameters and the client inputs its private data (or prompt) for inference. However, these frameworks impose significant overhead when the private inputs are forward propagated through the original LLMs. In this paper, we show that substituting the computation- and communication-heavy operators in the transformer architecture with privacy-computing friendly approximations can greatly reduce the private inference costs while incurring very minor impact on model performance. Compared to state-of-the-art Iron (NeurIPS 2022), our privacy-computing friendly model inference pipeline achieves a $5\times$ acceleration in computation and an 80% reduction in communication overhead, while retaining nearly identical accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08688v2">TigerBot: An Open Multilingual Multitask LLM</a></div>
    <div class="paper-meta">
      📅 2023-12-15
    </div>
    <details class="paper-abstract">
      We release and introduce the TigerBot family of large language models (LLMs), consisting of base and chat models, sized from 7, 13, 70 and 180 billion parameters. We develop our models embarking from Llama-2 and BLOOM, and push the boundary further in data, training algorithm, infrastructure, and application tools. Our models yield meaningful performance gain over SOTA open-source models, e.g., Llama-2, specifically 6% gain in English and 20% gain in Chinese. TigerBot model family also achieves leading performance in major academic and industrial benchmarks and leaderboards. We believe that TigerBot represents just a snapshot of lightning-fast progression in LLM open-source community. Therefore, we are thrilled to give back by publicly releasing our models and reporting our approach behind, with additional emphases on building SOTA LLMs in a democratized way and making LLMs of use in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.10101v1">A Review of Repository Level Prompting for LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-15
      | 💬 15 figures/charts, 7 pages, Submitted as an NLP project at Northeastern. Focuses on comparing two papers, there are many more papers that could be included
    </div>
    <details class="paper-abstract">
      As coding challenges become more complex, recent advancements in Large Language Models (LLMs) have led to notable successes, such as achieving a 94.6\% solve rate on the HumanEval benchmark. Concurrently, there is an increasing commercial push for repository-level inline code completion tools, such as GitHub Copilot and Tab Nine, aimed at enhancing developer productivity. This paper delves into the transition from individual coding problems to repository-scale solutions, presenting a thorough review of the current literature on effective LLM prompting for code generation at the repository level. We examine approaches that will work with black-box LLMs such that they will be useful and applicable to commercial use cases, and their applicability in interpreting code at a repository scale. We juxtapose the Repository-Level Prompt Generation technique with RepoCoder, an iterative retrieval and generation method, to highlight the trade-offs inherent in each approach and to establish best practices for their application in cutting-edge coding benchmarks. The interplay between iterative refinement of prompts and the development of advanced retrieval systems forms the core of our discussion, offering a pathway to significantly improve LLM performance in code generation tasks. Insights from this study not only guide the application of these methods but also chart a course for future research to integrate such techniques into broader software engineering contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09366v1">Arabic Mini-ClimateGPT : A Climate Change and Sustainability Tailored Arabic LLM</a></div>
    <div class="paper-meta">
      📅 2023-12-14
      | 💬 Accepted to EMNLP 2023 (Findings)
    </div>
    <details class="paper-abstract">
      Climate change is one of the most significant challenges we face together as a society. Creating awareness and educating policy makers the wide-ranging impact of climate change is an essential step towards a sustainable future. Recently, Large Language Models (LLMs) like ChatGPT and Bard have shown impressive conversational abilities and excel in a wide variety of NLP tasks. While these models are close-source, recently alternative open-source LLMs such as Stanford Alpaca and Vicuna have shown promising results. However, these open-source models are not specifically tailored for climate related domain specific information and also struggle to generate meaningful responses in other languages such as, Arabic. To this end, we propose a light-weight Arabic Mini-ClimateGPT that is built on an open-source LLM and is specifically fine-tuned on a conversational-style instruction tuning curated Arabic dataset Clima500-Instruct with over 500k instructions about climate change and sustainability. Further, our model also utilizes a vector embedding based retrieval mechanism during inference. We validate our proposed model through quantitative and qualitative evaluations on climate-related queries. Our model surpasses the baseline LLM in 88.3% of cases during ChatGPT-based evaluation. Furthermore, our human expert evaluation reveals an 81.6% preference for our model's responses over multiple popular open-source models. Our open-source demos, code-base and models are available here https://github.com/mbzuai-oryx/ClimateGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08725v1">A Comparative Analysis of Fine-Tuned LLMs and Few-Shot Learning of LLMs for Financial Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2023-12-14
    </div>
    <details class="paper-abstract">
      Financial sentiment analysis plays a crucial role in uncovering latent patterns and detecting emerging trends, enabling individuals to make well-informed decisions that may yield substantial advantages within the constantly changing realm of finance. Recently, Large Language Models (LLMs) have demonstrated their effectiveness in diverse domains, showcasing remarkable capabilities even in zero-shot and few-shot in-context learning for various Natural Language Processing (NLP) tasks. Nevertheless, their potential and applicability in the context of financial sentiment analysis have not been thoroughly explored yet. To bridge this gap, we employ two approaches: in-context learning (with a focus on gpt-3.5-turbo model) and fine-tuning LLMs on a finance-domain dataset. Given the computational costs associated with fine-tuning LLMs with large parameter sizes, our focus lies on smaller LLMs, spanning from 250M to 3B parameters for fine-tuning. We then compare the performances with state-of-the-art results to evaluate their effectiveness in the finance-domain. Our results demonstrate that fine-tuned smaller LLMs can achieve comparable performance to state-of-the-art fine-tuned LLMs, even with models having fewer parameters and a smaller training dataset. Additionally, the zero-shot and one-shot performance of LLMs produces comparable results with fine-tuned smaller LLMs and state-of-the-art outcomes. Furthermore, our analysis demonstrates that there is no observed enhancement in performance for finance-domain sentiment analysis when the number of shots for in-context learning is increased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08629v1">ChatSOS: LLM-based knowledge Q&A system for safety engineering</a></div>
    <div class="paper-meta">
      📅 2023-12-14
      | 💬 in Chinese language
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have notably propelled natural language processing (NLP) capabilities, demonstrating significant potential in safety engineering applications. Despite these advancements, LLMs face constraints in processing specialized tasks, attributed to factors such as corpus size, input processing limitations, and privacy concerns. Obtaining useful information from reliable sources in a limited time is crucial for LLM. Addressing this, our study introduces an LLM-based Q&A system for safety engineering, enhancing the comprehension and response accuracy of the model. We employed prompt engineering to incorporate external knowledge databases, thus enriching the LLM with up-to-date and reliable information. The system analyzes historical incident reports through statistical methods, utilizes vector embedding to construct a vector database, and offers an efficient similarity-based search functionality. Our findings indicate that the integration of external knowledge significantly augments the capabilities of LLM for in-depth problem analysis and autonomous task assignment. It effectively summarizes accident reports and provides pertinent recommendations. This integration approach not only expands LLM applications in safety engineering but also sets a precedent for future developments towards automation and intelligent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08056v1">Knowledge-Aware Artifact Image Synthesis with LLM-Enhanced Prompting and Multi-Source Supervision</a></div>
    <div class="paper-meta">
      📅 2023-12-13
    </div>
    <details class="paper-abstract">
      Ancient artifacts are an important medium for cultural preservation and restoration. However, many physical copies of artifacts are either damaged or lost, leaving a blank space in archaeological and historical studies that calls for artifact image generation techniques. Despite the significant advancements in open-domain text-to-image synthesis, existing approaches fail to capture the important domain knowledge presented in the textual description, resulting in errors in recreated images such as incorrect shapes and patterns. In this paper, we propose a novel knowledge-aware artifact image synthesis approach that brings lost historical objects accurately into their visual forms. We use a pretrained diffusion model as backbone and introduce three key techniques to enhance the text-to-image generation framework: 1) we construct prompts with explicit archaeological knowledge elicited from large language models (LLMs); 2) we incorporate additional textual guidance to correlated historical expertise in a contrastive manner; 3) we introduce further visual-semantic constraints on edge and perceptual features that enable our model to learn more intricate visual details of the artifacts. Compared to existing approaches, our proposed model produces higher-quality artifact images that align better with the implicit details and historical knowledge contained within written documents, thus achieving significant improvements across automatic metrics and in human evaluation. Our code and data are available at https://github.com/danielwusg/artifact_diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08400v1">Beyond English: Evaluating LLMs for Arabic Grammatical Error Correction</a></div>
    <div class="paper-meta">
      📅 2023-12-13
      | 💬 arXiv admin note: text overlap with arXiv:2308.04492
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) finetuned to follow human instruction have recently exhibited significant capabilities in various English NLP tasks. However, their performance in grammatical error correction (GEC), especially on languages other than English, remains significantly unexplored. In this work, we evaluate the abilities of instruction finetuned LLMs in Arabic GEC, a complex task due to Arabic's rich morphology. Our findings suggest that various prompting methods, coupled with (in-context) few-shot learning, demonstrate considerable effectiveness, with GPT-4 achieving up to $65.49$ F$_{1}$ score under expert prompting (approximately $5$ points higher than our established baseline). Despite these positive results, we find that instruction finetuned models, regardless of their size, are still outperformed by fully finetuned ones, even if they are significantly smaller in size. This disparity highlights substantial room for improvements for LLMs. Inspired by methods used in low-resource machine translation, we also develop a method exploiting synthetic data that significantly outperforms previous models on two standard Arabic benchmarks. Our best model achieves a new SOTA on Arabic GEC, with $73.29$ and $73.26$ F$_{1}$ on the 2014 and 2015 QALB datasets, respectively, compared to peer-reviewed published baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06720v2">Audio-Visual LLM for Video Understanding</a></div>
    <div class="paper-meta">
      📅 2023-12-13
    </div>
    <details class="paper-abstract">
      This paper presents Audio-Visual LLM, a Multimodal Large Language Model that takes both visual and auditory inputs for holistic video understanding. A key design is the modality-augmented training, which involves the integration of modality-specific tokens engineered to activate the appropriate visual and/or auditory encoder selectively. This mechanism is pivotal in enabling end-to-end joint training with video data at different modalities, including visual-only, audio-only, and audio-visual formats. Moreover, we introduce a high-quality video instruction dataset, derived from GPT-4. This dataset allows Audio-Visual LLM to adeptly process a variety of task-oriented video instructions, ranging from multi-turn conversations and audio-visual narratives to complex reasoning tasks. Extensive experiments demonstrate that Audio-Visual LLM impressively achieves strong zero-shot results across a range of video understanding tasks. For example, Audio-Visual LLM achieves an accuracy of 53.7% on MSRVTT-QA, outperforming non-LLM-based InterVideo by 6.6% and LLM-based Valley by 4.4%, respectively. Additionally, our Audio-Visual LLM also achieves competitive performance on audio tasks (e.g., AudioCaps).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07886v1">Modality Plug-and-Play: Elastic Modality Adaptation in Multimodal LLMs for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-12-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are capable of reasoning over diverse input data modalities through pre-trained encoders. However, the growing diversity of input data modalities prevents incorporating all modalities into LLMs, especially when LLMs are deployed on resource-constrained edge devices for embodied AI applications. Instead, a better option is to adaptively involve only the useful modalities at runtime, depending on the current environmental contexts and task requirements. For such modality adaptation, existing work adopts fixed connections between encoders and the LLM's input layer, leading to high training cost at runtime and ineffective cross-modal interaction. In this paper, we address these limitations by presenting mPnP-LLM, a new technique that allows fully elastic, automated and prompt runtime modality adaptation, by connecting unimodal encoders to a flexible set of last LLM blocks and making such latent connections fully trainable at runtime. Experiments over the nuScenes-QA dataset show that mPnP-LLM can achieve up to 3.7x FLOPs reduction and 30% GPU memory usage reduction, while retaining on-par accuracy with the existing schemes. Under the same compute budget, mPnP-LLM improves the task accuracy by up to 4% compared to the best existing scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07848v1">Finetuning an LLM on Contextual Knowledge of Classics for Q&A</a></div>
    <div class="paper-meta">
      📅 2023-12-13
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The open-source publishing of large language models (LLMs) has created many possibilities for how anyone who understands language and has access to a computer can interact with significant tools of artificial intelligence, particularly in the context of learning and knowledge dissemination. However, the utility of these models in specialized fields like Classics is still largely unexplored. This project is an attempt to merge the knowledge of Classics with the capabilities of artificial intelligence by finetuning an LLM to cater to the specific needs of learners and professionals. The goal of this project is to develop an LLM that not only reproduces contextual knowledge accurately but also exhibits a consistent "personality" - and, indeed, has consistent propriety - to appeal to a diverse audience who possess differing levels of knowledge. A significant portion of this project was dedicated to refining the dataset, following the principle of "garbage in, garbage out," to ensure the model generates relevant, useful, and creative responses when given a prompt (a statement, question, or single word). After training and evaluation, my model's ability to handle a vast array of different types of inputs and prompting exceeded expectations for a 355M parameter model, though its occasional hallucinations (especially when set with a high temperature), particularly in its assertions about historical events or its own identity, make it seem somewhat capricious and more work in the form of continuous finetuning will be undertaken.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07779v1">Tell, don't show: Declarative facts influence how LLMs generalize</a></div>
    <div class="paper-meta">
      📅 2023-12-12
    </div>
    <details class="paper-abstract">
      We examine how large language models (LLMs) generalize from abstract declarative statements in their training data. As an illustration, consider an LLM that is prompted to generate weather reports for London in 2050. One possibility is that the temperatures in the reports match the mean and variance of reports from 2023 (i.e. matching the statistics of pretraining). Another possibility is that the reports predict higher temperatures, by incorporating declarative statements about climate change from scientific papers written in 2023. An example of such a declarative statement is "global temperatures will increase by $1^{\circ} \mathrm{C}$ by 2050". To test the influence of abstract declarative statements, we construct tasks in which LLMs are finetuned on both declarative and procedural information. We find that declarative statements influence model predictions, even when they conflict with procedural information. In particular, finetuning on a declarative statement $S$ increases the model likelihood for logical consequences of $S$. The effect of declarative statements is consistent across three domains: aligning an AI assistant, predicting weather, and predicting demographic features. Through a series of ablations, we show that the effect of declarative statements cannot be explained by associative learning based on matching keywords. Nevertheless, the effect of declarative statements on model likelihoods is small in absolute terms and increases surprisingly little with model size (i.e. from 330 million to 175 billion parameters). We argue that these results have implications for AI risk (in relation to the "treacherous turn") and for fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07763v1">Can LLM find the green circle? Investigation and Human-guided tool manipulation for compositional generalization</a></div>
    <div class="paper-meta">
      📅 2023-12-12
      | 💬 Accepted by ICASSP 2024
    </div>
    <details class="paper-abstract">
      The meaning of complex phrases in natural language is composed of their individual components. The task of compositional generalization evaluates a model's ability to understand new combinations of components. Previous studies trained smaller, task-specific models, which exhibited poor generalization. While large language models (LLMs) exhibit impressive generalization abilities on many tasks through in-context learning (ICL), their potential for compositional generalization remains unexplored. In this paper, we first empirically investigate prevailing ICL methods in compositional generalization. We find that they struggle with complex compositional questions due to cumulative errors in long reasoning steps and intricate logic required for tool-making. Consequently, we propose a human-guided tool manipulation framework (HTM) that generates tools for sub-questions and integrates multiple tools. Our method enhances the effectiveness of tool creation and usage with minimal human effort. Experiments show that our method achieves state-of-the-art performance on two compositional generalization benchmarks and outperforms existing methods on the most challenging test split by 70%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.04730v2">DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions</a></div>
    <div class="paper-meta">
      📅 2023-12-12
    </div>
    <details class="paper-abstract">
      With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07420v1">FairSISA: Ensemble Post-Processing to Improve Fairness of Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-12
      | 💬 Accepted in NeurIPS 2023 Workshop on Socially Responsible Language Modelling Research (SoLaR)
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) is a costly endeavour in terms of time and computational resources. The large amount of training data used during the unsupervised pre-training phase makes it difficult to verify all data and, unfortunately, undesirable data may be ingested during training. Re-training from scratch is impractical and has led to the creation of the 'unlearning' discipline where models are modified to "unlearn" undesirable information without retraining. However, any modification can alter the behaviour of LLMs, especially on key dimensions such as fairness. This is the first work that examines this interplay between unlearning and fairness for LLMs. In particular, we focus on a popular unlearning framework known as SISA [Bourtoule et al., 2021], which creates an ensemble of models trained on disjoint shards. We evaluate the performance-fairness trade-off for SISA, and empirically demsontrate that SISA can indeed reduce fairness in LLMs. To remedy this, we propose post-processing bias mitigation techniques for ensemble models produced by SISA. We adapt the post-processing fairness improvement technique from [Hardt et al., 2016] to design three methods that can handle model ensembles, and prove that one of the methods is an optimal fair predictor for ensemble of models. Through experimental results, we demonstrate the efficacy of our post-processing framework called 'FairSISA'.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07368v1">Sequential Planning in Large Partially Observable Environments guided by LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-12
      | 💬 8 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      Sequential planning in large state space and action space quickly becomes intractable due to combinatorial explosion of the search space. Heuristic methods, like monte-carlo tree search, though effective for large state space, but struggle if action space is large. Pure reinforcement learning methods, relying only on reward signals, needs prohibitively large interactions with the environment to device a viable plan. If the state space, observations and actions can be represented in natural language then Large Language models (LLM) can be used to generate action plans. Recently several such goal-directed agents like Reflexion, CLIN, SayCan were able to surpass the performance of other state-of-the-art methods with minimum or no task specific training. But they still struggle with exploration and get stuck in local optima. Their planning capabilities are limited by the limited reasoning capability of the foundational LLMs on text data. We propose a hybrid agent "neoplanner", that synergizes both state space search with queries to foundational LLM to get the best action plan. The reward signals are quantitatively used to drive the search. A balance of exploration and exploitation is maintained by maximizing upper confidence bounds of values of states. In places where random exploration is needed, the LLM is queried to generate an action plan. Learnings from each trial are stored as entity relationships in text format. Those are used in future queries to the LLM for continual improvement. Experiments in the Scienceworld environment reveals a 124% improvement from the current best method in terms of average reward gained across multiple tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07110v1">LLMs Perform Poorly at Concept Extraction in Cyber-security Research Literature</a></div>
    <div class="paper-meta">
      📅 2023-12-12
      | 💬 24 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The cybersecurity landscape evolves rapidly and poses threats to organizations. To enhance resilience, one needs to track the latest developments and trends in the domain. It has been demonstrated that standard bibliometrics approaches show their limits in such a fast-evolving domain. For this purpose, we use large language models (LLMs) to extract relevant knowledge entities from cybersecurity-related texts. We use a subset of arXiv preprints on cybersecurity as our data and compare different LLMs in terms of entity recognition (ER) and relevance. The results suggest that LLMs do not produce good knowledge entities that reflect the cybersecurity context, but our results show some potential for noun extractors. For this reason, we developed a noun extractor boosted with some statistical analysis to extract specific and relevant compound nouns from the domain. Later, we tested our model to identify trends in the LLM domain. We observe some limitations, but it offers promising results to monitor the evolution of emergent trends.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.03241v2">Early Weight Averaging meets High Learning Rates for LLM Pre-training</a></div>
    <div class="paper-meta">
      📅 2023-12-11
      | 💬 17 pages, 13 figures, presented at NeurIPs 2023 WANT workshop
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) incurs significant cost; hence, any strategy that accelerates model convergence is helpful. In this paper, we investigate the ability of a simple idea checkpoint averaging along the trajectory of a training run to improve both convergence and generalization quite early on during training. Here we show that models trained with high learning rates observe higher gains due to checkpoint averaging. Furthermore, these gains are amplified when checkpoints are sampled with considerable spacing in training steps. Our training recipe outperforms conventional training and popular checkpoint averaging baselines such as exponential moving average (EMA) and stochastic moving average (SWA). We evaluate our training recipe by pre-training LLMs, where high learning rates are inherently preferred due to extremely large batch sizes. Specifically, we pre-trained nanoGPT-2 models of varying sizes, small (125M), medium (335M), and large (770M)on the OpenWebText dataset, comprised of 9B tokens. Additionally, we present results for publicly available Pythia LLMs, ranging from 1B to 12B, which were trained on the PILE-deduped dataset containing 207B tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06820v1">Extracting Self-Consistent Causal Insights from Users Feedback with LLMs and In-context Learning</a></div>
    <div class="paper-meta">
      📅 2023-12-11
    </div>
    <details class="paper-abstract">
      Microsoft Windows Feedback Hub is designed to receive customer feedback on a wide variety of subjects including critical topics such as power and battery. Feedback is one of the most effective ways to have a grasp of users' experience with Windows and its ecosystem. However, the sheer volume of feedback received by Feedback Hub makes it immensely challenging to diagnose the actual cause of reported issues. To better understand and triage issues, we leverage Double Machine Learning (DML) to associate users' feedback with telemetry signals. One of the main challenges we face in the DML pipeline is the necessity of domain knowledge for model design (e.g., causal graph), which sometimes is either not available or hard to obtain. In this work, we take advantage of reasoning capabilities in Large Language Models (LLMs) to generate a prior model that which to some extent compensates for the lack of domain knowledge and could be used as a heuristic for measuring feedback informativeness. Our LLM-based approach is able to extract previously known issues, uncover new bugs, and identify sequences of events that lead to a bug, while minimizing out-of-domain outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06652v1">Building Domain-Specific LLMs Faithful To The Islamic Worldview: Mirage or Technical Possibility?</a></div>
    <div class="paper-meta">
      📅 2023-12-11
      | 💬 Accepted for Muslims in ML workshop at the 37th Conference on Neural Information Processing Systems (NeurIPS 2023)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across numerous natural language understanding use cases. However, this impressive performance comes with inherent limitations, such as the tendency to perpetuate stereotypical biases or fabricate non-existent facts. In the context of Islam and its representation, accurate and factual representation of its beliefs and teachings rooted in the Quran and Sunnah is key. This work focuses on the challenge of building domain-specific LLMs faithful to the Islamic worldview and proposes ways to build and evaluate such systems. Firstly, we define this open-ended goal as a technical problem and propose various solutions. Subsequently, we critically examine known challenges inherent to each approach and highlight evaluation methodologies that can be used to assess such systems. This work highlights the need for high-quality datasets, evaluations, and interdisciplinary work blending machine learning with Islamic scholarship.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06550v1">LLM360: Towards Fully Transparent Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-11
    </div>
    <details class="paper-abstract">
      The recent surge in open-source Large Language Models (LLMs), such as LLaMA, Falcon, and Mistral, provides diverse options for AI practitioners and researchers. However, most LLMs have only released partial artifacts, such as the final model weights or inference code, and technical reports increasingly limit their scope to high-level design choices and surface statistics. These choices hinder progress in the field by degrading transparency into the training of LLMs and forcing teams to rediscover many details in the training process. We present LLM360, an initiative to fully open-source LLMs, which advocates for all training code and data, model checkpoints, and intermediate results to be made available to the community. The goal of LLM360 is to support open and collaborative AI research by making the end-to-end LLM training process transparent and reproducible by everyone. As a first step of LLM360, we release two 7B parameter LLMs pre-trained from scratch, Amber and CrystalCoder, including their training code, data, intermediate checkpoints, and analyses (at https://www.llm360.ai). We are committed to continually pushing the boundaries of LLMs through this open-source effort. More large-scale and stronger models are underway and will be released in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.17793v2">"You Are An Expert Linguistic Annotator": Limits of LLMs as Analyzers of Abstract Meaning Representation</a></div>
    <div class="paper-meta">
      📅 2023-12-11
      | 💬 EMNLP 2023 Findings (short)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show amazing proficiency and fluency in the use of language. Does this mean that they have also acquired insightful linguistic knowledge about the language, to an extent that they can serve as an "expert linguistic annotator"? In this paper, we examine the successes and limitations of the GPT-3, ChatGPT, and GPT-4 models in analysis of sentence meaning structure, focusing on the Abstract Meaning Representation (AMR; Banarescu et al. 2013) parsing formalism, which provides rich graphical representations of sentence meaning structure while abstracting away from surface forms. We compare models' analysis of this semantic structure across two settings: 1) direct production of AMR parses based on zero- and few-shot prompts, and 2) indirect partial reconstruction of AMR via metalinguistic natural language queries (e.g., "Identify the primary event of this sentence, and the predicate corresponding to that event."). Across these settings, we find that models can reliably reproduce the basic format of AMR, and can often capture core event, argument, and modifier structure -- however, model outputs are prone to frequent and major errors, and holistic analysis of parse acceptability shows that even with few-shot demonstrations, models have virtually 0% success in producing fully accurate parses. Eliciting natural language responses produces similar patterns of errors. Overall, our findings indicate that these models out-of-the-box can capture aspects of semantic structure, but there remain key limitations in their ability to support fully accurate semantic analyses or parses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06147v1">"What's important here?": Opportunities and Challenges of Using LLMs in Retrieving Information from Web Interfaces</a></div>
    <div class="paper-meta">
      📅 2023-12-11
      | 💬 Accepted to NeurIPS 2023 R0-FoMo Workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) that have been trained on a corpus that includes large amount of code exhibit a remarkable ability to understand HTML code. As web interfaces are primarily constructed using HTML, we design an in-depth study to see how LLMs can be used to retrieve and locate important elements for a user given query (i.e. task description) in a web interface. In contrast with prior works, which primarily focused on autonomous web navigation, we decompose the problem as an even atomic operation - Can LLMs identify the important information in the web page for a user given query? This decomposition enables us to scrutinize the current capabilities of LLMs and uncover the opportunities and challenges they present. Our empirical experiments show that while LLMs exhibit a reasonable level of performance in retrieving important UI elements, there is still a substantial room for improvement. We hope our investigation will inspire follow-up works in overcoming the current challenges in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06121v1">Can LLMs Configure Software Tools</a></div>
    <div class="paper-meta">
      📅 2023-12-11
    </div>
    <details class="paper-abstract">
      In software engineering, the meticulous configuration of software tools is crucial in ensuring optimal performance within intricate systems. However, the complexity inherent in selecting optimal configurations is exacerbated by the high-dimensional search spaces presented in modern applications. Conventional trial-and-error or intuition-driven methods are both inefficient and error-prone, impeding scalability and reproducibility. In this study, we embark on an exploration of leveraging Large-Language Models (LLMs) to streamline the software configuration process. We identify that the task of hyperparameter configuration for machine learning components within intelligent applications is particularly challenging due to the extensive search space and performance-critical nature. Existing methods, including Bayesian optimization, have limitations regarding initial setup, computational cost, and convergence efficiency. Our work presents a novel approach that employs LLMs, such as Chat-GPT, to identify starting conditions and narrow down the search space, improving configuration efficiency. We conducted a series of experiments to investigate the variability of LLM-generated responses, uncovering intriguing findings such as potential response caching and consistent behavior based on domain-specific keywords. Furthermore, our results from hyperparameter optimization experiments reveal the potential of LLMs in expediting initialization processes and optimizing configurations. While our initial insights are promising, they also indicate the need for further in-depth investigations and experiments in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.05693v1">Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge</a></div>
    <div class="paper-meta">
      📅 2023-12-09
      | 💬 Accepted by AAAI 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) stand out for their impressive performance in intricate language modeling tasks. However, their demanding computational and memory needs pose obstacles for broad use on edge devices. Quantization is then introduced to boost LLMs' on-device efficiency. Recent works show that 8-bit or lower weight quantization is feasible with minimal impact on end-to-end task performance, while the activation is still not quantized. On the other hand, mainstream commodity edge devices still struggle to execute these sub-8-bit quantized networks effectively. In this paper, we propose Agile-Quant, an activation-guided quantization framework for popular Large Language Models (LLMs), and implement an end-to-end accelerator on multiple edge devices for faster inference. Considering the hardware profiling and activation analysis, we first introduce a basic activation quantization strategy to balance the trade-off of task performance and real inference speed. Then we leverage the activation-aware token pruning technique to reduce the outliers and the adverse impact on attentivity. Ultimately, we utilize the SIMD-based 4-bit multiplier and our efficient TRIP matrix multiplication to implement the accelerator for LLMs on the edge. We apply our framework on different scales of LLMs including LLaMA, OPT, and BLOOM with 4-bit or 8-bit for the activation and 4-bit for the weight quantization. Experiments show that Agile-Quant achieves simultaneous quantization of model weights and activations while maintaining task performance comparable to existing weight-only quantization methods. Moreover, in the 8- and 4-bit scenario, Agile-Quant achieves an on-device speedup of up to 2.55x compared to its FP16 counterparts across multiple edge devices, marking a pioneering advancement in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.03815v2">LLM as OS, Agents as Apps: Envisioning AIOS, Agents and the AIOS-Agent Ecosystem</a></div>
    <div class="paper-meta">
      📅 2023-12-09
      | 💬 35 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This paper envisions a revolutionary AIOS-Agent ecosystem, where Large Language Model (LLM) serves as the (Artificial) Intelligent Operating System (IOS, or AIOS)--an operating system "with soul". Upon this foundation, a diverse range of LLM-based AI Agent Applications (Agents, or AAPs) are developed, enriching the AIOS-Agent ecosystem and signaling a paradigm shift from the traditional OS-APP ecosystem. We envision that LLM's impact will not be limited to the AI application level, instead, it will in turn revolutionize the design and implementation of computer system, architecture, software, and programming language, featured by several main concepts: LLM as OS (system-level), Agents as Applications (application-level), Natural Language as Programming Interface (user-level), and Tools as Devices/Libraries (hardware/middleware-level). We begin by introducing the architecture of traditional OS. Then we formalize a conceptual framework for AIOS through "LLM as OS (LLMOS)", drawing analogies between AIOS and traditional OS: LLM is likened to OS kernel, context window to memory, external storage to file system, hardware tools to peripheral devices, software tools to programming libraries, and user prompts to user commands. Subsequently, we introduce the new AIOS-Agent Ecosystem, where users can easily program Agent Applications (AAPs) using natural language, democratizing the development of software, which is different from the traditional OS-APP ecosystem. Following this, we explore the diverse scope of Agent Applications. We delve into both single-agent and multi-agent systems, as well as human-agent interaction. Lastly, drawing on the insights from traditional OS-APP ecosystem, we propose a roadmap for the evolution of the AIOS-Agent ecosystem. This roadmap is designed to guide the future research and development, suggesting systematic progresses of AIOS and its Agent applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.10075v1">Assessing LLMs for Moral Value Pluralism</a></div>
    <div class="paper-meta">
      📅 2023-12-08
      | 💬 Accepted Paper to workshop on "AI meets Moral Philosophy and Moral Psychology: An Interdisciplinary Dialogue about Computational Ethics" at NeurIPS 2023
    </div>
    <details class="paper-abstract">
      The fields of AI current lacks methods to quantitatively assess and potentially alter the moral values inherent in the output of large language models (LLMs). However, decades of social science research has developed and refined widely-accepted moral value surveys, such as the World Values Survey (WVS), eliciting value judgments from direct questions in various geographies. We have turned those questions into value statements and use NLP to compute to how well popular LLMs are aligned with moral values for various demographics and cultures. While the WVS is accepted as an explicit assessment of values, we lack methods for assessing implicit moral and cultural values in media, e.g., encountered in social media, political rhetoric, narratives, and generated by AI systems such as LLMs that are increasingly present in our daily lives. As we consume online content and utilize LLM outputs, we might ask, which moral values are being implicitly promoted or undercut, or -- in the case of LLMs -- if they are intending to represent a cultural identity, are they doing so consistently? In this paper we utilize a Recognizing Value Resonance (RVR) NLP model to identify WVS values that resonate and conflict with a given passage of output text. We apply RVR to the text generated by LLMs to characterize implicit moral values, allowing us to quantify the moral/cultural distance between LLMs and various demographics that have been surveyed using the WVS. In line with other work we find that LLMs exhibit several Western-centric value biases; they overestimate how conservative people in non-Western countries are, they are less accurate in representing gender for non-Western countries, and portray older populations as having more traditional values. Our results highlight value misalignment and age groups, and a need for social science informed technological solutions addressing value plurality in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11830v2">Goal-Oriented Prompt Attack and Safety Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) presents significant priority in text understanding and generation. However, LLMs suffer from the risk of generating harmful contents especially while being employed to applications. There are several black-box attack methods, such as Prompt Attack, which can change the behaviour of LLMs and induce LLMs to generate unexpected answers with harmful contents. Researchers are interested in Prompt Attack and Defense with LLMs, while there is no publicly available dataset with high successful attacking rate to evaluate the abilities of defending prompt attack. In this paper, we introduce a pipeline to construct high-quality prompt attack samples, along with a Chinese prompt attack dataset called CPAD. Our prompts aim to induce LLMs to generate unexpected outputs with several carefully designed prompt attack templates and widely concerned attacking contents. Different from previous datasets involving safety estimation, we construct the prompts considering three dimensions: contents, attacking methods and goals. Especially, the attacking goals indicate the behaviour expected after successfully attacking the LLMs, thus the responses can be easily evaluated and analysed. We run several popular Chinese LLMs on our dataset, and the results show that our prompts are significantly harmful to LLMs, with around 70% attack success rate to GPT-3.5. CPAD is publicly available at https://github.com/liuchengyuan123/CPAD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.04782v1">Make Them Spill the Beans! Coercive Knowledge Extraction from (Production) LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are now widely used in various applications, making it crucial to align their ethical standards with human values. However, recent jail-breaking methods demonstrate that this alignment can be undermined using carefully constructed prompts. In our study, we reveal a new threat to LLM alignment when a bad actor has access to the model's output logits, a common feature in both open-source LLMs and many commercial LLM APIs (e.g., certain GPT models). It does not rely on crafting specific prompts. Instead, it exploits the fact that even when an LLM rejects a toxic request, a harmful response often hides deep in the output logits. By forcefully selecting lower-ranked output tokens during the auto-regressive generation process at a few critical output positions, we can compel the model to reveal these hidden responses. We term this process model interrogation. This approach differs from and outperforms jail-breaking methods, achieving 92% effectiveness compared to 62%, and is 10 to 20 times faster. The harmful content uncovered through our method is more relevant, complete, and clear. Additionally, it can complement jail-breaking strategies, with which results in further boosting attack performance. Our findings indicate that interrogation can extract toxic knowledge even from models specifically designed for coding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.15766v2">Knowledge Unlearning for LLMs: Tasks, Methods, and Challenges</a></div>
    <div class="paper-meta">
      📅 2023-12-08
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have spurred a new research paradigm in natural language processing. Despite their excellent capability in knowledge-based question answering and reasoning, their potential to retain faulty or even harmful knowledge poses risks of malicious application. The challenge of mitigating this issue and transforming these models into purer assistants is crucial for their widespread applicability. Unfortunately, Retraining LLMs repeatedly to eliminate undesirable knowledge is impractical due to their immense parameters. Knowledge unlearning, derived from analogous studies on machine unlearning, presents a promising avenue to address this concern and is notably advantageous in the context of LLMs. It allows for the removal of harmful knowledge in an efficient manner, without affecting unrelated knowledge in the model. To this end, we provide a survey of knowledge unlearning in the era of LLMs. Firstly, we formally define the knowledge unlearning problem and distinguish it from related works. Subsequently, we categorize existing knowledge unlearning methods into three classes: those based on parameter optimization, parameter merging, and in-context learning, and introduce details of these unlearning methods. We further present evaluation datasets used in existing methods, and finally conclude this survey by presenting the ongoing challenges and future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14591v3">ALGO: Synthesizing Algorithmic Programs with LLM-Generated Oracle Verifiers</a></div>
    <div class="paper-meta">
      📅 2023-12-08
      | 💬 NeurIPS 2023
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at implementing code from functionality descriptions but struggle with algorithmic problems that require not only implementation but also identification of the suitable algorithm. Moreover, LLM-generated programs lack guaranteed correctness and require human verification. To address these challenges, we propose ALGO, a framework that synthesizes Algorithmic programs with LLM-Generated Oracles to guide the generation and verify their correctness. ALGO first generates a reference oracle by prompting an LLM to exhaustively enumerate all the combinations of relevant variables. This oracle is then utilized to guide an arbitrary search strategy in exploring the algorithm space and to verify the synthesized algorithms. Our study shows that the LLM-generated oracles are correct for 88% of the cases. With the oracles as verifiers, ALGO can be integrated with any existing code generation model in a model-agnostic manner to enhance its performance. Experiments show that when equipped with ALGO, we achieve an 8x better one-submission pass rate over the Codex model and a 2.6x better one-submission pass rate over CodeT, the current state-of-the-art model on CodeContests. We can also get 1.3x better pass rate over the ChatGPT Code Interpreter on unseen problems. The problem set we used for testing, the prompts we used, the verifier and solution programs, and the test cases generated by ALGO are available at https://github.com/zkx06111/ALGO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14938v2">Do LLMs Understand Social Knowledge? Evaluating the Sociability of Large Language Models with SocKET Benchmark</a></div>
    <div class="paper-meta">
      📅 2023-12-07
      | 💬 Camera-ready version for EMNLP'23. First two authors contributed equally
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to perform well at a variety of syntactic, discourse, and reasoning tasks. While LLMs are increasingly deployed in many forms including conversational agents that interact with humans, we lack a grounded benchmark to measure how well LLMs understand \textit{social} language. Here, we introduce a new theory-driven benchmark, SocKET, that contains 58 NLP tasks testing social knowledge which we group into five categories: humor & sarcasm, offensiveness, sentiment & emotion, and trustworthiness. In tests on the benchmark, we demonstrate that current models attain only moderate performance but reveal significant potential for task transfer among different types and categories of tasks, which were predicted from theory. Through zero-shot evaluations, we show that pretrained models already possess some innate but limited capabilities of social language understanding and training on one category of tasks can improve zero-shot testing on others. Our benchmark provides a systematic way to analyze model performance on an important dimension of language and points to clear room for improvement to build more socially-aware LLMs. The associated resources are released at https://github.com/minjechoi/SOCKET.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06674v1">Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations</a></div>
    <div class="paper-meta">
      📅 2023-12-07
    </div>
    <details class="paper-abstract">
      We introduce Llama Guard, an LLM-based input-output safeguard model geared towards Human-AI conversation use cases. Our model incorporates a safety risk taxonomy, a valuable tool for categorizing a specific set of safety risks found in LLM prompts (i.e., prompt classification). This taxonomy is also instrumental in classifying the responses generated by LLMs to these prompts, a process we refer to as response classification. For the purpose of both prompt and response classification, we have meticulously gathered a dataset of high quality. Llama Guard, a Llama2-7b model that is instruction-tuned on our collected dataset, albeit low in volume, demonstrates strong performance on existing benchmarks such as the OpenAI Moderation Evaluation dataset and ToxicChat, where its performance matches or exceeds that of currently available content moderation tools. Llama Guard functions as a language model, carrying out multi-class classification and generating binary decision scores. Furthermore, the instruction fine-tuning of Llama Guard allows for the customization of tasks and the adaptation of output formats. This feature enhances the model's capabilities, such as enabling the adjustment of taxonomy categories to align with specific use cases, and facilitating zero-shot or few-shot prompting with diverse taxonomies at the input. We are making Llama Guard model weights available and we encourage researchers to further develop and adapt them to meet the evolving needs of the community for AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.04613v1">Testing LLM performance on the Physics GRE: some observations</a></div>
    <div class="paper-meta">
      📅 2023-12-07
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      With the recent developments in large language models (LLMs) and their widespread availability through open source models and/or low-cost APIs, several exciting products and applications are emerging, many of which are in the field of STEM educational technology for K-12 and university students. There is a need to evaluate these powerful language models on several benchmarks, in order to understand their risks and limitations. In this short paper, we summarize and analyze the performance of Bard, a popular LLM-based conversational service made available by Google, on the standardized Physics GRE examination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.00502v2">Efficient LLM Inference on CPUs</a></div>
    <div class="paper-meta">
      📅 2023-12-07
      | 💬 NeurIPS'2023 on Efficient Natural Language and Speech Processing
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance and tremendous potential across a wide range of tasks. However, deploying these models has been challenging due to the astronomical amount of model parameters, which requires a demand for large memory capacity and high memory bandwidth. In this paper, we propose an effective approach that can make the deployment of LLMs more efficiently. We support an automatic INT4 weight-only quantization flow and design a special LLM runtime with highly-optimized kernels to accelerate the LLM inference on CPUs. We demonstrate the general applicability of our approach on popular LLMs including Llama2, Llama, GPT-NeoX, and showcase the extreme inference efficiency on CPUs. The code is publicly available at: https://github.com/intel/intel-extension-for-transformers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13255v2">Steve-Eye: Equipping LLM-based Embodied Agents with Visual Perception in Open Worlds</a></div>
    <div class="paper-meta">
      📅 2023-12-07
      | 💬 19 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Recent studies have presented compelling evidence that large language models (LLMs) can equip embodied agents with the self-driven capability to interact with the world, which marks an initial step toward versatile robotics. However, these efforts tend to overlook the visual richness of open worlds, rendering the entire interactive process akin to "a blindfolded text-based game." Consequently, LLM-based agents frequently encounter challenges in intuitively comprehending their surroundings and producing responses that are easy to understand. In this paper, we propose Steve-Eye, an end-to-end trained large multimodal model designed to address this limitation. Steve-Eye integrates the LLM with a visual encoder which enables it to process visual-text inputs and generate multimodal feedback. In addition, we use a semi-automatic strategy to collect an extensive dataset comprising 850K open-world instruction pairs, empowering our model to encompass three essential functions for an agent: multimodal perception, foundational knowledge base, and skill prediction and planning. Lastly, we develop three open-world evaluation benchmarks, then carry out extensive experiments from a wide range of perspectives to validate our model's capability to strategically act and plan. Codes and datasets will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.03788v1">SmoothQuant+: Accurate and Efficient 4-bit Post-Training WeightQuantization for LLM</a></div>
    <div class="paper-meta">
      📅 2023-12-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in various tasks. However their huge model size and the consequent demand for computational and memory resources also pose challenges to model deployment. Currently, 4-bit post-training quantization (PTQ) has achieved some success in LLMs, reducing the memory footprint by approximately 75% compared to FP16 models, albeit with some accuracy loss. In this paper, we propose SmoothQuant+, an accurate and efficient 4-bit weight-only PTQ that requires no additional training, which enables lossless in accuracy for LLMs for the first time. Based on the fact that the loss of weight quantization is amplified by the activation outliers, SmoothQuant+ smoothes the activation outliers by channel before quantization, while adjusting the corresponding weights for mathematical equivalence, and then performs group-wise 4-bit weight quantization for linear layers. We have integrated SmoothQuant+ into the vLLM framework, an advanced high-throughput inference engine specially developed for LLMs, and equipped it with an efficient W4A16 CUDA kernels, so that vLLM can seamlessly support SmoothQuant+ 4-bit weight quantization. Our results show that, with SmoothQuant+, the Code Llama-34B model can be quantized and deployed on a A100 40GB GPU, achieving lossless accuracy and a throughput increase of 1.9 to 4.0 times compared to the FP16 model deployed on two A100 40GB GPUs. Moreover, the latency per token is only 68% of the FP16 model deployed on two A100 40GB GPUs. This is the state-of-the-art 4-bit weight quantization for LLMs as we know.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.03088v1">LLMs for Multi-Modal Knowledge Extraction and Analysis in Intelligence/Safety-Critical Applications</a></div>
    <div class="paper-meta">
      📅 2023-12-05
      | 💬 initial draft
    </div>
    <details class="paper-abstract">
      Large Language Models have seen rapid progress in capability in recent years; this progress has been accelerating and their capabilities, measured by various benchmarks, are beginning to approach those of humans. There is a strong demand to use such models in a wide variety of applications but, due to unresolved vulnerabilities and limitations, great care needs to be used before applying them to intelligence and safety-critical applications. This paper reviews recent literature related to LLM assessment and vulnerabilities to synthesize the current research landscape and to help understand what advances are most critical to enable use of of these technologies in intelligence and safety-critical applications. The vulnerabilities are broken down into ten high-level categories and overlaid onto a high-level life cycle of an LLM. Some general categories of mitigations are reviewed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02913v1">Let the LLMs Talk: Simulating Human-to-Human Conversational QA via Zero-Shot LLM-to-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2023-12-05
      | 💬 Accepted at WSDM 2024
    </div>
    <details class="paper-abstract">
      Conversational question-answering (CQA) systems aim to create interactive search systems that effectively retrieve information by interacting with users. To replicate human-to-human conversations, existing work uses human annotators to play the roles of the questioner (student) and the answerer (teacher). Despite its effectiveness, challenges exist as human annotation is time-consuming, inconsistent, and not scalable. To address this issue and investigate the applicability of large language models (LLMs) in CQA simulation, we propose a simulation framework that employs zero-shot learner LLMs for simulating teacher-student interactions. Our framework involves two LLMs interacting on a specific topic, with the first LLM acting as a student, generating questions to explore a given search topic. The second LLM plays the role of a teacher by answering questions and is equipped with additional information, including a text on the given topic. We implement both the student and teacher by zero-shot prompting the GPT-4 model. To assess the effectiveness of LLMs in simulating CQA interactions and understand the disparities between LLM- and human-generated conversations, we evaluate the simulated data from various perspectives. We begin by evaluating the teacher's performance through both automatic and human assessment. Next, we evaluate the performance of the student, analyzing and comparing the disparities between questions generated by the LLM and those generated by humans. Furthermore, we conduct extensive analyses to thoroughly examine the LLM performance by benchmarking state-of-the-art reading comprehension models on both datasets. Our results reveal that the teacher LLM generates lengthier answers that tend to be more accurate and complete. The student LLM generates more diverse questions, covering more aspects of a given topic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02798v1">Weakly Supervised Detection of Hallucinations in LLM Activations</a></div>
    <div class="paper-meta">
      📅 2023-12-05
    </div>
    <details class="paper-abstract">
      We propose an auditing method to identify whether a large language model (LLM) encodes patterns such as hallucinations in its internal states, which may propagate to downstream tasks. We introduce a weakly supervised auditing technique using a subset scanning approach to detect anomalous patterns in LLM activations from pre-trained models. Importantly, our method does not need knowledge of the type of patterns a-priori. Instead, it relies on a reference dataset devoid of anomalies during testing. Further, our approach enables the identification of pivotal nodes responsible for encoding these patterns, which may offer crucial insights for fine-tuning specific sub-networks for bias mitigation. We introduce two new scanning methods to handle LLM activations for anomalous sentences that may deviate from the expected distribution in either direction. Our results confirm prior findings of BERT's limited internal capacity for encoding hallucinations, while OPT appears capable of encoding hallucination information internally. Importantly, our scanning approach, without prior exposure to false statements, performs comparably to a fully supervised out-of-distribution classifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02441v1">MedDM:LLM-executable clinical guidance tree for clinical decision-making</a></div>
    <div class="paper-meta">
      📅 2023-12-05
    </div>
    <details class="paper-abstract">
      It is becoming increasingly emphasis on the importance of LLM participating in clinical diagnosis decision-making. However, the low specialization refers to that current medical LLMs can not provide specific medical advice, which are more like a medical Q\&A. And there is no suitable clinical guidance tree data set that can be used directly with LLM. To address this issue, we first propose LLM-executavle clinical guidance tree(CGT), which can be directly used by large language models, and construct medical diagnostic decision-making dataset (MedDM), from flowcharts in clinical practice guidelines. We propose an approach to screen flowcharts from medical literature, followed by their identification and conversion into standardized diagnostic decision trees. Constructed a knowledge base with 1202 decision trees, which came from 5000 medical literature and covered 12 hospital departments, including internal medicine, surgery, psychiatry, and over 500 diseases.Moreover, we propose a method for reasoning on LLM-executable CGT and a Patient-LLM multi-turn dialogue framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02382v1">New Evaluation Metrics Capture Quality Degradation due to LLM Watermarking</a></div>
    <div class="paper-meta">
      📅 2023-12-04
    </div>
    <details class="paper-abstract">
      With the increasing use of large-language models (LLMs) like ChatGPT, watermarking has emerged as a promising approach for tracing machine-generated content. However, research on LLM watermarking often relies on simple perplexity or diversity-based measures to assess the quality of watermarked text, which can mask important limitations in watermarking. Here we introduce two new easy-to-use methods for evaluating watermarking algorithms for LLMs: 1) evaluation by LLM-judger with specific guidelines; and 2) binary classification on text embeddings to distinguish between watermarked and unwatermarked text. We apply these methods to characterize the effectiveness of current watermarking techniques. Our experiments, conducted across various datasets, reveal that current watermarking methods are detectable by even simple classifiers, challenging the notion of watermarking subtlety. We also found, through the LLM judger, that watermarking impacts text quality, especially in degrading the coherence and depth of the response. Our findings underscore the trade-off between watermark robustness and text quality and highlight the importance of having more informative metrics to assess watermarking quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02310v1">VaQuitA: Enhancing Alignment in LLM-Assisted Video Understanding</a></div>
    <div class="paper-meta">
      📅 2023-12-04
    </div>
    <details class="paper-abstract">
      Recent advancements in language-model-based video understanding have been progressing at a remarkable pace, spurred by the introduction of Large Language Models (LLMs). However, the focus of prior research has been predominantly on devising a projection layer that maps video features to tokens, an approach that is both rudimentary and inefficient. In our study, we introduce a cutting-edge framework, VaQuitA, designed to refine the synergy between video and textual information. At the data level, instead of sampling frames uniformly, we implement a sampling method guided by CLIP-score rankings, which enables a more aligned selection of frames with the given question. At the feature level, we integrate a trainable Video Perceiver alongside a Visual-Query Transformer (abbreviated as VQ-Former), which bolsters the interplay between the input question and the video features. We also discover that incorporating a simple prompt, "Please be critical", into the LLM input can substantially enhance its video comprehension capabilities. Our experimental results indicate that VaQuitA consistently sets a new benchmark for zero-shot video question-answering tasks and is adept at producing high-quality, multi-turn video dialogues with users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02296v1">LLMs Accelerate Annotation for Medical Information Extraction</a></div>
    <div class="paper-meta">
      📅 2023-12-04
      | 💬 Published in proceedings of the Machine Learning for Health (ML4H) Symposium 2023
    </div>
    <details class="paper-abstract">
      The unstructured nature of clinical notes within electronic health records often conceals vital patient-related information, making it challenging to access or interpret. To uncover this hidden information, specialized Natural Language Processing (NLP) models are required. However, training these models necessitates large amounts of labeled data, a process that is both time-consuming and costly when relying solely on human experts for annotation. In this paper, we propose an approach that combines Large Language Models (LLMs) with human expertise to create an efficient method for generating ground truth labels for medical text annotation. By utilizing LLMs in conjunction with human annotators, we significantly reduce the human annotation burden, enabling the rapid creation of labeled datasets. We rigorously evaluate our method on a medical information extraction task, demonstrating that our approach not only substantially cuts down on human intervention but also maintains high accuracy. The results highlight the potential of using LLMs to improve the utilization of unstructured clinical data, allowing for the swift deployment of tailored NLP solutions in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.07944v2">AutoRepo: A general framework for multi-modal LLM-based automated construction reporting</a></div>
    <div class="paper-meta">
      📅 2023-12-04
      | 💬 We believe that keeping this version of the paper publicly available may lead to confusion or misinterpretation regarding our current research direction and findings
    </div>
    <details class="paper-abstract">
      Ensuring the safety, quality, and timely completion of construction projects is paramount, with construction inspections serving as a vital instrument towards these goals. Nevertheless, the predominantly manual approach of present-day inspections frequently results in inefficiencies and inadequate information management. Such methods often fall short of providing holistic, exhaustive assessments, consequently engendering regulatory oversights and potential safety hazards. To address this issue, this paper presents a novel framework named AutoRepo for automated generation of construction inspection reports. The unmanned vehicles efficiently perform construction inspections and collect scene information, while the multimodal large language models (LLMs) are leveraged to automatically generate the inspection reports. The framework was applied and tested on a real-world construction site, demonstrating its potential to expedite the inspection process, significantly reduce resource allocation, and produce high-quality, regulatory standard-compliant inspection reports. This research thus underscores the immense potential of multimodal large language models in revolutionizing construction inspection practices, signaling a significant leap forward towards a more efficient and safer construction management paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02065v1">Know Your Audience: Do LLMs Adapt to Different Age and Education Levels?</a></div>
    <div class="paper-meta">
      📅 2023-12-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer a range of new possibilities, including adapting the text to different audiences and their reading needs. But how well do they adapt? We evaluate the readability of answers generated by four state-of-the-art LLMs (commercial and open-source) to science questions when prompted to target different age groups and education levels. To assess the adaptability of LLMs to diverse audiences, we compare the readability scores of the generated responses against the recommended comprehension level of each age and education group. We find large variations in the readability of the answers by different LLMs. Our results suggest LLM answers need to be better adapted to the intended audience demographics to be more comprehensible. They underline the importance of enhancing the adaptability of LLMs in education settings to cater to diverse age and education levels. Overall, current LLMs have set readability ranges and do not adapt well to different audiences, even when prompted. That limits their potential for educational purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.12509v2">Joint Prompt Optimization of Stacked LLMs using Variational Inference</a></div>
    <div class="paper-meta">
      📅 2023-12-04
      | 💬 NeurIPS 2023
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be seen as atomic units of computation mapping sequences to a distribution over sequences. Thus, they can be seen as stochastic language layers in a language network, where the learnable parameters are the natural language prompts at each layer. By stacking two such layers and feeding the output of one layer to the next, we obtain a Deep Language Network (DLN). We first show how to effectively perform prompt optimization for a 1-Layer language network (DLN-1). Then, we present an extension that applies to 2-layer DLNs (DLN-2), where two prompts must be learned. The key idea is to consider the output of the first layer as a latent variable, which requires inference, and prompts to be learned as the parameters of the generative distribution. We first test the effectiveness of DLN-1 in multiple reasoning and natural language understanding tasks. Then, we show that DLN-2 can reach higher performance than a single layer, showing promise that we might reach comparable performance to GPT-4, even when each LLM in the network is smaller and less powerful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15896v2">BianQue: Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT</a></div>
    <div class="paper-meta">
      📅 2023-12-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have performed well in providing general and extensive health suggestions in single-turn conversations, exemplified by systems such as ChatGPT, ChatGLM, ChatDoctor, DoctorGLM, and etc. However, the limited information provided by users during single turn results in inadequate personalization and targeting of the generated suggestions, which requires users to independently select the useful part. It is mainly caused by the missing ability to engage in multi-turn questioning. In real-world medical consultations, doctors usually employ a series of iterative inquiries to comprehend the patient's condition thoroughly, enabling them to provide effective and personalized suggestions subsequently, which can be defined as chain of questioning (CoQ) for LLMs. To improve the CoQ of LLMs, we propose BianQue, a ChatGLM-based LLM finetuned with the self-constructed health conversation dataset BianQueCorpus that is consist of multiple turns of questioning and health suggestions polished by ChatGPT. Experimental results demonstrate that the proposed BianQue can simultaneously balance the capabilities of both questioning and health suggestions, which will help promote the research and application of LLMs in the field of proactive health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06677v1">Intelligent Virtual Assistants with LLM-based Process Automation</a></div>
    <div class="paper-meta">
      📅 2023-12-04
    </div>
    <details class="paper-abstract">
      While intelligent virtual assistants like Siri, Alexa, and Google Assistant have become ubiquitous in modern life, they still face limitations in their ability to follow multi-step instructions and accomplish complex goals articulated in natural language. However, recent breakthroughs in large language models (LLMs) show promise for overcoming existing barriers by enhancing natural language processing and reasoning capabilities. Though promising, applying LLMs to create more advanced virtual assistants still faces challenges like ensuring robust performance and handling variability in real-world user commands. This paper proposes a novel LLM-based virtual assistant that can automatically perform multi-step operations within mobile apps based on high-level user requests. The system represents an advance in assistants by providing an end-to-end solution for parsing instructions, reasoning about goals, and executing actions. LLM-based Process Automation (LLMPA) has modules for decomposing instructions, generating descriptions, detecting interface elements, predicting next actions, and error checking. Experiments demonstrate the system completing complex mobile operation tasks in Alipay based on natural language instructions. This showcases how large language models can enable automated assistants to accomplish real-world tasks. The main contributions are the novel LLMPA architecture optimized for app process automation, the methodology for applying LLMs to mobile apps, and demonstrations of multi-step task completion in a real-world environment. Notably, this work represents the first real-world deployment and extensive evaluation of a large language model-based virtual assistant in a widely used mobile application with an enormous user base numbering in the hundreds of millions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.01552v1">The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2023-12-04
      | 💬 26 pages, 8 figures. Project website: https://allenai.github.io/re-align/
    </div>
    <details class="paper-abstract">
      The alignment tuning process of large language models (LLMs) typically involves instruction learning through supervised fine-tuning (SFT) and preference tuning via reinforcement learning from human feedback (RLHF). A recent study, LIMA (Zhou et al. 2023), shows that using merely 1K examples for SFT can achieve significant alignment performance as well, suggesting that the effect of alignment tuning might be "superficial." This raises questions about how exactly the alignment tuning transforms a base LLM. We analyze the effect of alignment tuning by examining the token distribution shift between base LLMs and their aligned counterpart. Our findings reveal that base LLMs and their alignment-tuned versions perform nearly identically in decoding on the majority of token positions. Most distribution shifts occur with stylistic tokens. These direct evidence strongly supports the Superficial Alignment Hypothesis suggested by LIMA. Based on these findings, we rethink the alignment of LLMs by posing the research question: how effectively can we align base LLMs without SFT or RLHF? To address this, we introduce a simple, tuning-free alignment method, URIAL. URIAL achieves effective alignment purely through in-context learning (ICL) with base LLMs, requiring as few as three constant stylistic examples and a system prompt. We conduct a fine-grained and interpretable evaluation on a diverse set of examples, named JUST-EVAL-INSTRUCT. Results demonstrate that base LLMs with URIAL can match or even surpass the performance of LLMs aligned with SFT or SFT+RLHF. We show that the gap between tuning-free and tuning-based alignment methods can be significantly reduced through strategic prompting and ICL. Our findings on the superficial nature of alignment tuning and results with URIAL suggest that deeper analysis and theoretical understanding of alignment is crucial to future LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2210.14986v2">The Goldilocks of Pragmatic Understanding: Fine-Tuning Strategy Matters for Implicature Resolution by LLMs</a></div>
    <div class="paper-meta">
      📅 2023-12-03
      | 💬 Accepted as Spotlight at NeurIPS 2023
    </div>
    <details class="paper-abstract">
      Despite widespread use of LLMs as conversational agents, evaluations of performance fail to capture a crucial aspect of communication: interpreting language in context -- incorporating its pragmatics. Humans interpret language using beliefs and prior knowledge about the world. For example, we intuitively understand the response "I wore gloves" to the question "Did you leave fingerprints?" as meaning "No". To investigate whether LLMs have the ability to make this type of inference, known as an implicature, we design a simple task and evaluate four categories of widely used state-of-the-art models. We find that, despite only evaluating on utterances that require a binary inference (yes or no), models in three of these categories perform close to random. However, LLMs instruction-tuned at the example-level perform significantly better. These results suggest that certain fine-tuning strategies are far better at inducing pragmatic understanding in models. We present our findings as the starting point for further research into evaluating how LLMs interpret language in context and to drive the development of more pragmatic and useful models of human discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.13743v2">FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design</a></div>
    <div class="paper-meta">
      📅 2023-12-03
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have exhibited notable efficacy in question-answering (QA) tasks across diverse domains. Their prowess in integrating extensive web knowledge has fueled interest in developing LLM-based autonomous agents. While LLMs are efficient in decoding human instructions and deriving solutions by holistically processing historical inputs, transitioning to purpose-driven agents requires a supplementary rational architecture to process multi-source information, establish reasoning chains, and prioritize critical tasks. Addressing this, we introduce \textsc{FinMem}, a novel LLM-based agent framework devised for financial decision-making. It encompasses three core modules: Profiling, to customize the agent's characteristics; Memory, with layered message processing, to aid the agent in assimilating hierarchical financial data; and Decision-making, to convert insights gained from memories into investment decisions. Notably, \textsc{FinMem}'s memory module aligns closely with the cognitive structure of human traders, offering robust interpretability and real-time tuning. Its adjustable cognitive span allows for the retention of critical information beyond human perceptual limits, thereby enhancing trading outcomes. This framework enables the agent to self-evolve its professional knowledge, react agilely to new investment cues, and continuously refine trading decisions in the volatile financial environment. We first compare \textsc{FinMem} with various algorithmic agents on a scalable real-world financial dataset, underscoring its leading trading performance in stocks. We then fine-tuned the agent's perceptual span and character setting to achieve a significantly enhanced trading performance. Collectively, \textsc{FinMem} presents a cutting-edge LLM agent framework for automated trading, boosting cumulative investment returns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02213v1">JarviX: A LLM No code Platform for Tabular Data Analysis and Optimization</a></div>
    <div class="paper-meta">
      📅 2023-12-03
    </div>
    <details class="paper-abstract">
      In this study, we introduce JarviX, a sophisticated data analytics framework. JarviX is designed to employ Large Language Models (LLMs) to facilitate an automated guide and execute high-precision data analyzes on tabular datasets. This framework emphasizes the significance of varying column types, capitalizing on state-of-the-art LLMs to generate concise data insight summaries, propose relevant analysis inquiries, visualize data effectively, and provide comprehensive explanations for results drawn from an extensive data analysis pipeline. Moreover, JarviX incorporates an automated machine learning (AutoML) pipeline for predictive modeling. This integration forms a comprehensive and automated optimization cycle, which proves particularly advantageous for optimizing machine configuration. The efficacy and adaptability of JarviX are substantiated through a series of practical use case studies.
    </details>
</div>
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
