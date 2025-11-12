# llm - 2024_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05527v4">GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM</a></div>
    <div class="paper-meta">
      📅 2024-09-30
    </div>
    <details class="paper-abstract">
      Key-value (KV) caching has become the de-facto to accelerate generation speed for large language models (LLMs) inference. However, the growing cache demand with increasing sequence length has transformed LLM inference to be a memory bound problem, significantly constraining the system throughput. Existing methods rely on dropping unimportant tokens or quantizing all entries uniformly. Such methods, however, often incur high approximation errors to represent the compressed matrices. The autoregressive decoding process further compounds the error of each step, resulting in critical deviation in model generation and deterioration of performance. To tackle this challenge, we propose GEAR, an efficient KV cache compression framework that achieves near-lossless high-ratio compression. GEAR first applies quantization to majority of entries of similar magnitudes to ultra-low precision. It then employs a low rank matrix to approximate the quantization error, and a sparse matrix to remedy individual errors from outlier entries. By adeptly integrating three techniques, GEAR is able to fully exploit their synergistic potentials. Our experiments demonstrate that compared to alternatives, GEAR achieves near-lossless 4-bit KV cache compression with up to 2.38x throughput improvement, while reducing peak-memory size up to 2.29x. Our code is publicly available at https://github.com/HaoKang-Timmy/GEAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11807v4">How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 11 pages of main text; 19 pages of appendices. Included models: GPT-3.5-{0613, 1106, 0125}, GPT-4-0125, Gemini-{1.0, 1.5)-Pro, LLaMA-3.1-{7, 70, 405}B, Mixtral-8x{7, 22}B, Qwen-2-72B
    </div>
    <details class="paper-abstract">
      Decision-making is a complex process requiring diverse abilities, making it an excellent framework for evaluating Large Language Models (LLMs). Researchers have examined LLMs' decision-making through the lens of Game Theory. However, existing evaluation mainly focus on two-player scenarios where an LLM competes against another. Additionally, previous benchmarks suffer from test set leakage due to their static design. We introduce GAMA($\gamma$)-Bench, a new framework for evaluating LLMs' Gaming Ability in Multi-Agent environments. It includes eight classical game theory scenarios and a dynamic scoring scheme specially designed to quantitatively assess LLMs' performance. $\gamma$-Bench allows flexible game settings and adapts the scoring system to different game parameters, enabling comprehensive evaluation of robustness, generalizability, and strategies for improvement. Our results indicate that GPT-3.5 demonstrates strong robustness but limited generalizability, which can be enhanced using methods like Chain-of-Thought. We also evaluate twelve LLMs from six model families, including GPT-3.5, GPT-4, Gemini, LLaMA-3.1, Mixtral, and Qwen-2. Gemini-1.5-Pro outperforms others, scoring of $68.1$ out of $100$, followed by LLaMA-3.1-70B ($64.5$) and Mixtral-8x22B ($61.4$). All code and experimental results are publicly available via https://github.com/CUHK-ARISE/GAMABench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00194v1">"Real Learner Data Matters" Exploring the Design of LLM-Powered Question Generation for Deaf and Hard of Hearing Learners</a></div>
    <div class="paper-meta">
      📅 2024-09-30
    </div>
    <details class="paper-abstract">
      Deaf and Hard of Hearing (DHH) learners face unique challenges in learning environments, often due to a lack of tailored educational materials that address their specific needs. This study explores the potential of Large Language Models (LLMs) to generate personalized quiz questions to enhance DHH students' video-based learning experiences. We developed a prototype leveraging LLMs to generate questions with emphasis on two unique strategies: Visual Questions, which identify video segments where visual information might be misrepresented, and Emotion Questions, which highlight moments where previous DHH learners experienced learning difficulty manifested in emotional responses. Through user studies with DHH undergraduates, we evaluated the effectiveness of these LLM-generated questions in supporting the learning experience. Our findings indicate that while LLMs offer significant potential for personalized learning, challenges remain in the interaction accessibility for the diverse DHH community. The study highlights the importance of considering language diversity and culture in LLM-based educational technology design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00163v1">Adapting LLMs for the Medical Domain in Portuguese: A Study on Fine-Tuning and Model Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      This study evaluates the performance of large language models (LLMs) as medical agents in Portuguese, aiming to develop a reliable and relevant virtual assistant for healthcare professionals. The HealthCareMagic-100k-en and MedQuAD datasets, translated from English using GPT-3.5, were used to fine-tune the ChatBode-7B model using the PEFT-QLoRA method. The InternLM2 model, with initial training on medical data, presented the best overall performance, with high precision and adequacy in metrics such as accuracy, completeness and safety. However, DrBode models, derived from ChatBode, exhibited a phenomenon of catastrophic forgetting of acquired medical knowledge. Despite this, these models performed frequently or even better in aspects such as grammaticality and coherence. A significant challenge was low inter-rater agreement, highlighting the need for more robust assessment protocols. This work paves the way for future research, such as evaluating multilingual models specific to the medical field, improving the quality of training data, and developing more consistent evaluation methodologies for the medical field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00153v1">Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 28 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Probing learned concepts in large language models (LLMs) is crucial for understanding how semantic knowledge is encoded internally. Training linear classifiers on probing tasks is a principle approach to denote the vector of a certain concept in the representation space. However, the single vector identified for a concept varies with both data and training, making it less robust and weakening its effectiveness in real-world applications. To address this challenge, we propose an approach to approximate the subspace representing a specific concept. Built on linear probing classifiers, we extend the concept vectors into Gaussian Concept Subspace (GCS). We demonstrate GCS's effectiveness through measuring its faithfulness and plausibility across multiple LLMs with different sizes and architectures. Additionally, we use representation intervention tasks to showcase its efficacy in real-world applications such as emotion steering. Experimental results indicate that GCS concept vectors have the potential to balance steering performance and maintaining the fluency in natural language generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20565v1">Ranking Over Scoring: Towards Reliable and Robust Automated Evaluation of LLM-Generated Medical Explanatory Arguments</a></div>
    <div class="paper-meta">
      📅 2024-09-30
    </div>
    <details class="paper-abstract">
      Evaluating LLM-generated text has become a key challenge, especially in domain-specific contexts like the medical field. This work introduces a novel evaluation methodology for LLM-generated medical explanatory arguments, relying on Proxy Tasks and rankings to closely align results with human evaluation criteria, overcoming the biases typically seen in LLMs used as judges. We demonstrate that the proposed evaluators are robust against adversarial attacks, including the assessment of non-argumentative text. Additionally, the human-crafted arguments needed to train the evaluators are minimized to just one example per Proxy Task. By examining multiple LLM-generated arguments, we establish a methodology for determining whether a Proxy Task is suitable for evaluating LLM-generated medical explanatory arguments, requiring only five examples and two human experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20557v1">Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 Accepted by ECCV 2024 (Oral)
    </div>
    <details class="paper-abstract">
      Goal-oriented planning, or anticipating a series of actions that transition an agent from its current state to a predefined objective, is crucial for developing intelligent assistants aiding users in daily procedural tasks. The problem presents significant challenges due to the need for comprehensive knowledge of temporal and hierarchical task structures, as well as strong capabilities in reasoning and planning. To achieve this, prior work typically relies on extensive training on the target dataset, which often results in significant dataset bias and a lack of generalization to unseen tasks. In this work, we introduce VidAssist, an integrated framework designed for zero/few-shot goal-oriented planning in instructional videos. VidAssist leverages large language models (LLMs) as both the knowledge base and the assessment tool for generating and evaluating action plans, thus overcoming the challenges of acquiring procedural knowledge from small-scale, low-diversity datasets. Moreover, VidAssist employs a breadth-first search algorithm for optimal plan generation, in which a composite of value functions designed for goal-oriented planning is utilized to assess the predicted actions at each step. Extensive experiments demonstrate that VidAssist offers a unified framework for different goal-oriented planning setups, e.g., visual planning for assistance (VPA) and procedural planning (PP), and achieves remarkable performance in zero-shot and few-shot setups. Specifically, our few-shot model outperforms the prior fully supervised state-of-the-art method by +7.7% in VPA and +4.81% PP task on the COIN dataset while predicting 4 future actions. Code, and models are publicly available at https://sites.google.com/view/vidassist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12807v1">A Hierarchical conv-LSTM and LLM Integrated Model for Holistic Stock Forecasting</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 8 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The financial domain presents a complex environment for stock market prediction, characterized by volatile patterns and the influence of multifaceted data sources. Traditional models have leveraged either Convolutional Neural Networks (CNN) for spatial feature extraction or Long Short-Term Memory (LSTM) networks for capturing temporal dependencies, with limited integration of external textual data. This paper proposes a novel Two-Level Conv-LSTM Neural Network integrated with a Large Language Model (LLM) for comprehensive stock advising. The model harnesses the strengths of Conv-LSTM for analyzing time-series data and LLM for processing and understanding textual information from financial news, social media, and reports. In the first level, convolutional layers are employed to identify local patterns in historical stock prices and technical indicators, followed by LSTM layers to capture the temporal dynamics. The second level integrates the output with an LLM that analyzes sentiment and contextual information from textual data, providing a holistic view of market conditions. The combined approach aims to improve prediction accuracy and provide contextually rich stock advising.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01270v2">The African Woman is Rhythmic and Soulful: An Investigation of Implicit Biases in LLM Open-ended Text Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-30
    </div>
    <details class="paper-abstract">
      This paper investigates the subtle and often concealed biases present in Large Language Models (LLMs), focusing on implicit biases that may remain despite passing explicit bias tests. Implicit biases are significant because they influence the decisions made by these systems, potentially perpetuating stereotypes and discrimination, even when LLMs appear to function fairly. Traditionally, explicit bias tests or embedding-based methods are employed to detect bias, but these approaches can overlook more nuanced, implicit forms of bias. To address this, we introduce two novel psychological-inspired methodologies: the LLM Implicit Association Test (IAT) Bias and the LLM Decision Bias, designed to reveal and measure implicit biases through prompt-based and decision-making tasks. Additionally, open-ended generation tasks with thematic analysis of word generations and storytelling provide qualitative insights into the model's behavior. Our findings demonstrate that the LLM IAT Bias correlates with traditional methods and more effectively predicts downstream behaviors, as measured by the LLM Decision Bias, offering a more comprehensive framework for detecting subtle biases in AI systems. This research advances the field of AI ethics by proposing new methods to continually assess and mitigate biases in LLMs, highlighting the importance of qualitative and decision-focused evaluations to address challenges that previous approaches have not fully captured.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14744v2">LINKAGE: Listwise Ranking among Varied-Quality References for Non-Factoid QA Evaluation via LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 Published as a conference paper at EMNLP Findings 2024
    </div>
    <details class="paper-abstract">
      Non-Factoid (NF) Question Answering (QA) is challenging to evaluate due to diverse potential answers and no objective criterion. The commonly used automatic evaluation metrics like ROUGE or BERTScore cannot accurately measure semantic similarities or answers from different perspectives. Recently, Large Language Models (LLMs) have been resorted to for NFQA evaluation due to their compelling performance on various NLP tasks. Common approaches include pointwise scoring of each candidate answer and pairwise comparisons between answers. Inspired by the evolution from pointwise to pairwise to listwise in learning-to-rank methods, we propose a novel listwise NFQA evaluation approach, that utilizes LLMs to rank candidate answers in a list of reference answers sorted by descending quality. Moreover, for NF questions that do not have multi-grade or any golden answers, we leverage LLMs to generate the reference answer list of various quality to facilitate the listwise evaluation. Extensive experimental results on three NFQA datasets, i.e., ANTIQUE, the TREC-DL-NF, and WebGLM show that our method has significantly higher correlations with human annotations compared to automatic scores and common pointwise and pairwise approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09794v2">AutoML-guided Fusion of Entity and LLM-based Representations for Document Classification</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 Accepted at the 2024 Discovery Science Conference, oral presentation track
    </div>
    <details class="paper-abstract">
      Large semantic knowledge bases are grounded in factual knowledge. However, recent approaches to dense text representations (i.e. embeddings) do not efficiently exploit these resources. Dense and robust representations of documents are essential for effectively solving downstream classification and retrieval tasks. This work demonstrates that injecting embedded information from knowledge bases can augment the performance of contemporary Large Language Model (LLM)-based representations for the task of text classification. Further, by considering automated machine learning (AutoML) with the fused representation space, we demonstrate it is possible to improve classification accuracy even if we use low-dimensional projections of the original representation space obtained via efficient matrix factorization. This result shows that significantly faster classifiers can be achieved with minimal or no loss in predictive performance, as demonstrated using five strong LLM baselines on six diverse real-life datasets. The code is freely available at \url{https://github.com/bkolosk1/bablfusion.git}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20296v1">PersonalLLM: Tailoring LLMs to Individual Preferences</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 28 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As LLMs become capable of complex tasks, there is growing potential for personalized interactions tailored to the subtle and idiosyncratic preferences of the user. We present a public benchmark, PersonalLLM, focusing on adapting LLMs to provide maximal benefits for a particular user. Departing from existing alignment benchmarks that implicitly assume uniform preferences, we curate open-ended prompts paired with many high-quality answers over which users would be expected to display heterogeneous latent preferences. Instead of persona-prompting LLMs based on high-level attributes (e.g., user's race or response length), which yields homogeneous preferences relative to humans, we develop a method that can simulate a large user base with diverse preferences from a set of pre-trained reward models. Our dataset and generated personalities offer an innovative testbed for developing personalization algorithms that grapple with continual data sparsity--few relevant feedback from the particular user--by leveraging historical data from other (similar) users. We explore basic in-context learning and meta-learning baselines to illustrate the utility of PersonalLLM and highlight the need for future methodological development. Our dataset is available at https://huggingface.co/datasets/namkoong-lab/PersonalLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20247v1">Resource Allocation for Stable LLM Training in Mobile Edge Computing</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 This paper appears in the 2024 International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc)
    </div>
    <details class="paper-abstract">
      As mobile devices increasingly become focal points for advanced applications, edge computing presents a viable solution to their inherent computational limitations, particularly in deploying large language models (LLMs). However, despite the advancements in edge computing, significant challenges remain in efficient training and deploying LLMs due to the computational demands and data privacy concerns associated with these models. This paper explores a collaborative training framework that integrates mobile users with edge servers to optimize resource allocation, thereby enhancing both performance and efficiency. Our approach leverages parameter-efficient fine-tuning (PEFT) methods, allowing mobile users to adjust the initial layers of the LLM while edge servers handle the more demanding latter layers. Specifically, we formulate a multi-objective optimization problem to minimize the total energy consumption and delay during training. We also address the common issue of instability in model performance by incorporating stability enhancements into our objective function. Through novel fractional programming technique, we achieve a stationary point for the formulated problem. Simulations demonstrate that our method reduces the energy consumption as well as the latency, and increases the reliability of LLMs across various mobile settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20163v1">MemSim: A Bayesian Simulator for Evaluating Memory of LLM-based Personal Assistants</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 26 pages, 25 tables, 1 figure
    </div>
    <details class="paper-abstract">
      LLM-based agents have been widely applied as personal assistants, capable of memorizing information from user messages and responding to personal queries. However, there still lacks an objective and automatic evaluation on their memory capability, largely due to the challenges in constructing reliable questions and answers (QAs) according to user messages. In this paper, we propose MemSim, a Bayesian simulator designed to automatically construct reliable QAs from generated user messages, simultaneously keeping their diversity and scalability. Specifically, we introduce the Bayesian Relation Network (BRNet) and a causal generation mechanism to mitigate the impact of LLM hallucinations on factual information, facilitating the automatic creation of an evaluation dataset. Based on MemSim, we generate a dataset in the daily-life scenario, named MemDaily, and conduct extensive experiments to assess the effectiveness of our approach. We also provide a benchmark for evaluating different memory mechanisms in LLM-based agents with the MemDaily dataset. To benefit the research community, we have released our project at https://github.com/nuster1128/MemSim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20089v1">Robust LLM safeguarding via refusal feature adversarial training</a></div>
    <div class="paper-meta">
      📅 2024-09-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20059v1">Is Preference Alignment Always the Best Option to Enhance LLM-Based Translation? An Empirical Analysis</a></div>
    <div class="paper-meta">
      📅 2024-09-30
    </div>
    <details class="paper-abstract">
      Neural metrics for machine translation (MT) evaluation have become increasingly prominent due to their superior correlation with human judgments compared to traditional lexical metrics. Researchers have therefore utilized neural metrics through quality-informed decoding strategies, achieving better results than likelihood-based methods. With the rise of Large Language Models (LLMs), preference-based alignment techniques have gained attention for their potential to enhance translation quality by optimizing model weights directly on preferences induced by quality estimators. This study focuses on Contrastive Preference Optimization (CPO) and conducts extensive experiments to evaluate the impact of preference-based alignment on translation quality. Our findings indicate that while CPO consistently outperforms Supervised Fine-Tuning (SFT) on high-quality data with regard to the alignment metric, it may lead to instability across downstream evaluation metrics, particularly between neural and lexical ones. Additionally, we demonstrate that relying solely on the base model for generating candidate translations achieves performance comparable to using multiple external systems, while ensuring better consistency across downstream metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11263v2">Understanding the Collapse of LLMs in Model Editing</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 Accepted at Findings of EMNLP 2024 (Camera-Ready Version)
    </div>
    <details class="paper-abstract">
      Despite significant progress in model editing methods, their application in real-world scenarios remains challenging as they often cause large language models (LLMs) to collapse. Among them, ROME is particularly concerning, as it could disrupt LLMs with only a single edit. In this paper, we study the root causes of such collapse. Through extensive analysis, we identify two primary factors that contribute to the collapse: i) inconsistent handling of prefixed and unprefixed keys in the parameter update equation may result in very small denominators, causing excessively large parameter updates; ii) the subject of collapse cases is usually the first token, whose unprefixed key distribution significantly differs from the prefixed key distribution in autoregressive transformers, causing the aforementioned issue to materialize. To validate our findings, we propose a simple yet effective approach: uniformly using prefixed keys during editing phase and adding prefixes during testing phase to ensure the consistency between training and testing. The experimental results show that the proposed solution can prevent model collapse while maintaining the effectiveness of the edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14767v2">I Need Help! Evaluating LLM's Ability to Ask for Users' Support: A Case Study on Text-to-SQL Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 Accepted by EMNLP 2024 Main Conference
    </div>
    <details class="paper-abstract">
      This study explores the proactive ability of LLMs to seek user support. We propose metrics to evaluate the trade-off between performance improvements and user burden, and investigate whether LLMs can determine when to request help under varying information availability. Our experiments show that without external feedback, many LLMs struggle to recognize their need for user support. The findings highlight the importance of external signals and provide insights for future research on improving support-seeking strategies. Source code: https://github.com/appier-research/i-need-help
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19854v1">The Construction of Instruction-tuned LLMs for Finance without Instruction Data Using Continual Pretraining and Model Merging</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      This paper proposes a novel method for constructing instruction-tuned large language models (LLMs) for finance without instruction data. Traditionally, developing such domain-specific LLMs has been resource-intensive, requiring a large dataset and significant computational power for continual pretraining and instruction tuning. Our study proposes a simpler approach that combines domain-specific continual pretraining with model merging. Given that general-purpose pretrained LLMs and their instruction-tuned LLMs are often publicly available, they can be leveraged to obtain the necessary instruction task vector. By merging this with a domain-specific pretrained vector, we can effectively create instruction-tuned LLMs for finance without additional instruction data. Our process involves two steps: first, we perform continual pretraining on financial data; second, we merge the instruction-tuned vector with the domain-specific pretrained vector. Our experiments demonstrate the successful construction of instruction-tuned LLMs for finance. One major advantage of our method is that the instruction-tuned and domain-specific pretrained vectors are nearly independent. This independence makes our approach highly effective. The Japanese financial instruction-tuned LLMs we developed in this study are available at https://huggingface.co/pfnet/nekomata-14b-pfn-qfin-inst-merge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01882v2">Investigating Expert-in-the-Loop LLM Discourse Patterns for Ancient Intertextual Analysis</a></div>
    <div class="paper-meta">
      📅 2024-09-29
    </div>
    <details class="paper-abstract">
      This study explores the potential of large language models (LLMs) for identifying and examining intertextual relationships within biblical, Koine Greek texts. By evaluating the performance of LLMs on various intertextuality scenarios the study demonstrates that these models can detect direct quotations, allusions, and echoes between texts. The LLM's ability to generate novel intertextual observations and connections highlights its potential to uncover new insights. However, the model also struggles with long query passages and the inclusion of false intertextual dependences, emphasizing the importance of expert evaluation. The expert-in-the-loop methodology presented offers a scalable approach for intertextual research into the complex web of intertextuality within and beyond the biblical corpus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19750v1">AstroMLab 2: AstroLLaMA-2-70B Model and Benchmarking Specialised LLMs for Astronomy</a></div>
    <div class="paper-meta">
      📅 2024-09-29
      | 💬 10 pages, 1 figure, 1 table, accepted to AI4S: The 5th Workshop on Artificial Intelligence and Machine Learning for Scientific Applications at the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC24). Models will be released at https://huggingface.co/AstroMLab. AstroMLab homepage: https://astromlab.org/
    </div>
    <details class="paper-abstract">
      Continual pretraining of large language models on domain-specific data has been proposed to enhance performance on downstream tasks. In astronomy, the previous absence of astronomy-focused benchmarks has hindered objective evaluation of these specialized LLM models. Leveraging a recent initiative to curate high-quality astronomical MCQs, this study aims to quantitatively assess specialized LLMs in astronomy. We find that the previously released AstroLLaMA series, based on LLaMA-2-7B, underperforms compared to the base model. We demonstrate that this performance degradation can be partially mitigated by utilizing high-quality data for continual pretraining, such as summarized text from arXiv. Despite the observed catastrophic forgetting in smaller models, our results indicate that continual pretraining on the 70B model can yield significant improvements. However, the current supervised fine-tuning dataset still constrains the performance of instruct models. In conjunction with this study, we introduce a new set of models, AstroLLaMA-3-8B and AstroLLaMA-2-70B, building upon the previous AstroLLaMA series.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19710v1">A multimodal LLM for the non-invasive decoding of spoken text from brain recordings</a></div>
    <div class="paper-meta">
      📅 2024-09-29
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Brain-related research topics in artificial intelligence have recently gained popularity, particularly due to the expansion of what multimodal architectures can do from computer vision to natural language processing. Our main goal in this work is to explore the possibilities and limitations of these architectures in spoken text decoding from non-invasive fMRI recordings. Contrary to vision and textual data, fMRI data represent a complex modality due to the variety of brain scanners, which implies (i) the variety of the recorded signal formats, (ii) the low resolution and noise of the raw signals, and (iii) the scarcity of pretrained models that can be leveraged as foundation models for generative learning. These points make the problem of the non-invasive decoding of text from fMRI recordings very challenging. In this paper, we propose and end-to-end multimodal LLM for decoding spoken text from fMRI signals. The proposed architecture is founded on (i) an encoder derived from a specific transformer incorporating an augmented embedding layer for the encoder and a better-adjusted attention mechanism than that present in the state of the art, and (ii) a frozen large language model adapted to align the embedding of the input text and the encoded embedding of brain activity to decode the output text. A benchmark in performed on a corpus consisting of a set of interactions human-human and human-robot interactions where fMRI and conversational signals are recorded synchronously. The obtained results are very promising, as our proposal outperforms the evaluated models, and is able to generate text capturing more accurate semantics present in the ground truth. The implementation code is provided in https://github.com/Hmamouche/brain_decode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03282v2">LLM Internal States Reveal Hallucination Risk Faced With a Query</a></div>
    <div class="paper-meta">
      📅 2024-09-29
    </div>
    <details class="paper-abstract">
      The hallucination problem of Large Language Models (LLMs) significantly limits their reliability and trustworthiness. Humans have a self-awareness process that allows us to recognize what we don't know when faced with queries. Inspired by this, our paper investigates whether LLMs can estimate their own hallucination risk before response generation. We analyze the internal mechanisms of LLMs broadly both in terms of training data sources and across 15 diverse Natural Language Generation (NLG) tasks, spanning over 700 datasets. Our empirical analysis reveals two key insights: (1) LLM internal states indicate whether they have seen the query in training data or not; and (2) LLM internal states show they are likely to hallucinate or not regarding the query. Our study explores particular neurons, activation layers, and tokens that play a crucial role in the LLM perception of uncertainty and hallucination risk. By a probing estimator, we leverage LLM self-assessment, achieving an average hallucination estimation accuracy of 84.32\% at run time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17972v2">BEATS: Optimizing LLM Mathematical Capabilities with BackVerify and Adaptive Disambiguate based Efficient Tree Search</a></div>
    <div class="paper-meta">
      📅 2024-09-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited exceptional performance across a broad range of tasks and domains. However, they still encounter difficulties in solving mathematical problems due to the rigorous and logical nature of mathematics. Previous studies have employed techniques such as supervised fine-tuning (SFT), prompt engineering, and search-based methods to improve the mathematical problem-solving abilities of LLMs. Despite these efforts, their performance remains suboptimal and demands substantial computational resources. To address this issue, we propose a novel approach, BEATS, to enhance mathematical problem-solving abilities. Our method leverages newly designed prompts that guide the model to iteratively rewrite, advance by one step, and generate answers based on previous steps. Additionally, we introduce a new back-verification technique that uses LLMs to validate the correctness of the generated answers. Furthermore, we employ a pruning tree search to optimize search time while achieving strong performance. Notably, our method improves Qwen2-7b-Instruct's score from 36.94 to 61.52, outperforming GPT4's 42.5 on the MATH benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12585v2">Breaking the Ceiling of the LLM Community by Treating Token Generation as a Classification for Ensembling</a></div>
    <div class="paper-meta">
      📅 2024-09-29
      | 💬 Accepted to EMNLP 2024
    </div>
    <details class="paper-abstract">
      Ensembling multiple models has always been an effective approach to push the limits of existing performance and is widely used in classification tasks by simply averaging the classification probability vectors from multiple classifiers to achieve better accuracy. However, in the thriving open-source Large Language Model (LLM) community, ensembling methods are rare and typically limited to ensembling the full-text outputs of LLMs, such as selecting the best output using a ranker, which leads to underutilization of token-level probability information. In this paper, we treat the Generation of each token by LLMs as a Classification (GaC) for ensembling. This approach fully exploits the probability information at each generation step and better prevents LLMs from producing early incorrect tokens that lead to snowballing errors. In experiments, we ensemble state-of-the-art LLMs on several benchmarks, including exams, mathematics and reasoning, and observe that our method breaks the existing community performance ceiling. Furthermore, we observed that most of the tokens in the answer are simple and do not affect the correctness of the final answer. Therefore, we also experimented with ensembling only key tokens, and the results showed better performance with lower latency across benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09785v2">GoNoGo: An Efficient LLM-based Multi-Agent System for Streamlining Automotive Software Release Decision-Making</a></div>
    <div class="paper-meta">
      📅 2024-09-29
    </div>
    <details class="paper-abstract">
      Traditional methods for making software deployment decisions in the automotive industry typically rely on manual analysis of tabular software test data. These methods often lead to higher costs and delays in the software release cycle due to their labor-intensive nature. Large Language Models (LLMs) present a promising solution to these challenges. However, their application generally demands multiple rounds of human-driven prompt engineering, which limits their practical deployment, particularly for industrial end-users who need reliable and efficient results. In this paper, we propose GoNoGo, an LLM agent system designed to streamline automotive software deployment while meeting both functional requirements and practical industrial constraints. Unlike previous systems, GoNoGo is specifically tailored to address domain-specific and risk-sensitive systems. We evaluate GoNoGo's performance across different task difficulties using zero-shot and few-shot examples taken from industrial practice. Our results show that GoNoGo achieves a 100% success rate for tasks up to Level 2 difficulty with 3-shot examples, and maintains high performance even for more complex tasks. We find that GoNoGo effectively automates decision-making for simpler tasks, significantly reducing the need for manual intervention. In summary, GoNoGo represents an efficient and user-friendly LLM-based solution currently employed in our industrial partner's company to assist with software release decision-making, supporting more informed and timely decisions in the release process for risk-sensitive vehicle systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17287v3">When to Trust LLMs: Aligning Confidence with Response Quality</a></div>
    <div class="paper-meta">
      📅 2024-09-29
      | 💬 Accepted by ACL 2024. Code: https://github.com/TaoShuchang/CONQORD
    </div>
    <details class="paper-abstract">
      Despite the success of large language models (LLMs) in natural language generation, much evidence shows that LLMs may produce incorrect or nonsensical text. This limitation highlights the importance of discerning when to trust LLMs, especially in safety-critical domains. Existing methods often express reliability by confidence level, however, their effectiveness is limited by the lack of objective guidance. To address this, we propose CONfidence-Quality-ORDer-preserving alignment approach (CONQORD), which leverages reinforcement learning guided by a tailored dual-component reward function. This function integrates quality reward and order-preserving alignment reward functions. Specifically, the order-preserving reward incentivizes the model to verbalize greater confidence for responses of higher quality to align the order of confidence and quality. Experiments demonstrate that CONQORD significantly improves the alignment performance between confidence and response accuracy, without causing over-cautious. Furthermore, the aligned confidence provided by CONQORD informs when to trust LLMs, and acts as a determinant for initiating the retrieval process of external knowledge. Aligning confidence with response quality ensures more transparent and reliable responses, providing better trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14320v6">Triad: A Framework Leveraging a Multi-Role LLM-based Agent to Solve Knowledge Base Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-09-29
      | 💬 8 pages, Accepted by EMNLP 2024
    </div>
    <details class="paper-abstract">
      Recent progress with LLM-based agents has shown promising results across various tasks. However, their use in answering questions from knowledge bases remains largely unexplored. Implementing a KBQA system using traditional methods is challenging due to the shortage of task-specific training data and the complexity of creating task-focused model structures. In this paper, we present Triad, a unified framework that utilizes an LLM-based agent with three roles for KBQA tasks. The agent is assigned three roles to tackle different KBQA subtasks: agent as a generalist for mastering various subtasks, as a decision maker for the selection of candidates, and as an advisor for answering questions with knowledge. Our KBQA framework is executed in four phases, involving the collaboration of the agent's multiple roles. We evaluated the performance of our framework using three benchmark datasets, and the results show that our framework outperforms state-of-the-art systems on the LC-QuAD and YAGO-QA benchmarks, yielding F1 scores of 11.8% and 20.7%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19523v1">LANDeRMT: Detecting and Routing Language-Aware Neurons for Selectively Finetuning LLMs to Machine Translation</a></div>
    <div class="paper-meta">
      📅 2024-09-29
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown promising results in multilingual translation even with limited bilingual supervision. The major challenges are catastrophic forgetting and parameter interference for finetuning LLMs when provided parallel training data. To address these challenges, we propose LANDeRMT, a \textbf{L}anguage-\textbf{A}ware \textbf{N}euron \textbf{De}tecting and \textbf{R}outing framework that selectively finetunes LLMs to \textbf{M}achine \textbf{T}ranslation with diverse translation training data. In LANDeRMT, we evaluate the awareness of neurons to MT tasks and categorize them into language-general and language-specific neurons. This categorization enables selective parameter updates during finetuning, mitigating parameter interference and catastrophic forgetting issues. For the detected neurons, we further propose a conditional awareness-based routing mechanism to dynamically adjust language-general and language-specific capacity within LLMs, guided by translation signals. Experimental results demonstrate that the proposed LANDeRMT is very effective in learning translation knowledge, significantly improving translation quality over various strong baselines for multiple language pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19510v1">CoT-ST: Enhancing LLM-based Speech Translation with Multimodal Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2024-09-29
    </div>
    <details class="paper-abstract">
      Speech Language Models (SLMs) have demonstrated impressive performance on speech translation tasks. However, existing research primarily focuses on direct instruction fine-tuning and often overlooks the inherent reasoning capabilities of SLMs. In this paper, we introduce a three-stage training framework designed to activate the chain-of-thought (CoT) capabilities of SLMs. We propose CoT-ST, a speech translation model that utilizes multimodal CoT to decompose speech translation into sequential steps of speech recognition and translation. We validated the effectiveness of our method on two datasets: the CoVoST-2 dataset and MuST-C dataset. The experimental results demonstrate that CoT-ST outperforms previous state-of-the-art methods, achieving higher BLEU scores (CoVoST-2 en-ja: 30.5->30.8, en-zh: 45.2->47.7, MuST-C en-zh: 19.6->21.2). This work is open sourced at https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/st_covost2 .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10065v3">Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-28
      | 💬 EMNLP Main 2024. Code, prompt templates, prompts, and outputs are publicly available at https://github.com/UKPLab/arxiv2024-conditional-reasoning-llms
    </div>
    <details class="paper-abstract">
      Reasoning is a fundamental component of language understanding. Recent prompting techniques, such as chain of thought, have consistently improved LLMs' performance on various reasoning tasks. Nevertheless, there is still little understanding of what triggers reasoning abilities in LLMs in the inference stage. In this paper, we introduce code prompting, a chain of prompts that transforms a natural language problem into code and directly prompts the LLM using the generated code without resorting to external code execution. We hypothesize that code prompts can elicit certain reasoning capabilities of LLMs trained on text and code and utilize the proposed method to improve conditional reasoning, the ability to infer different conclusions depending on the fulfillment of certain conditions. We find that code prompting exhibits a high-performance boost for multiple LLMs (up to 22.52 percentage points on GPT 3.5, 7.75 on Mixtral, and 16.78 on Mistral) across multiple conditional reasoning datasets. We then conduct comprehensive experiments to understand how code prompts trigger reasoning abilities and which capabilities are elicited in the underlying models. Our analysis of GPT 3.5 reveals that the code formatting of the input problem is essential for performance improvement. Furthermore, code prompts improve sample efficiency of in-context learning and facilitate state tracking of variables or entities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19338v1">Decoding Echo Chambers: LLM-Powered Simulations Revealing Polarization in Social Networks</a></div>
    <div class="paper-meta">
      📅 2024-09-28
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The impact of social media on critical issues such as echo chambers needs to be addressed, as these phenomena can have disruptive consequences for our society. Traditional research often oversimplifies emotional tendencies and opinion evolution into numbers and formulas, neglecting that news and communication are conveyed through text, which limits these approaches. Hence, in this work, we propose an LLM-based simulation for the social opinion network to evaluate and counter polarization phenomena. We first construct three typical network structures to simulate different characteristics of social interactions. Then, agents interact based on recommendation algorithms and update their strategies through reasoning and analysis. By comparing these interactions with the classic Bounded Confidence Model (BCM), the Friedkin Johnsen (FJ) model, and using echo chamber-related indices, we demonstrate the effectiveness of our framework in simulating opinion dynamics and reproducing phenomena such as opinion polarization and echo chambers. We propose two mitigation methods, active and passive nudges, that can help reduce echo chambers, specifically within language-based simulations. We hope our work will offer valuable insights and guidance for social polarization mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11065v2">Can LLMs Understand the Implication of Emphasized Sentences in Dialogue?</a></div>
    <div class="paper-meta">
      📅 2024-09-28
      | 💬 Accepted by EMNLP 2024 Findings
    </div>
    <details class="paper-abstract">
      Emphasis is a crucial component in human communication, which indicates the speaker's intention and implication beyond pure text in dialogue. While Large Language Models (LLMs) have revolutionized natural language processing, their ability to understand emphasis in dialogue remains unclear. This paper introduces Emphasized-Talk, a benchmark with emphasis-annotated dialogue samples capturing the implications of emphasis. We evaluate various LLMs, both open-source and commercial, to measure their performance in understanding emphasis. Additionally, we propose an automatic evaluation pipeline using GPT-4, which achieves a high correlation with human rating. Our findings reveal that although commercial LLMs generally perform better, there is still significant room for improvement in comprehending emphasized sentences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04281v2">Similar Data Points Identification with LLM: A Human-in-the-loop Strategy Using Summarization and Hidden State Insights</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      This study introduces a simple yet effective method for identifying similar data points across non-free text domains, such as tabular and image data, using Large Language Models (LLMs). Our two-step approach involves data point summarization and hidden state extraction. Initially, data is condensed via summarization using an LLM, reducing complexity and highlighting essential information in sentences. Subsequently, the summarization sentences are fed through another LLM to extract hidden states, serving as compact, feature-rich representations. This approach leverages the advanced comprehension and generative capabilities of LLMs, offering a scalable and efficient strategy for similarity identification across diverse datasets. We demonstrate the effectiveness of our method in identifying similar data points on multiple datasets. Additionally, our approach enables non-technical domain experts, such as fraud investigators or marketing operators, to quickly identify similar data points tailored to specific scenarios, demonstrating its utility in practical applications. In general, our results open new avenues for leveraging LLMs in data analysis across various domains
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19151v1">Can LLMs Really Learn to Translate a Low-Resource Language from One Grammar Book?</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      Extremely low-resource (XLR) languages lack substantial corpora for training NLP models, motivating the use of all available resources such as dictionaries and grammar books. Machine Translation from One Book (Tanzer et al., 2024) suggests prompting long-context LLMs with one grammar book enables English-Kalamang translation, an unseen XLR language - a noteworthy case of linguistic knowledge helping an NLP task. We investigate whether the book's grammatical explanations or its parallel examples are most effective for learning XLR translation, finding almost all improvement stems from the parallel examples. Further, we find similar results for Nepali, a seen low-resource language, and achieve performance comparable to an LLM with a grammar book by simply fine-tuning an encoder-decoder translation model. We then investigate where grammar books help by testing two linguistic tasks, grammaticality judgment and gloss prediction, and we explore what kind of grammatical knowledge helps by introducing a typological feature prompt that achieves leading results on these more relevant tasks. We thus emphasise the importance of task-appropriate data for XLR languages: parallel examples for translation, and grammatical data for linguistic tasks. As we find no evidence that long-context LLMs can make effective use of grammatical explanations for XLR translation, we suggest data collection for multilingual XLR tasks such as translation is best focused on parallel data over linguistic description.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19100v1">Outlining the Borders for LLM Applications in Patient Education: Developing an Expert-in-the-Loop LLM-Powered Chatbot for Prostate Cancer Patient Education</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      Cancer patients often struggle to transition swiftly to treatment due to limited institutional resources, lack of sophisticated professional guidance, and low health literacy. The emergence of Large Language Models (LLMs) offers new opportunities for such patients to access the wealth of existing patient education materials. The current paper presents the development process for an LLM-based chatbot focused on prostate cancer education, including needs assessment, co-design, and usability studies. The resulting application, MedEduChat, integrates with patients' electronic health record data and features a closed-domain, semi-structured, patient-centered approach to address real-world needs. This paper contributes to the growing field of patient-LLM interaction by demonstrating the potential of LLM-based chatbots to enhance prostate cancer patient education and by offering co-design guidelines for future LLM-based healthcare downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19097v1">Implementing LLMs in industrial process modeling: Addressing Categorical Variables</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      Important variables of processes are, in many occasions, categorical, i.e. names or labels representing, e.g. categories of inputs, or types of reactors or a sequence of steps. In this work, we use Large Language Models (LLMs) to derive embeddings of such inputs that represent their actual meaning, or reflect the ``distances" between categories, i.e. how similar or dissimilar they are. This is a marked difference from the current standard practice of using binary, or one-hot encoding to replace categorical variables with sequences of ones and zeros. Combined with dimensionality reduction techniques, either linear such as Principal Components Analysis (PCA), or nonlinear such as Uniform Manifold Approximation and Projection (UMAP), the proposed approach leads to a \textit{meaningful}, low-dimensional feature space. The significance of obtaining meaningful embeddings is illustrated in the context of an industrial coating process for cutting tools that includes both numerical and categorical inputs. The proposed approach enables feature importance which is a marked improvement compared to the current state-of-the-art (SotA) in the encoding of categorical variables.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19058v1">CLLMate: A Multimodal LLM for Weather and Climate Events Forecasting</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      Forecasting weather and climate events is crucial for making appropriate measures to mitigate environmental hazards and minimize associated losses. Previous research on environmental forecasting focuses on predicting numerical meteorological variables related to closed-set events rather than forecasting open-set events directly, which limits the comprehensiveness of event forecasting. We propose Weather and Climate Event Forecasting (WCEF), a new task that leverages meteorological raster data and textual event data to predict potential weather and climate events. However, due to difficulties in aligning multimodal data and the lack of sufficient supervised datasets, this task is challenging to accomplish. Therefore, we first propose a framework to align historical meteorological data with past weather and climate events using the large language model (LLM). In this framework, we construct a knowledge graph by using LLM to extract information about weather and climate events from a corpus of over 41k highly environment-focused news articles. Subsequently, we mapped these events with meteorological raster data, creating a supervised dataset, which is the largest and most novel for LLM tuning on the WCEF task. Finally, we introduced our aligned models, CLLMate (LLM for climate), a multimodal LLM to forecast weather and climate events using meteorological raster data. In evaluating CLLMate, we conducted extensive experiments. The results indicate that CLLMate surpasses both the baselines and other multimodal LLMs, showcasing the potential of utilizing LLM to align weather and climate events with meteorological data and highlighting the promising future for research on the WCEF task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03291v2">LLM Detectors Still Fall Short of Real World: Case of LLM-Generated Short News-Like Posts</a></div>
    <div class="paper-meta">
      📅 2024-09-27
      | 💬 20 pages, 7 tables, 13 figures, under consideration for EMNLP
    </div>
    <details class="paper-abstract">
      With the emergence of widely available powerful LLMs, disinformation generated by large Language Models (LLMs) has become a major concern. Historically, LLM detectors have been touted as a solution, but their effectiveness in the real world is still to be proven. In this paper, we focus on an important setting in information operations -- short news-like posts generated by moderately sophisticated attackers. We demonstrate that existing LLM detectors, whether zero-shot or purpose-trained, are not ready for real-world use in that setting. All tested zero-shot detectors perform inconsistently with prior benchmarks and are highly vulnerable to sampling temperature increase, a trivial attack absent from recent benchmarks. A purpose-trained detector generalizing across LLMs and unseen attacks can be developed, but it fails to generalize to new human-written texts. We argue that the former indicates domain-specific benchmarking is needed, while the latter suggests a trade-off between the adversarial evasion resilience and overfitting to the reference human text, with both needing evaluation in benchmarks and currently absent. We believe this suggests a re-consideration of current LLM detector benchmarking approaches and provides a dynamically extensible benchmark to allow it (https://github.com/Reliable-Information-Lab-HEVS/benchmark_llm_texts_detection).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14744v2">Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to face hallucination issues due to the data they trained on often containing human bias; whether this is reflected in the decision-making process of LLM Agents remains under-explored. As LLM Agents are increasingly employed in intricate social environments, a pressing and natural question emerges: Can we utilize LLM Agents' systematic hallucinations to mirror human cognitive biases, thus exhibiting irrational social intelligence? In this paper, we probe the irrational behavior among contemporary LLM Agents by melding practical social science experiments with theoretical insights. Specifically, We propose CogMir, an open-ended Multi-LLM Agents framework that utilizes hallucination properties to assess and enhance LLM Agents' social intelligence through cognitive biases. Experimental results on CogMir subsets show that LLM Agents and humans exhibit high consistency in irrational and prosocial decision-making under uncertain conditions, underscoring the prosociality of LLM Agents as social entities and highlighting the significance of hallucination properties. Additionally, the CogMir framework demonstrates its potential as a valuable platform for encouraging more research into the social intelligence of LLM Agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18794v1">Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) tasks require an agent to follow textual instructions to navigate through 3D environments. Traditional approaches use supervised learning methods, relying heavily on domain-specific datasets to train VLN models. Recent methods try to utilize closed-source large language models (LLMs) like GPT-4 to solve VLN tasks in zero-shot manners, but face challenges related to expensive token costs and potential data breaches in real-world applications. In this work, we introduce Open-Nav, a novel study that explores open-source LLMs for zero-shot VLN in the continuous environment. Open-Nav employs a spatial-temporal chain-of-thought (CoT) reasoning approach to break down tasks into instruction comprehension, progress estimation, and decision-making. It enhances scene perceptions with fine-grained object and spatial knowledge to improve LLM's reasoning in navigation. Our extensive experiments in both simulated and real-world environments demonstrate that Open-Nav achieves competitive performance compared to using closed-source LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18764v1">Charting the Future: Using Chart Question-Answering for Scalable Evaluation of LLM-Driven Data Visualizations</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      We propose a novel framework that leverages Visual Question Answering (VQA) models to automate the evaluation of LLM-generated data visualizations. Traditional evaluation methods often rely on human judgment, which is costly and unscalable, or focus solely on data accuracy, neglecting the effectiveness of visual communication. By employing VQA models, we assess data representation quality and the general communicative clarity of charts. Experiments were conducted using two leading VQA benchmark datasets, ChartQA and PlotQA, with visualizations generated by OpenAI's GPT-3.5 Turbo and Meta's Llama 3.1 70B-Instruct models. Our results indicate that LLM-generated charts do not match the accuracy of the original non-LLM-generated charts based on VQA performance measures. Moreover, while our results demonstrate that few-shot prompting significantly boosts the accuracy of chart generation, considerable progress remains to be made before LLMs can fully match the precision of human-generated graphs. This underscores the importance of our work, which expedites the research process by enabling rapid iteration without the need for human annotation, thus accelerating advancements in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14277v2">QPaug: Question and Passage Augmentation for Open-Domain Question Answering of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-27
      | 💬 The 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), Findings
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has received much attention for Open-domain question-answering (ODQA) tasks as a means to compensate for the parametric knowledge of large language models (LLMs). While previous approaches focused on processing retrieved passages to remove irrelevant context, they still rely heavily on the quality of retrieved passages which can degrade if the question is ambiguous or complex. In this paper, we propose a simple yet efficient method called question and passage augmentation (QPaug) via LLMs for open-domain QA. QPaug first decomposes the original questions into multiple-step sub-questions. By augmenting the original question with detailed sub-questions and planning, we are able to make the query more specific on what needs to be retrieved, improving the retrieval performance. In addition, to compensate for the case where the retrieved passages contain distracting information or divided opinions, we augment the retrieved passages with self-generated passages by LLMs to guide the answer extraction. Experimental results show that QPaug outperforms the previous state-of-the-art and achieves significant performance gain over existing RAG methods. The source code is available at \url{https://github.com/kmswin1/QPaug}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18661v1">Not the Silver Bullet: LLM-enhanced Programming Error Messages are Ineffective in Practice</a></div>
    <div class="paper-meta">
      📅 2024-09-27
      | 💬 To appear in the proceedings of the 2024 UK and Ireland Computing Education Research conference (UKICER '24)
    </div>
    <details class="paper-abstract">
      The sudden emergence of large language models (LLMs) such as ChatGPT has had a disruptive impact throughout the computing education community. LLMs have been shown to excel at producing correct code to CS1 and CS2 problems, and can even act as friendly assistants to students learning how to code. Recent work shows that LLMs demonstrate unequivocally superior results in being able to explain and resolve compiler error messages -- for decades, one of the most frustrating parts of learning how to code. However, LLM-generated error message explanations have only been assessed by expert programmers in artificial conditions. This work sought to understand how novice programmers resolve programming error messages (PEMs) in a more realistic scenario. We ran a within-subjects study with $n$ = 106 participants in which students were tasked to fix six buggy C programs. For each program, participants were randomly assigned to fix the problem using either a stock compiler error message, an expert-handwritten error message, or an error message explanation generated by GPT-4. Despite promising evidence on synthetic benchmarks, we found that GPT-4 generated error messages outperformed conventional compiler error messages in only 1 of the 6 tasks, measured by students' time-to-fix each problem. Handwritten explanations still outperform LLM and conventional error messages, both on objective and subjective measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06349v2">CausalBench: A Comprehensive Benchmark for Causal Learning Capability of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      The ability to understand causality significantly impacts the competence of large language models (LLMs) in output explanation and counterfactual reasoning, as causality reveals the underlying data distribution. However, the lack of a comprehensive benchmark currently limits the evaluation of LLMs' causal learning capabilities. To fill this gap, this paper develops CausalBench based on data from the causal research community, enabling comparative evaluations of LLMs against traditional causal learning algorithms. To provide a comprehensive investigation, we offer three tasks of varying difficulties, including correlation, causal skeleton, and causality identification. Evaluations of 19 leading LLMs reveal that, while closed-source LLMs show potential for simple causal relationships, they significantly lag behind traditional algorithms on larger-scale networks ($>50$ nodes). Specifically, LLMs struggle with collider structures but excel at chain structures, especially at long-chain causality analogous to Chains-of-Thought techniques. This supports the current prompt approaches while suggesting directions to enhance LLMs' causal reasoning capability. Furthermore, CausalBench incorporates background knowledge and training data into prompts to thoroughly unlock LLMs' text-comprehension ability during evaluation, whose findings indicate that, LLM understand causality through semantic associations with distinct entities, rather than directly from contextual information or numerical distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18602v1">Do LLMs suffer from Multi-Party Hangover? A Diagnostic Approach to Addressee Recognition and Response Selection in Conversations</a></div>
    <div class="paper-meta">
      📅 2024-09-27
      | 💬 Accepted to EMNLP 2024 main conference
    </div>
    <details class="paper-abstract">
      Assessing the performance of systems to classify Multi-Party Conversations (MPC) is challenging due to the interconnection between linguistic and structural characteristics of conversations. Conventional evaluation methods often overlook variances in model behavior across different levels of structural complexity on interaction graphs. In this work, we propose a methodological pipeline to investigate model performance across specific structural attributes of conversations. As a proof of concept we focus on Response Selection and Addressee Recognition tasks, to diagnose model weaknesses. To this end, we extract representative diagnostic subdatasets with a fixed number of users and a good structural variety from a large and open corpus of online MPCs. We further frame our work in terms of data minimization, avoiding the use of original usernames to preserve privacy, and propose alternatives to using original text messages. Results show that response selection relies more on the textual content of conversations, while addressee recognition requires capturing their structural dimension. Using an LLM in a zero-shot setting, we further highlight how sensitivity to prompt variations is task-dependent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08424v2">Comparing Apples to Oranges: LLM-powered Multimodal Intention Prediction in an Object Categorization Task</a></div>
    <div class="paper-meta">
      📅 2024-09-27
      | 💬 Accepted at ICSR 2024,14 pages,5 figures,2 tables; work was co-funded by Horizon Europe project TERAIS under Grant agreement number 101079338
    </div>
    <details class="paper-abstract">
      Human intention-based systems enable robots to perceive and interpret user actions to interact with humans and adapt to their behavior proactively. Therefore, intention prediction is pivotal in creating a natural interaction with social robots in human-designed environments. In this paper, we examine using Large Language Models (LLMs) to infer human intention in a collaborative object categorization task with a physical robot. We propose a novel multimodal approach that integrates user non-verbal cues, like hand gestures, body poses, and facial expressions, with environment states and user verbal cues to predict user intentions in a hierarchical architecture. Our evaluation of five LLMs shows the potential for reasoning about verbal and non-verbal user cues, leveraging their context-understanding and real-world knowledge to support intention prediction while collaborating on a task with a social robot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18594v1">"Oh LLM, I'm Asking Thee, Please Give Me a Decision Tree": Zero-Shot Decision Tree Induction and Embedding with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) provide powerful means to leverage prior knowledge for predictive modeling when data is limited. In this work, we demonstrate how LLMs can use their compressed world knowledge to generate intrinsically interpretable machine learning models, i.e., decision trees, without any training data. We find that these zero-shot decision trees can surpass data-driven trees on some small-sized tabular datasets and that embeddings derived from these trees perform on par with data-driven tree-based embeddings on average. Our knowledge-driven decision tree induction and embedding approaches therefore serve as strong new baselines for data-driven machine learning methods in the low-data regime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14364v2">More Effective LLM Compressed Tokens with Uniformly Spread Position Identifiers and Compression Loss</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      Compressing Transformer inputs into compressd tokens allows running LLMs with improved speed and cost efficiency. Based on the compression method ICAE, we carefully examine the position identifier choices for compressed tokens and also propose a new compression loss. We demonstrate empirically that our proposed methods achieve significantly higher compression ratios (15x compared to 4x for ICAE), while being able to attain comparable reconstruction performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13997v2">"Global is Good, Local is Bad?": Understanding Brand Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-27
      | 💬 Accepted at EMNLP-2024 (main)
    </div>
    <details class="paper-abstract">
      Many recent studies have investigated social biases in LLMs but brand bias has received little attention. This research examines the biases exhibited by LLMs towards different brands, a significant concern given the widespread use of LLMs in affected use cases such as product recommendation and market analysis. Biased models may perpetuate societal inequalities, unfairly favoring established global brands while marginalizing local ones. Using a curated dataset across four brand categories, we probe the behavior of LLMs in this space. We find a consistent pattern of bias in this space -- both in terms of disproportionately associating global brands with positive attributes and disproportionately recommending luxury gifts for individuals in high-income countries. We also find LLMs are subject to country-of-origin effects which may boost local brand preference in LLM outputs in specific contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18433v1">Easy2Hard-Bench: Standardized Difficulty Labels for Profiling LLM Performance and Generalization</a></div>
    <div class="paper-meta">
      📅 2024-09-27
      | 💬 NeurIPS 2024 Datasets and Benchmarks Track
    </div>
    <details class="paper-abstract">
      While generalization over tasks from easy to hard is crucial to profile language models (LLMs), the datasets with fine-grained difficulty annotations for each problem across a broad range of complexity are still blank. Aiming to address this limitation, we present Easy2Hard-Bench, a consistently formatted collection of 6 benchmark datasets spanning various domains, such as mathematics and programming problems, chess puzzles, and reasoning questions. Each problem within these datasets is annotated with numerical difficulty scores. To systematically estimate problem difficulties, we collect abundant performance data on attempts to each problem by humans in the real world or LLMs on the prominent leaderboard. Leveraging the rich performance data, we apply well-established difficulty ranking systems, such as Item Response Theory (IRT) and Glicko-2 models, to uniformly assign numerical difficulty scores to problems. Moreover, datasets in Easy2Hard-Bench distinguish themselves from previous collections by a higher proportion of challenging problems. Through extensive experiments with six state-of-the-art LLMs, we provide a comprehensive analysis of their performance and generalization capabilities across varying levels of difficulty, with the aim of inspiring future research in LLM generalization. The datasets are available at https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08044v2">RoLoRA: Fine-tuning Rotated Outlier-free LLMs for Effective Weight-Activation Quantization</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 EMNLP 2024 Findings, Codes: https://github.com/HuangOwen/RoLoRA, Models: https://huggingface.co/collections/ScarletAce/rolora-66f5f228a90681c7c4512b28
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA), as a representative Parameter-Efficient Fine-Tuning (PEFT)method, significantly enhances the training efficiency by updating only a small portion of the weights in Large Language Models (LLMs). Recently, weight-only quantization techniques have also been applied to LoRA methods to reduce the memory footprint of fine-tuning. However, applying weight-activation quantization to the LoRA pipeline is under-explored, and we observe substantial performance degradation primarily due to the presence of activation outliers. In this work, we propose RoLoRA, the first LoRA-based scheme for effective weight-activation quantization. RoLoRA utilizes rotation for outlier elimination and proposes rotation-aware fine-tuning to preserve the outlier-free characteristics in rotated LLMs. Experimental results show RoLoRA consistently improves low-bit LoRA convergence and post-training quantization robustness in weight-activation settings. We evaluate RoLoRA across LLaMA2-7B/13B, LLaMA3-8B models, achieving up to 29.5% absolute accuracy gain of 4-bit weight-activation quantized LLaMA2- 13B on commonsense reasoning tasks compared to LoRA baseline. We further demonstrate its effectiveness on Large Multimodal Models (LLaVA-1.5-7B). Codes are available at https://github.com/HuangOwen/RoLoRA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18345v1">A Generalized LLM-Augmented BIM Framework: Application to a Speech-to-BIM system</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 In Proceedings of the 41st International Conference of CIB W78. Marrakech, Morocco, 2024
    </div>
    <details class="paper-abstract">
      Performing building information modeling (BIM) tasks is a complex process that imposes a steep learning curve and a heavy cognitive load due to the necessity of remembering sequences of numerous commands. With the rapid advancement of large language models (LLMs), it is foreseeable that BIM tasks, including querying and managing BIM data, 4D and 5D BIM, design compliance checking, or authoring a design, using written or spoken natural language (i.e., text-to-BIM or speech-to-BIM), will soon supplant traditional graphical user interfaces. This paper proposes a generalized LLM-augmented BIM framework to expedite the development of LLM-enhanced BIM applications by providing a step-by-step development process. The proposed framework consists of six steps: interpret-fill-match-structure-execute-check. The paper demonstrates the applicability of the proposed framework through implementing a speech-to-BIM application, NADIA-S (Natural-language-based Architectural Detailing through Interaction with Artificial Intelligence via Speech), using exterior wall detailing as an example.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18290v1">Retrospective Comparative Analysis of Prostate Cancer In-Basket Messages: Responses from Closed-Domain LLM vs. Clinical Teams</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      In-basket message interactions play a crucial role in physician-patient communication, occurring during all phases (pre-, during, and post) of a patient's care journey. However, responding to these patients' inquiries has become a significant burden on healthcare workflows, consuming considerable time for clinical care teams. To address this, we introduce RadOnc-GPT, a specialized Large Language Model (LLM) powered by GPT-4 that has been designed with a focus on radiotherapeutic treatment of prostate cancer with advanced prompt engineering, and specifically designed to assist in generating responses. We integrated RadOnc-GPT with patient electronic health records (EHR) from both the hospital-wide EHR database and an internal, radiation-oncology-specific database. RadOnc-GPT was evaluated on 158 previously recorded in-basket message interactions. Quantitative natural language processing (NLP) analysis and two grading studies with clinicians and nurses were used to assess RadOnc-GPT's responses. Our findings indicate that RadOnc-GPT slightly outperformed the clinical care team in "Clarity" and "Empathy," while achieving comparable scores in "Completeness" and "Correctness." RadOnc-GPT is estimated to save 5.2 minutes per message for nurses and 2.4 minutes for clinicians, from reading the inquiry to sending the response. Employing RadOnc-GPT for in-basket message draft generation has the potential to alleviate the workload of clinical care teams and reduce healthcare costs by producing high-quality, timely responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18203v1">AI Policy Projector: Grounding LLM Policy Design in Iterative Mapmaking</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      Whether a large language model policy is an explicit constitution or an implicit reward model, it is challenging to assess coverage over the unbounded set of real-world situations that a policy must contend with. We introduce an AI policy design process inspired by mapmaking, which has developed tactics for visualizing and iterating on maps even when full coverage is not possible. With Policy Projector, policy designers can survey the landscape of model input-output pairs, define custom regions (e.g., "violence"), and navigate these regions with rules that can be applied to LLM outputs (e.g., if output contains "violence" and "graphic details," then rewrite without "graphic details"). Policy Projector supports interactive policy authoring using LLM classification and steering and a map visualization reflecting the policy designer's work. In an evaluation with 12 AI safety experts, our system helps policy designers to address problematic model behaviors extending beyond an existing, comprehensive harm taxonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19028v1">Exploring LLM-Driven Explanations for Quantum Algorithms</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      Background: Quantum computing is a rapidly growing new programming paradigm that brings significant changes to the design and implementation of algorithms. Understanding quantum algorithms requires knowledge of physics and mathematics, which can be challenging for software developers. Aims: In this work, we provide a first analysis of how LLMs can support developers' understanding of quantum code. Method: We empirically analyse and compare the quality of explanations provided by three widely adopted LLMs (Gpt3.5, Llama2, and Tinyllama) using two different human-written prompt styles for seven state-of-the-art quantum algorithms. We also analyse how consistent LLM explanations are over multiple rounds and how LLMs can improve existing descriptions of quantum algorithms. Results: Llama2 provides the highest quality explanations from scratch, while Gpt3.5 emerged as the LLM best suited to improve existing explanations. In addition, we show that adding a small amount of context to the prompt significantly improves the quality of explanations. Finally, we observe how explanations are qualitatively and syntactically consistent over multiple rounds. Conclusions: This work highlights promising results, and opens challenges for future research in the field of LLMs for quantum code explanation. Future work includes refining the methods through prompt optimisation and parsing of quantum code explanations, as well as carrying out a systematic assessment of the quality of explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13731v3">KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 33 pages
    </div>
    <details class="paper-abstract">
      The recently developed retrieval-augmented generation (RAG) technology has enabled the efficient construction of domain-specific applications. However, it also has limitations, including the gap between vector similarity and the relevance of knowledge reasoning, as well as insensitivity to knowledge logic, such as numerical values, temporal relations, expert rules, and others, which hinder the effectiveness of professional knowledge services. In this work, we introduce a professional domain knowledge service framework called Knowledge Augmented Generation (KAG). KAG is designed to address the aforementioned challenges with the motivation of making full use of the advantages of knowledge graph(KG) and vector retrieval, and to improve generation and reasoning performance by bidirectionally enhancing large language models (LLMs) and KGs through five key aspects: (1) LLM-friendly knowledge representation, (2) mutual-indexing between knowledge graphs and original chunks, (3) logical-form-guided hybrid reasoning engine, (4) knowledge alignment with semantic reasoning, and (5) model capability enhancement for KAG. We compared KAG with existing RAG methods in multihop question answering and found that it significantly outperforms state-of-theart methods, achieving a relative improvement of 19.6% on 2wiki and 33.5% on hotpotQA in terms of F1 score. We have successfully applied KAG to two professional knowledge Q&A tasks of Ant Group, including E-Government Q&A and E-Health Q&A, achieving significant improvement in professionalism compared to RAG methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18014v1">Role-RL: Online Long-Context Processing with Role Reinforcement Learning for Distinct LLMs in Their Optimal Roles</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with long-context processing are still challenging because of their implementation complexity, training efficiency and data sparsity. To address this issue, a new paradigm named Online Long-context Processing (OLP) is proposed when we process a document of unlimited length, which typically occurs in the information reception and organization of diverse streaming media such as automated news reporting, live e-commerce, and viral short videos. Moreover, a dilemma was often encountered when we tried to select the most suitable LLM from a large number of LLMs amidst explosive growth aiming for outstanding performance, affordable prices, and short response delays. In view of this, we also develop Role Reinforcement Learning (Role-RL) to automatically deploy different LLMs in their respective roles within the OLP pipeline according to their actual performance. Extensive experiments are conducted on our OLP-MINI dataset and it is found that OLP with Role-RL framework achieves OLP benchmark with an average recall rate of 93.2% and the LLM cost saved by 79.4%. The code and dataset are publicly available at: https://anonymous.4open.science/r/Role-RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14788v2">On the Design and Analysis of LLM-Based Algorithms</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      We initiate a formal investigation into the design and analysis of LLM-based algorithms, i.e. algorithms that contain one or multiple calls of large language models (LLMs) as sub-routines and critically rely on the capabilities of LLMs. While LLM-based algorithms, ranging from basic LLM calls with prompt engineering to complicated LLM-powered agent systems and compound AI systems, have achieved remarkable empirical success, the design and optimization of them have mostly relied on heuristics and trial-and-errors, which is largely due to a lack of formal and analytical study for these algorithms. To fill this gap, we start by identifying the computational-graph representation of LLM-based algorithms, the design principle of task decomposition, and some key abstractions, which then facilitate our formal analysis for the accuracy and efficiency of LLM-based algorithms, despite the black-box nature of LLMs. Through extensive analytical and empirical investigation in a series of case studies, we demonstrate that the proposed framework is broadly applicable to a wide range of scenarios and diverse patterns of LLM-based algorithms, such as parallel, hierarchical and recursive task decomposition. Our proposed framework holds promise for advancing LLM-based algorithms, by revealing the reasons behind curious empirical phenomena, guiding the choices of hyperparameters, predicting the empirical performance of algorithms, and inspiring new algorithm design. To promote further study of LLM-based algorithms, we release our source code at https://github.com/modelscope/agentscope/tree/main/examples/paper_llm_based_algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17655v1">AssistantX: An LLM-Powered Proactive Assistant in Collaborative Human-Populated Environment</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 6 pages, 8 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The increasing demand for intelligent assistants in human-populated environments has motivated significant research in autonomous robotic systems. Traditional service robots and virtual assistants, however, struggle with real-world task execution due to their limited capacity for dynamic reasoning and interaction, particularly when human collaboration is required. Recent developments in Large Language Models have opened new avenues for improving these systems, enabling more sophisticated reasoning and natural interaction capabilities. In this paper, we introduce AssistantX, an LLM-powered proactive assistant designed to operate autonomously in a physical office environment. Unlike conventional service robots, AssistantX leverages a novel multi-agent architecture, PPDR4X, which provides advanced inference capabilities and comprehensive collaboration awareness. By effectively bridging the gap between virtual operations and physical interactions, AssistantX demonstrates robust performance in managing complex real-world scenarios. Our evaluation highlights the architecture's effectiveness, showing that AssistantX can respond to clear instructions, actively retrieve supplementary information from memory, and proactively seek collaboration from team members to ensure successful task completion. More details and videos can be found at https://assistantx-agent.github.io/AssistantX/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16341v2">Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) for external tool usage is a rapidly expanding field, with recent research focusing on generating synthetic data to address the shortage of available data. However, the absence of systematic data quality checks poses complications for properly training and testing models. To that end, we propose two approaches for assessing the reliability of data for training LLMs to use external tools. The first approach uses intuitive, human-defined correctness criteria. The second approach uses a model-driven assessment with in-context evaluation. We conduct a thorough evaluation of data quality on two popular benchmarks, followed by an extrinsic evaluation that showcases the impact of data quality on model performance. Our results demonstrate that models trained on high-quality data outperform those trained on unvalidated data, even when trained with a smaller quantity of data. These findings empirically support the significance of assessing and ensuring the reliability of training data for tool-using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10267v2">Unused information in token probability distribution of generative LLM: improving LLM reading comprehension through calculation of expected values</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 7 pages, 1 figure, presented at FEDCSIS 2024 conference,
    </div>
    <details class="paper-abstract">
      LLM text decoding is key component for perceived LLM quality. We demonstrate two experiments showing that decoding methods could be improved by manipulation of token probabilities. First, we test few LLM on SummEval summary scoring dataset, to measure reading comprehension. We compare scores from greedy decoding to expected values over the next token distribution. We scale logits by large temperature to increase the entropy of scores. This allows strong improvement of performance on SummEval (in terms of correlations to human judgement). We see improvement from 6-8% to 13-28% for 7B Mistral and from 20%-46% to 37%-56% for Mixtral, beating GPT 4 0314 result on two metrics. Part of the gain seems related to positional bias. Secondly, we use probability-based tree sampling algorithm, to examine all most probable generations for given prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17572v1">Dr. GPT in Campus Counseling: Understanding Higher Education Students' Opinions on LLM-assisted Mental Health Services</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      In response to the increasing mental health challenges faced by college students, we sought to understand their perspectives on how AI applications, particularly Large Language Models (LLMs), can be leveraged to enhance their mental well-being. Through pilot interviews with ten diverse students, we explored their opinions on the use of LLMs across five fictional scenarios: General Information Inquiry, Initial Screening, Reshaping Patient-Expert Dynamics, Long-term Care, and Follow-up Care. Our findings revealed that students' acceptance of LLMs varied by scenario, with participants highlighting both potential benefits, such as proactive engagement and personalized follow-up care, and concerns, including limitations in training data and emotional support. These insights inform how AI technology should be designed and implemented to effectively support and enhance students' mental well-being, particularly in scenarios where LLMs can complement traditional methods, while maintaining empathy and respecting individual preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17504v1">HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 NeurIPS 2024 Spotlight
    </div>
    <details class="paper-abstract">
      The surge in applications of large language models (LLMs) has prompted concerns about the generation of misleading or fabricated information, known as hallucinations. Therefore, detecting hallucinations has become critical to maintaining trust in LLM-generated content. A primary challenge in learning a truthfulness classifier is the lack of a large amount of labeled truthful and hallucinated data. To address the challenge, we introduce HaloScope, a novel learning framework that leverages the unlabeled LLM generations in the wild for hallucination detection. Such unlabeled data arises freely upon deploying LLMs in the open world, and consists of both truthful and hallucinated information. To harness the unlabeled data, we present an automated membership estimation score for distinguishing between truthful and untruthful generations within unlabeled mixture data, thereby enabling the training of a binary truthfulness classifier on top. Importantly, our framework does not require extra data collection and human annotations, offering strong flexibility and practicality for real-world applications. Extensive experiments show that HaloScope can achieve superior hallucination detection performance, outperforming the competitive rivals by a significant margin. Code is available at https://github.com/deeplearningwisc/haloscope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10669v5">Humans or LLMs as the Judge? A Study on Judgement Biases</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 EMNLP2024
    </div>
    <details class="paper-abstract">
      Adopting human and large language models (LLM) as judges (a.k.a human- and LLM-as-a-judge) for evaluating the performance of LLMs has recently gained attention. Nonetheless, this approach concurrently introduces potential biases from human and LLMs, questioning the reliability of the evaluation results. In this paper, we propose a novel framework that is free from referencing groundtruth annotations for investigating Misinformation Oversight Bias, Gender Bias, Authority Bias and Beauty Bias on LLM and human judges. We curate a dataset referring to the revised Bloom's Taxonomy and conduct thousands of evaluations. Results show that human and LLM judges are vulnerable to perturbations to various degrees, and that even the cutting-edge judges possess considerable biases. We further exploit these biases to conduct attacks on LLM judges. We hope that our work can notify the community of the bias and vulnerability of human- and LLM-as-a-judge, as well as the urgency of developing robust evaluation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17433v1">HDFlow: Enhancing LLM Complex Problem-Solving with Hybrid Thinking and Dynamic Workflows</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 27 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Despite recent advancements in large language models (LLMs), their performance on complex reasoning problems requiring multi-step thinking and combining various skills is still limited. To address this, we propose a novel framework HDFlow for complex reasoning with LLMs that combines fast and slow thinking modes in an adaptive manner. Our approach consists of two key components: 1) a new approach for slow, deliberate reasoning called Dynamic Workflow, which automatically decomposes complex problems into more manageable sub-tasks and dynamically designs a workflow to assemble specialized LLM or symbolic reasoning tools to solve sub-tasks; 2) Hybrid Thinking, a general framework that dynamically combines fast and slow thinking based on problem complexity. Finally, we propose an easy-to-scale method for automatically synthesizing a large-scale dataset of 27K challenging reasoning problems for complex reasoning and a hybrid thinking tuning method that trains smaller LLMs on this dataset to internalize the fast/slow hybrid reasoning strategies. Experiments on four reasoning benchmark datasets demonstrate that our slow thinking with dynamic workflows significantly outperforms Chain-of-Thought, and hybrid thinking achieves the highest accuracy while providing an effective balance between computational efficiency and performance. Fine-tuning using our hybrid thinking approach also significantly boosts the complex reasoning capabilities of open-source language models. The results showcase the promise of slow thinking, dynamic workflows, and hybrid thinking in expanding the frontier of complex problem-solving with LLMs\footnote{Code and data will be released at \url{https://github.com/wenlinyao/HDFlow}.}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17422v1">Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in handling long context inputs, but this comes at the cost of increased computational resources and latency. Our research introduces a novel approach for the long context bottleneck to accelerate LLM inference and reduce GPU memory consumption. Our research demonstrates that LLMs can identify relevant tokens in the early layers before generating answers to a query. Leveraging this insight, we propose an algorithm that uses early layers of an LLM as filters to select and compress input tokens, significantly reducing the context length for subsequent processing. Our method, GemFilter, demonstrates substantial improvements in both speed and memory efficiency compared to existing techniques, such as standard attention and SnapKV/H2O. Notably, it achieves a 2.4$\times$ speedup and 30\% reduction in GPU memory usage compared to SOTA methods. Evaluation on the Needle in a Haystack task shows that GemFilter significantly outperforms standard attention, SnapKV and demonstrates comparable performance on the LongBench challenge. GemFilter is simple, training-free, and broadly applicable across different LLMs. Crucially, it provides interpretability by allowing humans to inspect the selected input sequence. These findings not only offer practical benefits for LLM deployment, but also enhance our understanding of LLM internal mechanisms, paving the way for further optimizations in LLM design and inference. Our code is available at \url{https://github.com/SalesforceAIResearch/GemFilter}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17397v1">Severity Prediction in Mental Health: LLM-based Creation, Analysis, Evaluation of a Novel Multilingual Dataset</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into various medical fields, including mental health support systems. However, there is a gap in research regarding the effectiveness of LLMs in non-English mental health support applications. To address this problem, we present a novel multilingual adaptation of widely-used mental health datasets, translated from English into six languages (Greek, Turkish, French, Portuguese, German, and Finnish). This dataset enables a comprehensive evaluation of LLM performance in detecting mental health conditions and assessing their severity across multiple languages. By experimenting with GPT and Llama, we observe considerable variability in performance across languages, despite being evaluated on the same translated dataset. This inconsistency underscores the complexities inherent in multilingual mental health support, where language-specific nuances and mental health data coverage can affect the accuracy of the models. Through comprehensive error analysis, we emphasize the risks of relying exclusively on large language models (LLMs) in medical settings (e.g., their potential to contribute to misdiagnoses). Moreover, our proposed approach offers significant cost savings for multilingual tasks, presenting a major advantage for broad-scale implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17289v1">Steering LLM Summarization with Visual Workspaces for Sensemaking</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 11 figures, 7 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been widely applied in summarization due to their speedy and high-quality text generation. Summarization for sensemaking involves information compression and insight extraction. Human guidance in sensemaking tasks can prioritize and cluster relevant information for LLMs. However, users must translate their cognitive thinking into natural language to communicate with LLMs. Can we use more readable and operable visual representations to guide the summarization process for sensemaking? Therefore, we propose introducing an intermediate step--a schematic visual workspace for human sensemaking--before the LLM generation to steer and refine the summarization process. We conduct a series of proof-of-concept experiments to investigate the potential for enhancing the summarization by GPT-4 through visual workspaces. Leveraging a textual sensemaking dataset with a ground truth summary, we evaluate the impact of a human-generated visual workspace on LLM-generated summarization of the dataset and assess the effectiveness of space-steered summarization. We categorize several types of extractable information from typical human workspaces that can be injected into engineered prompts to steer the LLM summarization. The results demonstrate how such workspaces can help align an LLM with the ground truth, leading to more accurate summarization results than without the workspaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17264v1">Mnemosyne: Parallelization Strategies for Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve to handle increasingly longer contexts, serving inference requests for context lengths in the range of millions of tokens presents unique challenges. While existing techniques are effective for training, they fail to address the unique challenges of inference, such as varying prefill and decode phases and their associated latency constraints - like Time to First Token (TTFT) and Time Between Tokens (TBT). Furthermore, there are no long context inference solutions that allow batching requests to increase the hardware utilization today. In this paper, we propose three key innovations for efficient interactive long context LLM inference, without resorting to any approximation: adaptive chunking to reduce prefill overheads in mixed batching, Sequence Pipeline Parallelism (SPP) to lower TTFT, and KV Cache Parallelism (KVP) to minimize TBT. These contributions are combined into a 3D parallelism strategy, enabling Mnemosyne to scale interactive inference to context lengths at least up to 10 million tokens with high throughput enabled with batching. To our knowledge, Mnemosyne is the first to be able to achieve support for 10 million long context inference efficiently, while satisfying production-grade SLOs on TBT (30ms) on contexts up to and including 10 million.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17140v1">Turn Every Application into an Agent: Towards Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have enabled LLM-based agents to directly interact with application user interfaces (UIs), enhancing agents' performance in complex tasks. However, these agents often suffer from high latency and low reliability due to the extensive sequential UI interactions. To address this issue, we propose AXIS, a novel LLM-based agents framework prioritize actions through application programming interfaces (APIs) over UI actions. This framework also facilitates the creation and expansion of APIs through automated exploration of applications. Our experiments on Office Word demonstrate that AXIS reduces task completion time by 65%-70% and cognitive workload by 38%-53%, while maintaining accuracy of 97%-98% compare to humans. Our work contributes to a new human-agent-computer interaction (HACI) framework and a fresh UI design principle for application providers in the era of LLMs. It also explores the possibility of turning every applications into agents, paving the way towards an agent-centric operating system (Agent OS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17054v1">Using LLM for Real-Time Transcription and Summarization of Doctor-Patient Interactions into ePuskesmas in Indonesia</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      One of the key issues contributing to inefficiency in Puskesmas is the time-consuming nature of doctor-patient interactions. Doctors need to conduct thorough consultations, which include diagnosing the patient's condition, providing treatment advice, and transcribing detailed notes into medical records. In regions with diverse linguistic backgrounds, doctors often have to ask clarifying questions, further prolonging the process. While diagnosing is essential, transcription and summarization can often be automated using AI to improve time efficiency and help doctors enhance care quality and enable early diagnosis and intervention. This paper proposes a solution using a localized large language model (LLM) to transcribe, translate, and summarize doctor-patient conversations. We utilize the Whisper model for transcription and GPT-3 to summarize them into the ePuskemas medical records format. This system is implemented as an add-on to an existing web browser extension, allowing doctors to fill out patient forms while talking. By leveraging this solution for real-time transcription, translation, and summarization, doctors can improve the turnaround time for patient care while enhancing the quality of records, which become more detailed and insightful for future visits. This innovation addresses challenges like overcrowded facilities and the administrative burden on healthcare providers in Indonesia. We believe this solution will help doctors save time, provide better care, and produce more accurate medical records, representing a significant step toward modernizing healthcare and ensuring patients receive timely, high-quality care, even in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16984v1">AXCEL: Automated eXplainable Consistency Evaluation using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in both industry and academia for various tasks, yet evaluating the consistency of generated text responses continues to be a challenge. Traditional metrics like ROUGE and BLEU show a weak correlation with human judgment. More sophisticated metrics using Natural Language Inference (NLI) have shown improved correlations but are complex to implement, require domain-specific training due to poor cross-domain generalization, and lack explainability. More recently, prompt-based metrics using LLMs as evaluators have emerged; while they are easier to implement, they still lack explainability and depend on task-specific prompts, which limits their generalizability. This work introduces Automated eXplainable Consistency Evaluation using LLMs (AXCEL), a prompt-based consistency metric which offers explanations for the consistency scores by providing detailed reasoning and pinpointing inconsistent text spans. AXCEL is also a generalizable metric which can be adopted to multiple tasks without changing the prompt. AXCEL outperforms both non-prompt and prompt-based state-of-the-art (SOTA) metrics in detecting inconsistencies across summarization by 8.7%, free text generation by 6.2%, and data-to-text conversion tasks by 29.4%. We also evaluate the influence of underlying LLMs on prompt based metric performance and recalibrate the SOTA prompt-based metrics with the latest LLMs for fair comparison. Further, we show that AXCEL demonstrates strong performance using open source LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16973v1">Adaptive Self-Supervised Learning Strategies for Dynamic On-Device LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 First ASLS
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized how we interact with technology, but their personalization to individual user preferences remains a significant challenge, particularly in on-device applications. Traditional methods often depend heavily on labeled datasets and can be resource-intensive. To address these issues, we present Adaptive Self-Supervised Learning Strategies (ASLS), which utilizes self-supervised learning techniques to personalize LLMs dynamically. The framework comprises a user profiling layer for collecting interaction data and a neural adaptation layer for real-time model fine-tuning. This innovative approach enables continuous learning from user feedback, allowing the model to generate responses that align closely with user-specific contexts. The adaptive mechanisms of ASLS minimize computational demands and enhance personalization efficiency. Experimental results across various user scenarios illustrate the superior performance of ASLS in boosting user engagement and satisfaction, highlighting its potential to redefine LLMs as highly responsive and context-aware systems on-device.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16949v1">DALDA: Data Augmentation Leveraging Diffusion Model and LLM with Adaptive Guidance Scaling</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 Accepted to ECCV Synthetic Data for Computer Vision Workshop (Oral)
    </div>
    <details class="paper-abstract">
      In this paper, we present an effective data augmentation framework leveraging the Large Language Model (LLM) and Diffusion Model (DM) to tackle the challenges inherent in data-scarce scenarios. Recently, DMs have opened up the possibility of generating synthetic images to complement a few training images. However, increasing the diversity of synthetic images also raises the risk of generating samples outside the target distribution. Our approach addresses this issue by embedding novel semantic information into text prompts via LLM and utilizing real images as visual prompts, thus generating semantically rich images. To ensure that the generated images remain within the target distribution, we dynamically adjust the guidance weight based on each image's CLIPScore to control the diversity. Experimental results show that our method produces synthetic images with enhanced diversity while maintaining adherence to the target distribution. Consequently, our approach proves to be more efficient in the few-shot setting on several benchmarks. Our code is available at https://github.com/kkyuhun94/dalda .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16914v1">Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 To appear at the main conference of EMNLP 2024
    </div>
    <details class="paper-abstract">
      The increasing capability and widespread usage of large language models (LLMs) highlight the desirability of automatic detection of LLM-generated text. Zero-shot detectors, due to their training-free nature, have received considerable attention and notable success. In this paper, we identify a new feature, token cohesiveness, that is useful for zero-shot detection, and we demonstrate that LLM-generated text tends to exhibit higher token cohesiveness than human-written text. Based on this observation, we devise TOCSIN, a generic dual-channel detection paradigm that uses token cohesiveness as a plug-and-play module to improve existing zero-shot detectors. To calculate token cohesiveness, TOCSIN only requires a few rounds of random token deletion and semantic difference measurement, making it particularly suitable for a practical black-box setting where the source model used for generation is not accessible. Extensive experiments with four state-of-the-art base detectors on various datasets, source models, and evaluation settings demonstrate the effectiveness and generality of the proposed approach. Code available at: \url{https://github.com/Shixuan-Ma/TOCSIN}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16900v1">A Roadmap for Embodied and Social Grounding in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 Accepted Version of a conference paper presented at Robophilosophy Conference 2024
    </div>
    <details class="paper-abstract">
      The fusion of Large Language Models (LLMs) and robotic systems has led to a transformative paradigm in the robotic field, offering unparalleled capabilities not only in the communication domain but also in skills like multimodal input handling, high-level reasoning, and plan generation. The grounding of LLMs knowledge into the empirical world has been considered a crucial pathway to exploit the efficiency of LLMs in robotics. Nevertheless, connecting LLMs' representations to the external world with multimodal approaches or with robots' bodies is not enough to let them understand the meaning of the language they are manipulating. Taking inspiration from humans, this work draws attention to three necessary elements for an agent to grasp and experience the world. The roadmap for LLMs grounding is envisaged in an active bodily system as the reference point for experiencing the environment, a temporally structured experience for a coherent, self-related interaction with the external world, and social skills to acquire a common-grounded shared experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16879v1">GRACE: Generating Socially Appropriate Robot Actions Leveraging LLMs and Human Explanations</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 Under review for 2025 IEEE International Conference on Robotics & Automation (ICRA), Supplementary video: https://youtu.be/3gP3euwNBjQ
    </div>
    <details class="paper-abstract">
      When operating in human environments, robots need to handle complex tasks while both adhering to social norms and accommodating individual preferences. For instance, based on common sense knowledge, a household robot can predict that it should avoid vacuuming during a social gathering, but it may still be uncertain whether it should vacuum before or after having guests. In such cases, integrating common-sense knowledge with human preferences, often conveyed through human explanations, is fundamental yet a challenge for existing systems. In this paper, we introduce GRACE, a novel approach addressing this while generating socially appropriate robot actions. GRACE leverages common sense knowledge from Large Language Models (LLMs), and it integrates this knowledge with human explanations through a generative network architecture. The bidirectional structure of GRACE enables robots to refine and enhance LLM predictions by utilizing human explanations and makes robots capable of generating such explanations for human-specified actions. Our experimental evaluations show that integrating human explanations boosts GRACE's performance, where it outperforms several baselines and provides sensible explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13781v2">GenOnet: Generative Open xG Network Simulation with Multi-Agent LLM and ns-3</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 3 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The move toward Sixth-Generation (6G) networks relies on open interfaces and protocols for seamless interoperability across devices, vendors, and technologies. In this context, open 6G development involves multiple disciplines and requires advanced simulation approaches for testing. In this demo paper, we propose a generative simulation approach based on a multi-agent Large Language Model (LLM) and Network Simulator 3 (ns-3), called Generative Open xG Network Simulation (GenOnet), to effectively generate, debug, execute, and interpret simulated Open Fifth-Generation (5G) environments. The first version of GenOnet application represents a specialized adaptation of the OpenAI GPT models. It incorporates supplementary tools, agents, 5G standards, and seamless integration with ns-3 simulation capabilities, supporting both C++ variants and Python implementations. This release complies with the latest Open Radio Access Network (O-RAN) and 3GPP standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05200v2">Are LLMs Ready for Real-World Materials Discovery?</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) create exciting possibilities for powerful language processing tools to accelerate research in materials science. While LLMs have great potential to accelerate materials understanding and discovery, they currently fall short in being practical materials science tools. In this position paper, we show relevant failure cases of LLMs in materials science that reveal current limitations of LLMs related to comprehending and reasoning over complex, interconnected materials science knowledge. Given those shortcomings, we outline a framework for developing Materials Science LLMs (MatSci-LLMs) that are grounded in materials science knowledge and hypothesis generation followed by hypothesis testing. The path to attaining performant MatSci-LLMs rests in large part on building high-quality, multi-modal datasets sourced from scientific literature where various information extraction challenges persist. As such, we describe key materials science information extraction challenges which need to be overcome in order to build large-scale, multi-modal datasets that capture valuable materials science knowledge. Finally, we outline a roadmap for applying future MatSci-LLMs for real-world materials discovery via: 1. Automated Knowledge Base Generation; 2. Automated In-Silico Material Design; and 3. MatSci-LLM Integrated Self-Driving Materials Laboratories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16813v1">PeerArg: Argumentative Peer Review with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      Peer review is an essential process to determine the quality of papers submitted to scientific conferences or journals. However, it is subjective and prone to biases. Several studies have been conducted to apply techniques from NLP to support peer review, but they are based on black-box techniques and their outputs are difficult to interpret and trust. In this paper, we propose a novel pipeline to support and understand the reviewing and decision-making processes of peer review: the PeerArg system combining LLMs with methods from knowledge representation. PeerArg takes in input a set of reviews for a paper and outputs the paper acceptance prediction. We evaluate the performance of the PeerArg pipeline on three different datasets, in comparison with a novel end-2-end LLM that uses few-shot learning to predict paper acceptance given reviews. The results indicate that the end-2-end LLM is capable of predicting paper acceptance from reviews, but a variant of the PeerArg pipeline outperforms this LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16739v1">Context-Enhanced LLM-Based Framework for Automatic Test Refactoring</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      Test smells arise from poor design practices and insufficient domain knowledge, which can lower the quality of test code and make it harder to maintain and update. Manually refactoring test smells is time-consuming and error-prone, highlighting the necessity for automated approaches. Current rule-based refactoring methods often struggle in scenarios not covered by predefined rules and lack the flexibility needed to handle diverse cases effectively. In this paper, we propose a novel approach called UTRefactor, a context-enhanced, LLM-based framework for automatic test refactoring in Java projects. UTRefactor extracts relevant context from test code and leverages an external knowledge base that includes test smell definitions, descriptions, and DSL-based refactoring rules. By simulating the manual refactoring process through a chain-of-thought approach, UTRefactor guides the LLM to eliminate test smells in a step-by-step process, ensuring both accuracy and consistency throughout the refactoring. Additionally, we implement a checkpoint mechanism to facilitate comprehensive refactoring, particularly when multiple smells are present. We evaluate UTRefactor on 879 tests from six open-source Java projects, reducing the number of test smells from 2,375 to 265, achieving an 89% reduction. UTRefactor outperforms direct LLM-based refactoring methods by 61.82% in smell elimination and significantly surpasses the performance of a rule-based test smell refactoring tool. Our results demonstrate the effectiveness of UTRefactor in enhancing test code quality while minimizing manual involvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00781v3">ChatDiet: Empowering Personalized Nutrition-Oriented Food Recommender Chatbots through an LLM-Augmented Framework</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 Published on Smart Health
    </div>
    <details class="paper-abstract">
      The profound impact of food on health necessitates advanced nutrition-oriented food recommendation services. Conventional methods often lack the crucial elements of personalization, explainability, and interactivity. While Large Language Models (LLMs) bring interpretability and explainability, their standalone use falls short of achieving true personalization. In this paper, we introduce ChatDiet, a novel LLM-powered framework designed specifically for personalized nutrition-oriented food recommendation chatbots. ChatDiet integrates personal and population models, complemented by an orchestrator, to seamlessly retrieve and process pertinent information. The personal model leverages causal discovery and inference techniques to assess personalized nutritional effects for a specific user, whereas the population model provides generalized information on food nutritional content. The orchestrator retrieves, synergizes and delivers the output of both models to the LLM, providing tailored food recommendations designed to support targeted health outcomes. The result is a dynamic delivery of personalized and explainable food recommendations, tailored to individual user preferences. Our evaluation of ChatDiet includes a compelling case study, where we establish a causal personal model to estimate individual nutrition effects. Our assessments, including a food recommendation test showcasing a 92\% effectiveness rate, coupled with illustrative dialogue examples, underscore ChatDiet's strengths in explainability, personalization, and interactivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16621v1">Entailment-Driven Privacy Policy Classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 8 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      While many online services provide privacy policies for end users to read and understand what personal data are being collected, these documents are often lengthy and complicated. As a result, the vast majority of users do not read them at all, leading to data collection under uninformed consent. Several attempts have been made to make privacy policies more user friendly by summarising them, providing automatic annotations or labels for key sections, or by offering chat interfaces to ask specific questions. With recent advances in Large Language Models (LLMs), there is an opportunity to develop more effective tools to parse privacy policies and help users make informed decisions. In this paper, we propose an entailment-driven LLM based framework to classify paragraphs of privacy policies into meaningful labels that are easily understood by users. The results demonstrate that our framework outperforms traditional LLM methods, improving the F1 score in average by 11.2%. Additionally, our framework provides inherently explainable and meaningful predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.02374v5">Conversational Health Agents: A Personalized LLM-Powered Agent Framework</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 23 pages, 6 figures, 2 tables, 4 appendices, journal paper
    </div>
    <details class="paper-abstract">
      Conversational Health Agents (CHAs) are interactive systems that provide healthcare services, such as assistance and diagnosis. Current CHAs, especially those utilizing Large Language Models (LLMs), primarily focus on conversation aspects. However, they offer limited agent capabilities, specifically lacking multi-step problem-solving, personalized conversations, and multimodal data analysis. Our aim is to overcome these limitations. We propose openCHA, an open-source LLM-powered framework, to empower conversational agents to generate a personalized response for users' healthcare queries. This framework enables developers to integrate external sources including data sources, knowledge bases, and analysis models, into their LLM-based solutions. openCHA includes an orchestrator to plan and execute actions for gathering information from external sources, essential for formulating responses to user inquiries. It facilitates knowledge acquisition, problem-solving capabilities, multilingual and multimodal conversations, and fosters interaction with various AI platforms. We illustrate the framework's proficiency in handling complex healthcare tasks via two demonstrations and four use cases. Moreover, we release openCHA as open source available to the community via GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16563v1">Enhancing disease detection in radiology reports through fine-tuning lightweight LLM on weak labels</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      Despite significant progress in applying large language models (LLMs) to the medical domain, several limitations still prevent them from practical applications. Among these are the constraints on model size and the lack of cohort-specific labeled datasets. In this work, we investigated the potential of improving a lightweight LLM, such as Llama 3.1-8B, through fine-tuning with datasets using synthetic labels. Two tasks are jointly trained by combining their respective instruction datasets. When the quality of the task-specific synthetic labels is relatively high (e.g., generated by GPT4- o), Llama 3.1-8B achieves satisfactory performance on the open-ended disease detection task, with a micro F1 score of 0.91. Conversely, when the quality of the task-relevant synthetic labels is relatively low (e.g., from the MIMIC-CXR dataset), fine-tuned Llama 3.1-8B is able to surpass its noisy teacher labels (micro F1 score of 0.67 v.s. 0.63) when calibrated against curated labels, indicating the strong inherent underlying capability of the model. These findings demonstrate the potential of fine-tuning LLMs with synthetic labels, offering a promising direction for future research on LLM specialization in the medical domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16560v1">Dynamic-Width Speculative Beam Decoding for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown outstanding performance across numerous real-world tasks. However, the autoregressive nature of these models makes the inference process slow and costly. Speculative decoding has emerged as a promising solution, leveraging a smaller auxiliary model to draft future tokens, which are then validated simultaneously by the larger model, achieving a speed-up of 1-2x. Although speculative decoding matches the same distribution as multinomial sampling, multinomial sampling itself is prone to suboptimal outputs, whereas beam sampling is widely recognized for producing higher-quality results by maintaining multiple candidate sequences at each step. This paper explores the novel integration of speculative decoding with beam sampling. However, there are four key challenges: (1) how to generate multiple sequences from the larger model's distribution given drafts sequences from the small model; (2) how to dynamically optimize the number of beams to balance efficiency and accuracy; (3) how to efficiently verify the multiple drafts in parallel; and (4) how to address the extra memory costs inherent in beam sampling. To address these challenges, we propose dynamic-width speculative beam decoding (DSBD). Specifically, we first introduce a novel draft and verification scheme that generates multiple sequences following the large model's distribution based on beam sampling trajectories from the small model. Then, we introduce an adaptive mechanism to dynamically tune the number of beams based on the context, optimizing efficiency and effectiveness. Besides, we extend tree-based parallel verification to handle multiple trees simultaneously, accelerating the verification process. Finally, we illustrate a simple modification to our algorithm to mitigate the memory overhead of beam sampling...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16559v1">Demystifying Issues, Causes and Solutions in LLM Open-Source Projects</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 22 pages, 2 images, 6 tables, Manuscript submitted to a journal (2024)
    </div>
    <details class="paper-abstract">
      With the advancements of Large Language Models (LLMs), an increasing number of open-source software projects are using LLMs as their core functional component. Although research and practice on LLMs are capturing considerable interest, no dedicated studies explored the challenges faced by practitioners of LLM open-source projects, the causes of these challenges, and potential solutions. To fill this research gap, we conducted an empirical study to understand the issues that practitioners encounter when developing and using LLM open-source software, the possible causes of these issues, and potential solutions.We collected all closed issues from 15 LLM open-source projects and labelled issues that met our requirements. We then randomly selected 994 issues from the labelled issues as the sample for data extraction and analysis to understand the prevalent issues, their underlying causes, and potential solutions. Our study results show that (1) Model Issue is the most common issue faced by practitioners, (2) Model Problem, Configuration and Connection Problem, and Feature and Method Problem are identified as the most frequent causes of the issues, and (3) Optimize Model is the predominant solution to the issues. Based on the study results, we provide implications for practitioners and researchers of LLM open-source projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01556v2">Benchmarking Cognitive Domains for LLMs: Insights from Taiwanese Hakka Culture</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 Accepted to O-COCOSDA 2024
    </div>
    <details class="paper-abstract">
      This study introduces a comprehensive benchmark designed to evaluate the performance of large language models (LLMs) in understanding and processing cultural knowledge, with a specific focus on Hakka culture as a case study. Leveraging Bloom's Taxonomy, the study develops a multi-dimensional framework that systematically assesses LLMs across six cognitive domains: Remembering, Understanding, Applying, Analyzing, Evaluating, and Creating. This benchmark extends beyond traditional single-dimensional evaluations by providing a deeper analysis of LLMs' abilities to handle culturally specific content, ranging from basic recall of facts to higher-order cognitive tasks such as creative synthesis. Additionally, the study integrates Retrieval-Augmented Generation (RAG) technology to address the challenges of minority cultural knowledge representation in LLMs, demonstrating how RAG enhances the models' performance by dynamically incorporating relevant external information. The results highlight the effectiveness of RAG in improving accuracy across all cognitive domains, particularly in tasks requiring precise retrieval and application of cultural knowledge. However, the findings also reveal the limitations of RAG in creative tasks, underscoring the need for further optimization. This benchmark provides a robust tool for evaluating and comparing LLMs in culturally diverse contexts, offering valuable insights for future research and development in AI-driven cultural knowledge preservation and dissemination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2409.15654v1">Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 15 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Deploying advanced large language models on edge devices, such as smartphones and robotics, is a growing trend that enhances user data privacy and network connectivity resilience while preserving intelligent capabilities. However, such a task exhibits single-batch computing with incredibly low arithmetic intensity, which poses the significant challenges of huge memory footprint and bandwidth demands on limited edge resources. To address these issues, we introduce Cambricon-LLM, a chiplet-based hybrid architecture with NPU and a dedicated NAND flash chip to enable efficient on-device inference of 70B LLMs. Such a hybrid architecture utilizes both the high computing capability of NPU and the data capacity of the NAND flash chip, with the proposed hardware-tiling strategy that minimizes the data movement overhead between NPU and NAND flash chip. Specifically, the NAND flash chip, enhanced by our innovative in-flash computing and on-die ECC techniques, excels at performing precise lightweight on-die processing. Simultaneously, the NPU collaborates with the flash chip for matrix operations and handles special function computations beyond the flash's on-die processing capabilities. Overall, Cambricon-LLM enables the on-device inference of 70B LLMs at a speed of 3.44 token/s, and 7B LLMs at a speed of 36.34 token/s, which is over 22X to 45X faster than existing flash-offloading technologies, showing the potentiality of deploying powerful LLMs in edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16221v2">Do LLMs Know When to NOT Answer? Investigating Abstention Abilities of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 8 pages (excluding limitations, references and appendix) and 5 figures
    </div>
    <details class="paper-abstract">
      Abstention Ability (AA) is a critical aspect of Large Language Model (LLM) reliability, referring to an LLM's capability to withhold responses when uncertain or lacking a definitive answer, without compromising performance. Although previous studies have attempted to improve AA, they lack a standardised evaluation method and remain unsuitable for black-box models where token prediction probabilities are inaccessible. This makes comparative analysis challenging, especially for state-of-the-art closed-source commercial LLMs. This paper bridges this gap by introducing a black-box evaluation approach and a new dataset, Abstain-QA, crafted to rigorously assess AA across varied question types (answerable and unanswerable), domains (well-represented and under-represented), and task types (fact centric and reasoning). We also propose a new confusion matrix, the ''Answerable-Unanswerable Confusion Matrix'' (AUCM) which serves as the basis for evaluating AA, by offering a structured and precise approach for assessment. Finally, we explore the impact of three prompting strategies-Strict Prompting, Verbal Confidence Thresholding, and Chain-of-Thought (CoT)-on improving AA. Our results indicate that even powerful models like GPT-4, Mixtral 8x22b encounter difficulties with abstention; however, strategic approaches such as Strict prompting and CoT can enhance this capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18807v1">LLM With Tools: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The integration of tools in augmenting large language models presents a novel approach toward enhancing the efficiency and accuracy of these models in handling specific, complex tasks. This paper delves into the methodology,challenges, and developments in the realm of teaching LLMs to use external tools, thereby pushing the boundaries of their capabilities beyond pre-existing knowledge bases. We introduce a standardized paradigm for tool integration guided by a series of functions that map user instructions to actionable plans and their execution, emphasizing the significance of understanding user intent, tool selection, and dynamic plan adjustment. Our exploration reveals the various challenges encountered, such as tool invocation timing, selection accuracy, and the need for robust reasoning processes. In addressing these challenges, we investigate techniques within the context of fine-tuning and incontext learning paradigms, highlighting innovative approaches to ensure diversity, augment datasets, and improve generalization.Furthermore, we investigate a perspective on enabling LLMs to not only utilize but also autonomously create tools, which may redefine their role from mere tool users to tool creators. Finally,we reproduced Chameleon's results on ScienceQA and analyzed the code structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01833v2">Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03175v2">Beyond the Black Box: A Statistical Model for LLM Reasoning and Inference</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      This paper introduces a novel Bayesian learning model to explain the behavior of Large Language Models (LLMs), focusing on their core optimization metric of next token prediction. We develop a theoretical framework based on an ideal generative text model represented by a multinomial transition probability matrix with a prior, and examine how LLMs approximate this matrix. Key contributions include: (i) a continuity theorem relating embeddings to multinomial distributions, (ii) a demonstration that LLM text generation aligns with Bayesian learning principles, (iii) an explanation for the emergence of in-context learning in larger models, (iv) empirical validation using visualizations of next token probabilities from an instrumented Llama model Our findings provide new insights into LLM functioning, offering a statistical foundation for understanding their capabilities and limitations. This framework has implications for LLM design, training, and application, potentially guiding future developments in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15491v2">Open Conversational LLMs do not know most Spanish words</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 Procesamiento del Lenguaje Natural, 73, 95-108
    </div>
    <details class="paper-abstract">
      The growing interest in Large Language Models (LLMs) and in particular in conversational models with which users can interact has led to the development of a large number of open-source chat LLMs. These models are evaluated on a wide range of benchmarks to assess their capabilities in answering questions or solving problems on almost any possible topic or to test their ability to reason or interpret texts. Instead, the evaluation of the knowledge that these models have of the languages has received much less attention. For example, the words that they can recognize and use in different languages. In this paper, we evaluate the knowledge that open-source chat LLMs have of Spanish words by testing a sample of words in a reference dictionary. The results show that open-source chat LLMs produce incorrect meanings for an important fraction of the words and are not able to use most of the words correctly to write sentences with context. These results show how Spanish is left behind in the open-source LLM race and highlight the need to push for linguistic fairness in conversational LLMs ensuring that they provide similar performance across languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16005v1">Bridging Speech and Text: Enhancing ASR with Pinyin-to-Character Pre-training in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 Accepted by ISCSLP2024-Special session-Speech Processing in LLM Era
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) with pre-trained speech models has opened up new avenues in automatic speech recognition (ASR). While LLMs excel in multimodal understanding tasks, effectively leveraging their capabilities for ASR remains a significant challenge. This paper presents a novel training approach to enhance LLM performance in ASR tasks. We propose pre-training LLMs on Pinyin embedding sequences, which represent pronunciation features, to generate corresponding Chinese characters. This step enables the LLM to adapt to generating text from pronunciation features before encountering real speech data. Furthermore, we fine-tune the LoRA parameters to enhance the LLM's understanding of speech modality information. In AISHELL-1 corpus, our approach yields a 9.5% relative improvement in ASR tasks compared to the baseline without Pinyi-to-Character pre-training. Additionally, incorporating auxiliary text data for Pinyi-to-Character pre-training further boosts performance, achieving a 19.0% relative improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15979v1">Finetuning LLMs for Comparative Assessment Tasks</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 8 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Automated assessment in natural language generation is a challenging task. Instruction-tuned large language models (LLMs) have shown promise in reference-free evaluation, particularly through comparative assessment. However, the quadratic computational complexity of pairwise comparisons limits its scalability. To address this, efficient comparative assessment has been explored by applying comparative strategies on zero-shot LLM probabilities. We propose a framework for finetuning LLMs for comparative assessment to align the model's output with the target distribution of comparative probabilities. By training on soft probabilities, our approach improves state-of-the-art performance while maintaining high performance with an efficient subset of comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10553v2">"Flipped" University: LLM-Assisted Lifelong Learning Environment</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 Pre-print version, accepted for 31st International Conference on Neural Information Processing (ICONIP2024), 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The rapid development of artificial intelligence technologies, particularly Large Language Models (LLMs), has revolutionized the landscape of lifelong learning. This paper introduces a conceptual framework for a self-constructed lifelong learning environment supported by LLMs. It highlights the inadequacies of traditional education systems in keeping pace with the rapid deactualization of knowledge and skills. The proposed framework emphasizes the transformation from institutionalized education to personalized, self-driven learning. It leverages the natural language capabilities of LLMs to provide dynamic and adaptive learning experiences, facilitating the creation of personal intellectual agents that assist in knowledge acquisition. The framework integrates principles of lifelong learning, including the necessity of building personal world models, the dual modes of learning (training and exploration), and the creation of reusable learning artifacts. Additionally, it underscores the importance of curiosity-driven learning and reflective practices in maintaining an effective learning trajectory. The paper envisions the evolution of educational institutions into "flipped" universities, focusing on supporting global knowledge consistency rather than merely structuring and transmitting knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15915v1">Planning in the Dark: LLM-Symbolic Planning Pipeline without Experts</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 8 main body pages, 10 appendix pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in solving natural language-described planning tasks, but their direct use often leads to inconsistent reasoning and hallucination. While hybrid LLM-symbolic planning pipelines have emerged as a more robust alternative, they typically require extensive expert intervention to refine and validate generated action schemas. It not only limits scalability but also introduces a potential for biased interpretation, as a single expert's interpretation of ambiguous natural language descriptions might not align with the user's actual intent. To address this, we propose a novel approach that constructs an action schema library to generate multiple candidates, accounting for the diverse possible interpretations of natural language descriptions. We further introduce a semantic validation and ranking module that automatically filter and rank the generated schemas and plans without expert-in-the-loop. The experiments showed our pipeline maintains superiority in planning over the direct LLM planning approach. These findings demonstrate the feasibility of a fully automated end-to-end LLM-symbolic planner that requires no expert intervention, opening up the possibility for a broader audience to engage with AI planning with less prerequisite of domain expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15890v1">HLB: Benchmarking LLMs' Humanlikeness in Language Use</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      As synthetic data becomes increasingly prevalent in training language models, particularly through generated dialogue, concerns have emerged that these models may deviate from authentic human language patterns, potentially losing the richness and creativity inherent in human communication. This highlights the critical need to assess the humanlikeness of language models in real-world language use. In this paper, we present a comprehensive humanlikeness benchmark (HLB) evaluating 20 large language models (LLMs) using 10 psycholinguistic experiments designed to probe core linguistic aspects, including sound, word, syntax, semantics, and discourse (see https://huggingface.co/spaces/XufengDuan/HumanLikeness). To anchor these comparisons, we collected responses from over 2,000 human participants and compared them to outputs from the LLMs in these experiments. For rigorous evaluation, we developed a coding algorithm that accurately identified language use patterns, enabling the extraction of response distributions for each task. By comparing the response distributions between human participants and LLMs, we quantified humanlikeness through distributional similarity. Our results reveal fine-grained differences in how well LLMs replicate human responses across various linguistic levels. Importantly, we found that improvements in other performance metrics did not necessarily lead to greater humanlikeness, and in some cases, even resulted in a decline. By introducing psycholinguistic methods to model evaluation, this benchmark offers the first framework for systematically assessing the humanlikeness of LLMs in language use.
    </details>
</div>
