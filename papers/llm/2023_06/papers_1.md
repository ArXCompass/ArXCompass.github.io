# llm - 2023_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.13911v2">Federated Prompting and Chain-of-Thought Reasoning for Improving LLMs Answering</a></div>
    <div class="paper-meta">
      📅 2023-06-30
    </div>
    <details class="paper-abstract">
      We investigate how to enhance answer precision in frequently asked questions posed by distributed users using cloud-based Large Language Models (LLMs). Our study focuses on a typical situations where users ask similar queries that involve identical mathematical reasoning steps and problem-solving procedures. Due to the unsatisfactory accuracy of LLMs' zero-shot prompting with standalone questions, we propose to improve the distributed synonymous questions using Self-Consistency (SC) and Chain-of-Thought (CoT) techniques. Specifically, we first retrieve synonymous questions from a crowd-sourced database and create a federated question pool. We call these federated synonymous questions with the same or different parameters SP-questions or DP-questions, respectively. We refer to our methods as Fed-SP-SC and Fed-DP-CoT, which can generate significantly more accurate answers for all user queries without requiring sophisticated model-tuning. Through extensive experiments, we demonstrate that our proposed methods can significantly enhance question accuracy by fully exploring the synonymous nature of the questions and the consistency of the answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.16931v1">UMASS_BioNLP at MEDIQA-Chat 2023: Can LLMs generate high-quality synthetic note-oriented doctor-patient conversations?</a></div>
    <div class="paper-meta">
      📅 2023-06-29
    </div>
    <details class="paper-abstract">
      This paper presents UMASS_BioNLP team participation in the MEDIQA-Chat 2023 shared task for Task-A and Task-C. We focus especially on Task-C and propose a novel LLMs cooperation system named a doctor-patient loop to generate high-quality conversation data sets. The experiment results demonstrate that our approaches yield reasonable performance as evaluated by automatic metrics such as ROUGE, medical concept recall, BLEU, and Self-BLEU. Furthermore, we conducted a comparative analysis between our proposed method and ChatGPT and GPT-4. This analysis also investigates the potential of utilizing cooperation LLMs to generate high-quality datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.14077v1">Full Automation of Goal-driven LLM Dialog Threads with And-Or Recursors and Refiner Oracles</a></div>
    <div class="paper-meta">
      📅 2023-06-24
      | 💬 23 pages, 1 figure, more information at https://github.com/ptarau/recursors
    </div>
    <details class="paper-abstract">
      We automate deep step-by step reasoning in an LLM dialog thread by recursively exploring alternatives (OR-nodes) and expanding details (AND-nodes) up to a given depth. Starting from a single succinct task-specific initiator we steer the automated dialog thread to stay focussed on the task by synthesizing a prompt that summarizes the depth-first steps taken so far. Our algorithm is derived from a simple recursive descent implementation of a Horn Clause interpreter, except that we accommodate our logic engine to fit the natural language reasoning patterns LLMs have been trained on. Semantic similarity to ground-truth facts or oracle advice from another LLM instance is used to restrict the search space and validate the traces of justification steps returned as answers. At the end, the unique minimal model of a generated Horn Clause program collects the results of the reasoning process. As applications, we sketch implementations of consequence predictions, causal explanations, recommendation systems and topic-focussed exploration of scientific literature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.14924v1">LLM-Assisted Content Analysis: Using Large Language Models to Support Deductive Coding</a></div>
    <div class="paper-meta">
      📅 2023-06-23
    </div>
    <details class="paper-abstract">
      Deductive coding is a widely used qualitative research method for determining the prevalence of themes across documents. While useful, deductive coding is often burdensome and time consuming since it requires researchers to read, interpret, and reliably categorize a large body of unstructured text documents. Large language models (LLMs), like ChatGPT, are a class of quickly evolving AI tools that can perform a range of natural language processing and reasoning tasks. In this study, we explore the use of LLMs to reduce the time it takes for deductive coding while retaining the flexibility of a traditional content analysis. We outline the proposed approach, called LLM-assisted content analysis (LACA), along with an in-depth case study using GPT-3.5 for LACA on a publicly available deductive coding data set. Additionally, we conduct an empirical benchmark using LACA on 4 publicly available data sets to assess the broader question of how well GPT-3.5 performs across a range of deductive coding tasks. Overall, we find that GPT-3.5 can often perform deductive coding at levels of agreement comparable to human coders. Additionally, we demonstrate that LACA can help refine prompts for deductive coding, identify codes for which an LLM is randomly guessing, and help assess when to use LLMs vs. human coders for deductive coding. We conclude with several implications for future practice of deductive coding and related research methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.13781v1">Retrieving Supporting Evidence for LLMs Generated Answers</a></div>
    <div class="paper-meta">
      📅 2023-06-23
    </div>
    <details class="paper-abstract">
      Current large language models (LLMs) can exhibit near-human levels of performance on many natural language tasks, including open-domain question answering. Unfortunately, they also convincingly hallucinate incorrect answers, so that responses to questions must be verified against external sources before they can be accepted at face value. In this paper, we report a simple experiment to automatically verify generated answers against a corpus. After presenting a question to an LLM and receiving a generated answer, we query the corpus with the combination of the question + generated answer. We then present the LLM with the combination of the question + generated answer + retrieved answer, prompting it to indicate if the generated answer can be supported by the retrieved answer. We base our experiment on questions and passages from the MS MARCO (V1) test collection, exploring three retrieval approaches ranging from standard BM25 to a full question answering stack, including a reader based on the LLM. For a large fraction of questions, we find that an LLM is capable of verifying its generated answer if appropriate supporting material is provided. However, with an accuracy of 70-80%, this approach cannot be fully relied upon to detect hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.13304v1">ToolQA: A Dataset for LLM Question Answering with External Tools</a></div>
    <div class="paper-meta">
      📅 2023-06-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance in various NLP tasks, but they still suffer from challenges such as hallucination and weak numerical reasoning. To overcome these challenges, external tools can be used to enhance LLMs' question-answering abilities. However, current evaluation methods do not distinguish between questions that can be answered using LLMs' internal knowledge and those that require external information through tool use. To address this issue, we introduce a new dataset called ToolQA, which is designed to faithfully evaluate LLMs' ability to use external tools for question answering. Our development of ToolQA involved a scalable, automated process for dataset curation, along with 13 specialized tools designed for interaction with external knowledge in order to answer questions. Importantly, we strive to minimize the overlap between our benchmark data and LLMs' pre-training data, enabling a more precise evaluation of LLMs' tool-use reasoning abilities. We conducted an in-depth diagnosis of existing tool-use LLMs to highlight their strengths, weaknesses, and potential improvements. Our findings set a new benchmark for evaluating LLMs and suggest new directions for future advancements. Our data and code are freely available to the broader scientific community on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.13298v1">Exploring Qualitative Research Using LLMs</a></div>
    <div class="paper-meta">
      📅 2023-06-23
    </div>
    <details class="paper-abstract">
      The advent of AI driven large language models (LLMs) have stirred discussions about their role in qualitative research. Some view these as tools to enrich human understanding, while others perceive them as threats to the core values of the discipline. This study aimed to compare and contrast the comprehension capabilities of humans and LLMs. We conducted an experiment with small sample of Alexa app reviews, initially classified by a human analyst. LLMs were then asked to classify these reviews and provide the reasoning behind each classification. We compared the results with human classification and reasoning. The research indicated a significant alignment between human and ChatGPT 3.5 classifications in one third of cases, and a slightly lower alignment with GPT4 in over a quarter of cases. The two AI models showed a higher alignment, observed in more than half of the instances. However, a consensus across all three methods was seen only in about one fifth of the classifications. In the comparison of human and LLMs reasoning, it appears that human analysts lean heavily on their individual experiences. As expected, LLMs, on the other hand, base their reasoning on the specific word choices found in app reviews and the functional components of the app itself. Our results highlight the potential for effective human LLM collaboration, suggesting a synergistic rather than competitive relationship. Researchers must continuously evaluate LLMs role in their work, thereby fostering a future where AI and humans jointly enrich qualitative research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11932v1">Opportunities and Risks of LLMs for Scalable Deliberation with Polis</a></div>
    <div class="paper-meta">
      📅 2023-06-20
      | 💬 31 pages (main body; 45 with Bibliography and Appendix), 6 figures
    </div>
    <details class="paper-abstract">
      Polis is a platform that leverages machine intelligence to scale up deliberative processes. In this paper, we explore the opportunities and risks associated with applying Large Language Models (LLMs) towards challenges with facilitating, moderating and summarizing the results of Polis engagements. In particular, we demonstrate with pilot experiments using Anthropic's Claude that LLMs can indeed augment human intelligence to help more efficiently run Polis conversations. In particular, we find that summarization capabilities enable categorically new methods with immense promise to empower the public in collective meaning-making exercises. And notably, LLM context limitations have a significant impact on insight and quality of these results. However, these opportunities come with risks. We discuss some of these risks, as well as principles and techniques for characterizing and mitigating them, and the implications for other deliberative or political systems that may employ LLMs. Finally, we conclude with several open future research directions for augmenting tools like Polis with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11593v1">Improving Image Captioning Descriptiveness by Ranking and LLM-based Fusion</a></div>
    <div class="paper-meta">
      📅 2023-06-20
    </div>
    <details class="paper-abstract">
      State-of-The-Art (SoTA) image captioning models often rely on the Microsoft COCO (MS-COCO) dataset for training. This dataset contains annotations provided by human annotators, who typically produce captions averaging around ten tokens. However, this constraint presents a challenge in effectively capturing complex scenes and conveying detailed information. Furthermore, captioning models tend to exhibit bias towards the ``average'' caption, which captures only the more general aspects. What would happen if we were able to automatically generate longer captions, thereby making them more detailed? Would these captions, evaluated by humans, be more or less representative of the image content compared to the original MS-COCO captions? In this paper, we present a novel approach to address previous challenges by showcasing how captions generated from different SoTA models can be effectively fused, resulting in richer captions. Our proposed method leverages existing models from the literature, eliminating the need for additional training. Instead, it utilizes an image-text based metric to rank the captions generated by SoTA models for a given image. Subsequently, the top two captions are fused using a Large Language Model (LLM). Experimental results demonstrate the effectiveness of our approach, as the captions generated by our model exhibit higher consistency with human judgment when evaluated on the MS-COCO test set. By combining the strengths of various SoTA models, our method enhances the quality and appeal of image captions, bridging the gap between automated systems and the rich, informative nature of human-generated descriptions. This advance opens up new possibilities for generating captions that are more suitable for the training of both vision-language and captioning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14449v3">Graph Meets LLM: A Novel Approach to Collaborative Filtering for Robust Conversational Understanding</a></div>
    <div class="paper-meta">
      📅 2023-06-19
    </div>
    <details class="paper-abstract">
      Conversational AI systems such as Alexa need to understand defective queries to ensure robust conversational understanding and reduce user friction. These defective queries often arise from user ambiguities, mistakes, or errors in automatic speech recognition (ASR) and natural language understanding (NLU). Personalized query rewriting is an approach that focuses on reducing defects in queries by taking into account the user's individual behavior and preferences. It typically relies on an index of past successful user interactions with the conversational AI. However, unseen interactions within the user's history present additional challenges for personalized query rewriting. This paper presents our "Collaborative Query Rewriting" approach, which specifically addresses the task of rewriting new user interactions that have not been previously observed in the user's history. This approach builds a "User Feedback Interaction Graph" (FIG) of historical user-entity interactions and leverages multi-hop graph traversal to enrich each user's index to cover future unseen defective queries. The enriched user index is called a Collaborative User Index and contains hundreds of additional entries. To counteract precision degradation from the enlarged index, we add additional transformer layers to the L1 retrieval model and incorporate graph-based and guardrail features into the L2 ranking model. Since the user index can be pre-computed, we further investigate the utilization of a Large Language Model (LLM) to enhance the FIG for user-entity link prediction in the Video/Music domains. Specifically, this paper investigates the Dolly-V2 7B model. We found that the user index augmented by the fine-tuned Dolly-V2 generation significantly enhanced the coverage of future unseen user interactions, thereby boosting QR performance on unseen queries compared with the graph traversal only approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11025v1">Temporal Data Meets LLM -- Explainable Financial Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2023-06-19
    </div>
    <details class="paper-abstract">
      This paper presents a novel study on harnessing Large Language Models' (LLMs) outstanding knowledge and reasoning abilities for explainable financial time series forecasting. The application of machine learning models to financial time series comes with several challenges, including the difficulty in cross-sequence reasoning and inference, the hurdle of incorporating multi-modal signals from historical news, financial knowledge graphs, etc., and the issue of interpreting and explaining the model results. In this paper, we focus on NASDAQ-100 stocks, making use of publicly accessible historical stock price data, company metadata, and historical economic/financial news. We conduct experiments to illustrate the potential of LLMs in offering a unified solution to the aforementioned challenges. Our experiments include trying zero-shot/few-shot inference with GPT-4 and instruction-based fine-tuning with a public LLM model Open LLaMA. We demonstrate our approach outperforms a few baselines, including the widely applied classic ARMA-GARCH model and a gradient-boosting tree model. Through the performance comparison results and a few examples, we find LLMs can make a well-thought decision by reasoning over information from both textual news and price time series and extracting insights, leveraging cross-sequence information, and utilizing the inherent knowledge embedded within the LLM. Additionally, we show that a publicly available LLM such as Open-LLaMA, after fine-tuning, can comprehend the instruction to generate explainable forecasts and achieve reasonable performance, albeit relatively inferior in comparison to GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.10765v1">Path to Medical AGI: Unify Domain-specific Medical LLMs with the Lowest Cost</a></div>
    <div class="paper-meta">
      📅 2023-06-19
    </div>
    <details class="paper-abstract">
      Medical artificial general intelligence (AGI) is an emerging field that aims to develop systems specifically designed for medical applications that possess the ability to understand, learn, and apply knowledge across a wide range of tasks and domains. Large language models (LLMs) represent a significant step towards AGI. However, training cross-domain LLMs in the medical field poses significant challenges primarily attributed to the requirement of collecting data from diverse domains. This task becomes particularly difficult due to privacy restrictions and the scarcity of publicly available medical datasets. Here, we propose Medical AGI (MedAGI), a paradigm to unify domain-specific medical LLMs with the lowest cost, and suggest a possible path to achieve medical AGI. With an increasing number of domain-specific professional multimodal LLMs in the medical field being developed, MedAGI is designed to automatically select appropriate medical models by analyzing users' questions with our novel adaptive expert selection algorithm. It offers a unified approach to existing LLMs in the medical field, eliminating the need for retraining regardless of the introduction of new models. This characteristic renders it a future-proof solution in the dynamically advancing medical domain. To showcase the resilience of MedAGI, we conducted an evaluation across three distinct medical domains: dermatology diagnosis, X-ray diagnosis, and analysis of pathology pictures. The results demonstrated that MedAGI exhibited remarkable versatility and scalability, delivering exceptional performance across diverse domains. Our code is publicly available to facilitate further research at https://github.com/JoshuaChou2018/MedAGI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.14910v1">The Importance of Human-Labeled Data in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-06-18
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has brought about a revolution in the development of tailored machine learning models and sparked debates on redefining data requirements. The automation facilitated by the training and implementation of LLMs has led to discussions and aspirations that human-level labeling interventions may no longer hold the same level of importance as in the era of supervised learning. This paper presents compelling arguments supporting the ongoing relevance of human-labeled data in the era of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.08871v1">Med-MMHL: A Multi-Modal Dataset for Detecting Human- and LLM-Generated Misinformation in the Medical Domain</a></div>
    <div class="paper-meta">
      📅 2023-06-15
    </div>
    <details class="paper-abstract">
      The pervasive influence of misinformation has far-reaching and detrimental effects on both individuals and society. The COVID-19 pandemic has witnessed an alarming surge in the dissemination of medical misinformation. However, existing datasets pertaining to misinformation predominantly focus on textual information, neglecting the inclusion of visual elements, and tend to center solely on COVID-19-related misinformation, overlooking misinformation surrounding other diseases. Furthermore, the potential of Large Language Models (LLMs), such as the ChatGPT developed in late 2022, in generating misinformation has been overlooked in previous works. To overcome these limitations, we present Med-MMHL, a novel multi-modal misinformation detection dataset in a general medical domain encompassing multiple diseases. Med-MMHL not only incorporates human-generated misinformation but also includes misinformation generated by LLMs like ChatGPT. Our dataset aims to facilitate comprehensive research and development of methodologies for detecting misinformation across diverse diseases and various scenarios, including human and LLM-generated misinformation detection at the sentence, document, and multi-modal levels. To access our dataset and code, visit our GitHub repository: \url{https://github.com/styxsys0927/Med-MMHL}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.01248v2">How Ready are Pre-trained Abstractive Models and LLMs for Legal Case Judgement Summarization?</a></div>
    <div class="paper-meta">
      📅 2023-06-14
      | 💬 Accepted for presentation at the 3rd Workshop on Artificial Intelligence and Intelligent Assistance for Legal Professionals in the Digital Workplace (LegalAIIA 2023), co-located with the ICAIL 2023 conference
    </div>
    <details class="paper-abstract">
      Automatic summarization of legal case judgements has traditionally been attempted by using extractive summarization methods. However, in recent years, abstractive summarization models are gaining popularity since they can generate more natural and coherent summaries. Legal domain-specific pre-trained abstractive summarization models are now available. Moreover, general-domain pre-trained Large Language Models (LLMs), such as ChatGPT, are known to generate high-quality text and have the capacity for text summarization. Hence it is natural to ask if these models are ready for off-the-shelf application to automatically generate abstractive summaries for case judgements. To explore this question, we apply several state-of-the-art domain-specific abstractive summarization models and general-domain LLMs on Indian court case judgements, and check the quality of the generated summaries. In addition to standard metrics for summary quality, we check for inconsistencies and hallucinations in the summaries. We see that abstractive summarization models generally achieve slightly higher scores than extractive models in terms of standard summary evaluation metrics such as ROUGE and BLEU. However, we often find inconsistent or hallucinated information in the generated abstractive summaries. Overall, our investigation indicates that the pre-trained abstractive summarization models and LLMs are not yet ready for fully automatic deployment for case judgement summarization; rather a human-in-the-loop approach including manual checks for inconsistencies is more suitable at present.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.09349v4">LLM as A Robotic Brain: Unifying Egocentric Memory and Control</a></div>
    <div class="paper-meta">
      📅 2023-06-12
      | 💬 This early project is now integrated to: Mindstorms in Natural Language-Based Societies of Mind, arXiv:2305.17066
    </div>
    <details class="paper-abstract">
      Embodied AI focuses on the study and development of intelligent systems that possess a physical or virtual embodiment (i.e. robots) and are able to dynamically interact with their environment. Memory and control are the two essential parts of an embodied system and usually require separate frameworks to model each of them. In this paper, we propose a novel and generalizable framework called LLM-Brain: using Large-scale Language Model as a robotic brain to unify egocentric memory and control. The LLM-Brain framework integrates multiple multimodal language models for robotic tasks, utilizing a zero-shot learning approach. All components within LLM-Brain communicate using natural language in closed-loop multi-round dialogues that encompass perception, planning, control, and memory. The core of the system is an embodied LLM to maintain egocentric memory and control the robot. We demonstrate LLM-Brain by examining two downstream tasks: active exploration and embodied question answering. The active exploration tasks require the robot to extensively explore an unknown environment within a limited number of actions. Meanwhile, the embodied question answering tasks necessitate that the robot answers questions based on observations acquired during prior explorations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.07032v1">Mitigating Prior Errors in Causal Structure Learning: Towards LLM driven Prior Knowledge</a></div>
    <div class="paper-meta">
      📅 2023-06-12
      | 💬 14 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Causal structure learning, a prominent technique for encoding cause and effect relationships among variables, through Bayesian Networks (BNs). Merely recovering causal structures from real-world observed data lacks precision, while the development of Large Language Models (LLM) is opening a new frontier of causality. LLM presents strong capability in discovering causal relationships between variables with the "text" inputs defining the investigated variables, leading to a potential new hierarchy and new ladder of causality. We aim an critical issue in the emerging topic of LLM based causal structure learning, to tackle erroneous prior causal statements from LLM, which is seldom considered in the current context of expert dominating prior resources. As a pioneer attempt, we propose a BN learning strategy resilient to prior errors without need of human intervention. Focusing on the edge-level prior, we classify the possible prior errors into three types: order-consistent, order-reversed, and irrelevant, and provide their theoretical impact on the Structural Hamming Distance (SHD) under the presumption of sufficient data. Intriguingly, we discover and prove that only the order-reversed error contributes to an increase in a unique acyclic closed structure, defined as a "quasi-circle". Leveraging this insight, a post-hoc strategy is employed to identify the order-reversed prior error by its impact on the increment of "quasi-circles". Through empirical evaluation on both real and synthetic datasets, we demonstrate our strategy's robustness against prior errors. Specifically, we highlight its substantial ability to resist order-reversed errors while maintaining the majority of correct prior knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.06923v1">On the Viability of using LLMs for SW/HW Co-Design: An Example in Designing CiM DNN Accelerators</a></div>
    <div class="paper-meta">
      📅 2023-06-12
    </div>
    <details class="paper-abstract">
      Deep Neural Networks (DNNs) have demonstrated impressive performance across a wide range of tasks. However, deploying DNNs on edge devices poses significant challenges due to stringent power and computational budgets. An effective solution to this issue is software-hardware (SW-HW) co-design, which allows for the tailored creation of DNN models and hardware architectures that optimally utilize available resources. However, SW-HW co-design traditionally suffers from slow optimization speeds because their optimizers do not make use of heuristic knowledge, also known as the ``cold start'' problem. In this study, we present a novel approach that leverages Large Language Models (LLMs) to address this issue. By utilizing the abundant knowledge of pre-trained LLMs in the co-design optimization process, we effectively bypass the cold start problem, substantially accelerating the design process. The proposed method achieves a significant speedup of 25x. This advancement paves the way for the rapid and efficient deployment of DNNs on edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.06297v1">Protect Your Prompts: Protocols for IP Protection in LLM Applications</a></div>
    <div class="paper-meta">
      📅 2023-06-09
      | 💬 5 pages, 2 figures
    </div>
    <details class="paper-abstract">
      With the rapid adoption of AI in the form of large language models (LLMs), the potential value of carefully engineered prompts has become significant. However, to realize this potential, prompts should be tradable on an open market. Since prompts are, at present, generally economically non-excludable, by virtue of their nature as text, no general competitive market has yet been established. This note discusses two protocols intended to provide protection of prompts, elevating their status as intellectual property, thus confirming the intellectual property rights of prompt engineers, and potentially supporting the flourishing of an open market for LLM prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.06085v1">Trapping LLM Hallucinations Using Tagged Context Prompts</a></div>
    <div class="paper-meta">
      📅 2023-06-09
      | 💬 13 pages, 3 Figures, 2 Tables
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs), such as ChatGPT, have led to highly sophisticated conversation agents. However, these models suffer from "hallucinations," where the model generates false or fabricated information. Addressing this challenge is crucial, particularly with AI-driven platforms being adopted across various sectors. In this paper, we propose a novel method to recognize and flag instances when LLMs perform outside their domain knowledge, and ensuring users receive accurate information. We find that the use of context combined with embedded tags can successfully combat hallucinations within generative language models. To do this, we baseline hallucination frequency in no-context prompt-response pairs using generated URLs as easily-tested indicators of fabricated data. We observed a significant reduction in overall hallucination when context was supplied along with question prompts for tested generative engines. Lastly, we evaluated how placing tags within contexts impacted model responses and were able to eliminate hallucinations in responses with 98.88% effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.05827v1">Towards the Exploitation of LLM-based Chatbot for Providing Legal Support to Palestinian Cooperatives</a></div>
    <div class="paper-meta">
      📅 2023-06-09
    </div>
    <details class="paper-abstract">
      With the ever-increasing utilization of natural language processing (NLP), we started to witness over the past few years a significant transformation in our interaction with legal texts. This technology has advanced the analysis and enhanced the understanding of complex legal terminology and contexts. The development of recent large language models (LLMs), particularly ChatGPT, has also introduced a revolutionary contribution to the way that legal texts can be processed and comprehended. In this paper, we present our work on a cooperative-legal question-answering LLM-based chatbot, where we developed a set of legal questions about Palestinian cooperatives, associated with their regulations and compared the auto-generated answers by the chatbot to their correspondences that are designed by a legal expert. To evaluate the proposed chatbot, we have used 50 queries generated by the legal expert and compared the answers produced by the chart to their relevance judgments. Finding demonstrated that an overall accuracy rate of 82% has been achieved when answering the queries, while exhibiting an F1 score equivalent to 79%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.07944v1">Speech-to-Text Adapter and Speech-to-Entity Retriever Augmented LLMs for Speech Understanding</a></div>
    <div class="paper-meta">
      📅 2023-06-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been applied in the speech domain, often incurring a performance drop due to misaligned between speech and language representations. To bridge this gap, we propose a joint speech and language model (SLM) using a Speech2Text adapter, which maps speech into text token embedding space without speech information loss. Additionally, using a CTC-based blank-filtering, we can reduce the speech sequence length to that of text. In speech MultiWoz dataset (DSTC11 challenge), SLM largely improves the dialog state tracking (DST) performance (24.7% to 28.4% accuracy). Further to address errors on rare entities, we augment SLM with a Speech2Entity retriever, which uses speech to retrieve relevant entities, and then adds them to the original SLM input as a prefix. With this retrieval-augmented SLM (ReSLM), the DST performance jumps to 34.6% accuracy. Moreover, augmenting the ASR task with the dialog understanding task improves the ASR performance from 9.4% to 8.5% WER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.07622v3">PALR: Personalization Aware LLMs for Recommendation</a></div>
    <div class="paper-meta">
      📅 2023-06-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently received significant attention for their exceptional capabilities. Despite extensive efforts in developing general-purpose LLMs that can be utilized in various natural language processing (NLP) tasks, there has been less research exploring their potential in recommender systems. In this paper, we propose a novel framework, named PALR, which aiming to combine user history behaviors (such as clicks, purchases, ratings, etc.) with LLMs to generate user preferred items. Specifically, we first use user/item interactions as guidance for candidate retrieval. Then we adopt a LLM-based ranking model to generate recommended items. Unlike existing approaches that typically adopt general-purpose LLMs for zero/few-shot recommendation testing or training on small-sized language models (with less than 1 billion parameters), which cannot fully elicit LLMs' reasoning abilities and leverage rich item side parametric knowledge, we fine-tune a 7 billion parameters LLM for the ranking purpose. This model takes retrieval candidates in natural language format as input, with instruction which explicitly asking to select results from input candidates during inference. Our experimental results demonstrate that our solution outperforms state-of-the-art models on various sequential recommendation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.03901v2">ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory</a></div>
    <div class="paper-meta">
      📅 2023-06-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with memory are computationally universal. However, mainstream LLMs are not taking full advantage of memory, and the designs are heavily influenced by biological brains. Due to their approximate nature and proneness to the accumulation of errors, conventional neural memory mechanisms cannot support LLMs to simulate complex reasoning. In this paper, we seek inspiration from modern computer architectures to augment LLMs with symbolic memory for complex multi-hop reasoning. Such a symbolic memory framework is instantiated as an LLM and a set of SQL databases, where the LLM generates SQL instructions to manipulate the SQL databases. We validate the effectiveness of the proposed memory framework on a synthetic dataset requiring complex reasoning. The project website is available at https://chatdatabase.github.io/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.03314v1">Multi-Agent Collaboration: Harnessing the Power of Intelligent LLM Agents</a></div>
    <div class="paper-meta">
      📅 2023-06-05
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel framework for enhancing the capabilities of large language models (LLMs) by leveraging the power of multi-agent systems. Our framework introduces a collaborative environment where multiple intelligent agent components, each with distinctive attributes and roles, work together to handle complex tasks more efficiently and effectively. We demonstrate the practicality and versatility of our framework through case studies in artificial general intelligence (AGI), specifically focusing on the Auto-GPT and BabyAGI models. We also examine the "Gorilla" model, which integrates external APIs into the LLM. Our framework addresses limitations and challenges such as looping issues, security risks, scalability, system evaluation, and ethical considerations. By modeling various domains such as courtroom simulations and software development scenarios, we showcase the potential applications and benefits of our proposed multi-agent system. Our framework provides an avenue for advancing the capabilities and performance of LLMs through collaboration and knowledge exchange among intelligent agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.03264v1">shs-nlp at RadSum23: Domain-Adaptive Pre-training of Instruction-tuned LLMs for Radiology Report Impression Generation</a></div>
    <div class="paper-meta">
      📅 2023-06-05
      | 💬 1st Place in Task 1B: Radiology Report Summarization at BioNLP 2023
    </div>
    <details class="paper-abstract">
      Instruction-tuned generative Large language models (LLMs) like ChatGPT and Bloomz possess excellent generalization abilities, but they face limitations in understanding radiology reports, particularly in the task of generating the IMPRESSIONS section from the FINDINGS section. They tend to generate either verbose or incomplete IMPRESSIONS, mainly due to insufficient exposure to medical text data during training. We present a system which leverages large-scale medical text data for domain-adaptive pre-training of instruction-tuned LLMs to enhance its medical knowledge and performance on specific medical tasks. We show that this system performs better in a zero-shot setting than a number of pretrain-and-finetune adaptation methods on the IMPRESSIONS generation task, and ranks 1st among participating systems in Task 1B: Radiology Report Summarization at the BioNLP 2023 workshop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.03078v1">SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression</a></div>
    <div class="paper-meta">
      📅 2023-06-05
      | 💬 Extended preprint
    </div>
    <details class="paper-abstract">
      Recent advances in large language model (LLM) pretraining have led to high-quality LLMs with impressive abilities. By compressing such LLMs via quantization to 3-4 bits per parameter, they can fit into memory-limited devices such as laptops and mobile phones, enabling personalized use. However, quantization down to 3-4 bits per parameter usually leads to moderate-to-high accuracy losses, especially for smaller models in the 1-10B parameter range, which are well-suited for edge deployments. To address this accuracy issue, we introduce the Sparse-Quantized Representation (SpQR), a new compressed format and quantization technique which enables for the first time near-lossless compression of LLMs across model scales, while reaching similar compression levels to previous methods. SpQR works by identifying and isolating outlier weights, which cause particularly-large quantization errors, and storing them in higher precision, while compressing all other weights to 3-4 bits, and achieves relative accuracy losses of less than 1% in perplexity for highly-accurate LLaMA and Falcon LLMs. This makes it possible to run 33B parameter LLM on a single 24 GB consumer GPU without any performance degradation at 15% speedup thus making powerful LLMs available to consumer without any downsides. SpQR comes with efficient algorithms for both encoding weights into its format, as well as decoding them efficiently at runtime. Specifically, we provide an efficient GPU inference algorithm for SpQR which yields faster inference than 16-bit baselines at similar accuracy, while enabling memory compression gains of more than 4x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.02776v1">Cheap-fake Detection with LLM using Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2023-06-05
      | 💬 ICME2023 Workshop
    </div>
    <details class="paper-abstract">
      The misuse of real photographs with conflicting image captions in news items is an example of the out-of-context (OOC) misuse of media. In order to detect OOC media, individuals must determine the accuracy of the statement and evaluate whether the triplet (~\textit{i.e.}, the image and two captions) relates to the same event. This paper presents a novel learnable approach for detecting OOC media in ICME'23 Grand Challenge on Detecting Cheapfakes. The proposed method is based on the COSMOS structure, which assesses the coherence between an image and captions, as well as between two captions. We enhance the baseline algorithm by incorporating a Large Language Model (LLM), GPT3.5, as a feature extractor. Specifically, we propose an innovative approach to feature extraction utilizing prompt engineering to develop a robust and reliable feature extractor with GPT3.5 model. The proposed method captures the correlation between two captions and effectively integrates this module into the COSMOS baseline model, which allows for a deeper understanding of the relationship between captions. By incorporating this module, we demonstrate the potential for significant improvements in cheap-fakes detection performance. The proposed methodology holds promising implications for various applications such as natural language processing, image captioning, and text-to-image synthesis. Docker for submission is available at https://hub.docker.com/repository/docker/mulns/ acmmmcheapfakes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.02230v1">Prompt Sapper: LLM-Empowered Software Engineering Infrastructure for AI-Native Services</a></div>
    <div class="paper-meta">
      📅 2023-06-04
    </div>
    <details class="paper-abstract">
      Foundation models, such as GPT-4, DALL-E have brought unprecedented AI "operating system" effect and new forms of human-AI interaction, sparking a wave of innovation in AI-native services, where natural language prompts serve as executable "code" directly (prompt as executable code), eliminating the need for programming language as an intermediary and opening up the door to personal AI. Prompt Sapper has emerged in response, committed to support the development of AI-native services by AI chain engineering. It creates a large language model (LLM) empowered software engineering infrastructure for authoring AI chains through human-AI collaborative intelligence, unleashing the AI innovation potential of every individual, and forging a future where everyone can be a master of AI innovation. This article will introduce the R\&D motivation behind Prompt Sapper, along with its corresponding AI chain engineering methodology and technical practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2303.07205v3">The Science of Detecting LLM-Generated Texts</a></div>
    <div class="paper-meta">
      📅 2023-06-02
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has resulted in the production of LLM-generated texts that is highly sophisticated and almost indistinguishable from texts written by humans. However, this has also sparked concerns about the potential misuse of such texts, such as spreading misinformation and causing disruptions in the education system. Although many detection approaches have been proposed, a comprehensive understanding of the achievements and challenges is still lacking. This survey aims to provide an overview of existing LLM-generated text detection techniques and enhance the control and regulation of language generation models. Furthermore, we emphasize crucial considerations for future research, including the development of comprehensive evaluation metrics and the threat posed by open-source LLMs, to drive progress in the area of LLM-generated text detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.01499v1">Can LLMs like GPT-4 outperform traditional AI tools in dementia diagnosis? Maybe, but not today</a></div>
    <div class="paper-meta">
      📅 2023-06-02
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent investigations show that large language models (LLMs), specifically GPT-4, not only have remarkable capabilities in common Natural Language Processing (NLP) tasks but also exhibit human-level performance on various professional and academic benchmarks. However, whether GPT-4 can be directly used in practical applications and replace traditional artificial intelligence (AI) tools in specialized domains requires further experimental validation. In this paper, we explore the potential of LLMs such as GPT-4 to outperform traditional AI tools in dementia diagnosis. Comprehensive comparisons between GPT-4 and traditional AI tools are conducted to examine their diagnostic accuracy in a clinical setting. Experimental results on two real clinical datasets show that, although LLMs like GPT-4 demonstrate potential for future advancements in dementia diagnosis, they currently do not surpass the performance of traditional AI tools. The interpretability and faithfulness of GPT-4 are also evaluated by comparison with real doctors. We discuss the limitations of GPT-4 in its current state and propose future research directions to enhance GPT-4 in dementia diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.01815v1">Prototyping the use of Large Language Models (LLMs) for adult learning content creation at scale</a></div>
    <div class="paper-meta">
      📅 2023-06-02
      | 💬 1 figure
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) and other forms of Generative AI permeate various aspects of our lives, their application for learning and education has provided opportunities and challenges. This paper presents an investigation into the use of LLMs in asynchronous course creation, particularly within the context of adult learning, training and upskilling. We developed a course prototype leveraging an LLM, implementing a robust human-in-the-loop process to ensure the accuracy and clarity of the generated content. Our research questions focus on the feasibility of LLMs to produce high-quality adult learning content with reduced human involvement. Initial findings indicate that taking this approach can indeed facilitate faster content creation without compromising on accuracy or clarity, marking a promising advancement in the field of Generative AI for education. Despite some limitations, the study underscores the potential of LLMs to transform the landscape of learning and education, necessitating further research and nuanced discussions about their strategic and ethical use in learning design.
    </details>
</div>
