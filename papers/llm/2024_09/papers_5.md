# llm - 2024_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5

## Papers

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
    <div class="paper-title"><a href="https://arxiv.org/pdf/2310.05191v2">LLM-as-a-tutor in EFL Writing Education: Focusing on Evaluation of Student-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2024-09-04
    </div>
    <details class="paper-abstract">
      In the context of English as a Foreign Language (EFL) writing education, LLM-as-a-tutor can assist students by providing real-time feedback on their essays. However, challenges arise in assessing LLM-as-a-tutor due to differing standards between educational and general use cases. To bridge this gap, we integrate pedagogical principles to assess student-LLM interaction. First, we explore how LLMs can function as English tutors, providing effective essay feedback tailored to students. Second, we propose three metrics to evaluate LLM-as-a-tutor specifically designed for EFL writing education, emphasizing pedagogical aspects. In this process, EFL experts evaluate the feedback from LLM-as-a-tutor regarding quality and characteristics. On the other hand, EFL learners assess their learning outcomes from interaction with LLM-as-a-tutor. This approach lays the groundwork for developing LLMs-as-a-tutor tailored to the needs of EFL learners, advancing the effectiveness of writing education in this context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2409.01140v1">LLM-PQA: LLM-enhanced Prediction Query Answering</a></div>
    <div class="paper-meta">
      📅 2024-09-04
      | 💬 This paper is accepted as a demo at CIKM 2024
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) provides an opportunity to change the way queries are processed, moving beyond the constraints of conventional SQL-based database systems. However, using an LLM to answer a prediction query is still challenging, since an external ML model has to be employed and inference has to be performed in order to provide an answer. This paper introduces LLM-PQA, a novel tool that addresses prediction queries formulated in natural language. LLM-PQA is the first to combine the capabilities of LLMs and retrieval-augmented mechanism for the needs of prediction queries by integrating data lakes and model zoos. This integration provides users with access to a vast spectrum of heterogeneous data and diverse ML models, facilitating dynamic prediction query answering. In addition, LLM-PQA can dynamically train models on demand, based on specific query requirements, ensuring reliable and relevant results even when no pre-trained model in a model zoo, available for the task.
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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16465v2">Human and LLM-Based Voice Assistant Interaction: An Analytical Framework for User Verbal and Nonverbal Behaviors</a></div>
    <div class="paper-meta">
      📅 2024-09-03
    </div>
    <details class="paper-abstract">
      Recent progress in large language model (LLM) technology has significantly enhanced the interaction experience between humans and voice assistants (VAs). This project aims to explore a user's continuous interaction with LLM-based VA (LLM-VA) during a complex task. We recruited 12 participants to interact with an LLM-VA during a cooking task, selected for its complexity and the requirement for continuous interaction. We observed that users show both verbal and nonverbal behaviors, though they know that the LLM-VA can not capture those nonverbal signals. Despite the prevalence of nonverbal behavior in human-human communication, there is no established analytical methodology or framework for exploring it in human-VA interactions. After analyzing 3 hours and 39 minutes of video recordings, we developed an analytical framework with three dimensions: 1) behavior characteristics, including both verbal and nonverbal behaviors, 2) interaction stages--exploration, conflict, and integration--that illustrate the progression of user interactions, and 3) stage transition throughout the task. This analytical framework identifies key verbal and nonverbal behaviors that provide a foundation for future research and practical applications in optimizing human and LLM-VA interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01907v1">Focus Agent: LLM-Powered Virtual Focus Group</a></div>
    <div class="paper-meta">
      📅 2024-09-03
      | 💬 8 pages, the 24th Intelligent Virtual Agent Conference
    </div>
    <details class="paper-abstract">
      In the domain of Human-Computer Interaction, focus groups represent a widely utilised yet resource-intensive methodology, often demanding the expertise of skilled moderators and meticulous preparatory efforts. This study introduces the ``Focus Agent,'' a Large Language Model (LLM) powered framework that simulates both the focus group (for data collection) and acts as a moderator in a focus group setting with human participants. To assess the data quality derived from the Focus Agent, we ran five focus group sessions with a total of 23 human participants as well as deploying the Focus Agent to simulate these discussions with AI participants. Quantitative analysis indicates that Focus Agent can generate opinions similar to those of human participants. Furthermore, the research exposes some improvements associated with LLMs acting as moderators in focus group discussions that include human participants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04465v1">Here's Charlie! Realising the Semantic Web vision of Agents in the age of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-03
      | 💬 The 23rd International Semantic Web Conference, November 11--15, 2024, Hanover, MD - Posters and Demos track
    </div>
    <details class="paper-abstract">
      This paper presents our research towards a near-term future in which legal entities, such as individuals and organisations can entrust semi-autonomous AI-driven agents to carry out online interactions on their behalf. The author's research concerns the development of semi-autonomous Web agents, which consult users if and only if the system does not have sufficient context or confidence to proceed working autonomously. This creates a user-agent dialogue that allows the user to teach the agent about the information sources they trust, their data-sharing preferences, and their decision-making preferences. Ultimately, this enables the user to maximise control over their data and decisions while retaining the convenience of using agents, including those driven by LLMs. In view of developing near-term solutions, the research seeks to answer the question: "How do we build a trustworthy and reliable network of semi-autonomous agents which represent individuals and organisations on the Web?". After identifying key requirements, the paper presents a demo for a sample use case of a generic personal assistant. This is implemented using (Notation3) rules to enforce safety guarantees around belief, data sharing and data usage and LLMs to allow natural language interaction with users and serendipitous dialogues between software agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01252v3">Towards Scalable Automated Alignment of LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-09-03
      | 💬 Paper List: https://github.com/cascip/awesome-auto-alignment
    </div>
    <details class="paper-abstract">
      Alignment is the most critical step in building large language models (LLMs) that meet human needs. With the rapid development of LLMs gradually surpassing human capabilities, traditional alignment methods based on human-annotation are increasingly unable to meet the scalability demands. Therefore, there is an urgent need to explore new sources of automated alignment signals and technical approaches. In this paper, we systematically review the recently emerging methods of automated alignment, attempting to explore how to achieve effective, scalable, automated alignment once the capabilities of LLMs exceed those of humans. Specifically, we categorize existing automated alignment methods into 4 major categories based on the sources of alignment signals and discuss the current status and potential development of each category. Additionally, we explore the underlying mechanisms that enable automated alignment and discuss the essential factors that make automated alignment technologies feasible and effective from the fundamental role of alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01605v1">Laser: Parameter-Efficient LLM Bi-Tuning for Sequential Recommendation with Collaborative Information</a></div>
    <div class="paper-meta">
      📅 2024-09-03
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Sequential recommender systems are essential for discerning user preferences from historical interactions and facilitating targeted recommendations. Recent innovations employing Large Language Models (LLMs) have advanced the field by encoding item semantics, yet they often necessitate substantial parameter tuning and are resource-demanding. Moreover, these works fails to consider the diverse characteristics of different types of users and thus diminishes the recommendation accuracy. In this paper, we propose a parameter-efficient Large Language Model Bi-Tuning framework for sequential recommendation with collaborative information (Laser). Specifically, Bi-Tuning works by inserting trainable virtual tokens at both the prefix and suffix of the input sequence and freezing the LLM parameters, thus optimizing the LLM for the sequential recommendation. In our Laser, the prefix is utilized to incorporate user-item collaborative information and adapt the LLM to the recommendation task, while the suffix converts the output embeddings of the LLM from the language space to the recommendation space for the follow-up item recommendation. Furthermore, to capture the characteristics of different types of users when integrating the collaborative information via the prefix, we introduce M-Former, a lightweight MoE-based querying transformer that uses a set of query experts to integrate diverse user-specific collaborative information encoded by frozen ID-based sequential recommender systems, significantly improving the accuracy of recommendations. Extensive experiments on real-world datasets demonstrate that Laser can parameter-efficiently adapt LLMs to effective recommender systems, significantly outperforming state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01575v1">An Implementation of Werewolf Agent That does not Truly Trust LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-03
    </div>
    <details class="paper-abstract">
      Werewolf is an incomplete information game, which has several challenges when creating a computer agent as a player given the lack of understanding of the situation and individuality of utterance (e.g., computer agents are not capable of characterful utterance or situational lying). We propose a werewolf agent that solves some of those difficulties by combining a Large Language Model (LLM) and a rule-based algorithm. In particular, our agent uses a rule-based algorithm to select an output either from an LLM or a template prepared beforehand based on the results of analyzing conversation history using an LLM. It allows the agent to refute in specific situations, identify when to end the conversation, and behave with persona. This approach mitigated conversational inconsistencies and facilitated logical utterance as a result. We also conducted a qualitative evaluation, which resulted in our agent being perceived as more human-like compared to an unmodified LLM. The agent is freely available for contributing to advance the research in the field of Werewolf game.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01552v1">Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown success in generating high-quality responses. In order to achieve better alignment with LLMs with human preference, various works are proposed based on specific optimization process, which, however, is not suitable to Black-Box LLMs like GPT-4, due to inaccessible parameters. In Black-Box LLMs case, their performance is highly dependent on the quality of the provided prompts. Existing methods to enhance response quality often involve a prompt refinement model, yet these approaches potentially suffer from semantic inconsistencies between the refined and original prompts, and typically overlook the relationship between them. To address these challenges, we introduce a self-instructed in-context learning framework that empowers LLMs to deliver more effective responses by generating reliable derived prompts to construct informative contextual environments. Our approach incorporates a self-instructed reinforcement learning mechanism, enabling direct interaction with the response model during derived prompt generation for better alignment. We then formulate querying as an in-context learning task, using responses from LLMs combined with the derived prompts to establish a contextual demonstration for the original prompt. This strategy ensures alignment with the original query, reduces discrepancies from refined prompts, and maximizes the LLMs' in-context learning capability. Extensive experiments demonstrate that the proposed method not only generates more reliable derived prompts but also significantly enhances LLMs' ability to deliver more effective responses, including Black-Box models such as GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15092v2">Dissociation of Faithful and Unfaithful Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-02
      | 💬 code published at https://github.com/CoTErrorRecovery/CoTErrorRecovery
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often improve their performance in downstream tasks when they generate Chain of Thought reasoning text before producing an answer. We investigate how LLMs recover from errors in Chain of Thought. Through analysis of error recovery behaviors, we find evidence for unfaithfulness in Chain of Thought, which occurs when models arrive at the correct answer despite invalid reasoning text. We identify factors that shift LLM recovery behavior: LLMs recover more frequently from obvious errors and in contexts that provide more evidence for the correct answer. Critically, these factors have divergent effects on faithful and unfaithful recoveries. Our results indicate that there are distinct mechanisms driving faithful and unfaithful error recoveries. Selective targeting of these mechanisms may be able to drive down the rate of unfaithful reasoning and improve model interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01466v1">PoliPrompt: A High-Performance Cost-Effective LLM-Based Text Classification Framework for Political Science</a></div>
    <div class="paper-meta">
      📅 2024-09-02
      | 💬 23 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have opened new avenues for enhancing text classification efficiency in political science, surpassing traditional machine learning methods that often require extensive feature engineering, human labeling, and task-specific training. However, their effectiveness in achieving high classification accuracy remains questionable. This paper introduces a three-stage in-context learning approach that leverages LLMs to improve classification accuracy while minimizing experimental costs. Our method incorporates automatic enhanced prompt generation, adaptive exemplar selection, and a consensus mechanism that resolves discrepancies between two weaker LLMs, refined by an advanced LLM. We validate our approach using datasets from the BBC news reports, Kavanaugh Supreme Court confirmation, and 2018 election campaign ads. The results show significant improvements in classification F1 score (+0.36 for zero-shot classification) with manageable economic costs (-78% compared with human labeling), demonstrating that our method effectively addresses the limitations of traditional machine learning while offering a scalable and reliable solution for text analysis in political science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01382v1">Automatic Detection of LLM-generated Code: A Case Study of Claude 3 Haiku</a></div>
    <div class="paper-meta">
      📅 2024-09-02
      | 💬 Submitted to a journal for potential publication
    </div>
    <details class="paper-abstract">
      Using Large Language Models (LLMs) has gained popularity among software developers for generating source code. However, the use of LLM-generated code can introduce risks of adding suboptimal, defective, and vulnerable code. This makes it necessary to devise methods for the accurate detection of LLM-generated code. Toward this goal, we perform a case study of Claude 3 Haiku (or Claude 3 for brevity) on CodeSearchNet dataset. We divide our analyses into two parts: function-level and class-level. We extract 22 software metric features, such as Code Lines and Cyclomatic Complexity, for each level of granularity. We then analyze code snippets generated by Claude 3 and their human-authored counterparts using the extracted features to understand how unique the code generated by Claude 3 is. In the following step, we use the unique characteristics of Claude 3-generated code to build Machine Learning (ML) models and identify which features of the code snippets make them more detectable by ML models. Our results indicate that Claude 3 tends to generate longer functions, but shorter classes than humans, and this characteristic can be used to detect Claude 3-generated code with ML models with 82% and 66% accuracies for function-level and class-level snippets, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12288v3">An Investigation of Neuron Activation as a Unified Lens to Explain Chain-of-Thought Eliciting Arithmetic Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-02
      | 💬 9 pages, 1 figure, to be published in ACL 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong arithmetic reasoning capabilities when prompted with Chain-of-Thought (CoT) prompts. However, we have only a limited understanding of how they are processed by LLMs. To demystify it, prior work has primarily focused on ablating different components in the CoT prompt and empirically observing their resulting LLM performance change. Yet, the reason why these components are important to LLM reasoning is not explored. To fill this gap, in this work, we investigate ``neuron activation'' as a lens to provide a unified explanation to observations made by prior work. Specifically, we look into neurons within the feed-forward layers of LLMs that may have activated their arithmetic reasoning capabilities, using Llama2 as an example. To facilitate this investigation, we also propose an approach based on GPT-4 to automatically identify neurons that imply arithmetic reasoning. Our analyses revealed that the activation of reasoning neurons in the feed-forward layers of an LLM can explain the importance of various components in a CoT prompt, and future research can extend it for a more complete understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13152v2">Analyzing Diversity in Healthcare LLM Research: A Scientometric Perspective</a></div>
    <div class="paper-meta">
      📅 2024-09-02
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) in healthcare has demonstrated substantial potential for enhancing clinical decision-making, administrative efficiency, and patient outcomes. However, the underrepresentation of diverse groups in the development and application of these models can perpetuate biases, leading to inequitable healthcare delivery. This paper presents a comprehensive scientometric analysis of LLM research for healthcare, including data from January 1, 2021, to July 1, 2024. By analyzing metadata from PubMed and Dimensions, including author affiliations, countries, and funding sources, we assess the diversity of contributors to LLM research. Our findings highlight significant gender and geographic disparities, with a predominance of male authors and contributions primarily from high-income countries (HICs). We introduce a novel journal diversity index based on Gini diversity to measure the inclusiveness of scientific publications. Our results underscore the necessity for greater representation in order to ensure the equitable application of LLMs in healthcare. We propose actionable strategies to enhance diversity and inclusivity in artificial intelligence research, with the ultimate goal of fostering a more inclusive and equitable future in healthcare innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01145v1">LATEX-GCL: Large Language Models (LLMs)-Based Data Augmentation for Text-Attributed Graph Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2024-09-02
    </div>
    <details class="paper-abstract">
      Graph Contrastive Learning (GCL) is a potent paradigm for self-supervised graph learning that has attracted attention across various application scenarios. However, GCL for learning on Text-Attributed Graphs (TAGs) has yet to be explored. Because conventional augmentation techniques like feature embedding masking cannot directly process textual attributes on TAGs. A naive strategy for applying GCL to TAGs is to encode the textual attributes into feature embeddings via a language model and then feed the embeddings into the following GCL module for processing. Such a strategy faces three key challenges: I) failure to avoid information loss, II) semantic loss during the text encoding phase, and III) implicit augmentation constraints that lead to uncontrollable and incomprehensible results. In this paper, we propose a novel GCL framework named LATEX-GCL to utilize Large Language Models (LLMs) to produce textual augmentations and LLMs' powerful natural language processing (NLP) abilities to address the three limitations aforementioned to pave the way for applying GCL to TAG tasks. Extensive experiments on four high-quality TAG datasets illustrate the superiority of the proposed LATEX-GCL method. The source codes and datasets are released to ease the reproducibility, which can be accessed via this link: https://anonymous.4open.science/r/LATEX-GCL-0712.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01140v1">LLM-PQA: LLM-enhanced Prediction Query Answering</a></div>
    <div class="paper-meta">
      📅 2024-09-02
      | 💬 This paper is accepted as a demo at CIKM 2024
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) provides an opportunity to change the way queries are processed, moving beyond the constraints of conventional SQL-based database systems. However, using an LLM to answer a prediction query is still challenging, since an external ML model has to be employed and inference has to be performed in order to provide an answer. This paper introduces LLM-PQA, a novel tool that addresses prediction queries formulated in natural language. LLM-PQA is the first to combine the capabilities of LLMs and retrieval-augmented mechanism for the needs of prediction queries by integrating data lakes and model zoos. This integration provides users with access to a vast spectrum of heterogeneous data and diverse ML models, facilitating dynamic prediction query answering. In addition, LLM-PQA can dynamically train models on demand, based on specific query requirements, ensuring reliable and relevant results even when no pre-trained model in a model zoo, available for the task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02091v2">Physics simulation capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-02
      | 💬 Accepted for publication in Physica Scripta. Abridged abstract. 15 pages + appendix, 1 figure. Comments are welcome
    </div>
    <details class="paper-abstract">
      [Abridged abstract] Large Language Models (LLMs) can solve some undergraduate-level to graduate-level physics textbook problems and are proficient at coding. Combining these two capabilities could one day enable AI systems to simulate and predict the physical world. We present an evaluation of state-of-the-art (SOTA) LLMs on PhD-level to research-level computational physics problems. We condition LLM generation on the use of well-documented and widely-used packages to elicit coding capabilities in the physics and astrophysics domains. We contribute $\sim 50$ original and challenging problems in celestial mechanics (with REBOUND), stellar physics (with MESA), 1D fluid dynamics (with Dedalus) and non-linear dynamics (with SciPy). Since our problems do not admit unique solutions, we evaluate LLM performance on several soft metrics: counts of lines that contain different types of errors (coding, physics, necessity and sufficiency) as well as a more "educational" Pass-Fail metric focused on capturing the salient physical ingredients of the problem at hand. As expected, today's SOTA LLM (GPT4) zero-shot fails most of our problems, although about 40\% of the solutions could plausibly get a passing grade. About $70-90 \%$ of the code lines produced are necessary, sufficient and correct (coding \& physics). Physics and coding errors are the most common, with some unnecessary or insufficient lines. We observe significant variations across problem class and difficulty. We identify several failure modes of GPT4 in the computational physics domain. Our reconnaissance work provides a snapshot of current computational capabilities in classical physics and points to obvious improvement targets if AI systems are ever to reach a basic level of autonomy in physics simulation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01073v1">SCOPE: Sign Language Contextual Processing with Embedding from LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-02
    </div>
    <details class="paper-abstract">
      Sign languages, used by around 70 million Deaf individuals globally, are visual languages that convey visual and contextual information. Current methods in vision-based sign language recognition (SLR) and translation (SLT) struggle with dialogue scenes due to limited dataset diversity and the neglect of contextually relevant information. To address these challenges, we introduce SCOPE (Sign language Contextual Processing with Embedding from LLMs), a novel context-aware vision-based SLR and SLT framework. For SLR, we utilize dialogue contexts through a multi-modal encoder to enhance gloss-level recognition. For subsequent SLT, we further fine-tune a Large Language Model (LLM) by incorporating prior conversational context. We also contribute a new sign language dataset that contains 72 hours of Chinese sign language videos in contextual dialogues across various scenarios. Experimental results demonstrate that our SCOPE framework achieves state-of-the-art performance on multiple datasets, including Phoenix-2014T, CSL-Daily, and our SCOPE dataset. Moreover, surveys conducted with participants from the Deaf community further validate the robustness and effectiveness of our approach in real-world applications. Both our dataset and code will be open-sourced to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.09067v3">Contrasting Linguistic Patterns in Human and LLM-Generated News Text</a></div>
    <div class="paper-meta">
      📅 2024-09-02
      | 💬 Published at Artificial Intelligence Review vol. 57, 265
    </div>
    <details class="paper-abstract">
      We conduct a quantitative analysis contrasting human-written English news text with comparable large language model (LLM) output from six different LLMs that cover three different families and four sizes in total. Our analysis spans several measurable linguistic dimensions, including morphological, syntactic, psychometric, and sociolinguistic aspects. The results reveal various measurable differences between human and AI-generated texts. Human texts exhibit more scattered sentence length distributions, more variety of vocabulary, a distinct use of dependency and constituent types, shorter constituents, and more optimized dependency distances. Humans tend to exhibit stronger negative emotions (such as fear and disgust) and less joy compared to text generated by LLMs, with the toxicity of these models increasing as their size grows. LLM outputs use more numbers, symbols and auxiliaries (suggesting objective language) than human texts, as well as more pronouns. The sexist bias prevalent in human text is also expressed by LLMs, and even magnified in all of them but one. Differences between LLMs and humans are larger than between LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01001v1">Beyond ChatGPT: Enhancing Software Quality Assurance Tasks with Diverse LLMs and Validation Techniques</a></div>
    <div class="paper-meta">
      📅 2024-09-02
    </div>
    <details class="paper-abstract">
      With the advancement of Large Language Models (LLMs), their application in Software Quality Assurance (SQA) has increased. However, the current focus of these applications is predominantly on ChatGPT. There remains a gap in understanding the performance of various LLMs in this critical domain. This paper aims to address this gap by conducting a comprehensive investigation into the capabilities of several LLMs across two SQA tasks: fault localization and vulnerability detection. We conducted comparative studies using GPT-3.5, GPT-4o, and four other publicly available LLMs (LLaMA-3-70B, LLaMA-3-8B, Gemma-7B, and Mixtral-8x7B), to evaluate their effectiveness in these tasks. Our findings reveal that several LLMs can outperform GPT-3.5 in both tasks. Additionally, even the lower-performing LLMs provided unique correct predictions, suggesting the potential of combining different LLMs' results to enhance overall performance. By implementing a voting mechanism to combine the LLMs' results, we achieved more than a 10% improvement over the GPT-3.5 in both tasks. Furthermore, we introduced a cross-validation approach to refine the LLM answer by validating one LLM answer against another using a validation prompt. This approach led to performance improvements of 16% in fault localization and 12% in vulnerability detection compared to the GPT-3.5, with a 4% improvement compared to the best-performed LLMs. Our analysis also indicates that the inclusion of explanations in the LLMs' results affects the effectiveness of the cross-validation technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00993v1">Evolution of Social Norms in LLM Agents using Natural Language</a></div>
    <div class="paper-meta">
      📅 2024-09-02
      | 💬 5 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have spurred a surge of interest in leveraging these models for game-theoretical simulations, where LLMs act as individual agents engaging in social interactions. This study explores the potential for LLM agents to spontaneously generate and adhere to normative strategies through natural language discourse, building upon the foundational work of Axelrod's metanorm games. Our experiments demonstrate that through dialogue, LLM agents can form complex social norms, such as metanorms-norms enforcing the punishment of those who do not punish cheating-purely through natural language interaction. The results affirm the effectiveness of using LLM agents for simulating social interactions and understanding the emergence and evolution of complex strategies and norms through natural language. Future work may extend these findings by incorporating a wider range of scenarios and agent characteristics, aiming to uncover more nuanced mechanisms behind social norm formation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05191v2">LLM-as-a-tutor in EFL Writing Education: Focusing on Evaluation of Student-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2024-09-02
    </div>
    <details class="paper-abstract">
      In the context of English as a Foreign Language (EFL) writing education, LLM-as-a-tutor can assist students by providing real-time feedback on their essays. However, challenges arise in assessing LLM-as-a-tutor due to differing standards between educational and general use cases. To bridge this gap, we integrate pedagogical principles to assess student-LLM interaction. First, we explore how LLMs can function as English tutors, providing effective essay feedback tailored to students. Second, we propose three metrics to evaluate LLM-as-a-tutor specifically designed for EFL writing education, emphasizing pedagogical aspects. In this process, EFL experts evaluate the feedback from LLM-as-a-tutor regarding quality and characteristics. On the other hand, EFL learners assess their learning outcomes from interaction with LLM-as-a-tutor. This approach lays the groundwork for developing LLMs-as-a-tutor tailored to the needs of EFL learners, advancing the effectiveness of writing education in this context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00861v1">Harnessing the Power of Semi-Structured Knowledge and LLMs with Triplet-Based Prefiltering for Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-09-01
      | 💬 9 pages, published at IJCLR 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently lack domain-specific knowledge and even fine-tuned models tend to hallucinate. Hence, more reliable models that can include external knowledge are needed. We present a pipeline, 4StepFocus, and specifically a preprocessing step, that can substantially improve the answers of LLMs. This is achieved by providing guided access to external knowledge making use of the model's ability to capture relational context and conduct rudimentary reasoning by themselves. The method narrows down potentially correct answers by triplets-based searches in a semi-structured knowledge base in a direct, traceable fashion, before switching to latent representations for ranking those candidates based on unstructured data. This distinguishes it from related methods that are purely based on latent representations. 4StepFocus consists of the steps: 1) Triplet generation for extraction of relational data by an LLM, 2) substitution of variables in those triplets to narrow down answer candidates employing a knowledge graph, 3) sorting remaining candidates with a vector similarity search involving associated non-structured data, 4) reranking the best candidates by the LLM with background data provided. Experiments on a medical, a product recommendation, and an academic paper search test set demonstrate that this approach is indeed a powerful augmentation. It not only adds relevant traceable background information from information retrieval, but also improves performance considerably in comparison to state-of-the-art methods. This paper presents a novel, largely unexplored direction and therefore provides a wide range of future work opportunities. Used source code is available at https://github.com/kramerlab/4StepFocus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00856v1">Benchmarking LLM Code Generation for Audio Programming with Visual Dataflow Languages</a></div>
    <div class="paper-meta">
      📅 2024-09-01
    </div>
    <details class="paper-abstract">
      Node-based programming languages are increasingly popular in media arts coding domains. These languages are designed to be accessible to users with limited coding experience, allowing them to achieve creative output without an extensive programming background. Using LLM-based code generation to further lower the barrier to creative output is an exciting opportunity. However, the best strategy for code generation for visual node-based programming languages is still an open question. In particular, such languages have multiple levels of representation in text, each of which may be used for code generation. In this work, we explore the performance of LLM code generation in audio programming tasks in visual programming languages at multiple levels of representation. We explore code generation through metaprogramming code representations for these languages (i.e., coding the language using a different high-level text-based programming language), as well as through direct node generation with JSON. We evaluate code generated in this way for two visual languages for audio programming on a benchmark set of coding problems. We measure both correctness and complexity of the generated code. We find that metaprogramming results in more semantically correct generated code, given that the code is well-formed (i.e., is syntactically correct and runs). We also find that prompting for richer metaprogramming using randomness and loops led to more complex code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00800v1">Comparing Discrete and Continuous Space LLMs for Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2024-09-01
      | 💬 InterSpeech 2024
    </div>
    <details class="paper-abstract">
      This paper investigates discrete and continuous speech representations in Large Language Model (LLM)-based Automatic Speech Recognition (ASR), organizing them by feature continuity and training approach into four categories: supervised and unsupervised for both discrete and continuous types. We further classify LLMs based on their input and autoregressive feedback into continuous and discrete-space models. Using specialized encoders and comparative analysis with a Joint-Training-From-Scratch Language Model (JTFS LM) and pre-trained LLaMA2-7b, we provide a detailed examination of their effectiveness. Our work marks the first extensive comparison of speech representations in LLM-based ASR and explores various modeling techniques. We present an open-sourced achievement of a state-of-the-art Word Error Rate (WER) of 1.69\% on LibriSpeech using a HuBERT encoder, offering valuable insights for advancing ASR and natural language processing (NLP) research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15319v3">LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-01
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      In traditional RAG framework, the basic retrieval units are normally short. The common retrievers like DPR normally work with 100-word Wikipedia paragraphs. Such a design forces the retriever to search over a large corpus to find the `needle' unit. In contrast, the readers only need to generate answers from the short retrieved units. The imbalanced `heavy' retriever and `light' reader design can lead to sub-optimal performance. The loss of contextual information in the short, chunked units may increase the likelihood of introducing hard negatives during the retrieval stage. Additionally, the reader might not fully leverage the capabilities of recent advancements in LLMs. In order to alleviate the imbalance, we propose a new framework LongRAG, consisting of a `long retriever' and a `long reader'. In the two Wikipedia-based datasets, NQ and HotpotQA, LongRAG processes the entire Wikipedia corpus into 4K-token units by grouping related documents. By increasing the unit size, we significantly reduce the total number of units. This greatly reduces the burden on the retriever, resulting in strong retrieval performance with only a few (less than 8) top units. Without requiring any training, LongRAG achieves an EM of 62.7% on NQ and 64.3% on HotpotQA, which are on par with the (fully-trained) SoTA model. Furthermore, we test on two non-Wikipedia-based datasets, Qasper and MultiFieldQA-en. LongRAG processes each individual document as a single (long) unit rather than chunking them into smaller units. By doing so, we achieve an F1 score of 25.9% on Qasper and 57.5% on MultiFieldQA-en. Our study offers insights into the future roadmap for combining RAG with long-context LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.04565v2">A Versatile Graph Learning Approach through LLM-based Agent</a></div>
    <div class="paper-meta">
      📅 2024-09-01
    </div>
    <details class="paper-abstract">
      Designing versatile graph learning approaches is important, considering the diverse graphs and tasks existing in real-world applications. Existing methods have attempted to achieve this target through automated machine learning techniques, pre-training and fine-tuning strategies, and large language models. However, these methods are not versatile enough for graph learning, as they work on either limited types of graphs or a single task. In this paper, we propose to explore versatile graph learning approaches with LLM-based agents, and the key insight is customizing the graph learning procedures for diverse graphs and tasks. To achieve this, we develop several LLM-based agents, equipped with diverse profiles, tools, functions and human experience. They collaborate to configure each procedure with task and data-specific settings step by step towards versatile solutions, and the proposed method is dubbed GL-Agent. By evaluating on diverse tasks and graphs, the correct results of the agent and its comparable performance showcase the versatility of the proposed method, especially in complex scenarios.The low resource cost and the potential to use open-source LLMs highlight the efficiency of GL-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00661v1">Research on LLM Acceleration Using the High-Performance RISC-V Processor "Xiangshan" (Nanhu Version) Based on the Open-Source Matrix Instruction Set Extension (Vector Dot Product)</a></div>
    <div class="paper-meta">
      📅 2024-09-01
      | 💬 10 pages, in Chinese language, 6 figures
    </div>
    <details class="paper-abstract">
      Considering the high-performance and low-power requirements of edge AI, this study designs a specialized instruction set processor for edge AI based on the RISC-V instruction set architecture, addressing practical issues in digital signal processing for edge devices. This design enhances the execution efficiency of edge AI and reduces its energy consumption with limited hardware overhead, meeting the demands for efficient large language model (LLM) inference computation in edge AI applications. The main contributions of this paper are as follows: For the characteristics of large language models, custom instructions were extended based on the RISC-V instruction set to perform vector dot product calculations, accelerating the computation of large language models on dedicated vector dot product acceleration hardware. Based on the open-source high-performance RISC-V processor core XiangShan Nanhu architecture, the vector dot product specialized instruction set processor Nanhu-vdot was implemented, which adds vector dot product calculation units and pipeline processing logic on top of the XiangShan Nanhu.The Nanhu-vdot underwent FPGA hardware testing, achieving over four times the speed of scalar methods in vector dot product computation. Using a hardware-software co-design approach for second-generation Generative Pre-Trained Transformer (GPT-2) model inference, the speed improved by approximately 30% compared to pure software implementation with almost no additional consumption of hardware resources and power consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00630v1">LLMs as Evaluators: A Novel Approach to Evaluate Bug Report Summarization</a></div>
    <div class="paper-meta">
      📅 2024-09-01
    </div>
    <details class="paper-abstract">
      Summarizing software artifacts is an important task that has been thoroughly researched. For evaluating software summarization approaches, human judgment is still the most trusted evaluation. However, it is time-consuming and fatiguing for evaluators, making it challenging to scale and reproduce. Large Language Models (LLMs) have demonstrated remarkable capabilities in various software engineering tasks, motivating us to explore their potential as automatic evaluators for approaches that aim to summarize software artifacts. In this study, we investigate whether LLMs can evaluate bug report summarization effectively. We conducted an experiment in which we presented the same set of bug summarization problems to humans and three LLMs (GPT-4o, LLaMA-3, and Gemini) for evaluation on two tasks: selecting the correct bug report title and bug report summary from a set of options. Our results show that LLMs performed generally well in evaluating bug report summaries, with GPT-4o outperforming the other LLMs. Additionally, both humans and LLMs showed consistent decision-making, but humans experienced fatigue, impacting their accuracy over time. Our results indicate that LLMs demonstrate potential for being considered as automated evaluators for bug report summarization, which could allow scaling up evaluations while reducing human evaluators effort and fatigue.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12680v2">Can LLMs Understand Social Norms in Autonomous Driving Games?</a></div>
    <div class="paper-meta">
      📅 2024-09-01
    </div>
    <details class="paper-abstract">
      Social norm is defined as a shared standard of acceptable behavior in a society. The emergence of social norms fosters coordination among agents without any hard-coded rules, which is crucial for the large-scale deployment of AVs in an intelligent transportation system. This paper explores the application of LLMs in understanding and modeling social norms in autonomous driving games. We introduce LLMs into autonomous driving games as intelligent agents who make decisions according to text prompts. These agents are referred to as LLM-based agents. Our framework involves LLM-based agents playing Markov games in a multi-agent system (MAS), allowing us to investigate the emergence of social norms among individual agents. We aim to identify social norms by designing prompts and utilizing LLMs on textual information related to the environment setup and the observations of LLM-based agents. Using the OpenAI Chat API powered by GPT-4.0, we conduct experiments to simulate interactions and evaluate the performance of LLM-based agents in two driving scenarios: unsignalized intersection and highway platoon. The results show that LLM-based agents can handle dynamically changing environments in Markov games, and social norms evolve among LLM-based agents in both scenarios. In the intersection game, LLM-based agents tend to adopt a conservative driving policy when facing a potential car crash. The advantage of LLM-based agents in games lies in their strong operability and analyzability, which facilitate experimental design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00571v1">Enhancing Source Code Security with LLMs: Demystifying The Challenges and Generating Reliable Repairs</a></div>
    <div class="paper-meta">
      📅 2024-09-01
    </div>
    <details class="paper-abstract">
      With the recent unprecedented advancements in Artificial Intelligence (AI) computing, progress in Large Language Models (LLMs) is accelerating rapidly, presenting challenges in establishing clear guidelines, particularly in the field of security. That being said, we thoroughly identify and describe three main technical challenges in the security and software engineering literature that spans the entire LLM workflow, namely; \textbf{\textit{(i)}} Data Collection and Labeling; \textbf{\textit{(ii)}} System Design and Learning; and \textbf{\textit{(iii)}} Performance Evaluation. Building upon these challenges, this paper introduces \texttt{SecRepair}, an instruction-based LLM system designed to reliably \textit{identify}, \textit{describe}, and automatically \textit{repair} vulnerable source code. Our system is accompanied by a list of actionable guides on \textbf{\textit{(i)}} Data Preparation and Augmentation Techniques; \textbf{\textit{(ii)}} Selecting and Adapting state-of-the-art LLM Models; \textbf{\textit{(iii)}} Evaluation Procedures. \texttt{SecRepair} uses a reinforcement learning-based fine-tuning with a semantic reward that caters to the functionality and security aspects of the generated code. Our empirical analysis shows that \texttt{SecRepair} achieves a \textit{12}\% improvement in security code repair compared to other LLMs when trained using reinforcement learning. Furthermore, we demonstrate the capabilities of \texttt{SecRepair} in generating reliable, functional, and compilable security code repairs against real-world test cases using automated evaluation metrics.
    </details>
</div>
