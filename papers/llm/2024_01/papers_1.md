# llm - 2024_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17163v2">Learning Agent-based Modeling with LLM Companions: Experiences of Novices and Experts Using ChatGPT & NetLogo Chat</a></div>
    <div class="paper-meta">
      📅 2024-01-31
      | 💬 Conditionally accepted (with minor revisions) by Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI '24)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have the potential to fundamentally change the way people engage in computer programming. Agent-based modeling (ABM) has become ubiquitous in natural and social sciences and education, yet no prior studies have explored the potential of LLMs to assist it. We designed NetLogo Chat to support the learning and practice of NetLogo, a programming language for ABM. To understand how users perceive, use, and need LLM-based interfaces, we interviewed 30 participants from global academia, industry, and graduate schools. Experts reported more perceived benefits than novices and were more inclined to adopt LLMs in their workflow. We found significant differences between experts and novices in their perceptions, behaviors, and needs for human-AI collaboration. We surfaced a knowledge gap between experts and novices as a possible reason for the benefit gap. We identified guidance, personalization, and integration as major needs for LLM-based interfaces to support the programming of ABM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17839v1">Global-Liar: Factuality of LLMs over Time and Geographic Regions</a></div>
    <div class="paper-meta">
      📅 2024-01-31
      | 💬 24 pages, 12 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The increasing reliance on AI-driven solutions, particularly Large Language Models (LLMs) like the GPT series, for information retrieval highlights the critical need for their factuality and fairness, especially amidst the rampant spread of misinformation and disinformation online. Our study evaluates the factual accuracy, stability, and biases in widely adopted GPT models, including GPT-3.5 and GPT-4, contributing to reliability and integrity of AI-mediated information dissemination. We introduce 'Global-Liar,' a dataset uniquely balanced in terms of geographic and temporal representation, facilitating a more nuanced evaluation of LLM biases. Our analysis reveals that newer iterations of GPT models do not always equate to improved performance. Notably, the GPT-4 version from March demonstrates higher factual accuracy than its subsequent June release. Furthermore, a concerning bias is observed, privileging statements from the Global North over the Global South, thus potentially exacerbating existing informational inequities. Regions such as Africa and the Middle East are at a disadvantage, with much lower factual accuracy. The performance fluctuations over time suggest that model updates may not consistently benefit all regions equally. Our study also offers insights into the impact of various LLM configuration settings, such as binary decision forcing, model re-runs and temperature, on model's factuality. Models constrained to binary (true/false) choices exhibit reduced factuality compared to those allowing an 'unclear' option. Single inference at a low temperature setting matches the reliability of majority voting across various configurations. The insights gained highlight the need for culturally diverse and geographically inclusive model training and evaluation. This approach is key to achieving global equity in technology, distributing AI benefits fairly worldwide.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01765v1">LLMs Simulate Big Five Personality Traits: Further Evidence</a></div>
    <div class="paper-meta">
      📅 2024-01-31
    </div>
    <details class="paper-abstract">
      An empirical investigation into the simulation of the Big Five personality traits by large language models (LLMs), namely Llama2, GPT4, and Mixtral, is presented. We analyze the personality traits simulated by these models and their stability. This contributes to the broader understanding of the capabilities of LLMs to simulate personality traits and the respective implications for personalized human-computer interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16038v1">A Survey on Generative AI and LLM for Video Generation, Understanding, and Streaming</a></div>
    <div class="paper-meta">
      📅 2024-01-30
      | 💬 16 pages, 10 figures, 4 tables
    </div>
    <details class="paper-abstract">
      This paper offers an insightful examination of how currently top-trending AI technologies, i.e., generative artificial intelligence (Generative AI) and large language models (LLMs), are reshaping the field of video technology, including video generation, understanding, and streaming. It highlights the innovative use of these technologies in producing highly realistic videos, a significant leap in bridging the gap between real-world dynamics and digital creation. The study also delves into the advanced capabilities of LLMs in video understanding, demonstrating their effectiveness in extracting meaningful information from visual content, thereby enhancing our interaction with videos. In the realm of video streaming, the paper discusses how LLMs contribute to more efficient and user-centric streaming experiences, adapting content delivery to individual viewer preferences. This comprehensive review navigates through the current achievements, ongoing challenges, and future possibilities of applying Generative AI and LLMs to video-related tasks, underscoring the immense potential these technologies hold for advancing the field of video technology related to multimedia, networking, and AI communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.05934v3">Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encapsulate a vast amount of factual information within their pre-trained weights, as evidenced by their ability to answer diverse questions across different domains. However, this knowledge is inherently limited, relying heavily on the characteristics of the training data. Consequently, using external datasets to incorporate new information or refine the capabilities of LLMs on previously seen information poses a significant challenge. In this study, we compare two common approaches: unsupervised fine-tuning and retrieval-augmented generation (RAG). We evaluate both approaches on a variety of knowledge-intensive tasks across different topics. Our findings reveal that while unsupervised fine-tuning offers some improvement, RAG consistently outperforms it, both for existing knowledge encountered during training and entirely new knowledge. Moreover, we find that LLMs struggle to learn new factual information through unsupervised fine-tuning, and that exposing them to numerous variations of the same fact during training could alleviate this problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16788v1">Can Large Language Models be Trusted for Evaluation? Scalable Meta-Evaluation of LLMs as Evaluators via Agent Debate</a></div>
    <div class="paper-meta">
      📅 2024-01-30
    </div>
    <details class="paper-abstract">
      Despite the utility of Large Language Models (LLMs) across a wide range of tasks and scenarios, developing a method for reliably evaluating LLMs across varied contexts continues to be challenging. Modern evaluation approaches often use LLMs to assess responses generated by LLMs. However, the meta-evaluation conducted to assess the effectiveness of these LLMs as evaluators is typically constrained by the coverage of existing benchmarks or requires extensive human annotation. This underscores the urgency of methods for scalable meta-evaluation that can effectively, reliably, and efficiently evaluate the performance of LLMs as evaluators across diverse tasks and scenarios, particularly in potentially new, user-defined scenarios. To fill this gap, we propose ScaleEval, an agent-debate-assisted meta-evaluation framework that leverages the capabilities of multiple communicative LLM agents. This framework supports multi-round discussions to assist human annotators in discerning the most capable LLMs as evaluators, which significantly eases their workload in cases that used to require large-scale annotations during meta-evaluation. We release the code for our framework, which is publicly available at: \url{https://github.com/GAIR-NLP/scaleeval}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.02777v2">From LLM to Conversational Agent: A Memory Enhanced Architecture with Fine-Tuning of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-01-30
    </div>
    <details class="paper-abstract">
      This paper introduces RAISE (Reasoning and Acting through Scratchpad and Examples), an advanced architecture enhancing the integration of Large Language Models (LLMs) like GPT-4 into conversational agents. RAISE, an enhancement of the ReAct framework, incorporates a dual-component memory system, mirroring human short-term and long-term memory, to maintain context and continuity in conversations. It entails a comprehensive agent construction scenario, including phases like Conversation Selection, Scene Extraction, CoT Completion, and Scene Augmentation, leading to the LLMs Training phase. This approach appears to enhance agent controllability and adaptability in complex, multi-turn dialogues. Our preliminary evaluations in a real estate sales context suggest that RAISE has some advantages over traditional agents, indicating its potential for broader applications. This work contributes to the AI field by providing a robust framework for developing more context-aware and versatile conversational agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14698v2">Under the Surface: Tracking the Artifactuality of LLM-Generated Data</a></div>
    <div class="paper-meta">
      📅 2024-01-30
      | 💬 Core Authors: Debarati Das, Karin De Langis, Anna Martin-Boyle, Jaehyung Kim, Minhwa Lee and Zae Myung Kim | Project lead : Debarati Das | PI : Dongyeop Kang
    </div>
    <details class="paper-abstract">
      This work delves into the expanding role of large language models (LLMs) in generating artificial data. LLMs are increasingly employed to create a variety of outputs, including annotations, preferences, instruction prompts, simulated dialogues, and free text. As these forms of LLM-generated data often intersect in their application, they exert mutual influence on each other and raise significant concerns about the quality and diversity of the artificial data incorporated into training cycles, leading to an artificial data ecosystem. To the best of our knowledge, this is the first study to aggregate various types of LLM-generated text data, from more tightly constrained data like "task labels" to more lightly constrained "free-form text". We then stress test the quality and implications of LLM-generated artificial data, comparing it with human data across various existing benchmarks. Despite artificial data's capability to match human performance, this paper reveals significant hidden disparities, especially in complex tasks where LLMs often miss the nuanced understanding of intrinsic human-generated content. This study critically examines diverse LLM-generated data and emphasizes the need for ethical practices in data creation and when using LLMs. It highlights the LLMs' shortcomings in replicating human traits and behaviors, underscoring the importance of addressing biases and artifacts produced in LLM-generated content for future research and development. All data and code are available on our project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16638v1">Breaking Free Transformer Models: Task-specific Context Attribution Promises Improved Generalizability Without Fine-tuning Pre-trained LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-30
      | 💬 8 pages, 3 figures, 5 tables, To be published in 2024 AAAI workshop on Responsible Language Models (ReLM)
    </div>
    <details class="paper-abstract">
      Fine-tuning large pre-trained language models (LLMs) on particular datasets is a commonly employed strategy in Natural Language Processing (NLP) classification tasks. However, this approach usually results in a loss of models generalizability. In this paper, we present a framework that allows for maintaining generalizability, and enhances the performance on the downstream task by utilizing task-specific context attribution. We show that a linear transformation of the text representation from any transformer model using the task-specific concept operator results in a projection onto the latent concept space, referred to as context attribution in this paper. The specific concept operator is optimized during the supervised learning stage via novel loss functions. The proposed framework demonstrates that context attribution of the text representation for each task objective can improve the capacity of the discriminator function and thus achieve better performance for the classification task. Experimental results on three datasets, namely HateXplain, IMDB reviews, and Social Media Attributions, illustrate that the proposed model attains superior accuracy and generalizability. Specifically, for the non-fine-tuned BERT on the HateXplain dataset, we observe 8% improvement in accuracy and 10% improvement in F1-score. Whereas for the IMDB dataset, fine-tuned state-of-the-art XLNet is outperformed by 1% for both accuracy and F1-score. Furthermore, in an out-of-domain cross-dataset test, DistilBERT fine-tuned on the IMDB dataset in conjunction with the proposed model improves the F1-score on the HateXplain dataset by 7%. For the Social Media Attributions dataset of YouTube comments, we observe 5.2% increase in F1-metric. The proposed framework is implemented with PyTorch and provided open-source on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16603v1">LeftoverLocals: Listening to LLM Responses Through Leaked GPU Local Memory</a></div>
    <div class="paper-meta">
      📅 2024-01-29
    </div>
    <details class="paper-abstract">
      This paper describes LeftoverLocals: a vulnerability that allows data recovery from GPU memory created by another process on Apple, Qualcomm, and AMD GPUs. LeftoverLocals impacts the security posture of GPU applications, with particular significance to LLMs and ML models that run on impacted GPUs. By recovering local memory, an optimized GPU memory region, we built a PoC where an attacker can listen into another user's interactive LLM session (e.g., llama.cpp) across process or container boundaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16577v1">LLMs as On-demand Customizable Service</a></div>
    <div class="paper-meta">
      📅 2024-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable language understanding and generation capabilities. However, training, deploying, and accessing these models pose notable challenges, including resource-intensive demands, extended training durations, and scalability issues. To address these issues, we introduce a concept of hierarchical, distributed LLM architecture that aims at enhancing the accessibility and deployability of LLMs across heterogeneous computing platforms, including general-purpose computers (e.g., laptops) and IoT-style devices (e.g., embedded systems). By introducing a "layered" approach, the proposed architecture enables on-demand accessibility to LLMs as a customizable service. This approach also ensures optimal trade-offs between the available computational resources and the user's application needs. We envision that the concept of hierarchical LLM will empower extensive, crowd-sourced user bases to harness the capabilities of LLMs, thereby fostering advancements in AI technology in general.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16558v1">Diverse, but Divisive: LLMs Can Exaggerate Gender Differences in Opinion Related to Harms of Misinformation</a></div>
    <div class="paper-meta">
      📅 2024-01-29
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      The pervasive spread of misinformation and disinformation poses a significant threat to society. Professional fact-checkers play a key role in addressing this threat, but the vast scale of the problem forces them to prioritize their limited resources. This prioritization may consider a range of factors, such as varying risks of harm posed to specific groups of people. In this work, we investigate potential implications of using a large language model (LLM) to facilitate such prioritization. Because fact-checking impacts a wide range of diverse segments of society, it is important that diverse views are represented in the claim prioritization process. This paper examines whether a LLM can reflect the views of various groups when assessing the harms of misinformation, focusing on gender as a primary variable. We pose two central questions: (1) To what extent do prompts with explicit gender references reflect gender differences in opinion in the United States on topics of social relevance? and (2) To what extent do gender-neutral prompts align with gendered viewpoints on those topics? To analyze these questions, we present the TopicMisinfo dataset, containing 160 fact-checked claims from diverse topics, supplemented by nearly 1600 human annotations with subjective perceptions and annotator demographics. Analyzing responses to gender-specific and neutral prompts, we find that GPT 3.5-Turbo reflects empirically observed gender differences in opinion but amplifies the extent of these differences. These findings illuminate AI's complex role in moderating online communication, with implications for fact-checkers, algorithm designers, and the use of crowd-workers as annotators. We also release the TopicMisinfo dataset to support continuing research in the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01742v1">Towards Optimizing the Costs of LLM Usage</a></div>
    <div class="paper-meta">
      📅 2024-01-29
      | 💬 8 pages + Appendix, Total 12 pages
    </div>
    <details class="paper-abstract">
      Generative AI and LLMs in particular are heavily used nowadays for various document processing tasks such as question answering and summarization. However, different LLMs come with different capabilities for different tasks as well as with different costs, tokenization, and latency. In fact, enterprises are already incurring huge costs of operating or using LLMs for their respective use cases. In this work, we propose optimizing the usage costs of LLMs by estimating their output quality (without actually invoking the LLMs), and then solving an optimization routine for the LLM selection to either keep costs under a budget, or minimize the costs, in a quality and latency aware manner. We propose a model to predict the output quality of LLMs on document processing tasks like summarization, followed by an LP rounding algorithm to optimize the selection of LLMs. We study optimization problems trading off the quality and costs, both theoretically and empirically. We further propose a sentence simplification model for reducing the number of tokens in a controlled manner. Additionally, we propose several deterministic heuristics for reducing tokens in a quality aware manner, and study the related optimization problem of applying the heuristics optimizing the quality and cost trade-off. We perform extensive empirical validation of our methods on not only enterprise datasets but also on open-source datasets, annotated by us, and show that we perform much better compared to closest baselines. Our methods reduce costs by 40%- 90% while improving quality by 4%-7%. We will release the annotated open source datasets to the community for further research and exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16186v1">An Empirical Study on Usage and Perceptions of LLMs in a Software Engineering Project</a></div>
    <div class="paper-meta">
      📅 2024-01-29
      | 💬 8 pages, 6 figures, accepted for publication at the LLM4Code workshop @ ICSE 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) represent a leap in artificial intelligence, excelling in tasks using human language(s). Although the main focus of general-purpose LLMs is not code generation, they have shown promising results in the domain. However, the usefulness of LLMs in an academic software engineering project has not been fully explored yet. In this study, we explore the usefulness of LLMs for 214 students working in teams consisting of up to six members. Notably, in the academic course through which this study is conducted, students were encouraged to integrate LLMs into their development tool-chain, in contrast to most other academic courses that explicitly prohibit the use of LLMs. In this paper, we analyze the AI-generated code, prompts used for code generation, and the human intervention levels to integrate the code into the code base. We also conduct a perception study to gain insights into the perceived usefulness, influencing factors, and future outlook of LLM from a computer science student's perspective. Our findings suggest that LLMs can play a crucial role in the early stages of software development, especially in generating foundational code structures, and helping with syntax and error debugging. These insights provide us with a framework on how to effectively utilize LLMs as a tool to enhance the productivity of software engineering students, and highlight the necessity of shifting the educational focus toward preparing students for successful human-AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16107v1">Beyond Direct Diagnosis: LLM-based Multi-Specialist Agent Consultation for Automatic Diagnosis</a></div>
    <div class="paper-meta">
      📅 2024-01-29
    </div>
    <details class="paper-abstract">
      Automatic diagnosis is a significant application of AI in healthcare, where diagnoses are generated based on the symptom description of patients. Previous works have approached this task directly by modeling the relationship between the normalized symptoms and all possible diseases. However, in the clinical diagnostic process, patients are initially consulted by a general practitioner and, if necessary, referred to specialists in specific domains for a more comprehensive evaluation. The final diagnosis often emerges from a collaborative consultation among medical specialist groups. Recently, large language models have shown impressive capabilities in natural language understanding. In this study, we adopt tuning-free LLM-based agents as medical practitioners and propose the Agent-derived Multi-Specialist Consultation (AMSC) framework to model the diagnosis process in the real world by adaptively fusing probability distributions of agents over potential diseases. Experimental results demonstrate the superiority of our approach compared with baselines. Notably, our approach requires significantly less parameter updating and training time, enhancing efficiency and practical utility. Furthermore, we delve into a novel perspective on the role of implicit symptoms within the context of automatic diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.13340v2">Towards LLM-guided Causal Explainability for Black-box Text Classifiers</a></div>
    <div class="paper-meta">
      📅 2024-01-29
      | 💬 Camera-ready for AAAI ReLM 2024
    </div>
    <details class="paper-abstract">
      With the advent of larger and more complex deep learning models, such as in Natural Language Processing (NLP), model qualities like explainability and interpretability, albeit highly desirable, are becoming harder challenges to tackle and solve. For example, state-of-the-art models in text classification are black-box by design. Although standard explanation methods provide some degree of explainability, these are mostly correlation-based methods and do not provide much insight into the model. The alternative of causal explainability is more desirable to achieve but extremely challenging in NLP due to a variety of reasons. Inspired by recent endeavors to utilize Large Language Models (LLMs) as experts, in this work, we aim to leverage the instruction-following and textual understanding capabilities of recent state-of-the-art LLMs to facilitate causal explainability via counterfactual explanation generation for black-box text classifiers. To do this, we propose a three-step pipeline via which, we use an off-the-shelf LLM to: (1) identify the latent or unobserved features in the input text, (2) identify the input features associated with the latent features, and finally (3) use the identified input features to generate a counterfactual explanation. We experiment with our pipeline on multiple NLP text classification datasets, with several recent LLMs, and present interesting and promising findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.13500v3">Enhancing Student Performance Prediction on Learnersourced Questions with SGNN-LLM Synergy</a></div>
    <div class="paper-meta">
      📅 2024-01-29
    </div>
    <details class="paper-abstract">
      Learnersourcing offers great potential for scalable education through student content creation. However, predicting student performance on learnersourced questions, which is essential for personalizing the learning experience, is challenging due to the inherent noise in student-generated data. Moreover, while conventional graph-based methods can capture the complex network of student and question interactions, they often fall short under cold start conditions where limited student engagement with questions yields sparse data. To address both challenges, we introduce an innovative strategy that synergizes the potential of integrating Signed Graph Neural Networks (SGNNs) and Large Language Model (LLM) embeddings. Our methodology employs a signed bipartite graph to comprehensively model student answers, complemented by a contrastive learning framework that enhances noise resilience. Furthermore, LLM's contribution lies in generating foundational question embeddings, proving especially advantageous in addressing cold start scenarios characterized by limited graph data. Validation across five real-world datasets sourced from the PeerWise platform underscores our approach's effectiveness. Our method outperforms baselines, showcasing enhanced predictive accuracy and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01730v1">Evaluating LLM -- Generated Multimodal Diagnosis from Medical Images and Symptom Analysis</a></div>
    <div class="paper-meta">
      📅 2024-01-28
      | 💬 Department of Informatics, University of Piraeus, Greece
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) constitute a breakthrough state-of-the-art Artificial Intelligence technology which is rapidly evolving and promises to aid in medical diagnosis. However, the correctness and the accuracy of their returns has not yet been properly evaluated. In this work, we propose an LLM evaluation paradigm that incorporates two independent steps of a novel methodology, namely (1) multimodal LLM evaluation via structured interactions and (2) follow-up, domain-specific analysis based on data extracted via the previous interactions. Using this paradigm, (1) we evaluate the correctness and accuracy of LLM-generated medical diagnosis with publicly available multimodal multiple-choice questions(MCQs) in the domain of Pathology and (2) proceed to a systemic and comprehensive analysis of extracted results. We used GPT-4-Vision-Preview as the LLM to respond to complex, medical questions consisting of both images and text, and we explored a wide range of diseases, conditions, chemical compounds, and related entity types that are included in the vast knowledge domain of Pathology. GPT-4-Vision-Preview performed quite well, scoring approximately 84\% of correct diagnoses. Next, we further analyzed the findings of our work, following an analytical approach which included Image Metadata Analysis, Named Entity Recognition and Knowledge Graphs. Weaknesses of GPT-4-Vision-Preview were revealed on specific knowledge paths, leading to a further understanding of its shortcomings in specific areas. Our methodology and findings are not limited to the use of GPT-4-Vision-Preview, but a similar approach can be followed to evaluate the usefulness and accuracy of other LLMs and, thus, improve their use with further optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15589v1">OpineBot: Class Feedback Reimagined Using a Conversational LLM</a></div>
    <div class="paper-meta">
      📅 2024-01-28
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Conventional class feedback systems often fall short, relying on static, unengaging surveys offering little incentive for student participation. To address this, we present OpineBot, a novel system employing large language models (LLMs) to conduct personalized, conversational class feedback via chatbot interface. We assessed OpineBot's effectiveness in a user study with 20 students from an Indian university's Operating-Systems class, utilizing surveys and interviews to analyze their experiences. Findings revealed a resounding preference for OpineBot compared to conventional methods, highlighting its ability to engage students, produce deeper feedback, offering a dynamic survey experience. This research represents a work in progress, providing early results, marking a significant step towards revolutionizing class feedback through LLM-based technology, promoting student engagement, and leading to richer data for instructors. This ongoing research presents preliminary findings and marks a notable advancement in transforming classroom feedback using LLM-based technology to enhance student engagement and generate comprehensive data for educators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15463v1">DataFrame QA: A Universal LLM Framework on DataFrame Question Answering Without Data Exposure</a></div>
    <div class="paper-meta">
      📅 2024-01-27
    </div>
    <details class="paper-abstract">
      This paper introduces DataFrame question answering (QA), a novel task that utilizes large language models (LLMs) to generate Pandas queries for information retrieval and data analysis on dataframes, emphasizing safe and non-revealing data handling. Our method, which solely relies on dataframe column names, not only ensures data privacy but also significantly reduces the context window in the prompt, streamlining information processing and addressing major challenges in LLM-based data analysis. We propose DataFrame QA as a comprehensive framework that includes safe Pandas query generation and code execution. Various LLMs, notably GPT-4, are evaluated using the pass@1 metric on the renowned WikiSQL and our newly developed 'UCI-DataFrameQA', tailored for complex data analysis queries. Our findings indicate that GPT-4 achieves pass@1 rates of 86% on WikiSQL and 97% on UCI-DataFrameQA, underscoring its capability in securely retrieving and aggregating dataframe values and conducting sophisticated data analyses. This approach, deployable in a zero-shot manner without prior training or adjustments, proves to be highly adaptable and secure for diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15449v1">Learning to Trust Your Feelings: Leveraging Self-awareness in LLMs for Hallucination Mitigation</a></div>
    <div class="paper-meta">
      📅 2024-01-27
    </div>
    <details class="paper-abstract">
      We evaluate the ability of Large Language Models (LLMs) to discern and express their internal knowledge state, a key factor in countering factual hallucination and ensuring reliable application of LLMs. We observe a robust self-awareness of internal knowledge state in LLMs, evidenced by over 85% accuracy in knowledge probing. However, LLMs often fail to express their internal knowledge during generation, leading to factual hallucinations. We develop an automated hallucination annotation tool, Dreamcatcher, which merges knowledge probing and consistency checking methods to rank factual preference data. Using knowledge preference as reward, We propose a Reinforcement Learning from Knowledge Feedback (RLKF) training framework, leveraging reinforcement learning to enhance the factuality and honesty of LLMs. Our experiments across multiple models show that RLKF training effectively enhances the ability of models to utilize their internal knowledge state, boosting performance in a variety of knowledge-based and honesty-related tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.04892v2">Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-27
      | 💬 Project page: https://allenai.github.io/persona-bias. Paper to appear at ICLR 2024. Added results for other LLMs in v2 (similar findings)
    </div>
    <details class="paper-abstract">
      Recent works have showcased the ability of LLMs to embody diverse personas in their responses, exemplified by prompts like 'You are Yoda. Explain the Theory of Relativity.' While this ability allows personalization of LLMs and enables human behavior simulation, its effect on LLMs' capabilities remains unclear. To fill this gap, we present the first extensive study of the unintended side-effects of persona assignment on the ability of LLMs to perform basic reasoning tasks. Our study covers 24 reasoning datasets, 4 LLMs, and 19 diverse personas (e.g. an Asian person) spanning 5 socio-demographic groups. Our experiments unveil that LLMs harbor deep rooted bias against various socio-demographics underneath a veneer of fairness. While they overtly reject stereotypes when explicitly asked ('Are Black people less skilled at mathematics?'), they manifest stereotypical and erroneous presumptions when asked to answer questions while adopting a persona. These can be observed as abstentions in responses, e.g., 'As a Black person, I can't answer this question as it requires math knowledge', and generally result in a substantial performance drop. Our experiments with ChatGPT-3.5 show that this bias is ubiquitous - 80% of our personas demonstrate bias; it is significant - some datasets show performance drops of 70%+; and can be especially harmful for certain groups - some personas suffer statistically significant drops on 80%+ of the datasets. Overall, all 4 LLMs exhibit this bias to varying extents, with GPT-4-Turbo showing the least but still a problematic amount of bias (evident in 42% of the personas). Further analysis shows that these persona-induced errors can be hard-to-discern and hard-to-avoid. Our findings serve as a cautionary tale that the practice of assigning personas to LLMs - a trend on the rise - can surface their deep-rooted biases and have unforeseeable and detrimental side-effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14935v1">Appropriateness of LLM-equipped Robotic Well-being Coach Language in the Workplace: A Qualitative Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-01-26
    </div>
    <details class="paper-abstract">
      Robotic coaches have been recently investigated to promote mental well-being in various contexts such as workplaces and homes. With the widespread use of Large Language Models (LLMs), HRI researchers are called to consider language appropriateness when using such generated language for robotic mental well-being coaches in the real world. Therefore, this paper presents the first work that investigated the language appropriateness of robot mental well-being coach in the workplace. To this end, we conducted an empirical study that involved 17 employees who interacted over 4 weeks with a robotic mental well-being coach equipped with LLM-based capabilities. After the study, we individually interviewed them and we conducted a focus group of 1.5 hours with 11 of them. The focus group consisted of: i) an ice-breaking activity, ii) evaluation of robotic coach language appropriateness in various scenarios, and iii) listing shoulds and shouldn'ts for designing appropriate robotic coach language for mental well-being. From our qualitative evaluation, we found that a language-appropriate robotic coach should (1) ask deep questions which explore feelings of the coachees, rather than superficial questions, (2) express and show emotional and empathic understanding of the context, and (3) not make any assumptions without clarifying with follow-up questions to avoid bias and stereotyping. These results can inform the design of language-appropriate robotic coach to promote mental well-being in real-world contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14931v1">Do LLMs Dream of Ontologies?</a></div>
    <div class="paper-meta">
      📅 2024-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently revolutionized automated text understanding and generation. The performance of these models relies on the high number of parameters of the underlying neural architectures, which allows LLMs to memorize part of the vast quantity of data seen during the training. This paper investigates whether and to what extent general-purpose pre-trained LLMs have memorized information from known ontologies. Our results show that LLMs partially know ontologies: they can, and do indeed, memorize concepts from ontologies mentioned in the text, but the level of memorization of their concepts seems to vary proportionally to their popularity on the Web, the primary source of their training material. We additionally propose new metrics to estimate the degree of memorization of ontological information in LLMs by measuring the consistency of the output produced across different prompt repetitions, query languages, and degrees of determinism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.12060v3">FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-01-26
      | 💬 Accepted as AAAI-24 Oral paper; Knowledge Base Question Answering; Large Language Model; Data Generation; Few-Shot & Zero-Shot
    </div>
    <details class="paper-abstract">
      Knowledge base question answering (KBQA) is a critical yet challenging task due to the vast number of entities within knowledge bases and the diversity of natural language questions posed by users. Unfortunately, the performance of most KBQA models tends to decline significantly in real-world scenarios where high-quality annotated data is insufficient. To mitigate the burden associated with manual annotation, we introduce FlexKBQA by utilizing Large Language Models (LLMs) as program translators for addressing the challenges inherent in the few-shot KBQA task. Specifically, FlexKBQA leverages automated algorithms to sample diverse programs, such as SPARQL queries, from the knowledge base, which are subsequently converted into natural language questions via LLMs. This synthetic dataset facilitates training a specialized lightweight model for the KB. Additionally, to reduce the barriers of distribution shift between synthetic data and real user questions, FlexKBQA introduces an executionguided self-training method to iterative leverage unlabeled user questions. Furthermore, we explore harnessing the inherent reasoning capability of LLMs to enhance the entire framework. Consequently, FlexKBQA delivers substantial flexibility, encompassing data annotation, deployment, and being domain agnostic. Through extensive experiments on GrailQA, WebQSP, and KQA Pro, we observe that under the few-shot even the more challenging zero-shot scenarios, FlexKBQA achieves impressive results with a few annotations, surpassing all previous baselines and even approaching the performance of supervised models, achieving a remarkable 93% performance relative to the fully-supervised models. We posit that FlexKBQA represents a significant advancement towards exploring better integration of large and lightweight models. The code is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18628v2">Personalised Distillation: Empowering Open-Sourced LLMs with Adaptive Learning for Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-01-26
      | 💬 Accepted to EMNLP 2023; Codes at: https://github.com/SalesforceAIResearch/PersDistill
    </div>
    <details class="paper-abstract">
      With the rise of powerful closed-sourced LLMs (ChatGPT, GPT-4), there are increasing interests in distilling the capabilies of close-sourced LLMs to smaller open-sourced LLMs. Previous distillation methods usually prompt ChatGPT to generate a set of instructions and answers, for the student model to learn. However, such standard distillation approach neglects the merits and conditions of the student model. Inspired by modern teaching principles, we design a personalised distillation process, in which the student attempts to solve a task first, then the teacher provides an adaptive refinement for the student to improve. Instead of feeding the student with teacher's prior, personalised distillation enables personalised learning for the student model, as it only learns on examples it makes mistakes upon and learns to improve its own solution. On code generation, personalised distillation consistently outperforms standard distillation with only one third of the data. With only 2.5-3K personalised examples that incur a data-collection cost of 4-6$, we boost CodeGen-mono-16B by 7% to achieve 36.4% pass@1 and StarCoder by 12.2% to achieve 45.8% pass@1 on HumanEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14523v1">Empathy and the Right to Be an Exception: What LLMs Can and Cannot Do</a></div>
    <div class="paper-meta">
      📅 2024-01-25
    </div>
    <details class="paper-abstract">
      Advances in the performance of large language models (LLMs) have led some researchers to propose the emergence of theory of mind (ToM) in artificial intelligence (AI). LLMs can attribute beliefs, desires, intentions, and emotions, and they will improve in their accuracy. Rather than employing the characteristically human method of empathy, they learn to attribute mental states by recognizing linguistic patterns in a dataset that typically do not include that individual. We ask whether LLMs' inability to empathize precludes them from honoring an individual's right to be an exception, that is, from making assessments of character and predictions of behavior that reflect appropriate sensitivity to a person's individuality. Can LLMs seriously consider an individual's claim that their case is different based on internal mental states like beliefs, desires, and intentions, or are they limited to judging that case based on its similarities to others? We propose that the method of empathy has special significance for honoring the right to be an exception that is distinct from the value of predictive accuracy, at which LLMs excel. We conclude by considering whether using empathy to consider exceptional cases has intrinsic or merely practical value and we introduce conceptual and empirical avenues for advancing this investigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10920v1">Designing Silicon Brains using LLM: Leveraging ChatGPT for Automated Description of a Spiking Neuron Array</a></div>
    <div class="paper-meta">
      📅 2024-01-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made headlines for synthesizing correct-sounding responses to a variety of prompts, including code generation. In this paper, we present the prompts used to guide ChatGPT4 to produce a synthesizable and functional verilog description for the entirety of a programmable Spiking Neuron Array ASIC. This design flow showcases the current state of using ChatGPT4 for natural language driven hardware design. The AI-generated design was verified in simulation using handcrafted testbenches and has been submitted for fabrication in Skywater 130nm through Tiny Tapeout 5 using an open-source EDA flow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01711v1">LLM on FHIR -- Demystifying Health Records</a></div>
    <div class="paper-meta">
      📅 2024-01-25
      | 💬 Pre-print of the paper submitted to the Call for Papers for the Special Focus Issue on ChatGPT and Large Language Models (LLMs) in Biomedicine and Health at the Journal of the American Medical Informatics Association: https://academic.oup.com/jamia/pages/call-for-papers-for-special-focus-issue
    </div>
    <details class="paper-abstract">
      Objective: To enhance health literacy and accessibility of health information for a diverse patient population by developing a patient-centered artificial intelligence (AI) solution using large language models (LLMs) and Fast Healthcare Interoperability Resources (FHIR) application programming interfaces (APIs). Materials and Methods: The research involved developing LLM on FHIR, an open-source mobile application allowing users to interact with their health records using LLMs. The app is built on Stanford's Spezi ecosystem and uses OpenAI's GPT-4. A pilot study was conducted with the SyntheticMass patient dataset and evaluated by medical experts to assess the app's effectiveness in increasing health literacy. The evaluation focused on the accuracy, relevance, and understandability of the LLM's responses to common patient questions. Results: LLM on FHIR demonstrated varying but generally high degrees of accuracy and relevance in providing understandable health information to patients. The app effectively translated medical data into patient-friendly language and was able to adapt its responses to different patient profiles. However, challenges included variability in LLM responses and the need for precise filtering of health data. Discussion and Conclusion: LLMs offer significant potential in improving health literacy and making health records more accessible. LLM on FHIR, as a pioneering application in this field, demonstrates the feasibility and challenges of integrating LLMs into patient care. While promising, the implementation and pilot also highlight risks such as inconsistent responses and the importance of replicable output. Future directions include better resource identification mechanisms and executing LLMs on-device to enhance privacy and reduce costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.02981v2">Fine-tuning and Utilization Methods of Domain-specific LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-24
    </div>
    <details class="paper-abstract">
      Recent releases of pre-trained Large Language Models (LLMs) have gained considerable traction, yet research on fine-tuning and employing domain-specific LLMs remains scarce. This study investigates approaches for fine-tuning and leveraging domain-specific LLMs, highlighting trends in LLMs, foundational models, and methods for domain-specific pre-training. Focusing on the financial sector, it details dataset selection, preprocessing, model choice, and considerations crucial for LLM fine-tuning in finance. Addressing the unique characteristics of financial data, the study explores the construction of domain-specific vocabularies and considerations for security and regulatory compliance. In the practical application of LLM fine-tuning, the study outlines the procedure and implementation for generating domain-specific LLMs in finance. Various financial cases, including stock price prediction, sentiment analysis of financial news, automated document processing, research, information extraction, and customer service enhancement, are exemplified. The study explores the potential of LLMs in the financial domain, identifies limitations, and proposes directions for improvement, contributing valuable insights for future research. Ultimately, it advances natural language processing technology in business, suggesting proactive LLM utilization in financial services across industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.13598v1">Consistency Guided Knowledge Retrieval and Denoising in LLMs for Zero-shot Document-level Relation Triplet Extraction</a></div>
    <div class="paper-meta">
      📅 2024-01-24
      | 💬 Accepted by WWW 2024
    </div>
    <details class="paper-abstract">
      Document-level Relation Triplet Extraction (DocRTE) is a fundamental task in information systems that aims to simultaneously extract entities with semantic relations from a document. Existing methods heavily rely on a substantial amount of fully labeled data. However, collecting and annotating data for newly emerging relations is time-consuming and labor-intensive. Recent advanced Large Language Models (LLMs), such as ChatGPT and LLaMA, exhibit impressive long-text generation capabilities, inspiring us to explore an alternative approach for obtaining auto-labeled documents with new relations. In this paper, we propose a Zero-shot Document-level Relation Triplet Extraction (ZeroDocRTE) framework, which generates labeled data by retrieval and denoising knowledge from LLMs, called GenRDK. Specifically, we propose a chain-of-retrieval prompt to guide ChatGPT to generate labeled long-text data step by step. To improve the quality of synthetic data, we propose a denoising strategy based on the consistency of cross-document knowledge. Leveraging our denoised synthetic data, we proceed to fine-tune the LLaMA2-13B-Chat for extracting document-level relation triplets. We perform experiments for both zero-shot document-level relation and triplet extraction on two public datasets. The experimental results illustrate that our GenRDK framework outperforms strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.06082v2">VELMA: Verbalization Embodiment of LLM Agents for Vision and Language Navigation in Street View</a></div>
    <div class="paper-meta">
      📅 2024-01-24
      | 💬 Accepted at AAAI 2024
    </div>
    <details class="paper-abstract">
      Incremental decision making in real-world environments is one of the most challenging tasks in embodied artificial intelligence. One particularly demanding scenario is Vision and Language Navigation~(VLN) which requires visual and natural language understanding as well as spatial and temporal reasoning capabilities. The embodied agent needs to ground its understanding of navigation instructions in observations of a real-world environment like Street View. Despite the impressive results of LLMs in other research areas, it is an ongoing problem of how to best connect them with an interactive visual environment. In this work, we propose VELMA, an embodied LLM agent that uses a verbalization of the trajectory and of visual environment observations as contextual prompt for the next action. Visual information is verbalized by a pipeline that extracts landmarks from the human written navigation instructions and uses CLIP to determine their visibility in the current panorama view. We show that VELMA is able to successfully follow navigation instructions in Street View with only two in-context examples. We further finetune the LLM agent on a few thousand examples and achieve 25%-30% relative improvement in task completion over the previous state-of-the-art for two datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.13504v1">Research about the Ability of LLM in the Tamper-Detection Area</a></div>
    <div class="paper-meta">
      📅 2024-01-24
    </div>
    <details class="paper-abstract">
      In recent years, particularly since the early 2020s, Large Language Models (LLMs) have emerged as the most powerful AI tools in addressing a diverse range of challenges, from natural language processing to complex problem-solving in various domains. In the field of tamper detection, LLMs are capable of identifying basic tampering activities.To assess the capabilities of LLMs in more specialized domains, we have collected five different LLMs developed by various companies: GPT-4, LLaMA, Bard, ERNIE Bot 4.0, and Tongyi Qianwen. This diverse range of models allows for a comprehensive evaluation of their performance in detecting sophisticated tampering instances.We devised two domains of detection: AI-Generated Content (AIGC) detection and manipulation detection. AIGC detection aims to test the ability to distinguish whether an image is real or AI-generated. Manipulation detection, on the other hand, focuses on identifying tampered images. According to our experiments, most LLMs can identify composite pictures that are inconsistent with logic, and only more powerful LLMs can distinguish logical, but visible signs of tampering to the human eye. All of the LLMs can't identify carefully forged images and very realistic images generated by AI. In the area of tamper detection, LLMs still have a long way to go, particularly in reliably identifying highly sophisticated forgeries and AI-generated images that closely mimic reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08517v3">Supporting Student Decisions on Learning Recommendations: An LLM-Based Chatbot with Knowledge Graph Contextualization for Conversational Explainability and Mentoring</a></div>
    <div class="paper-meta">
      📅 2024-01-24
    </div>
    <details class="paper-abstract">
      Student commitment towards a learning recommendation is not separable from their understanding of the reasons it was recommended to them; and their ability to modify it based on that understanding. Among explainability approaches, chatbots offer the potential to engage the student in a conversation, similar to a discussion with a peer or a mentor. The capabilities of chatbots, however, are still not sufficient to replace a human mentor, despite the advancements of generative AI (GenAI) and large language models (LLM). Therefore, we propose an approach to utilize chatbots as mediators of the conversation and sources of limited and controlled generation of explanations, to harvest the potential of LLMs while reducing their potential risks at the same time. The proposed LLM-based chatbot supports students in understanding learning-paths recommendations. We use a knowledge graph (KG) as a human-curated source of information, to regulate the LLM's output through defining its prompt's context. A group chat approach is developed to connect students with human mentors, either on demand or in cases that exceed the chatbot's pre-defined tasks. We evaluate the chatbot with a user study, to provide a proof-of-concept and highlight the potential requirements and limitations of utilizing chatbots in conversational explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.13245v1">GraphiMind: LLM-centric Interface for Information Graphics Design</a></div>
    <div class="paper-meta">
      📅 2024-01-24
    </div>
    <details class="paper-abstract">
      Information graphics are pivotal in effective information dissemination and storytelling. However, creating such graphics is extremely challenging for non-professionals, since the design process requires multifaceted skills and comprehensive knowledge. Thus, despite the many available authoring tools, a significant gap remains in enabling non-experts to produce compelling information graphics seamlessly, especially from scratch. Recent breakthroughs show that Large Language Models (LLMs), especially when tool-augmented, can autonomously engage with external tools, making them promising candidates for enabling innovative graphic design applications. In this work, we propose a LLM-centric interface with the agent GraphiMind for automatic generation, recommendation, and composition of information graphics design resources, based on user intent expressed through natural language. Our GraphiMind integrates a Textual Conversational Interface, powered by tool-augmented LLM, with a traditional Graphical Manipulation Interface, streamlining the entire design process from raw resource curation to composition and refinement. Extensive evaluations highlight our tool's proficiency in simplifying the design process, opening avenues for its use by non-professional users. Moreover, we spotlight the potential of LLMs in reshaping the domain of information graphics design, offering a blend of automation, versatility, and user-centric interactivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.08535v3">Formally Specifying the High-Level Behavior of LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2024-01-24
      | 💬 Preprint under review
    </div>
    <details class="paper-abstract">
      Autonomous, goal-driven agents powered by LLMs have recently emerged as promising tools for solving challenging problems without the need for task-specific finetuned models that can be expensive to procure. Currently, the design and implementation of such agents is ad hoc, as the wide variety of tasks that LLM-based agents may be applied to naturally means there can be no one-size-fits-all approach to agent design. In this work we aim to alleviate the difficulty of designing and implementing new agents by proposing a minimalistic generation framework that simplifies the process of building agents. The framework we introduce allows the user to define desired agent behaviors in a high-level, declarative specification that is then used to construct a decoding monitor which guarantees the LLM will produce an output exhibiting the desired behavior. Our declarative approach, in which the behavior is described without concern for how it should be implemented or enforced, enables rapid design, implementation, and experimentation with different LLM-based agents. We demonstrate how the proposed framework can be used to implement recent LLM-based agents (e.g., ReACT), and show how the flexibility of our approach can be leveraged to define a new agent with more complex behavior, the Plan-Act-Summarize-Solve (PASS) agent. Lastly, we demonstrate that our method outperforms other agents on multiple popular reasoning-centric question-answering benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.13218v1">ULTRA: Unleash LLMs' Potential for Event Argument Extraction through Hierarchical Modeling and Pair-wise Refinement</a></div>
    <div class="paper-meta">
      📅 2024-01-24
    </div>
    <details class="paper-abstract">
      Structural extraction of events within discourse is critical since it avails a deeper understanding of communication patterns and behavior trends. Event argument extraction (EAE), at the core of event-centric understanding, is the task of identifying role-specific text spans (i.e., arguments) for a given event. Document-level EAE (DocEAE) focuses on arguments that are scattered across an entire document. In this work, we explore the capabilities of open source Large Language Models (LLMs), i.e., Flan-UL2, for the DocEAE task. To this end, we propose ULTRA, a hierarchical framework that extracts event arguments more cost-effectively -- the method needs as few as 50 annotations and doesn't require hitting costly API endpoints. Further, it alleviates the positional bias issue intrinsic to LLMs. ULTRA first sequentially reads text chunks of a document to generate a candidate argument set, upon which ULTRA learns to drop non-pertinent candidates through self-refinement. We further introduce LEAFER to address the challenge LLMs face in locating the exact boundary of an argument span. ULTRA outperforms strong baselines, which include strong supervised models and ChatGPT, by 9.8% when evaluated by the exact match (EM) metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.13136v1">The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts</a></div>
    <div class="paper-meta">
      📅 2024-01-23
    </div>
    <details class="paper-abstract">
      As the influence of large language models (LLMs) spans across global communities, their safety challenges in multilingual settings become paramount for alignment research. This paper examines the variations in safety challenges faced by LLMs across different languages and discusses approaches to alleviating such concerns. By comparing how state-of-the-art LLMs respond to the same set of malicious prompts written in higher- vs. lower-resource languages, we observe that (1) LLMs tend to generate unsafe responses much more often when a malicious prompt is written in a lower-resource language, and (2) LLMs tend to generate more irrelevant responses to malicious prompts in lower-resource languages. To understand where the discrepancy can be attributed, we study the effect of instruction tuning with reinforcement learning from human feedback (RLHF) or supervised finetuning (SFT) on the HH-RLHF dataset. Surprisingly, while training with high-resource languages improves model alignment, training in lower-resource languages yields minimal improvement. This suggests that the bottleneck of cross-lingual alignment is rooted in the pretraining stage. Our findings highlight the challenges in cross-lingual LLM safety, and we hope they inform future research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06373v2">How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-23
      | 💬 14 pages of the main text, qualitative examples of jailbreaks may be harmful in nature
    </div>
    <details class="paper-abstract">
      Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11803v2">NLP for Maternal Healthcare: Perspectives and Guiding Principles in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-23
    </div>
    <details class="paper-abstract">
      Ethical frameworks for the use of natural language processing (NLP) are urgently needed to shape how large language models (LLMs) and similar tools are used for healthcare applications. Healthcare faces existing challenges including the balance of power in clinician-patient relationships, systemic health disparities, historical injustices, and economic constraints. Drawing directly from the voices of those most affected, and focusing on a case study of a specific healthcare setting, we propose a set of guiding principles for the use of NLP in maternal healthcare. We led an interactive session centered on an LLM-based chatbot demonstration during a full-day workshop with 39 participants, and additionally surveyed 30 healthcare workers and 30 birthing people about their values, needs, and perceptions of NLP tools in the context of maternal health. We conducted quantitative and qualitative analyses of the survey results and interactive discussions to consolidate our findings into a set of guiding principles. We propose nine principles for ethical use of NLP for maternal healthcare, grouped into three themes: (i) recognizing contextual significance (ii) holistic measurements, and (iii) who/what is valued. For each principle, we describe its underlying rationale and provide practical advice. This set of principles can provide a methodological pattern for other researchers and serve as a resource to practitioners working on maternal health and other healthcare fields to emphasize the importance of technical nuance, historical context, and inclusive design when developing NLP technologies for clinical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10067v1">LLM-based policy generation for intent-based management of applications</a></div>
    <div class="paper-meta">
      📅 2024-01-22
      | 💬 This article has been accepted for publication in 2023 19th International Conference on Network and Service Management (CNSM), 3rd International Workshop on Analytics for Service and Application Management (AnServApp 2023)
    </div>
    <details class="paper-abstract">
      Automated management requires decomposing high-level user requests, such as intents, to an abstraction that the system can understand and execute. This is challenging because even a simple intent requires performing a number of ordered steps. And the task of identifying and adapting these steps (as conditions change) requires a decomposition approach that cannot be exactly pre-defined beforehand. To tackle these challenges and support automated intent decomposition and execution, we explore the few-shot capability of Large Language Models (LLMs). We propose a pipeline that progressively decomposes intents by generating the required actions using a policy-based abstraction. This allows us to automate the policy execution by creating a closed control loop for the intent deployment. To do so, we generate and map the policies to APIs and form application management loops that perform the necessary monitoring, analysis, planning and execution. We evaluate our proposal with a use-case to fulfill and assure an application service chain of virtual network functions. Using our approach, we can generalize and generate the necessary steps to realize intents, thereby enabling intent automation for application management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01386v2">Who is ChatGPT? Benchmarking LLMs' Psychological Portrayal Using PsychoBench</a></div>
    <div class="paper-meta">
      📅 2024-01-22
      | 💬 Accepted for ICLR 2024 Oral Presentation. 15 pages (main text) and 5 pages (appendix)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently showcased their remarkable capacities, not only in natural language processing tasks but also across diverse domains such as clinical medicine, legal consultation, and education. LLMs become more than mere applications, evolving into assistants capable of addressing diverse user requests. This narrows the distinction between human beings and artificial intelligence agents, raising intriguing questions regarding the potential manifestation of personalities, temperaments, and emotions within LLMs. In this paper, we propose a framework, PsychoBench, for evaluating diverse psychological aspects of LLMs. Comprising thirteen scales commonly used in clinical psychology, PsychoBench further classifies these scales into four distinct categories: personality traits, interpersonal relationships, motivational tests, and emotional abilities. Our study examines five popular models, namely text-davinci-003, gpt-3.5-turbo, gpt-4, LLaMA-2-7b, and LLaMA-2-13b. Additionally, we employ a jailbreak approach to bypass the safety alignment protocols and test the intrinsic natures of LLMs. We have made PsychoBench openly accessible via https://github.com/CUHK-ARISE/PsychoBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01684v1">A Framework to Implement 1+N Multi-task Fine-tuning Pattern in LLMs Using the CGC-LORA Algorithm</a></div>
    <div class="paper-meta">
      📅 2024-01-22
    </div>
    <details class="paper-abstract">
      With the productive evolution of large language models (LLMs) in the field of natural language processing (NLP), tons of effort has been made to effectively fine-tune common pre-trained LLMs to fulfill a variety of tasks in one or multiple specific domain. In practice, there are two prevailing ways, in which the adaptation can be achieved: (i) Multiple Independent Models: Pre-trained LLMs are fine-tuned a few times independently using the corresponding training samples from each task. (ii) An Integrated Model: Samples from all tasks are employed to fine-tune a pre-trianed LLM unitedly. To address the high computing cost and seesawing issue simultaneously, we propose a unified framework that implements a 1 + N mutli-task fine-tuning pattern in LLMs using a novel Customized Gate Control (CGC) Low-rank Adaptation (LoRA) algorithm. Our work aims to take an advantage of both MTL (i.e., CGC) and PEFT (i.e., LoRA) scheme. For a given cluster of tasks, we design an innovative layer that contains two types of experts as additional trainable parameters to make LoRA be compatible with MTL. To comprehensively evaluate the proposed framework, we conduct well-designed experiments on two public datasets. The experimental results demonstrate that the unified framework with CGC-LoRA modules achieves higher evaluation scores than all benchmarks on both two datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.11240v1">CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-01-20
    </div>
    <details class="paper-abstract">
      Pre-trained large language models (LLMs) often need specialization for domain-specific tasks. Low-Rank Adaptation (LoRA) is a popular approach that adapts a base model to multiple tasks by adding lightweight trainable adapters. In this paper, we present CaraServe, a system that efficiently serves many LoRA adapters derived from a common base model. CaraServe maintains the base model on GPUs and dynamically loads activated LoRA adapters from main memory. As GPU loading results in a cold-start that substantially delays token generation, CaraServe employs a CPU-assisted approach. It early starts the activated adapters on CPUs for prefilling as they are being loaded onto GPUs; after loading completes, it then switches to the GPUs for generative LoRA inference. CaraServe develops a highly optimized synchronization mechanism to efficiently coordinate LoRA computation on the CPU and GPU. Moreover, CaraServe employs a rank-aware scheduling algorithm to optimally schedule heterogeneous LoRA requests for maximum service-level objective (SLO) attainment. We have implemented CaraServe and evaluated it against state-of-the-art LoRA serving systems. Our results demonstrate that CaraServe can speed up the average request serving latency by up to 1.4$\times$ and achieve an SLO attainment of up to 99%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.02500v2">On the Prospects of Incorporating Large Language Models (LLMs) in Automated Planning and Scheduling (APS)</a></div>
    <div class="paper-meta">
      📅 2024-01-20
    </div>
    <details class="paper-abstract">
      Automated Planning and Scheduling is among the growing areas in Artificial Intelligence (AI) where mention of LLMs has gained popularity. Based on a comprehensive review of 126 papers, this paper investigates eight categories based on the unique applications of LLMs in addressing various aspects of planning problems: language translation, plan generation, model construction, multi-agent planning, interactive planning, heuristics optimization, tool integration, and brain-inspired planning. For each category, we articulate the issues considered and existing gaps. A critical insight resulting from our review is that the true potential of LLMs unfolds when they are integrated with traditional symbolic planners, pointing towards a promising neuro-symbolic approach. This approach effectively combines the generative aspects of LLMs with the precision of classical planning methods. By synthesizing insights from existing literature, we underline the potential of this integration to address complex planning challenges. Our goal is to encourage the ICAPS community to recognize the complementary strengths of LLMs and symbolic planners, advocating for a direction in automated planning that leverages these synergistic capabilities to develop more advanced and intelligent planning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.11181v1">Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads</a></div>
    <div class="paper-meta">
      📅 2024-01-20
    </div>
    <details class="paper-abstract">
      Transformer-based large language model (LLM) inference serving is now the backbone of many cloud services. LLM inference consists of a prefill phase and a decode phase. However, existing LLM deployment practices often overlook the distinct characteristics of these phases, leading to significant interference. To mitigate interference, our insight is to carefully schedule and group inference requests based on their characteristics. We realize this idea in TetriInfer through three pillars. First, it partitions prompts into fixed-size chunks so that the accelerator always runs close to its computationsaturated limit. Second, it disaggregates prefill and decode instances so each can run independently. Finally, it uses a smart two-level scheduling algorithm augmented with predicted resource usage to avoid decode scheduling hotspots. Results show that TetriInfer improves time-to-first-token (TTFT), job completion time (JCT), and inference efficiency in turns of performance per dollar by a large margin, e.g., it uses 38% less resources all the while lowering average TTFT and average JCT by 97% and 47%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06634v1">SocraSynth: Multi-LLM Reasoning with Conditional Statistics</a></div>
    <div class="paper-meta">
      📅 2024-01-19
      | 💬 1 figure, 6 tables, 6 appendices
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), while promising, face criticisms for biases, hallucinations, and a lack of reasoning capability. This paper introduces SocraSynth, a multi-LLM agent reasoning platform developed to mitigate these issues. SocraSynth utilizes conditional statistics and systematic context enhancement through continuous arguments, alongside adjustable debate contentiousness levels. The platform typically involves a human moderator and two LLM agents representing opposing viewpoints on a given subject. SocraSynth operates in two main phases: knowledge generation and reasoning evaluation. In the knowledge generation phase, the moderator defines the debate topic and contentiousness level, prompting the agents to formulate supporting arguments for their respective stances. The reasoning evaluation phase then employs Socratic reasoning and formal logic principles to appraise the quality of the arguments presented. The dialogue concludes with the moderator adjusting the contentiousness from confrontational to collaborative, gathering final, conciliatory remarks to aid in human reasoning and decision-making. Through case studies in three distinct application domains, this paper showcases SocraSynth's effectiveness in fostering rigorous research, dynamic reasoning, comprehensive assessment, and enhanced collaboration. This underscores the value of multi-agent interactions in leveraging LLMs for advanced knowledge extraction and decision-making support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10506v1">FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis</a></div>
    <div class="paper-meta">
      📅 2024-01-19
      | 💬 13 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Text-to-SQL, which provides zero-code interface for operating relational databases, has gained much attention in financial analysis; because, financial professionals may not well-skilled in SQL programming. However, until now, there is no practical Text-to-SQL benchmark dataset for financial analysis, and existing Text-to-SQL methods have not considered the unique characteristics of databases in financial applications, such as commonly existing wide tables. To address these issues, we collect a practical Text-to-SQL benchmark dataset and propose a model-agnostic Large Language Model (LLMs)-based Text-to-SQL framework for financial analysis. The benchmark dataset, BULL, is collected from the practical financial analysis business of Hundsun Technologies Inc., including databases for fund, stock, and macro economy. Besides, the proposed LLMs-based Text-to-SQL framework, FinSQL, provides a systematic treatment for financial Text-to-SQL from the perspectives of prompt construction, parameter-efficient fine-tuning and output calibration. Extensive experimental results on BULL demonstrate that FinSQL achieves the state-of-the-art Text-to-SQL performance at a small cost; furthermore, FinSQL can bring up to 36.64% performance improvement in scenarios requiring few-shot cross-database model transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10444v1">Can A Cognitive Architecture Fundamentally Enhance LLMs? Or Vice Versa?</a></div>
    <div class="paper-meta">
      📅 2024-01-19
    </div>
    <details class="paper-abstract">
      The paper discusses what is needed to address the limitations of current LLM-centered AI systems. The paper argues that incorporating insights from human cognition and psychology, as embodied by a computational cognitive architecture, can help develop systems that are more capable, more reliable, and more human-like. It emphasizes the importance of the dual-process architecture and the hybrid neuro-symbolic approach in addressing the limitations of current LLMs. In the opposite direction, the paper also highlights the need for an overhaul of computational cognitive architectures to better reflect advances in AI and computing technology. Overall, the paper advocates for a multidisciplinary, mutually beneficial approach towards developing better models both for AI and for understanding the human mind.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10364v1">Using LLM such as ChatGPT for Designing and Implementing a RISC Processor: Execution,Challenges and Limitations</a></div>
    <div class="paper-meta">
      📅 2024-01-18
    </div>
    <details class="paper-abstract">
      This paper discusses the feasibility of using Large Language Models LLM for code generation with a particular application in designing an RISC. The paper also reviews the associated steps such as parsing, tokenization, encoding, attention mechanism, sampling the tokens and iterations during code generation. The generated code for the RISC components is verified through testbenches and hardware implementation on a FPGA board. Four metric parameters Correct output on the first iteration, Number of errors embedded in the code, Number of trials required to achieve the code and Failure to generate the code after three iterations, are used to compare the efficiency of using LLM in programming. In all the cases, the generated code had significant errors and human intervention was always required to fix the bugs. LLM can therefore be used to complement a programmer code design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.07382v2">Less is More for Long Document Summary Evaluation by LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-18
      | 💬 EACL (main)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promising performance in summary evaluation tasks, yet they face challenges such as high computational costs and the Lost-in-the-Middle problem where important information in the middle of long documents is often overlooked. To address these issues, this paper introduces a novel approach, Extract-then-Evaluate, which involves extracting key sentences from a long source document and then evaluating the summary by prompting LLMs. The results reveal that the proposed method not only significantly reduces evaluation costs but also exhibits a higher correlation with human evaluations. Furthermore, we provide practical recommendations for optimal document length and sentence extraction methods, contributing to the development of cost-effective yet more accurate methods for LLM-based text generation evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10184v1">Comparing Traditional and LLM-based Search for Image Geolocation</a></div>
    <div class="paper-meta">
      📅 2024-01-18
    </div>
    <details class="paper-abstract">
      Web search engines have long served as indispensable tools for information retrieval; user behavior and query formulation strategies have been well studied. The introduction of search engines powered by large language models (LLMs) suggested more conversational search and new types of query strategies. In this paper, we compare traditional and LLM-based search for the task of image geolocation, i.e., determining the location where an image was captured. Our work examines user interactions, with a particular focus on query formulation strategies. In our study, 60 participants were assigned either traditional or LLM-based search engines as assistants for geolocation. Participants using traditional search more accurately predicted the location of the image compared to those using the LLM-based search. Distinct strategies emerged between users depending on the type of assistant. Participants using the LLM-based search issued longer, more natural language queries, but had shorter search sessions. When reformulating their search queries, traditional search participants tended to add more terms to their initial queries, whereas participants using the LLM-based search consistently rephrased their initial queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10061v1">DiffusionGPT: LLM-Driven Text-to-Image Generation System</a></div>
    <div class="paper-meta">
      📅 2024-01-18
    </div>
    <details class="paper-abstract">
      Diffusion models have opened up new avenues for the field of image generation, resulting in the proliferation of high-quality models shared on open-source platforms. However, a major challenge persists in current text-to-image systems are often unable to handle diverse inputs, or are limited to single model results. Current unified attempts often fall into two orthogonal aspects: i) parse Diverse Prompts in input stage; ii) activate expert model to output. To combine the best of both worlds, we propose DiffusionGPT, which leverages Large Language Models (LLM) to offer a unified generation system capable of seamlessly accommodating various types of prompts and integrating domain-expert models. DiffusionGPT constructs domain-specific Trees for various generative models based on prior knowledge. When provided with an input, the LLM parses the prompt and employs the Trees-of-Thought to guide the selection of an appropriate model, thereby relaxing input constraints and ensuring exceptional performance across diverse domains. Moreover, we introduce Advantage Databases, where the Tree-of-Thought is enriched with human feedback, aligning the model selection process with human preferences. Through extensive experiments and comparisons, we demonstrate the effectiveness of DiffusionGPT, showcasing its potential for pushing the boundaries of image synthesis in diverse domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.09760v1">A Comparative Study on Annotation Quality of Crowdsourcing and LLM via Label Aggregation</a></div>
    <div class="paper-meta">
      📅 2024-01-18
      | 💬 Accepted in ICASSP 2024
    </div>
    <details class="paper-abstract">
      Whether Large Language Models (LLMs) can outperform crowdsourcing on the data annotation task is attracting interest recently. Some works verified this issue with the average performance of individual crowd workers and LLM workers on some specific NLP tasks by collecting new datasets. However, on the one hand, existing datasets for the studies of annotation quality in crowdsourcing are not yet utilized in such evaluations, which potentially provide reliable evaluations from a different viewpoint. On the other hand, the quality of these aggregated labels is crucial because, when utilizing crowdsourcing, the estimated labels aggregated from multiple crowd labels to the same instances are the eventually collected labels. Therefore, in this paper, we first investigate which existing crowdsourcing datasets can be used for a comparative study and create a benchmark. We then compare the quality between individual crowd labels and LLM labels and make the evaluations on the aggregated labels. In addition, we propose a Crowd-LLM hybrid label aggregation method and verify the performance. We find that adding LLM labels from good LLMs to existing crowdsourcing datasets can enhance the quality of the aggregated labels of the datasets, which is also higher than the quality of LLM labels themselves.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.08469v5">LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters</a></div>
    <div class="paper-meta">
      📅 2024-01-18
      | 💬 This paper is currently under review. The code will be made available upon acceptance
    </div>
    <details class="paper-abstract">
      Multivariate time-series forecasting is vital in various domains, e.g., economic planning and weather prediction. Deep train-from-scratch models have exhibited effective performance yet require large amounts of data, which limits real-world applicability. Recently, researchers have leveraged the representation learning transferability of pre-trained Large Language Models (LLMs) to handle limited non-linguistic datasets effectively. However, incorporating LLMs with time-series data presents challenges of limited adaptation due to different compositions between time-series and linguistic data, and the inability to process multi-scale temporal information. To tackle these challenges, we propose LLM4TS, a framework for time-series forecasting with pre-trained LLMs. LLM4TS consists of a two-stage fine-tuning strategy: the \textit{time-series alignment} stage to align LLMs with the nuances of time-series data, and the \textit{forecasting fine-tuning} stage for downstream time-series forecasting tasks. Furthermore, our framework features a novel two-level aggregation method that integrates multi-scale temporal data within pre-trained LLMs, enhancing their ability to interpret time-specific information. In experiments across 7 time-series forecasting datasets, LLM4TS is superior to existing state-of-the-art methods compared with trained-from-scratch models in full-shot scenarios, and also achieves an average improvement of 6.84% in MSE in few-shot scenarios. In addition, evaluations compared with different self-supervised learning approaches highlight LLM4TS's effectiveness with representation learning in forecasting tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.14345v2">Logic-Scaffolding: Personalized Aspect-Instructed Recommendation Explanation Generation using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-17
      | 💬 The 17th ACM International Conference on Web Search and Data Mining (WSDM 2024)
    </div>
    <details class="paper-abstract">
      The unique capabilities of Large Language Models (LLMs), such as the natural language text generation ability, position them as strong candidates for providing explanation for recommendations. However, despite the size of the LLM, most existing models struggle to produce zero-shot explanations reliably. To address this issue, we propose a framework called Logic-Scaffolding, that combines the ideas of aspect-based explanation and chain-of-thought prompting to generate explanations through intermediate reasoning steps. In this paper, we share our experience in building the framework and present an interactive demonstration for exploring our results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.05566v3">Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training</a></div>
    <div class="paper-meta">
      📅 2024-01-17
      | 💬 updated to add missing acknowledgements
    </div>
    <details class="paper-abstract">
      Humans are capable of strategically deceptive behavior: behaving helpfully in most situations, but then behaving very differently in order to pursue alternative objectives when given the opportunity. If an AI system learned such a deceptive strategy, could we detect it and remove it using current state-of-the-art safety training techniques? To study this question, we construct proof-of-concept examples of deceptive behavior in large language models (LLMs). For example, we train models that write secure code when the prompt states that the year is 2023, but insert exploitable code when the stated year is 2024. We find that such backdoor behavior can be made persistent, so that it is not removed by standard safety training techniques, including supervised fine-tuning, reinforcement learning, and adversarial training (eliciting unsafe behavior and then training to remove it). The backdoor behavior is most persistent in the largest models and in models trained to produce chain-of-thought reasoning about deceiving the training process, with the persistence remaining even when the chain-of-thought is distilled away. Furthermore, rather than removing backdoors, we find that adversarial training can teach models to better recognize their backdoor triggers, effectively hiding the unsafe behavior. Our results suggest that, once a model exhibits deceptive behavior, standard techniques could fail to remove such deception and create a false impression of safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.09092v1">BibSonomy Meets ChatLLMs for Publication Management: From Chat to Publication Management: Organizing your related work using BibSonomy & LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-17
      | 💬 Accepted at 2024 ACM SIGIR CHIIR, For a demo see here http://professor-x.de/demos/bibsonomy-chatgpt/demo.mp4
    </div>
    <details class="paper-abstract">
      The ever-growing corpus of scientific literature presents significant challenges for researchers with respect to discovery, management, and annotation of relevant publications. Traditional platforms like Semantic Scholar, BibSonomy, and Zotero offer tools for literature management, but largely require manual laborious and error-prone input of tags and metadata. Here, we introduce a novel retrieval augmented generation system that leverages chat-based large language models (LLMs) to streamline and enhance the process of publication management. It provides a unified chat-based interface, enabling intuitive interactions with various backends, including Semantic Scholar, BibSonomy, and the Zotero Webscraper. It supports two main use-cases: (1) Explorative Search & Retrieval - leveraging LLMs to search for and retrieve both specific and general scientific publications, while addressing the challenges of content hallucination and data obsolescence; and (2) Cataloguing & Management - aiding in the organization of personal publication libraries, in this case BibSonomy, by automating the addition of metadata and tags, while facilitating manual edits and updates. We compare our system to different LLM models in three different settings, including a user study, and we can show its advantages in different metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.09042v1">LLMs for Relational Reasoning: How Far are We?</a></div>
    <div class="paper-meta">
      📅 2024-01-17
      | 💬 Accepted by The First International Workshop on Large Language Models for Code (ICSE 2024)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized many areas (e.g. natural language processing, software engineering, etc.) by achieving state-of-the-art performance on extensive downstream tasks. Aiming to achieve robust and general artificial intelligence, there has been a surge of interest in investigating the reasoning ability of the LLMs. Whereas the textual and numerical reasoning benchmarks adopted by previous works are rather shallow and simple, it is hard to conclude that the LLMs possess strong reasoning ability by merely achieving positive results on these benchmarks. Recent efforts have demonstrated that the LLMs are poor at solving sequential decision-making problems that require common-sense planning by evaluating their performance on the reinforcement learning benchmarks. In this work, we conduct an in-depth assessment of several state-of-the-art LLMs' reasoning ability based on the inductive logic programming (ILP) benchmark, which is broadly recognized as a representative and challenging measurement for evaluating logic program induction/synthesis systems as it requires inducing strict cause-effect logic to achieve robust deduction on independent and identically distributed (IID) and out-of-distribution (OOD) test samples. Our evaluations illustrate that compared with the neural program induction systems which are much smaller in model size, the state-of-the-art LLMs are much poorer in terms of reasoning ability by achieving much lower performance and generalization using either natural language prompting or truth-value matrix prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08908v1">Herding LLaMaS: Using LLMs as an OS Module</a></div>
    <div class="paper-meta">
      📅 2024-01-17
      | 💬 ASPLOS 2023, Wild and Crazy Ideas session
    </div>
    <details class="paper-abstract">
      Computer systems are becoming increasingly heterogeneous with the emergence of new memory technologies and compute devices. GPUs alongside CPUs have become commonplace and CXL is poised to be a mainstay of cloud systems. The operating system is responsible for managing these hardware resources, requiring modification every time a new device is released. Years of research and development are sunk into tuning the OS for high performance with each new heterogeneous device. With the recent explosion in memory technologies and domain-specific accelerators, it would be beneficial to have an OS that could provide high performance for new devices without significant effort. We propose LLaMaS which can adapt to new devices easily. LLaMaS uses Large Language Models (LLMs) to extract the useful features of new devices from their textual description and uses these features to make operating system decisions at runtime. Adding support to LLaMaS for a new device is as simple as describing the system and new device properties in plaintext. LLaMaS reduces the burden on system administrators to enable easy integration of new devices into production systems. Preliminary evaluation using ChatGPT shows that LLMs are capable of extracting device features from text and make correct OS decisions based on those features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.08177v3">Using an LLM to Help With Code Understanding</a></div>
    <div class="paper-meta">
      📅 2024-01-16
    </div>
    <details class="paper-abstract">
      Understanding code is challenging, especially when working in new and complex development environments. Code comments and documentation can help, but are typically scarce or hard to navigate. Large language models (LLMs) are revolutionizing the process of writing code. Can they do the same for helping understand it? In this study, we provide a first investigation of an LLM-based conversational UI built directly in the IDE that is geared towards code understanding. Our IDE plugin queries OpenAI's GPT-3.5-turbo model with four high-level requests without the user having to write explicit prompts: to explain a highlighted section of code, provide details of API calls used in the code, explain key domain-specific terms, and provide usage examples for an API. The plugin also allows for open-ended prompts, which are automatically contextualized to the LLM with the program being edited. We evaluate this system in a user study with 32 participants, which confirms that using our plugin can aid task completion more than web search. We additionally provide a thorough analysis of the ways developers use, and perceive the usefulness of, our system, among others finding that the usage and benefits differ between students and professionals. We conclude that in-IDE prompt-less interaction with LLMs is a promising future direction for tool builders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.03393v4">Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs</a></div>
    <div class="paper-meta">
      📅 2024-01-16
      | 💬 To be appear on SIGKDD Explorations
    </div>
    <details class="paper-abstract">
      Learning on Graphs has attracted immense attention due to its wide real-world applications. The most popular pipeline for learning on graphs with textual node attributes primarily relies on Graph Neural Networks (GNNs), and utilizes shallow text embedding as initial node representations, which has limitations in general knowledge and profound semantic understanding. In recent years, Large Language Models (LLMs) have been proven to possess extensive common knowledge and powerful semantic comprehension abilities that have revolutionized existing workflows to handle text data. In this paper, we aim to explore the potential of LLMs in graph machine learning, especially the node classification task, and investigate two possible pipelines: LLMs-as-Enhancers and LLMs-as-Predictors. The former leverages LLMs to enhance nodes' text attributes with their massive knowledge and then generate predictions through GNNs. The latter attempts to directly employ LLMs as standalone predictors. We conduct comprehensive and systematical studies on these two pipelines under various settings. From comprehensive empirical results, we make original observations and find new insights that open new possibilities and suggest promising directions to leverage LLMs for learning on graphs. Our codes and datasets are available at https://github.com/CurryTang/Graph-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.05596v2">POMP: Probability-driven Meta-graph Prompter for LLMs in Low-resource Unsupervised Neural Machine Translation</a></div>
    <div class="paper-meta">
      📅 2024-01-16
    </div>
    <details class="paper-abstract">
      Low-resource languages (LRLs) face challenges in supervised neural machine translation due to limited parallel data, prompting research into unsupervised methods. Unsupervised neural machine translation (UNMT) methods, including back-translation, transfer learning, and pivot-based translation, offer practical solutions for LRL translation, but they are hindered by issues like synthetic data noise, language bias, and error propagation, which can potentially be mitigated by Large Language Models (LLMs). LLMs have advanced NMT with in-context learning (ICL) and supervised fine-tuning methods, but insufficient training data results in poor performance in LRLs. We argue that LLMs can mitigate the linguistic noise with auxiliary languages to improve translations in LRLs. In this paper, we propose Probability-driven Meta-graph Prompter (POMP), a novel approach employing a dynamic, sampling-based graph of multiple auxiliary languages to enhance LLMs' translation capabilities for LRLs. POMP involves constructing a directed acyclic meta-graph for each source language, from which we dynamically sample multiple paths to prompt LLMs to mitigate the linguistic noise and improve translations during training. We use the BLEURT metric to evaluate the translations and back-propagate rewards, estimated by scores, to update the probabilities of auxiliary languages in the paths. Our experiments show significant improvements in the translation quality of three LRLs, demonstrating the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.03220v4">ALYMPICS: LLM Agents Meet Game Theory -- Exploring Strategic Decision-Making with AI Agents</a></div>
    <div class="paper-meta">
      📅 2024-01-16
    </div>
    <details class="paper-abstract">
      This paper introduces Alympics (Olympics for Agents), a systematic simulation framework utilizing Large Language Model (LLM) agents for game theory research. Alympics creates a versatile platform for studying complex game theory problems, bridging the gap between theoretical game theory and empirical investigations by providing a controlled environment for simulating human-like strategic interactions with LLM agents. In our pilot case study, the "Water Allocation Challenge," we explore Alympics through a challenging strategic game focused on the multi-round auction on scarce survival resources. This study demonstrates the framework's ability to qualitatively and quantitatively analyze game determinants, strategies, and outcomes. Additionally, we conduct a comprehensive human assessment and an in-depth evaluation of LLM agents in strategic decision-making scenarios. Our findings not only expand the understanding of LLM agents' proficiency in emulating human strategic behavior but also highlight their potential in advancing game theory knowledge, thereby enriching our understanding of both game theory and empowering further research into strategic decision-making domains with LLM agents. Codes, prompts, and all related resources are available at https://github.com/microsoft/Alympics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08138v1">LLMs for Test Input Generation for Semantic Caches</a></div>
    <div class="paper-meta">
      📅 2024-01-16
      | 💬 Accepted in International Conference on AI Engineering Software Engineering (CAIN 2024)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enable state-of-the-art semantic capabilities to be added to software systems such as semantic search of unstructured documents and text generation. However, these models are computationally expensive. At scale, the cost of serving thousands of users increases massively affecting also user experience. To address this problem, semantic caches are used to check for answers to similar queries (that may have been phrased differently) without hitting the LLM service. Due to the nature of these semantic cache techniques that rely on query embeddings, there is a high chance of errors impacting user confidence in the system. Adopting semantic cache techniques usually requires testing the effectiveness of a semantic cache (accurate cache hits and misses) which requires a labelled test set of similar queries and responses which is often unavailable. In this paper, we present VaryGen, an approach for using LLMs for test input generation that produces similar questions from unstructured text documents. Our novel approach uses the reasoning capabilities of LLMs to 1) adapt queries to the domain, 2) synthesise subtle variations to queries, and 3) evaluate the synthesised test dataset. We evaluated our approach in the domain of a student question and answer system by qualitatively analysing 100 generated queries and result pairs, and conducting an empirical case study with an open source semantic cache. Our results show that query pairs satisfy human expectations of similarity and our generated data demonstrates failure cases of a semantic cache. Additionally, we also evaluate our approach on Qasper dataset. This work is an important first step into test input generation for semantic applications and presents considerations for practitioners when calibrating a semantic cache.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08046v1">Enhancing Robustness of LLM-Synthetic Text Detectors for Academic Writing: A Comprehensive Analysis</a></div>
    <div class="paper-meta">
      📅 2024-01-16
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs), such as Generative Pre-trained Transformer 4 (GPT-4) used by ChatGPT, has profoundly impacted the academic and broader community. While these models offer numerous advantages in terms of revolutionizing work and study methods, they have also garnered significant attention due to their potential negative consequences. One example is generating academic reports or papers with little to no human contribution. Consequently, researchers have focused on developing detectors to address the misuse of LLMs. However, most existing methods prioritize achieving higher accuracy on restricted datasets, neglecting the crucial aspect of generalizability. This limitation hinders their practical application in real-life scenarios where reliability is paramount. In this paper, we present a comprehensive analysis of the impact of prompts on the text generated by LLMs and highlight the potential lack of robustness in one of the current state-of-the-art GPT detectors. To mitigate these issues concerning the misuse of LLMs in academic writing, we propose a reference-based Siamese detector named Synthetic-Siamese which takes a pair of texts, one as the inquiry and the other as the reference. Our method effectively addresses the lack of robustness of previous detectors (OpenAI detector and DetectGPT) and significantly improves the baseline performances in realistic academic writing scenarios by approximately 67% to 95%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.07612v1">Signed-Prompt: A New Approach to Prevent Prompt Injection Attacks Against LLM-Integrated Applications</a></div>
    <div class="paper-meta">
      📅 2024-01-15
    </div>
    <details class="paper-abstract">
      The critical challenge of prompt injection attacks in Large Language Models (LLMs) integrated applications, a growing concern in the Artificial Intelligence (AI) field. Such attacks, which manipulate LLMs through natural language inputs, pose a significant threat to the security of these applications. Traditional defense strategies, including output and input filtering, as well as delimiter use, have proven inadequate. This paper introduces the 'Signed-Prompt' method as a novel solution. The study involves signing sensitive instructions within command segments by authorized users, enabling the LLM to discern trusted instruction sources. The paper presents a comprehensive analysis of prompt injection attack patterns, followed by a detailed explanation of the Signed-Prompt concept, including its basic architecture and implementation through both prompt engineering and fine-tuning of LLMs. Experiments demonstrate the effectiveness of the Signed-Prompt method, showing substantial resistance to various types of prompt injection attacks, thus validating its potential as a robust defense strategy in AI security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.07526v1">Editing Arbitrary Propositions in LLMs without Subject Labels</a></div>
    <div class="paper-meta">
      📅 2024-01-15
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) editing modifies factual information in LLMs. Locate-and-Edit (L\&E) methods accomplish this by finding where relevant information is stored within the neural network, and editing the weights at that location. The goal of editing is to modify the response of an LLM to a proposition independently of its phrasing, while not modifying its response to other related propositions. Existing methods are limited to binary propositions, which represent straightforward binary relations between a subject and an object. Furthermore, existing methods rely on semantic subject labels, which may not be available or even be well-defined in practice. In this paper, we show that both of these issues can be effectively skirted with a simple and fast localization method called Gradient Tracing (GT). This localization method allows editing arbitrary propositions instead of just binary ones, and does so without the need for subject labels. As propositions always have a truth value, our experiments prompt an LLM as a boolean classifier, and edit its T/F response to propositions. Our method applies GT for location tracing, and then edit the model at that location using a mild variant of Rank-One Model Editing (ROME). On datasets of binary propositions derived from the CounterFact dataset, we show that our method -- without access to subject labels -- performs close to state-of-the-art L\&E methods which has access subject labels. We then introduce a new dataset, Factual Accuracy Classification Test (FACT), which includes non-binary propositions and for which subject labels are not generally applicable, and therefore is beyond the scope of existing L\&E methods. Nevertheless, we show that with our method editing is possible on FACT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.07190v1">Inroads to a Structured Data Natural Language Bijection and the role of LLM annotation</a></div>
    <div class="paper-meta">
      📅 2024-01-14
      | 💬 Graduate Coursework
    </div>
    <details class="paper-abstract">
      This work finds limited evidence supporting the theory that using multiple tasks with sequence-to-sequence transformer language models can improve performance on some metrics. In particular, the multi-task generalist t5-small outperforms the specialist t5-small with a $F_1$ of $0.771$ up from $0.692$, which may point to underlying cross-task knowledge generalization. This further suggests that even with the same network, "re-using" the same data in a different way may lead to higher performance in some metrics. However, the inverse task alone is likely only an optimization strategy, since it does not yield a significant general improvement at the model sizes explored in this work. Also, adding $\approx 4500$ LLM annotated records (interlaced with the $12800$ WebNLG training records) does not substantially change automatic metric performance compared to the same t5-small model without the synthetic data. This may be due to a learning capacity bottleneck on account of model size, and decreases observed may be due to distributional differences in the corpora. Future research using larger models or human evaluation is required to more fully explain the mechanisms contributing to performance on these tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.07181v1">Reinforcement Learning from LLM Feedback to Counteract Goal Misgeneralization</a></div>
    <div class="paper-meta">
      📅 2024-01-14
    </div>
    <details class="paper-abstract">
      We introduce a method to address goal misgeneralization in reinforcement learning (RL), leveraging Large Language Model (LLM) feedback during training. Goal misgeneralization, a type of robustness failure in RL occurs when an agent retains its capabilities out-of-distribution yet pursues a proxy rather than the intended one. Our approach utilizes LLMs to analyze an RL agent's policies during training and identify potential failure scenarios. The RL agent is then deployed in these scenarios, and a reward model is learnt through the LLM preferences and feedback. This LLM-informed reward model is used to further train the RL agent on the original dataset. We apply our method to a maze navigation task, and show marked improvements in goal generalization, especially in cases where true and proxy goals are somewhat distinguishable and behavioral biases are pronounced. This study demonstrates how the LLM, despite its lack of task proficiency, can efficiently supervise RL agents, providing scalable oversight and valuable insights for enhancing goal-directed learning in RL through the use of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.07078v1">PUB: A Pragmatics Understanding Benchmark for Assessing LLMs' Pragmatics Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-01-13
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated remarkable capability for understanding semantics, but they often struggle with understanding pragmatics. To demonstrate this fact, we release a Pragmatics Understanding Benchmark (PUB) dataset consisting of fourteen tasks in four pragmatics phenomena, namely, Implicature, Presupposition, Reference, and Deixis. We curated high-quality test sets for each task, consisting of Multiple Choice Question Answers (MCQA). PUB includes a total of 28k data points, 6.1k of which have been created by us, and the rest are adapted from existing datasets. We evaluated nine models varying in the number of parameters and type of training. Our study indicates that fine-tuning for instruction-following and chat significantly enhances the pragmatics capabilities of smaller language models. However, for larger models, the base versions perform comparably with their chat-adapted counterparts. Additionally, there is a noticeable performance gap between human capabilities and model capabilities. Furthermore, unlike the consistent performance of humans across various tasks, the models demonstrate variability in their proficiency, with performance levels fluctuating due to different hints and the complexities of tasks within the same dataset. Overall, the benchmark aims to provide a comprehensive evaluation of LLM's ability to handle real-world language tasks that require pragmatic reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.07004v1">Extending LLMs' Context Window with 100 Samples</a></div>
    <div class="paper-meta">
      📅 2024-01-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to have limited extrapolation ability beyond their pre-trained context window, constraining their application in downstream tasks with lengthy inputs. Recent studies have sought to extend LLMs' context window by modifying rotary position embedding (RoPE), a popular position encoding method adopted by well-known LLMs such as LLaMA, PaLM, and GPT-NeoX. However, prior works like Position Interpolation (PI) and YaRN are resource-intensive and lack comparative experiments to assess their applicability. In this work, we identify the inherent need for LLMs' attention entropy (i.e. the information entropy of attention scores) to maintain stability and introduce a novel extension to RoPE which combines adjusting RoPE's base frequency and scaling the attention logits to help LLMs efficiently adapt to a larger context window. We validate the superiority of our method in both fine-tuning performance and robustness across different context window sizes on various context-demanding tasks. Notably, our method extends the context window of LLaMA-2-7B-Chat to 16,384 with only 100 samples and 6 training steps, showcasing extraordinary efficiency. Finally, we also explore how data compositions and training curricula affect context window extension for specific downstream tasks, suggesting fine-tuning LLMs with lengthy conversations as a good starting point. We release our code and SFT data at https://github.com/GAIR-NLP/Entropy-ABF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06761v1">APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding</a></div>
    <div class="paper-meta">
      📅 2024-01-12
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      The massive adoption of large language models (LLMs) demands efficient deployment strategies. However, the auto-regressive decoding process, which is fundamental to how most LLMs generate text, poses challenges to achieve efficient serving. In this work, we introduce a parallel auto-regressive generation method. By instruct-tuning on general domain data that contains hierarchical structures, we enable LLMs to independently plan their generation process and perform auto-parallel auto-regressive (APAR) generation, significantly reducing the number of generation steps. APAR alone can achieve up to 2x speed-up, and when combined with speculative decoding, the speed-up can reach up to 4x. In addition, APAR reduces the key-value cache consumption and attention computation during generation. This leads to a throughput increase of 20-70% and a latency reduce of 20-35% in high-throughput scenarios, compared to state-of-the-art serving frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06676v1">LLMRS: Unlocking Potentials of LLM-Based Recommender Systems for Software Purchase</a></div>
    <div class="paper-meta">
      📅 2024-01-12
    </div>
    <details class="paper-abstract">
      Recommendation systems are ubiquitous, from Spotify playlist suggestions to Amazon product suggestions. Nevertheless, depending on the methodology or the dataset, these systems typically fail to capture user preferences and generate general recommendations. Recent advancements in Large Language Models (LLM) offer promising results for analyzing user queries. However, employing these models to capture user preferences and efficiency remains an open question. In this paper, we propose LLMRS, an LLM-based zero-shot recommender system where we employ pre-trained LLM to encode user reviews into a review score and generate user-tailored recommendations. We experimented with LLMRS on a real-world dataset, the Amazon product reviews, for software purchase use cases. The results show that LLMRS outperforms the ranking-based baseline model while successfully capturing meaningful information from product reviews, thereby providing more reliable recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06301v1">Misconfidence-based Demonstration Selection for LLM In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2024-01-12
    </div>
    <details class="paper-abstract">
      In-context learning with large language models (LLMs) excels at adapting to various tasks rapidly. However, its success hinges on carefully selecting demonstrations, which remains an obstacle in practice. Current approaches to this problem either rely on hard-to-acquire external supervision or require frequent interactions with LLMs, resulting in high costs. We propose a new method called In-Context Reflection (ICR) to overcome these challenges. ICR strategically selects demonstrations to reduce the discrepancy between the LLM's outputs and the actual input-output mappings. Specifically, ICR starts with a random set of initial demonstrations, then iteratively refines it. In each step, it analyzes a pool of candidate examples and identifies the ones most likely to challenge the LLM's current understanding, measured by a new metric called misconfidence. These most confusing examples are then selected to replace the less informative demonstrations in the current set. Our comprehensive evaluation across five diverse datasets encompassing 13 subtasks shows the efficacy of ICR. Compared to existing methods, ICR achieves an average performance boost of 4%, while demonstrating remarkable cross-task generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06204v1">An Exploratory Assessment of LLM's Potential Toward Flight Trajectory Reconstruction Analysis</a></div>
    <div class="paper-meta">
      📅 2024-01-11
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold transformative potential in aviation, particularly in reconstructing flight trajectories. This paper investigates this potential, grounded in the notion that LLMs excel at processing sequential data and deciphering complex data structures. Utilizing the LLaMA 2 model, a pre-trained open-source LLM, the study focuses on reconstructing flight trajectories using Automatic Dependent Surveillance-Broadcast (ADS-B) data with irregularities inherent in real-world scenarios. The findings demonstrate the model's proficiency in filtering noise and estimating both linear and curved flight trajectories. However, the analysis also reveals challenges in managing longer data sequences, which may be attributed to the token length limitations of LLM models. The study's insights underscore the promise of LLMs in flight trajectory reconstruction and open new avenues for their broader application across the aviation and transportation sectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.05940v1">Mutation-based Consistency Testing for Evaluating the Code Understanding Capability of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-11
      | 💬 This is an author-preprint. The published version will be included in the proceedings of CAIN 2024 (co-located with ICSE 2024)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in processing both natural and programming languages, which have enabled various applications in software engineering, such as requirement engineering, code generation, and software testing. However, existing code generation benchmarks do not necessarily assess the code understanding performance of LLMs, especially for the subtle inconsistencies that may arise between code and its semantics described in natural language. In this paper, we propose a novel method to systematically assess the code understanding performance of LLMs, particularly focusing on subtle differences between code and its descriptions, by introducing code mutations to existing code generation datasets. Code mutations are small changes that alter the semantics of the original code, creating a mismatch with the natural language description. We apply different types of code mutations, such as operator replacement and statement deletion, to generate inconsistent code-description pairs. We then use these pairs to test the ability of LLMs to correctly detect the inconsistencies. We propose a new LLM testing method, called Mutation-based Consistency Testing (MCT), and conduct a case study on the two popular LLMs, GPT-3.5 and GPT-4, using the state-of-the-art code generation benchmark, HumanEval-X, which consists of six programming languages (Python, C++, Java, Go, JavaScript, and Rust). We compare the performance of the LLMs across different types of code mutations and programming languages and analyze the results. We find that the LLMs show significant variation in their code understanding performance and that they have different strengths and weaknesses depending on the mutation type and language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.05799v1">Designing Heterogeneous LLM Agents for Financial Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2024-01-11
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have drastically changed the possible ways to design intelligent systems, shifting the focuses from massive data acquisition and new modeling training to human alignment and strategical elicitation of the full potential of existing pre-trained models. This paradigm shift, however, is not fully realized in financial sentiment analysis (FSA), due to the discriminative nature of this task and a lack of prescriptive knowledge of how to leverage generative models in such a context. This study investigates the effectiveness of the new paradigm, i.e., using LLMs without fine-tuning for FSA. Rooted in Minsky's theory of mind and emotions, a design framework with heterogeneous LLM agents is proposed. The framework instantiates specialized agents using prior domain knowledge of the types of FSA errors and reasons on the aggregated agent discussions. Comprehensive evaluation on FSA datasets show that the framework yields better accuracies, especially when the discussions are substantial. This study contributes to the design foundations and paves new avenues for LLMs-based FSA. Implications on business and management are also discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.05033v1">Bootstrapping LLM-based Task-Oriented Dialogue Agents via Self-Talk</a></div>
    <div class="paper-meta">
      📅 2024-01-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are powerful dialogue agents, but specializing them towards fulfilling a specific function can be challenging. Instructing tuning, i.e. tuning models on instruction and sample responses generated by humans (Ouyang et al., 2022), has proven as an effective method to do so, yet requires a number of data samples that a) might not be available or b) costly to generate. Furthermore, this cost increases when the goal is to make the LLM follow a specific workflow within a dialogue instead of single instructions. Inspired by the self-play technique in reinforcement learning and the use of LLMs to simulate human agents, we propose a more effective method for data collection through LLMs engaging in a conversation in various roles. This approach generates a training data via "self-talk" of LLMs that can be refined and utilized for supervised fine-tuning. We introduce an automated way to measure the (partial) success of a dialogue. This metric is used to filter the generated conversational data that is fed back in LLM for training. Based on our automated and human evaluations of conversation quality, we demonstrate that such self-talk data improves results. In addition, we examine the various characteristics that showcase the quality of generated dialogues and how they can be connected to their potential utility as training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.04666v4">Pre-training LLMs using human-like development data corpus</a></div>
    <div class="paper-meta">
      📅 2024-01-10
      | 💬 Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning
    </div>
    <details class="paper-abstract">
      Pre-trained Large Language Models (LLMs) have shown success in a diverse set of language inference and understanding tasks. The pre-training stage of LLMs looks at a large corpus of raw textual data. The BabyLM shared task compares LLM pre-training to human language acquisition, where the number of tokens seen by 13-year-old kids is magnitudes smaller than the number of tokens seen by LLMs. In this work, we pre-train and evaluate LLMs on their ability to learn contextual word representations using roughly the same number of tokens as seen by children. We provide a strong set of baselines; with different architectures, evaluation of changes in performance across epochs, and reported pre-training metrics for the strict small and strict tracks of the task. We also try to loosely replicate the RoBERTa baseline given by the task organizers to observe the training robustness to hyperparameter selection and replicability. We provide the submission details to the strict and strict-small tracks in this report.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.17352v2">Improving Audio Captioning Models with Fine-grained Audio Features, Text Embedding Supervision, and LLM Mix-up Augmentation</a></div>
    <div class="paper-meta">
      📅 2024-01-10
      | 💬 ICASSP 2024 camera-ready paper. Winner of the DCASE 2023 Challenge Task 6A: Automated Audio Captioning (AAC)
    </div>
    <details class="paper-abstract">
      Automated audio captioning (AAC) aims to generate informative descriptions for various sounds from nature and/or human activities. In recent years, AAC has quickly attracted research interest, with state-of-the-art systems now relying on a sequence-to-sequence (seq2seq) backbone powered by strong models such as Transformers. Following the macro-trend of applied machine learning research, in this work, we strive to improve the performance of seq2seq AAC models by extensively leveraging pretrained models and large language models (LLMs). Specifically, we utilize BEATs to extract fine-grained audio features. Then, we employ Instructor LLM to fetch text embeddings of captions, and infuse their language-modality knowledge into BEATs audio features via an auxiliary InfoNCE loss function. Moreover, we propose a novel data augmentation method that uses ChatGPT to produce caption mix-ups (i.e., grammatical and compact combinations of two captions) which, together with the corresponding audio mixtures, increase not only the amount but also the complexity and diversity of training data. During inference, we propose to employ nucleus sampling and a hybrid reranking algorithm, which has not been explored in AAC research. Combining our efforts, our model achieves a new state-of-the-art 32.6 SPIDEr-FL score on the Clotho evaluation split, and wins the 2023 DCASE AAC challenge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01651v1">Informed AI Regulation: Comparing the Ethical Frameworks of Leading LLM Chatbots Using an Ethics-Based Audit to Assess Moral Reasoning and Normative Values</a></div>
    <div class="paper-meta">
      📅 2024-01-09
      | 💬 23 pages, 6 figures (3 as tables), 1 table (in LaTeX)
    </div>
    <details class="paper-abstract">
      With the rise of individual and collaborative networks of autonomous agents, AI is deployed in more key reasoning and decision-making roles. For this reason, ethics-based audits play a pivotal role in the rapidly growing fields of AI safety and regulation. This paper undertakes an ethics-based audit to probe the 8 leading commercial and open-source Large Language Models including GPT-4. We assess explicability and trustworthiness by a) establishing how well different models engage in moral reasoning and b) comparing normative values underlying models as ethical frameworks. We employ an experimental, evidence-based approach that challenges the models with ethical dilemmas in order to probe human-AI alignment. The ethical scenarios are designed to require a decision in which the particulars of the situation may or may not necessitate deviating from normative ethical principles. A sophisticated ethical framework was consistently elicited in one model, GPT-4. Nonetheless, troubling findings include underlying normative frameworks with clear bias towards particular cultural norms. Many models also exhibit disturbing authoritarian tendencies. Code is available at https://github.com/jonchun/llm-sota-chatbots-ethics-based-audit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08671v1">DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference</a></div>
    <div class="paper-meta">
      📅 2024-01-09
    </div>
    <details class="paper-abstract">
      The deployment and scaling of large language models (LLMs) have become critical as they permeate various applications, demanding high-throughput and low-latency serving systems. Existing frameworks struggle to balance these requirements, especially for workloads with long prompts. This paper introduces DeepSpeed-FastGen, a system that employs Dynamic SplitFuse, a novel prompt and generation composition strategy, to deliver up to 2.3x higher effective throughput, 2x lower latency on average, and up to 3.7x lower (token-level) tail latency, compared to state-of-the-art systems like vLLM. We leverage a synergistic combination of DeepSpeed-MII and DeepSpeed-Inference to provide an efficient and easy-to-use serving system for LLMs. DeepSpeed-FastGen's advanced implementation supports a range of models and offers both non-persistent and persistent deployment options, catering to diverse user scenarios from interactive sessions to long-running applications. We present a detailed benchmarking methodology, analyze the performance through latency-throughput curves, and investigate scalability via load balancing. Our evaluations demonstrate substantial improvements in throughput and latency across various models and hardware configurations. We discuss our roadmap for future enhancements, including broader model support and new hardware backends. The DeepSpeed-FastGen code is readily available for community engagement and contribution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15224v2">LLM-Powered Hierarchical Language Agent for Real-time Human-AI Coordination</a></div>
    <div class="paper-meta">
      📅 2024-01-09
      | 💬 This paper is accpeted by AAMAS 2024. More demonstrations can be seen on our website https://sites.google.com/view/overcooked-hla/
    </div>
    <details class="paper-abstract">
      AI agents powered by Large Language Models (LLMs) have made significant advances, enabling them to assist humans in diverse complex tasks and leading to a revolution in human-AI coordination. LLM-powered agents typically require invoking LLM APIs and employing artificially designed complex prompts, which results in high inference latency. While this paradigm works well in scenarios with minimal interactive demands, such as code generation, it is unsuitable for highly interactive and real-time applications, such as gaming. Traditional gaming AI often employs small models or reactive policies, enabling fast inference but offering limited task completion and interaction abilities. In this work, we consider Overcooked as our testbed where players could communicate with natural language and cooperate to serve orders. We propose a Hierarchical Language Agent (HLA) for human-AI coordination that provides both strong reasoning abilities while keeping real-time execution. In particular, HLA adopts a hierarchical framework and comprises three modules: a proficient LLM, referred to as Slow Mind, for intention reasoning and language interaction, a lightweight LLM, referred to as Fast Mind, for generating macro actions, and a reactive policy, referred to as Executor, for transforming macro actions into atomic actions. Human studies show that HLA outperforms other baseline agents, including slow-mind-only agents and fast-mind-only agents, with stronger cooperation abilities, faster responses, and more consistent language communications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.00812v2">If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents</a></div>
    <div class="paper-meta">
      📅 2024-01-08
    </div>
    <details class="paper-abstract">
      The prominent large language models (LLMs) of today differ from past language models not only in size, but also in the fact that they are trained on a combination of natural language and formal language (code). As a medium between humans and computers, code translates high-level goals into executable steps, featuring standard syntax, logical consistency, abstraction, and modularity. In this survey, we present an overview of the various benefits of integrating code into LLMs' training data. Specifically, beyond enhancing LLMs in code generation, we observe that these unique properties of code help (i) unlock the reasoning ability of LLMs, enabling their applications to a range of more complex natural language tasks; (ii) steer LLMs to produce structured and precise intermediate steps, which can then be connected to external execution ends through function calls; and (iii) take advantage of code compilation and execution environment, which also provides diverse feedback for model improvement. In addition, we trace how these profound capabilities of LLMs, brought by code, have led to their emergence as intelligent agents (IAs) in situations where the ability to understand instructions, decompose goals, plan and execute actions, and refine from feedback are crucial to their success on downstream tasks. Finally, we present several key challenges and future directions of empowering LLMs with code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08055v2">Breaking the Silence: the Threats of Using LLMs in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2024-01-08
      | 💬 Accepted at the ICSE'24 conference, NIER track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained considerable traction within the Software Engineering (SE) community, impacting various SE tasks from code completion to test generation, from program repair to code summarization. Despite their promise, researchers must still be careful as numerous intricate factors can influence the outcomes of experiments involving LLMs. This paper initiates an open discussion on potential threats to the validity of LLM-based research including issues such as closed-source models, possible data leakage between LLM training data and research evaluation, and the reproducibility of LLM-based findings. In response, this paper proposes a set of guidelines tailored for SE researchers and Language Model (LM) providers to mitigate these concerns. The implications of the guidelines are illustrated using existing good practices followed by LLM providers and a practical example for SE researchers in the context of test case generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.03851v1">Aligned with LLM: a new multi-modal training paradigm for encoding fMRI activity in visual cortex</a></div>
    <div class="paper-meta">
      📅 2024-01-08
    </div>
    <details class="paper-abstract">
      Recently, there has been a surge in the popularity of pre trained large language models (LLMs) (such as GPT-4), sweeping across the entire Natural Language Processing (NLP) and Computer Vision (CV) communities. These LLMs have demonstrated advanced multi-modal understanding capabilities and showcased strong performance across various benchmarks. The LLM has started to embody traits of artificial general intelligence, which holds vital guidance for enhancing brain-like characteristics within visual encoding models. Hence, This paper proposes a new multi-modal training paradigm, aligning with LLM, for encoding fMRI activity in visual cortex. Based on this paradigm, we trained an encoding model in fMRI data named the LLM-Visual Encoding Model (LLM-VEM). Specifically, we utilize LLM (miniGPT4) to generate descriptive text for all stimulus images, forming a high-quality textual description set. Moreover, we use the pre-trained text encoder (CLIP) to process these detailed descriptions, obtaining the text embedding features. Next, we use the contrast loss function to minimize the distance between the image embedding features and the text embedding features to complete the alignment operation of the stimulus image and text information. With the assistance of the pre-trained LLM, this alignment process facilitates better learning of the visual encoding model, resulting in higher precision. The final experimental results indicate that our training paradigm has significantly aided in enhancing the performance of the visual encoding model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.04138v1">Expanding Horizons in HCI Research Through LLM-Driven Qualitative Analysis</a></div>
    <div class="paper-meta">
      📅 2024-01-07
    </div>
    <details class="paper-abstract">
      How would research be like if we still needed to "send" papers typed with a typewriter? Our life and research environment have continually evolved, often accompanied by controversial opinions about new methodologies. In this paper, we embrace this change by introducing a new approach to qualitative analysis in HCI using Large Language Models (LLMs). We detail a method that uses LLMs for qualitative data analysis and present a quantitative framework using SBART cosine similarity for performance evaluation. Our findings indicate that LLMs not only match the efficacy of traditional analysis methods but also offer unique insights. Through a novel dataset and benchmark, we explore LLMs' characteristics in HCI research, suggesting potential avenues for further exploration and application in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.01040v3">From Beginner to Expert: Modeling Medical Knowledge into General LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-07
      | 💬 Developed by Ant Group for PubMedQA leaderboard
    </div>
    <details class="paper-abstract">
      Recently, large language model (LLM) based artificial intelligence (AI) systems have demonstrated remarkable capabilities in natural language understanding and generation. However, these models face a significant challenge when it comes to sensitive applications, such as reasoning over medical knowledge and answering medical questions in a physician-like manner. Prior studies attempted to overcome this challenge by increasing the model size (>100B) to learn more general medical knowledge, while there is still room for improvement in LLMs with smaller-scale model sizes (<100B). In this work, we start from a pre-trained general LLM model (AntGLM-10B) and fine-tune it from a medical beginner towards a medical expert (called AntGLM-Med-10B), which leverages a 3-stage optimization procedure, i.e., general medical knowledge injection, medical domain instruction tuning, and specific medical task adaptation. Our contributions are threefold: (1) We specifically investigate how to adapt a pre-trained general LLM in medical domain, especially for a specific medical task. (2) We collect and construct large-scale medical datasets for each stage of the optimization process. These datasets encompass various data types and tasks, such as question-answering, medical reasoning, multi-choice questions, and medical conversations. (3) Specifically for multi-choice questions in the medical domain, we propose a novel Verification-of-Choice approach for prompting engineering, which significantly enhances the reasoning ability of LLMs. Remarkably, by combining the above approaches, our AntGLM-Med-10B model can outperform the most of LLMs on PubMedQA, including both general and medical LLMs, even when these LLMs have larger model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.03388v1">LLMs for Robotic Object Disambiguation</a></div>
    <div class="paper-meta">
      📅 2024-01-07
    </div>
    <details class="paper-abstract">
      The advantages of pre-trained large language models (LLMs) are apparent in a variety of language processing tasks. But can a language model's knowledge be further harnessed to effectively disambiguate objects and navigate decision-making challenges within the realm of robotics? Our study reveals the LLM's aptitude for solving complex decision making challenges that are often previously modeled by Partially Observable Markov Decision Processes (POMDPs). A pivotal focus of our research is the object disambiguation capability of LLMs. We detail the integration of an LLM into a tabletop environment disambiguation task, a decision making problem where the robot's task is to discern and retrieve a user's desired object from an arbitrarily large and complex cluster of objects. Despite multiple query attempts with zero-shot prompt engineering (details can be found in the Appendix), the LLM struggled to inquire about features not explicitly provided in the scene description. In response, we have developed a few-shot prompt engineering system to improve the LLM's ability to pose disambiguating queries. The result is a model capable of both using given features when they are available and inferring new relevant features when necessary, to successfully generate and navigate down a precise decision tree to the correct object--even when faced with identical options.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.03239v1">Reflections on Inductive Thematic Saturation as a potential metric for measuring the validity of an inductive Thematic Analysis with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-01-06
    </div>
    <details class="paper-abstract">
      This paper presents a set of reflections on saturation and the use of Large Language Models (LLMs) for performing Thematic Analysis (TA). The paper suggests that initial thematic saturation (ITS) could be used as a metric to assess part of the transactional validity of TA with LLM, focusing on the initial coding. The paper presents the initial coding of two datasets of different sizes, and it reflects on how the LLM reaches some form of analytical saturation during the coding. The procedure proposed in this work leads to the creation of two codebooks, one comprising the total cumulative initial codes and the other the total unique codes. The paper proposes a metric to synthetically measure ITS using a simple mathematical calculation employing the ratio between slopes of cumulative codes and unique codes. The paper contributes to the initial body of work exploring how to perform qualitative analysis with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06785v1">Human-Instruction-Free LLM Self-Alignment with Limited Samples</a></div>
    <div class="paper-meta">
      📅 2024-01-06
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human values is a vital task for LLM practitioners. Current alignment techniques have several limitations: (1) requiring a large amount of annotated data; (2) demanding heavy human involvement; (3) lacking a systematic mechanism to continuously improve. In this work, we study aligning LLMs to a new domain with limited samples (e.g. < 100). We propose an algorithm that can self-align LLMs iteratively without active human involvement. Unlike existing works, our algorithm relies on neither human-crafted instructions nor labeled rewards, significantly reducing human involvement. In addition, our algorithm can self-improve the alignment continuously. The key idea is to first retrieve high-quality samples related to the target domain and use them as In-context Learning examples to generate more samples. Then we use the self-generated samples to finetune the LLM iteratively. We show that our method can unlock the LLMs' self-generalization ability to perform alignment with near-zero human supervision. We test our algorithm on three benchmarks in safety, truthfulness, and instruction-following, and show good performance in alignment, domain adaptability, and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.03217v1">Understanding Large-Language Model (LLM)-powered Human-Robot Interaction</a></div>
    <div class="paper-meta">
      📅 2024-01-06
      | 💬 10 pages, 4 figures. Callie Y. Kim and Christine P. Lee contributed equally to the work. To be published in Proceedings of the 2024 ACM/IEEE International Conference on Human-Robot Interaction (HRI '24), March 11--14, 2024, Boulder, CO, USA
    </div>
    <details class="paper-abstract">
      Large-language models (LLMs) hold significant promise in improving human-robot interaction, offering advanced conversational skills and versatility in managing diverse, open-ended user requests in various tasks and domains. Despite the potential to transform human-robot interaction, very little is known about the distinctive design requirements for utilizing LLMs in robots, which may differ from text and voice interaction and vary by task and context. To better understand these requirements, we conducted a user study (n = 32) comparing an LLM-powered social robot against text- and voice-based agents, analyzing task-based requirements in conversational tasks, including choose, generate, execute, and negotiate. Our findings show that LLM-powered robots elevate expectations for sophisticated non-verbal cues and excel in connection-building and deliberation, but fall short in logical communication and may induce anxiety. We provide design implications both for robots integrating LLMs and for fine-tuning LLMs for use with robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.02038v2">Understanding LLMs: A Comprehensive Overview from Training to Inference</a></div>
    <div class="paper-meta">
      📅 2024-01-06
      | 💬 30 pages,6 figures
    </div>
    <details class="paper-abstract">
      The introduction of ChatGPT has led to a significant increase in the utilization of Large Language Models (LLMs) for addressing downstream tasks. There's an increasing focus on cost-efficient training and deployment within this context. Low-cost training and deployment of LLMs represent the future development trend. This paper reviews the evolution of large language model training techniques and inference deployment technologies aligned with this emerging trend. The discussion on training includes various aspects, including data preprocessing, training architecture, pre-training tasks, parallel training, and relevant content related to model fine-tuning. On the inference side, the paper covers topics such as model compression, parallel computation, memory scheduling, and structural optimization. It also explores LLMs' utilization and provides insights into their future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.02954v1">DeepSeek LLM: Scaling Open-Source Language Models with Longtermism</a></div>
    <div class="paper-meta">
      📅 2024-01-05
    </div>
    <details class="paper-abstract">
      The rapid development of open-source large language models (LLMs) has been truly remarkable. However, the scaling law described in previous literature presents varying conclusions, which casts a dark cloud over scaling LLMs. We delve into the study of scaling laws and present our distinctive findings that facilitate scaling of large scale models in two commonly used open-source configurations, 7B and 67B. Guided by the scaling laws, we introduce DeepSeek LLM, a project dedicated to advancing open-source language models with a long-term perspective. To support the pre-training phase, we have developed a dataset that currently consists of 2 trillion tokens and is continuously expanding. We further conduct supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) on DeepSeek LLM Base models, resulting in the creation of DeepSeek Chat models. Our evaluation results demonstrate that DeepSeek LLM 67B surpasses LLaMA-2 70B on various benchmarks, particularly in the domains of code, mathematics, and reasoning. Furthermore, open-ended evaluations reveal that DeepSeek LLM 67B Chat exhibits superior performance compared to GPT-3.5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.02412v1">LLM Augmented LLMs: Expanding Capabilities through Composition</a></div>
    <div class="paper-meta">
      📅 2024-01-04
      | 💬 17 pages, 2 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Foundational models with billions of parameters which have been trained on large corpora of data have demonstrated non-trivial skills in a variety of domains. However, due to their monolithic structure, it is challenging and expensive to augment them or impart new skills. On the other hand, due to their adaptation abilities, several new instances of these models are being trained towards new domains and tasks. In this work, we study the problem of efficient and practical composition of existing foundation models with more specific models to enable newer capabilities. To this end, we propose CALM -- Composition to Augment Language Models -- which introduces cross-attention between models to compose their representations and enable new capabilities. Salient features of CALM are: (i) Scales up LLMs on new tasks by 're-using' existing LLMs along with a few additional parameters and data, (ii) Existing model weights are kept intact, and hence preserves existing capabilities, and (iii) Applies to diverse domains and settings. We illustrate that augmenting PaLM2-S with a smaller model trained on low-resource languages results in an absolute improvement of up to 13\% on tasks like translation into English and arithmetic reasoning for low-resource languages. Similarly, when PaLM2-S is augmented with a code-specific model, we see a relative improvement of 40\% over the base model for code generation and explanation tasks -- on-par with fully fine-tuned counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.02297v1">Are LLMs Robust for Spoken Dialogues?</a></div>
    <div class="paper-meta">
      📅 2024-01-04
    </div>
    <details class="paper-abstract">
      Large Pre-Trained Language Models have demonstrated state-of-the-art performance in different downstream tasks, including dialogue state tracking and end-to-end response generation. Nevertheless, most of the publicly available datasets and benchmarks on task-oriented dialogues focus on written conversations. Consequently, the robustness of the developed models to spoken interactions is unknown. In this work, we have evaluated the performance of LLMs for spoken task-oriented dialogues on the DSTC11 test sets. Due to the lack of proper spoken dialogue datasets, we have automatically transcribed a development set of spoken dialogues with a state-of-the-art ASR engine. We have characterized the ASR-error types and their distributions and simulated these errors in a large dataset of dialogues. We report the intrinsic (perplexity) and extrinsic (human evaluation) performance of fine-tuned GPT-2 and T5 models in two subtasks of response generation and dialogue state tracking, respectively. The results show that LLMs are not robust to spoken noise by default, however, fine-tuning/training such models on a proper dataset of spoken TODs can result in a more robust performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.02115v1">Using LLM to select the right SQL Query from candidates</a></div>
    <div class="paper-meta">
      📅 2024-01-04
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Text-to-SQL models can generate a list of candidate SQL queries, and the best query is often in the candidate list, but not at the top of the list. An effective re-rank method can select the right SQL query from the candidate list and improve the model's performance. Previous studies on code generation automatically generate test cases and use them to re-rank candidate codes. However, automatic test case generation for text-to-SQL is an understudied field. We propose an automatic test case generation method that first generates a database and then uses LLMs to predict the ground truth, which is the expected execution results of the ground truth SQL query on this database. To reduce the difficulty for LLMs to predict, we conduct experiments to search for ways to generate easy databases for LLMs and design easy-to-understand prompts. Based on our test case generation method, we propose a re-rank method to select the right SQL query from the candidate list. Given a candidate list, our method can generate test cases and re-rank the candidate list according to their pass numbers on these test cases and their generation probabilities. The experiment results on the validation dataset of Spider show that the performance of some state-of-the-art models can get a 3.6\% improvement after applying our re-rank method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.07878v4">Evaluating LLMs on Document-Based QA: Exact Answer Selection and Numerical Extraction using Cogtale dataset</a></div>
    <div class="paper-meta">
      📅 2024-01-03
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Document-based Question-Answering (QA) tasks are crucial for precise information retrieval. While some existing work focus on evaluating large language models performance on retrieving and answering questions from documents, assessing the LLMs performance on QA types that require exact answer selection from predefined options and numerical extraction is yet to be fully assessed. In this paper, we specifically focus on this underexplored context and conduct empirical analysis of LLMs (GPT-4 and GPT-3.5) on question types, including single-choice, yes-no, multiple-choice, and number extraction questions from documents in zero-shot setting. We use the CogTale dataset for evaluation, which provide human expert-tagged responses, offering a robust benchmark for precision and factual grounding. We found that LLMs, particularly GPT-4, can precisely answer many single-choice and yes-no questions given relevant context, demonstrating their efficacy in information retrieval tasks. However, their performance diminishes when confronted with multiple-choice and number extraction formats, lowering the overall performance of the model on this task, indicating that these models may not yet be sufficiently reliable for the task. This limits the applications of LLMs on applications demanding precise information extraction from documents, such as meta-analysis tasks. These findings hinge on the assumption that the retrievers furnish pertinent context necessary for accurate responses, emphasizing the need for further research. Our work offers a framework for ongoing dataset evaluation, ensuring that LLM applications for information retrieval and document analysis continue to meet evolving standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.08163v2">Self-Assessment Tests are Unreliable Measures of LLM Personality</a></div>
    <div class="paper-meta">
      📅 2024-01-02
    </div>
    <details class="paper-abstract">
      As large language models (LLM) evolve in their capabilities, various recent studies have tried to quantify their behavior using psychological tools created to study human behavior. One such example is the measurement of "personality" of LLMs using self-assessment personality tests developed to measure human personality. Yet almost none of these works verify the applicability of these tests on LLMs. In this paper, we analyze the reliability of LLM personality scores obtained from self-assessment personality tests using two simple experiments. We first introduce the property of prompt sensitivity, where three semantically equivalent prompts representing three intuitive ways of administering self-assessment tests on LLMs are used to measure the personality of the same LLM. We find that all three prompts lead to very different personality scores, a difference that is statistically significant for all traits in a large majority of scenarios. We then introduce the property of option-order symmetry for personality measurement of LLMs. Since most of the self-assessment tests exist in the form of multiple choice question (MCQ) questions, we argue that the scores should also be robust to not just the prompt template but also the order in which the options are presented. This test unsurprisingly reveals that the self-assessment test scores are not robust to the order of the options. These simple tests, done on ChatGPT and three Llama2 models of different sizes, show that self-assessment personality tests created for humans are unreliable measures of personality in LLMs.
    </details>
</div>
