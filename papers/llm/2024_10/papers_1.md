# llm - 2024_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00890v1">Rethinking Scale: The Efficacy of Fine-Tuned Open-Source LLMs in Large-Scale Reproducible Social Science Research</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are distinguished by their architecture, which dictates their parameter size and performance capabilities. Social scientists have increasingly adopted LLMs for text classification tasks, which are difficult to scale with human coders. While very large, closed-source models often deliver superior performance, their use presents significant risks. These include lack of transparency, potential exposure of sensitive data, challenges to replicability, and dependence on proprietary systems. Additionally, their high costs make them impractical for large-scale research projects. In contrast, open-source models, although available in various sizes, may underperform compared to commercial alternatives if used without further fine-tuning. However, open-source models offer distinct advantages: they can be run locally (ensuring data privacy), fine-tuned for specific tasks, shared within the research community, and integrated into reproducible workflows. This study demonstrates that small, fine-tuned open-source LLMs can achieve equal or superior performance to models such as ChatGPT-4. We further explore the relationship between training set size and fine-tuning efficacy in open-source models. Finally, we propose a hybrid workflow that leverages the strengths of both open and closed models, offering a balanced approach to performance, transparency, and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06423v3">Evaluating LLMs on Entity Disambiguation in Tables</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 13 pages, 6 figures; fixed avg. accuracy-over-price plot for GPT families, fixed typos in table referencing, added evaluation and inference subsubsection
    </div>
    <details class="paper-abstract">
      Tables are crucial containers of information, but understanding their meaning may be challenging. Over the years, there has been a surge in interest in data-driven approaches based on deep learning that have increasingly been combined with heuristic-based ones. In the last period, the advent of \acf{llms} has led to a new category of approaches for table annotation. However, these approaches have not been consistently evaluated on a common ground, making evaluation and comparison difficult. This work proposes an extensive evaluation of four STI SOTA approaches: Alligator (formerly s-elbat), Dagobah, TURL, and TableLlama; the first two belong to the family of heuristic-based algorithms, while the others are respectively encoder-only and decoder-only Large Language Models (LLMs). We also include in the evaluation both GPT-4o and GPT-4o-mini, since they excel in various public benchmarks. The primary objective is to measure the ability of these approaches to solve the entity disambiguation task with respect to both the performance achieved on a common-ground evaluation setting and the computational and cost requirements involved, with the ultimate aim of charting new research paths in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13959v2">FinQAPT: Empowering Financial Decisions with End-to-End LLM-driven Question Answering Pipeline</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 Accepted in ICAIF 2024, 8 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Financial decision-making hinges on the analysis of relevant information embedded in the enormous volume of documents in the financial domain. To address this challenge, we developed FinQAPT, an end-to-end pipeline that streamlines the identification of relevant financial reports based on a query, extracts pertinent context, and leverages Large Language Models (LLMs) to perform downstream tasks. To evaluate the pipeline, we experimented with various techniques to optimize the performance of each module using the FinQA dataset. We introduced a novel clustering-based negative sampling technique to enhance context extraction and a novel prompting method called Dynamic N-shot Prompting to boost the numerical question-answering capabilities of LLMs. At the module level, we achieved state-of-the-art accuracy on FinQA, attaining an accuracy of 80.6%. However, at the pipeline level, we observed decreased performance due to challenges in extracting relevant context from financial reports. We conducted a detailed error analysis of each module and the end-to-end pipeline, pinpointing specific challenges that must be addressed to develop a robust solution for handling complex financial tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00136v1">LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have propelled groundbreaking advancements across several domains and are commonly used for text generation applications. However, the computational demands of these complex models pose significant challenges, requiring efficient hardware acceleration. Benchmarking the performance of LLMs across diverse hardware platforms is crucial to understanding their scalability and throughput characteristics. We introduce LLM-Inference-Bench, a comprehensive benchmarking suite to evaluate the hardware inference performance of LLMs. We thoroughly analyze diverse hardware platforms, including GPUs from Nvidia and AMD and specialized AI accelerators, Intel Habana and SambaNova. Our evaluation includes several LLM inference frameworks and models from LLaMA, Mistral, and Qwen families with 7B and 70B parameters. Our benchmarking results reveal the strengths and limitations of various models, hardware platforms, and inference frameworks. We provide an interactive dashboard to help identify configurations for optimal performance for a given hardware platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.12794v3">Benchmarking LLMs via Uncertainty Quantification</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 30 pages, accepted to NeurIPS 2024
    </div>
    <details class="paper-abstract">
      The proliferation of open-source Large Language Models (LLMs) from various institutions has highlighted the urgent need for comprehensive evaluation methods. However, current evaluation platforms, such as the widely recognized HuggingFace open LLM leaderboard, neglect a crucial aspect -- uncertainty, which is vital for thoroughly assessing LLMs. To bridge this gap, we introduce a new benchmarking approach for LLMs that integrates uncertainty quantification. Our examination involves nine LLMs (LLM series) spanning five representative natural language processing tasks. Our findings reveal that: I) LLMs with higher accuracy may exhibit lower certainty; II) Larger-scale LLMs may display greater uncertainty compared to their smaller counterparts; and III) Instruction-finetuning tends to increase the uncertainty of LLMs. These results underscore the significance of incorporating uncertainty in the evaluation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02119v3">Tree of Attacks: Jailbreaking Black-Box LLMs Automatically</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 Accepted for presentation at NeurIPS 2024. Code: https://github.com/RICommunity/TAP
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an attacker LLM to iteratively refine candidate (attack) prompts until one of the refined prompts jailbreaks the target. In addition, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks, reducing the number of queries sent to the target LLM. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4-Turbo and GPT4o) for more than 80% of the prompts. This significantly improves upon the previous state-of-the-art black-box methods for generating jailbreaks while using a smaller number of queries than them. Furthermore, TAP is also capable of jailbreaking LLMs protected by state-of-the-art guardrails, e.g., LlamaGuard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.24034v1">Handwriting Recognition in Historical Documents with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      There is an immense quantity of historical and cultural documentation that exists only as handwritten manuscripts. At the same time, performing OCR across scripts and different handwriting styles has proven to be an enormously difficult problem relative to the process of digitizing print. While recent Transformer based models have achieved relatively strong performance, they rely heavily on manually transcribed training data and have difficulty generalizing across writers. Multimodal LLM, such as GPT-4v and Gemini, have demonstrated effectiveness in performing OCR and computer vision tasks with few shot prompting. In this paper, I evaluate the accuracy of handwritten document transcriptions generated by Gemini against the current state of the art Transformer based methods. Keywords: Optical Character Recognition, Multimodal Language Models, Cultural Preservation, Mass digitization, Handwriting Recognitio
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10476v2">Will LLMs Replace the Encoder-Only Models in Temporal Relation Classification?</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      The automatic detection of temporal relations among events has been mainly investigated with encoder-only models such as RoBERTa. Large Language Models (LLM) have recently shown promising performance in temporal reasoning tasks such as temporal question answering. Nevertheless, recent studies have tested the LLMs' performance in detecting temporal relations of closed-source models only, limiting the interpretability of those results. In this work, we investigate LLMs' performance and decision process in the Temporal Relation Classification task. First, we assess the performance of seven open and closed-sourced LLMs experimenting with in-context learning and lightweight fine-tuning approaches. Results show that LLMs with in-context learning significantly underperform smaller encoder-only models based on RoBERTa. Then, we delve into the possible reasons for this gap by applying explainable methods. The outcome suggests a limitation of LLMs in this task due to their autoregressive nature, which causes them to focus only on the last part of the sequence. Additionally, we evaluate the word embeddings of these two models to better understand their pre-training differences. The code and the fine-tuned models can be found respectively on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16434v2">Enhancing LLM's Cognition via Structurization</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 This paper has been accepted by NeurIPS 2024. Code is available at https://github.com/alibaba/struxgpt
    </div>
    <details class="paper-abstract">
      When reading long-form text, human cognition is complex and structurized. While large language models (LLMs) process input contexts through a causal and sequential perspective, this approach can potentially limit their ability to handle intricate and complex inputs effectively. To enhance LLM's cognition capability, this paper presents a novel concept of context structurization. Specifically, we transform the plain, unordered contextual sentences into well-ordered and hierarchically structurized elements. By doing so, LLMs can better grasp intricate and extended contexts through precise attention and information-seeking along the organized structures. Extensive evaluations are conducted across various model architectures and sizes (including a series of auto-regressive LLMs as well as BERT-like masking models) on a diverse set of NLP tasks (e.g., context-based question-answering, exhaustive hallucination evaluation, and passage-level dense retrieval). Empirical results show consistent and significant performance gains afforded by a single-round structurization. In particular, we boost the open-sourced LLaMA2-70B model to achieve comparable performance against GPT-3.5-Turbo as the hallucination evaluator. Besides, we show the feasibility of distilling advanced LLMs' language processing abilities to a smaller yet effective StruXGPT-7B to execute structurization, addressing the practicality of our approach. Code is available at https://github.com/alibaba/struxgpt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23890v1">Leveraging LLMs for MT in Crisis Scenarios: a blueprint for low-resource languages</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 arXiv admin note: text overlap with arXiv:2403.02370, arXiv:2403.01580
    </div>
    <details class="paper-abstract">
      In an evolving landscape of crisis communication, the need for robust and adaptable Machine Translation (MT) systems is more pressing than ever, particularly for low-resource languages. This study presents a comprehensive exploration of leveraging Large Language Models (LLMs) and Multilingual LLMs (MLLMs) to enhance MT capabilities in such scenarios. By focusing on the unique challenges posed by crisis situations where speed, accuracy, and the ability to handle a wide range of languages are paramount, this research outlines a novel approach that combines the cutting-edge capabilities of LLMs with fine-tuning techniques and community-driven corpus development strategies. At the core of this study is the development and empirical evaluation of MT systems tailored for two low-resource language pairs, illustrating the process from initial model selection and fine-tuning through to deployment. Bespoke systems are developed and modelled on the recent Covid-19 pandemic. The research highlights the importance of community involvement in creating highly specialised, crisis-specific datasets and compares custom GPTs with NLLB-adapted MLLM models. It identifies fine-tuned MLLM models as offering superior performance compared with their LLM counterparts. A scalable and replicable model for rapid MT system development in crisis scenarios is outlined. Our approach enhances the field of humanitarian technology by offering a blueprint for developing multilingual communication systems during emergencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23844v1">Commonsense Knowledge Editing Based on Free-Text in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Knowledge editing technology is crucial for maintaining the accuracy and timeliness of large language models (LLMs) . However, the setting of this task overlooks a significant portion of commonsense knowledge based on free-text in the real world, characterized by broad knowledge scope, long content and non instantiation. The editing objects of previous methods (e.g., MEMIT) were single token or entity, which were not suitable for commonsense knowledge in free-text form. To address the aforementioned challenges, we conducted experiments from two perspectives: knowledge localization and knowledge editing. Firstly, we introduced Knowledge Localization for Free-Text(KLFT) method, revealing the challenges associated with the distribution of commonsense knowledge in MLP and Attention layers, as well as in decentralized distribution. Next, we propose a Dynamics-aware Editing Method(DEM), which utilizes a Dynamics-aware Module to locate the parameter positions corresponding to commonsense knowledge, and uses Knowledge Editing Module to update knowledge. The DEM method fully explores the potential of the MLP and Attention layers, and successfully edits commonsense knowledge based on free-text. The experimental results indicate that the DEM can achieve excellent editing performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11393v2">LLM-Agent-UMF: LLM-based Agent Unified Modeling Framework for Seamless Integration of Multi Active/Passive Core-Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 36 pages, 19 figures, 3 tables
    </div>
    <details class="paper-abstract">
      In an era where vast amounts of data are collected and processed from diverse sources, there is a growing demand to develop sophisticated AI systems capable of intelligently fusing and analyzing this information. To address these challenges, researchers have turned towards integrating tools into LLM-powered agents to enhance the overall information fusion process. However, the conjunction of these technologies and the proposed enhancements in several state-of-the-art works followed a non-unified software architecture resulting in a lack of modularity and terminological inconsistencies among researchers. To address these issues, we propose a novel LLM-based Agent Unified Modeling Framework (LLM-Agent-UMF) that aims to establish a clear foundation for agent development from both functional and software architectural perspectives. Our framework distinguishes between the different components of an LLM-based agent, setting LLMs, and tools apart from a new element, the core-agent, playing the role of the central coordinator of the agent. This pivotal entity comprises five modules: planning, memory, profile, action, and security - the latter often neglected in previous works. By classifying core-agents into passive and active types based on their authoritative natures, we propose various multi-core agent architectures that combine unique characteristics of distinctive agents to tackle complex tasks more efficiently. We evaluate our framework by applying it to thirteen state-of-the-art agents, thereby demonstrating its alignment with their functionalities and clarifying the overlooked architectural aspects. Moreover, we thoroughly assess five of our proposed architectures through the integration of existing agents into new hybrid active/passive core-agents architectures. This analysis provides insights into potential improvements and highlights challenges involved in combining specific agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04181v2">Combining LLMs and Knowledge Graphs to Reduce Hallucinations in Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      Advancements in natural language processing have revolutionized the way we can interact with digital information systems, such as databases, making them more accessible. However, challenges persist, especially when accuracy is critical, as in the biomedical domain. A key issue is the hallucination problem, where models generate information unsupported by the underlying data, potentially leading to dangerous misinformation. This paper presents a novel approach designed to bridge this gap by combining Large Language Models (LLM) and Knowledge Graphs (KG) to improve the accuracy and reliability of question-answering systems, on the example of a biomedical KG. Built on the LangChain framework, our method incorporates a query checker that ensures the syntactical and semantic validity of LLM-generated queries, which are then used to extract information from a Knowledge Graph, substantially reducing errors like hallucinations. We evaluated the overall performance using a new benchmark dataset of 50 biomedical questions, testing several LLMs, including GPT-4 Turbo and llama3:70b. Our results indicate that while GPT-4 Turbo outperforms other models in generating accurate queries, open-source models like llama3:70b show promise with appropriate prompt engineering. To make this approach accessible, a user-friendly web-based interface has been developed, allowing users to input natural language queries, view generated and corrected Cypher queries, and verify the resulting paths for accuracy. Overall, this hybrid approach effectively addresses common issues such as data gaps and hallucinations, offering a reliable and intuitive solution for question answering systems. The source code for generating the results of this paper and for the user-interface can be found in our Git repository: https://git.zib.de/lpusch/cyphergenkg-gui
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11200v3">AvaTaR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 NeurIPS 2024 main conference
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated impressive capabilities in utilizing external tools and knowledge to boost accuracy and reduce hallucinations. However, developing prompting techniques that enable LLM agents to effectively use these tools and knowledge remains a heuristic and labor-intensive task. Here, we introduce AvaTaR, a novel and automated framework that optimizes an LLM agent to effectively leverage provided tools, improving performance on a given task. During optimization, we design a comparator module to iteratively deliver insightful and comprehensive prompts to the LLM agent by contrastively reasoning between positive and negative examples sampled from training data. We demonstrate AvaTaR on four complex multimodal retrieval datasets featuring textual, visual, and relational information, and three general question-answering (QA) datasets. We find AvaTaR consistently outperforms state-of-the-art approaches across all seven tasks, exhibiting strong generalization ability when applied to novel cases and achieving an average relative improvement of 14% on the Hit@1 metric for the retrieval datasets and 13% for the QA datasets. Code and dataset are available at https://github.com/zou-group/avatar.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17619v2">From PDFs to Structured Data: Utilizing LLM Analysis in Sports Database Management</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 11 pages, 1 figure; corrected the corresponding authors e-mail
    </div>
    <details class="paper-abstract">
      This study investigates the effectiveness of Large Language Models (LLMs) in processing semi-structured data from PDF documents into structured formats, specifically examining their application in updating the Finnish Sports Clubs Database. Through action research methodology, we developed and evaluated an AI-assisted approach utilizing OpenAI's GPT-4 and Anthropic's Claude 3 Opus models to process data from 72 sports federation membership reports. The system achieved a 90% success rate in automated processing, successfully handling 65 of 72 files without errors and converting over 7,900 rows of data. While the initial development time was comparable to traditional manual processing (three months), the implemented system shows potential for reducing future processing time by approximately 90%. Key challenges included handling multilingual content, processing multi-page datasets, and managing extraneous information. The findings suggest that while LLMs demonstrate significant potential for automating semi-structured data processing tasks, optimal results are achieved through a hybrid approach combining AI automation with selective human oversight. This research contributes to the growing body of literature on practical LLM applications in organizational data management and provides insights into the transformation of traditional data processing workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23769v1">The Potential of LLMs in Medical Education: Generating Questions and Answers for Qualification Exams</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      Recent research on large language models (LLMs) has primarily focused on their adaptation and application in specialized domains. The application of LLMs in the medical field is mainly concentrated on tasks such as the automation of medical report generation, summarization, diagnostic reasoning, and question-and-answer interactions between doctors and patients. The challenge of becoming a good teacher is more formidable than that of becoming a good student, and this study pioneers the application of LLMs in the field of medical education. In this work, we investigate the extent to which LLMs can generate medical qualification exam questions and corresponding answers based on few-shot prompts. Utilizing a real-world Chinese dataset of elderly chronic diseases, we tasked the LLMs with generating open-ended questions and answers based on a subset of sampled admission reports across eight widely used LLMs, including ERNIE 4, ChatGLM 4, Doubao, Hunyuan, Spark 4, Qwen, Llama 3, and Mistral. Furthermore, we engaged medical experts to manually evaluate these open-ended questions and answers across multiple dimensions. The study found that LLMs, after using few-shot prompts, can effectively mimic real-world medical qualification exam questions, whereas there is room for improvement in the correctness, evidence-based statements, and professionalism of the generated answers. Moreover, LLMs also demonstrate a decent level of ability to correct and rectify reference answers. Given the immense potential of artificial intelligence in the medical field, the task of generating questions and answers for medical qualification exams aimed at medical students, interns and residents can be a significant focus of future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03621v2">Attend First, Consolidate Later: On the Importance of Attention in Different LLM Layers</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      In decoder-based LLMs, the representation of a given layer serves two purposes: as input to the next layer during the computation of the current token; and as input to the attention mechanism of future tokens. In this work, we show that the importance of the latter role might be overestimated. To show that, we start by manipulating the representations of previous tokens; e.g. by replacing the hidden states at some layer k with random vectors. Our experimenting with four LLMs and four tasks show that this operation often leads to small to negligible drop in performance. Importantly, this happens if the manipulation occurs in the top part of the model-k is in the final 30-50% of the layers. In contrast, doing the same manipulation in earlier layers might lead to chance level performance. We continue by switching the hidden state of certain tokens with hidden states of other tokens from another prompt; e.g., replacing the word "Italy" with "France" in "What is the capital of Italy?". We find that when applying this switch in the top 1/3 of the model, the model ignores it (answering "Rome"). However if we apply it before, the model conforms to the switch ("Paris"). Our results hint at a two stage process in transformer-based LLMs: the first part gathers input from previous tokens, while the second mainly processes that information internally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23746v1">DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 Accepted to NeurIPS 2024 Dataset & Benchmarking Track
    </div>
    <details class="paper-abstract">
      Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating advanced prompt usages, human revisions like word substitutions, and writing errors. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23743v1">What Happened in LLMs Layers when Trained for Fast vs. Slow Thinking: A Gradient Perspective</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      What makes a difference in the post-training of LLMs? We investigate the training patterns of different layers in large language models (LLMs), through the lens of gradient, when training with different responses and initial models. We are specifically interested in how fast vs. slow thinking affects the layer-wise gradients, given the recent popularity of training LLMs on reasoning paths such as chain-of-thoughts (CoT) and process rewards. In our study, fast thinking without CoT leads to larger gradients and larger differences of gradients across layers than slow thinking (Detailed CoT), indicating the learning stability brought by the latter. Moreover, pre-trained LLMs are less affected by the instability of fast thinking than instruction-tuned LLMs. Additionally, we study whether the gradient patterns can reflect the correctness of responses when training different LLMs using slow vs. fast thinking paths. The results show that the gradients of slow thinking can distinguish correct and irrelevant reasoning paths. As a comparison, we conduct similar gradient analyses on non-reasoning knowledge learning tasks, on which, however, trivially increasing the response length does not lead to similar behaviors of slow thinking. Our study strengthens fundamental understandings of LLM training and sheds novel insights on its efficiency and stability, which pave the way towards building a generalizable System-2 agent. Our code, data, and gradient statistics can be found in: https://github.com/MingLiiii/Layer_Gradient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09136v2">Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      The recent development of chain-of-thought (CoT) decoding has enabled large language models (LLMs) to generate explicit logical reasoning paths for complex problem-solving. However, research indicates that these paths are not always deliberate and optimal. The tree-of-thought (ToT) method employs tree-searching to extensively explore the reasoning space and find better reasoning paths that CoT decoding might overlook. This deliberation, however, comes at the cost of significantly increased inference complexity. In this work, we demonstrate that fine-tuning LLMs leveraging the search tree constructed by ToT allows CoT to achieve similar or better performance, thereby avoiding the substantial inference burden. This is achieved through Chain of Preference Optimization (CPO), where LLMs are fine-tuned to align each step of the CoT reasoning paths with those of ToT using the inherent preference information in the tree-search process. Extensive experimental results show that CPO significantly improves LLM performance in solving a variety of complex problems, including question answering, fact verification, and arithmetic reasoning, demonstrating its effectiveness. Our code is available at https://github.com/sail-sg/CPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23678v1">Pseudo-Conversation Injection for LLM Goal Hijacking</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      Goal hijacking is a type of adversarial attack on Large Language Models (LLMs) where the objective is to manipulate the model into producing a specific, predetermined output, regardless of the user's original input. In goal hijacking, an attacker typically appends a carefully crafted malicious suffix to the user's prompt, which coerces the model into ignoring the user's original input and generating the target response. In this paper, we introduce a novel goal hijacking attack method called Pseudo-Conversation Injection, which leverages the weaknesses of LLMs in role identification within conversation contexts. Specifically, we construct the suffix by fabricating responses from the LLM to the user's initial prompt, followed by a prompt for a malicious new task. This leads the model to perceive the initial prompt and fabricated response as a completed conversation, thereby executing the new, falsified prompt. Following this approach, we propose three Pseudo-Conversation construction strategies: Targeted Pseudo-Conversation, Universal Pseudo-Conversation, and Robust Pseudo-Conversation. These strategies are designed to achieve effective goal hijacking across various scenarios. Our experiments, conducted on two mainstream LLM platforms including ChatGPT and Qwen, demonstrate that our proposed method significantly outperforms existing approaches in terms of attack effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23676v1">Web-Scale Visual Entity Recognition: An LLM-Driven Data Approach</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Web-scale visual entity recognition, the task of associating images with their corresponding entities within vast knowledge bases like Wikipedia, presents significant challenges due to the lack of clean, large-scale training data. In this paper, we propose a novel methodology to curate such a dataset, leveraging a multimodal large language model (LLM) for label verification, metadata generation, and rationale explanation. Instead of relying on the multimodal LLM to directly annotate data, which we found to be suboptimal, we prompt it to reason about potential candidate entity labels by accessing additional contextually relevant information (such as Wikipedia), resulting in more accurate annotations. We further use the multimodal LLM to enrich the dataset by generating question-answer pairs and a grounded finegrained textual description (referred to as "rationale") that explains the connection between images and their assigned entities. Experiments demonstrate that models trained on this automatically curated data achieve state-of-the-art performance on web-scale visual entity recognition tasks (e.g. +6.9% improvement in OVEN entity task), underscoring the importance of high-quality training data in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10862v3">Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP
    </div>
    <details class="paper-abstract">
      This paper investigates the impact of model compression on the way Large Language Models (LLMs) process prompts, particularly concerning jailbreak resistance. We show that moderate WANDA pruning can enhance resistance to jailbreaking attacks without fine-tuning, while maintaining performance on standard benchmarks. To systematically evaluate this safety enhancement, we introduce a dataset of 225 harmful tasks across five categories. Our analysis of LLaMA-2 Chat, Vicuna 1.3, and Mistral Instruct v0.2 reveals that pruning benefits correlate with initial model safety levels. We interpret these results by examining changes in attention patterns and perplexity shifts, demonstrating that pruned models exhibit sharper attention and increased sensitivity to artificial jailbreak constructs. We extend our evaluation to the AdvBench harmful behavior tasks and the GCG attack method. We find that LLaMA-2 is much safer on AdvBench prompts than on our dataset when evaluated with manual jailbreak attempts, and that pruning is effective against both automated attacks and manual jailbreaking on Advbench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23605v1">Dynamic Uncertainty Ranking: Enhancing In-Context Learning for Long-Tail Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can learn vast amounts of knowledge from diverse domains during pre-training. However, long-tail knowledge from specialized domains is often scarce and underrepresented, rarely appearing in the models' memorization. Prior work has shown that in-context learning (ICL) with retriever augmentation can help LLMs better capture long-tail knowledge, reducing their reliance on pre-trained data. Despite these advances, we observe that LLM predictions for long-tail questions remain uncertain to variations in retrieved samples. To take advantage of the uncertainty in ICL for guiding LLM predictions toward correct answers on long-tail samples, we propose a reinforcement learning-based dynamic uncertainty ranking method for ICL that accounts for the varying impact of each retrieved sample on LLM predictions. Our approach prioritizes more informative and stable samples while demoting misleading ones, updating rankings based on the feedback from the LLM w.r.t. each retrieved sample. To enhance training efficiency and reduce query costs, we introduce a learnable dynamic ranking threshold, adjusted when the model encounters negative prediction shifts. Experimental results on various question-answering datasets from different domains show that our method outperforms the best baseline by $2.76\%$, with a notable $5.96\%$ boost in accuracy on long-tail questions that elude zero-shot inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17202v3">Efficient multi-prompt evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Most popular benchmarks for comparing LLMs rely on a limited set of prompt templates, which may not fully capture the LLMs' abilities and can affect the reproducibility of results on leaderboards. Many recent works empirically verify prompt sensitivity and advocate for changes in LLM evaluation. In this paper, we consider the problem of estimating the performance distribution across many prompt variants instead of finding a single prompt to evaluate with. We introduce PromptEval, a method for estimating performance across a large set of prompts borrowing strength across prompts and examples to produce accurate estimates under practical evaluation budgets. The resulting distribution can be used to obtain performance quantiles to construct various robust performance metrics (e.g., top 95% quantile or median). We prove that PromptEval consistently estimates the performance distribution and demonstrate its efficacy empirically on three prominent LLM benchmarks: MMLU, BIG-bench Hard, and LMentry; for example, PromptEval can accurately estimate performance quantiles across 100 prompt templates on MMLU with a budget equivalent to two single-prompt evaluations. Moreover, we show how PromptEval can be useful in LLM-as-a-judge and best prompt identification applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15905v2">Boosting Code-Switching ASR with Mixture of Experts Enhanced Speech-Conditioned LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 Submitted to ICASSP 2025
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a speech-conditioned Large Language Model (LLM) integrated with a Mixture of Experts (MoE) based connector to address the challenge of Code-Switching (CS) in Automatic Speech Recognition (ASR). Specifically, we propose an Insertion and Deletion of Interruption Token (IDIT) mechanism for better transfer text generation ability of LLM to speech recognition task. We also present a connecter with MoE architecture that manages multiple languages efficiently. To further enhance the collaboration of multiple experts and leverage the understanding capabilities of LLM, we propose a two-stage progressive training strategy: 1) The connector is unfrozen and trained with language-specialized experts to map speech representations to the text space. 2) The connector and LLM LoRA adaptor are trained with the proposed IDIT mechanism and all experts are activated to learn general representations. Experimental results demonstrate that our method significantly outperforms state-of-the-art models, including end-to-end and large-scale audio-language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22041v2">An LLM-based Simulation Framework for Embodied Conversational Agents in Psychological Counseling</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 After careful consideration, we have decided to withdraw this version because there are still several details that need to be adjusted to ensure the accuracy and completeness of our work. We do not have an alternative version in the short term and will resubmit it after the revision is completed
    </div>
    <details class="paper-abstract">
      Simulation is crucial for validating algorithmic strategies in real-world scenarios. While LLM-based social simulation shows promise as a mainstream tool, simulating complex scenarios like psychological counseling remains challenging. We present ECAs (short for Embodied Conversational Agents), a framework for simulating psychological counseling clients' embodied memory, integrating embodied cognition and counseling theories. We formulate six design goals based on a comprehensive review of psychological counseling theories. Using LLMs, we expand real counseling case data into a nuanced embodied cognitive memory space and generate dialogues based on high-frequency counseling questions. We validate our framework using the D4 dataset, with evaluations by licensed counselors. Results show our approach significantly outperforms baselines in simulation authenticity and necessity. To demonstrate scalability, we created a public ECAs dataset through batch simulations. This research provides valuable insights for future social simulation studies in psychological counseling and Embodied Counseling Agents research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01563v2">LoFiT: Localized Fine-tuning on LLM Representations</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 NeurIPS 2024 Camera Ready
    </div>
    <details class="paper-abstract">
      Recent work in interpretability shows that large language models (LLMs) can be adapted for new tasks in a learning-free way: it is possible to intervene on LLM representations to elicit desired behaviors for alignment. For instance, adding certain bias vectors to the outputs of certain attention heads is reported to boost the truthfulness of models. In this work, we show that localized fine-tuning serves as an effective alternative to such representation intervention methods. We introduce a framework called Localized Fine-Tuning on LLM Representations (LoFiT), which identifies a subset of attention heads that are most important for learning a specific task, then trains offset vectors to add to the model's hidden representations at those selected heads. LoFiT localizes to a sparse set of heads (3%-10%) and learns the offset vectors from limited training data, comparable to the settings used for representation intervention. For truthfulness and reasoning tasks, we find that LoFiT's intervention vectors are more effective for LLM adaptation than vectors from representation intervention methods such as Inference-time Intervention. We also find that the localization step is important: selecting a task-specific set of attention heads can lead to higher performance than intervening on heads selected for a different task. Finally, across 7 tasks we study, LoFiT achieves comparable performance to other parameter-efficient fine-tuning methods such as LoRA, despite modifying 20x-200x fewer parameters than these methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05674v2">Coding Reliable LLM-based Integrated Task and Knowledge Agents with GenieWorksheets</a></div>
    <div class="paper-meta">
      📅 2024-10-31
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) present an opportunity to create automated assistants that can help users navigate complex tasks. However, existing approaches have limitations in handling conditional logic, integrating knowledge sources, and consistently following instructions. Researchers and industry professionals often employ ad hoc pipelines to construct conversational agents. These pipelines aim to maintain context, address failure cases, and minimize hallucinations, yet frequently fail to achieve these objectives. To this end, we present Genie - a programmable framework for creating task-oriented conversational agents that are designed to handle complex user interactions and knowledge queries. Unlike LLMs, Genie provides reliable grounded responses, with controllable agent policies through its expressive specification, Genie Worksheet. In contrast to dialog trees, it is resilient to diverse user queries, helpful with knowledge sources, and offers ease of programming policies through its declarative paradigm. The agents built using Genie outperforms the state-of-the-art method on complex logic domains in STARV2 dataset by up to 20.5%. Additionally, through a real-user study involving 62 participants, we show that Genie beats the GPT-4 with function calling baseline by 21.1%, 20.1%, and 61% on execution accuracy, dialogue act accuracy, and goal completion rate, respectively, on three diverse real-world domains
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23214v2">Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-10-31
    </div>
    <details class="paper-abstract">
      The hallucinations of large language models (LLMs) are increasingly mitigated by allowing LLMs to search for information and to ground their answers in real sources. Unfortunately, LLMs often struggle with posing the right search queries, especially when dealing with complex or otherwise indirect topics. Observing that LLMs can learn to search for relevant facts by $\textit{trying}$ different queries and learning to up-weight queries that successfully produce relevant results, we introduce $\underline{Le}$arning to $\underline{Re}$trieve by $\underline{T}$rying (LeReT), a reinforcement learning framework that explores search queries and uses preference-based optimization to improve their quality. LeReT can improve the absolute retrieval accuracy by up to 29% and the downstream generator evaluations by 17%. The simplicity and flexibility of LeReT allows it to be applied to arbitrary off-the-shelf retrievers and makes it a promising technique for improving general LLM pipelines. Project website: http://sherylhsu.com/LeReT/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23452v1">Graph-Augmented Relation Extraction Model with LLMs-Generated Support Document</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      This study introduces a novel approach to sentence-level relation extraction (RE) that integrates Graph Neural Networks (GNNs) with Large Language Models (LLMs) to generate contextually enriched support documents. By harnessing the power of LLMs to generate auxiliary information, our approach crafts an intricate graph representation of textual data. This graph is subsequently processed through a Graph Neural Network (GNN) to refine and enrich the embeddings associated with each entity ensuring a more nuanced and interconnected understanding of the data. This methodology addresses the limitations of traditional sentence-level RE models by incorporating broader contexts and leveraging inter-entity interactions, thereby improving the model's ability to capture complex relationships across sentences. Our experiments, conducted on the CrossRE dataset, demonstrate the effectiveness of our approach, with notable improvements in performance across various domains. The results underscore the potential of combining GNNs with LLM-generated context to advance the field of relation extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23426v1">Social Science Meets LLMs: How Reliable Are Large Language Models in Social Simulations?</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed for simulations, enabling applications in role-playing agents and Computational Social Science (CSS). However, the reliability of these simulations is under-explored, which raises concerns about the trustworthiness of LLMs in these applications. In this paper, we aim to answer ``How reliable is LLM-based simulation?'' To address this, we introduce TrustSim, an evaluation dataset covering 10 CSS-related topics, to systematically investigate the reliability of the LLM simulation. We conducted experiments on 14 LLMs and found that inconsistencies persist in the LLM-based simulated roles. In addition, the consistency level of LLMs does not strongly correlate with their general performance. To enhance the reliability of LLMs in simulation, we proposed Adaptive Learning Rate Based ORPO (AdaORPO), a reinforcement learning-based algorithm to improve the reliability in simulation across 7 LLMs. Our research provides a foundation for future studies to explore more robust and trustworthy LLM-based simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00863v1">Next-Token Prediction Task Assumes Optimal Data Ordering for LLM Training in Proof Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      In the field of large language model (LLM)-based proof generation, despite being trained on extensive corpora such as OpenWebMath and Arxiv, these models still exhibit only modest performance on proving tasks of moderate difficulty. We believe that this is partly due to the suboptimal order of each proof data used in training. Published proofs often follow a purely logical order, where each step logically proceeds from the previous steps based on the deductive rules. However, this order aims to facilitate the verification of the proof's soundness, rather than to help people and models learn the discovery process of the proof. In proof generation, we argue that the optimal order for one training data sample occurs when the relevant intermediate supervision for a particular proof step in the proof is always positioned to the left of that proof step. We call such order the intuitively sequential order. We validate our claims using two tasks: intuitionistic propositional logic theorem-proving and digit multiplication. Our experiments verify the order effect and provide support for our explanations. We demonstrate that training is most effective when the proof is in the intuitively sequential order. Moreover, the order effect and the performance gap between models trained on different data orders are substantial -- with an 11 percent improvement in proof success rate observed in the propositional logic theorem-proving task, between models trained on the optimal order compared to the worst order.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23331v1">Can Models Help Us Create Better Models? Evaluating LLMs as Data Scientists</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      We present a benchmark for large language models designed to tackle one of the most knowledge-intensive tasks in data science: writing feature engineering code, which requires domain knowledge in addition to a deep understanding of the underlying problem and data structure. The model is provided with a dataset description in a prompt and asked to generate code transforming it. The evaluation score is derived from the improvement achieved by an XGBoost model fit on the modified dataset compared to the original data. By an extensive evaluation of state-of-the-art models and comparison to well-established benchmarks, we demonstrate that the FeatEng of our proposal can cheaply and efficiently assess the broad capabilities of LLMs, in contrast to the existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23252v1">Evaluating Cultural and Social Awareness of LLM Web Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) expand into performing as agents for real-world applications beyond traditional NLP tasks, evaluating their robustness becomes increasingly important. However, existing benchmarks often overlook critical dimensions like cultural and social awareness. To address these, we introduce CASA, a benchmark designed to assess LLM agents' sensitivity to cultural and social norms across two web-based tasks: online shopping and social discussion forums. Our approach evaluates LLM agents' ability to detect and appropriately respond to norm-violating user queries and observations. Furthermore, we propose a comprehensive evaluation framework that measures awareness coverage, helpfulness in managing user queries, and the violation rate when facing misleading web content. Experiments show that current LLMs perform significantly better in non-agent than in web-based agent environments, with agents achieving less than 10% awareness coverage and over 40% violation rates. To improve performance, we explore two methods: prompting and fine-tuning, and find that combining both methods can offer complementary advantages -- fine-tuning on culture-specific datasets significantly enhances the agents' ability to generalize across different regions, while prompting boosts the agents' ability to navigate complex tasks. These findings highlight the importance of constantly benchmarking LLM agents' cultural and social awareness during the development cycle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.06755v4">CoTran: An LLM-based Code Translator using Reinforcement Learning with Feedback from Compiler and Symbolic Execution</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 The paper has been published at the 27th European Conference on Artificial Intelligence (ECAI-2024) and is available at https://ebooks.iospress.nl/doi/10.3233/FAIA240968. This arXiv version is the full version that includes the supplementary material (Appendix)
    </div>
    <details class="paper-abstract">
      In this paper, we present an LLM-based code translation method and an associated tool called CoTran, that translates whole-programs from one high-level programming language to another. Existing LLM-based code translation methods lack training to ensure that the translated code reliably compiles or bears substantial functional equivalence to the input code. In our work, we fine-tune an LLM using reinforcement learning, incorporating compiler feedback, and symbolic execution (symexec)-based testing feedback to assess functional equivalence between the input and output programs. The idea is to guide an LLM during fine-tuning, via compiler and symexec-based testing feedback, by letting it know how far it is from producing perfect translations. We conduct extensive experiments comparing CoTran with 14 other code translation tools, including human-written transpilers, LLM-based translation tools, and ChatGPT. Using a benchmark of over \num{57000} code pairs in Java and Python, we demonstrate that CoTran outperforms the other tools on relevant metrics such as compilation accuracy (CompAcc) and functional equivalence accuracy (FEqAcc). For example, in Python-to-Java translation, CoTran achieves 48.68% FEqAcc and 76.98% CompAcc, whereas the nearest competing tool (PLBART-base) gets 38.26% and 75.77% respectively. Additionally, CoTran, built on top of CodeT5, improves FEqAcc by +14.89% and CompAcc by +8.14% for Python-to-Java (resp., +12.94% and +4.30% for Java-to-Python).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10372v3">Instigating Cooperation among LLM Agents Using Adaptive Information Modulation</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      This paper introduces a novel framework combining LLM agents as proxies for human strategic behavior with reinforcement learning (RL) to engage these agents in evolving strategic interactions within team environments. Our approach extends traditional agent-based simulations by using strategic LLM agents (SLA) and introducing dynamic and adaptive governance through a pro-social promoting RL agent (PPA) that modulates information access across agents in a network, optimizing social welfare and promoting pro-social behavior. Through validation in iterative games, including the prisoner dilemma, we demonstrate that SLA agents exhibit nuanced strategic adaptations. The PPA agent effectively learns to adjust information transparency, resulting in enhanced cooperation rates. This framework offers significant insights into AI-mediated social dynamics, contributing to the deployment of AI in real-world team settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23180v1">ReasoningRec: Bridging Personalized Recommendations and Human-Interpretable Explanations through LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Large Language Model, Recommendation, Human-Interpretable Reasoning, Personalization
    </div>
    <details class="paper-abstract">
      This paper presents ReasoningRec, a reasoning-based recommendation framework that leverages Large Language Models (LLMs) to bridge the gap between recommendations and human-interpretable explanations. In contrast to conventional recommendation systems that rely on implicit user-item interactions, ReasoningRec employs LLMs to model users and items, focusing on preferences, aversions, and explanatory reasoning. The framework utilizes a larger LLM to generate synthetic explanations for user preferences, subsequently used to fine-tune a smaller LLM for enhanced recommendation accuracy and human-interpretable explanation. Our experimental study investigates the impact of reasoning and contextual information on personalized recommendations, revealing that the quality of contextual and personalized data significantly influences the LLM's capacity to generate plausible explanations. Empirical evaluations demonstrate that ReasoningRec surpasses state-of-the-art methods by up to 12.5\% in recommendation prediction while concurrently providing human-intelligible explanations. The code is available here: https://github.com/millenniumbismay/reasoningrec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14670v2">Exploring Design Choices for Building Language-Specific LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Accepted to EMNLP 2024 Findings
    </div>
    <details class="paper-abstract">
      Despite rapid progress in large language models (LLMs), their performance on a vast majority of languages remains unsatisfactory. In this paper, we study building language-specific LLMs by adapting monolingual and multilingual LLMs. We conduct systematic experiments on how design choices (base model selection, vocabulary extension, and continued pretraining) impact the adapted LLM, both in terms of efficiency (how many tokens are needed to encode the same amount of information) and end task performance. We find that (1) the initial performance of LLM does not always correlate with the final performance after the adaptation. Adapting an English-centric models can yield better results than adapting multilingual models despite their worse initial performance on low-resource languages. (2) Efficiency can easily improved with simple vocabulary extension and continued pretraining in most LLMs we study, and (3) The optimal adaptation method (choice of the base model, new vocabulary size, training data, initialization strategy) is highly language-dependent, and the simplest embedding initialization works well across various experimental settings. Together, our work lays foundations on efficiently building language-specific LLMs by adapting existing LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23166v1">SciPIP: An LLM-based Scientific Paper Idea Proposer</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 25 pages, 5 figures, 19 tables
    </div>
    <details class="paper-abstract">
      The exponential growth of knowledge and the increasing complexity of interdisciplinary research pose significant challenges for researchers, including information overload and difficulties in exploring novel ideas. The advancements in large language models (LLMs), such as GPT-4, have shown great potential in enhancing idea proposals, but how to effectively utilize large models for reasonable idea proposal has not been thoroughly explored. This paper proposes a scientific paper idea proposer (SciPIP). Based on a user-provided research background, SciPIP retrieves helpful papers from a literature database while leveraging the capabilities of LLMs to generate more novel and feasible ideas. To this end, 1) we construct a literature retrieval database, extracting lots of papers' multi-dimension information for fast access. Then, a literature retrieval method based on semantics, entity, and citation co-occurrences is proposed to search relevant literature from multiple aspects based on the user-provided background. 2) After literature retrieval, we introduce dual-path idea proposal strategies, where one path infers solutions from the retrieved literature and the other path generates original ideas through model brainstorming. We then combine the two to achieve a good balance between feasibility and originality. Through extensive experiments on the natural language processing (NLP) field, we demonstrate that SciPIP can retrieve citations similar to those of existing top conference papers and generate many ideas consistent with them. Additionally, we evaluate the originality of other ideas generated by SciPIP using large language models, further validating the effectiveness of our proposed method. The code and the database are released at https://github.com/cheerss/SciPIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03656v3">WildDESED: An LLM-Powered Dataset for Wild Domestic Environment Sound Event Detection System</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 DCASE WS 2024
    </div>
    <details class="paper-abstract">
      This work aims to advance sound event detection (SED) research by presenting a new large language model (LLM)-powered dataset namely wild domestic environment sound event detection (WildDESED). It is crafted as an extension to the original DESED dataset to reflect diverse acoustic variability and complex noises in home settings. We leveraged LLMs to generate eight different domestic scenarios based on target sound categories of the DESED dataset. Then we enriched the scenarios with a carefully tailored mixture of noises selected from AudioSet and ensured no overlap with target sound. We consider widely popular convolutional neural recurrent network to study WildDESED dataset, which depicts its challenging nature. We then apply curriculum learning by gradually increasing noise complexity to enhance the model's generalization capabilities across various noise levels. Our results with this approach show improvements within the noisy environment, validating the effectiveness on the WildDESED dataset promoting noise-robust SED advancements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23136v1">Real-Time Personalization for LLM-based Recommendation with Customized In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      Frequently updating Large Language Model (LLM)-based recommender systems to adapt to new user interests -- as done for traditional ones -- is impractical due to high training costs, even with acceleration methods. This work explores adapting to dynamic user interests without any model updates by leveraging In-Context Learning (ICL), which allows LLMs to learn new tasks from few-shot examples provided in the input. Using new-interest examples as the ICL few-shot examples, LLMs may learn real-time interest directly, avoiding the need for model updates. However, existing LLM-based recommenders often lose the in-context learning ability during recommendation tuning, while the original LLM's in-context learning lacks recommendation-specific focus. To address this, we propose RecICL, which customizes recommendation-specific in-context learning for real-time recommendations. RecICL organizes training examples in an in-context learning format, ensuring that in-context learning ability is preserved and aligned with the recommendation task during tuning. Extensive experiments demonstrate RecICL's effectiveness in delivering real-time recommendations without requiring model updates. Our code is available at https://github.com/ym689/rec_icl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18952v2">Dynamic Vocabulary Pruning in Early-Exit LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      Increasing the size of large language models (LLMs) has been shown to lead to better performance. However, this comes at the cost of slower and more expensive inference. Early-exiting is a promising approach for improving the efficiency of LLM inference by enabling next token prediction at intermediate layers. Yet, the large vocabulary size in modern LLMs makes the confidence estimation required for exit decisions computationally expensive, diminishing the efficiency gains. To address this, we propose dynamically pruning the vocabulary at test time for each token. Specifically, the vocabulary is pruned at one of the initial layers, and the smaller vocabulary is then used throughout the rest of the forward pass. Our experiments demonstrate that such post-hoc dynamic vocabulary pruning improves the efficiency of confidence estimation in early-exit LLMs while maintaining competitive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23099v1">Comparative Analysis of Demonstration Selection Algorithms for LLM In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In-context learning can help Large Language Models (LLMs) to adapt new tasks without additional training. However, this performance heavily depends on the quality of the demonstrations, driving research into effective demonstration selection algorithms to optimize this process. These algorithms assist users in selecting the best $k$ input-label pairs (demonstration examples) based on a given test input, enabling LLMs to in-context learn the relationship between the provided examples and the test inputs. Despite all the proposed demonstration selection algorithms, their efficiency and effectiveness remain unclear. This lack of clarity make it difficult to apply these algorithms in real-world scenarios and poses challenges for future research aimed at developing improved methods. This paper revisits six proposed algorithms, evaluating them on five datasets from both efficiency and effectiveness perspectives. Our experiments reveal significant variations in algorithm performance across different tasks, with some methods struggling to outperform random selection in certain scenarios. We also find that increasing the number of demonstrations does not always lead to better performance, and that there are often trade-offs between accuracy and computational efficiency. Our code is available at https://github.com/Tizzzzy/Demonstration_Selection_Overview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00856v1">AI in Investment Analysis: LLMs for Equity Stock Ratings</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 9 pages, 5 figures, ICAIF24: 5th ACM International Conference on AI in Finance
    </div>
    <details class="paper-abstract">
      Investment Analysis is a cornerstone of the Financial Services industry. The rapid integration of advanced machine learning techniques, particularly Large Language Models (LLMs), offers opportunities to enhance the equity rating process. This paper explores the application of LLMs to generate multi-horizon stock ratings by ingesting diverse datasets. Traditional stock rating methods rely heavily on the expertise of financial analysts, and face several challenges such as data overload, inconsistencies in filings, and delayed reactions to market events. Our study addresses these issues by leveraging LLMs to improve the accuracy and consistency of stock ratings. Additionally, we assess the efficacy of using different data modalities with LLMs for the financial domain. We utilize varied datasets comprising fundamental financial, market, and news data from January 2022 to June 2024, along with GPT-4-32k (v0613) (with a training cutoff in Sep. 2021 to prevent information leakage). Our results show that our benchmark method outperforms traditional stock rating methods when assessed by forward returns, specially when incorporating financial fundamentals. While integrating news data improves short-term performance, substituting detailed news summaries with sentiment scores reduces token use without loss of performance. In many cases, omitting news data entirely enhances performance by reducing bias. Our research shows that LLMs can be leveraged to effectively utilize large amounts of multimodal financial data, as showcased by their effectiveness at the stock rating prediction task. Our work provides a reproducible and efficient framework for generating accurate stock ratings, serving as a cost-effective alternative to traditional methods. Future work will extend to longer timeframes, incorporate diverse data, and utilize newer models for enhanced insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23079v1">BUZZ: Beehive-structured Sparse KV Cache with Segmented Heavy Hitters for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are essential in natural language processing but often struggle with inference speed and computational efficiency, limiting real-time deployment. The key-value (KV) cache mechanism reduces computational overhead in transformer models, but challenges in maintaining contextual understanding remain. In this paper, we propose BUZZ, a novel KV caching algorithm that leverages structured contextual information to minimize cache memory usage while enhancing inference speed. BUZZ employs a beehive-structured sparse cache, incorporating a sliding window to capture recent information and dynamically segmenting historical tokens into chunks to prioritize important tokens in local neighborhoods. We evaluate BUZZ on four real-world datasets: CNN/Daily Mail, XSUM, Wikitext, and 10-QA. Our results demonstrate that BUZZ (1) reduces cache memory usage by $\textbf{2.5}\times$ in LLM inference while maintaining over 99% accuracy in long-text summarization, and (2) surpasses state-of-the-art performance in multi-document question answering by $\textbf{7.69%}$ under the same memory limit, where full cache methods encounter out-of-memory issues. Additionally, BUZZ achieves significant inference speedup with a $\log{n}$ time complexity. The code is available at https://github.com/JunqiZhao888/buzz-llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02490v2">MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Accepted at NeurIPS 2024 (Spotlight)
    </div>
    <details class="paper-abstract">
      The computational challenges of Large Language Model (LLM) inference remain a significant barrier to their widespread deployment, especially as prompt lengths continue to increase. Due to the quadratic complexity of the attention computation, it takes 30 minutes for an 8B LLM to process a prompt of 1M tokens (i.e., the pre-filling stage) on a single A100 GPU. Existing methods for speeding up prefilling often fail to maintain acceptable accuracy or efficiency when applied to long-context LLMs. To address this gap, we introduce MInference (Milliontokens Inference), a sparse calculation method designed to accelerate pre-filling of long-sequence processing. Specifically, we identify three unique patterns in long-context attention matrices-the A-shape, Vertical-Slash, and Block-Sparsethat can be leveraged for efficient sparse computation on GPUs. We determine the optimal pattern for each attention head offline and dynamically build sparse indices based on the assigned pattern during inference. With the pattern and sparse indices, we perform efficient sparse attention calculations via our optimized GPU kernels to significantly reduce the latency in the pre-filling stage of long-context LLMs. Our proposed technique can be directly applied to existing LLMs without any modifications to the pre-training setup or additional fine-tuning. By evaluating on a wide range of downstream tasks, including InfiniteBench, RULER, PG-19, and Needle In A Haystack, and models including LLaMA-3-1M, GLM4-1M, Yi-200K, Phi-3-128K, and Qwen2-128K, we demonstrate that MInference effectively reduces inference latency by up to 10x for pre-filling on an A100, while maintaining accuracy. Our code is available at https://aka.ms/MInference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11387v3">LLM2Swarm: Robot Swarms that Responsively Reason, Plan, and Collaborate through LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Accepted at NeurIPS 2024 Workshop on Open-World Agents. Code: https://github.com/Pold87/LLM2Swarm/
    </div>
    <details class="paper-abstract">
      Robot swarms are composed of many simple robots that communicate and collaborate to fulfill complex tasks. Robot controllers usually need to be specified by experts on a case-by-case basis via programming code. This process is time-consuming, prone to errors, and unable to take into account all situations that may be encountered during deployment. On the other hand, recent Large Language Models (LLMs) have demonstrated reasoning and planning capabilities, introduced new ways to interact with and program machines, and incorporate both domain-specific and commonsense knowledge. Hence, we propose to address the aforementioned challenges by integrating LLMs with robot swarms and show the potential in proofs of concept (showcases). For this integration, we explore two approaches. The first approach is 'indirect integration,' where LLMs are used to synthesize and validate the robot controllers. This approach may reduce development time and human error before deployment. Moreover, during deployment, it could be used for on-the-fly creation of new robot behaviors. The second approach is 'direct integration,' where each robot locally executes a separate LLM instance during deployment for robot-robot collaboration and human-swarm interaction. These local LLM instances enable each robot to reason, plan, and collaborate using natural language, as demonstrated in our showcases where the robots are able to detect a variety of anomalies, without prior information about the nature of these anomalies. To enable further research on our mainly conceptual contribution, we release the software and videos for our LLM2Swarm system: https://github.com/Pold87/LLM2Swarm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14516v4">Do LLMs "know" internally when they follow instructions?</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      Instruction-following is crucial for building AI agents with large language models (LLMs), as these models must adhere strictly to user-provided constraints and guidelines. However, LLMs often fail to follow even simple and clear instructions. To improve instruction-following behavior and prevent undesirable outputs, a deeper understanding of how LLMs' internal states relate to these outcomes is required. Our analysis of LLM internal states reveal a dimension in the input embedding space linked to successful instruction-following. We demonstrate that modifying representations along this dimension improves instruction-following success rates compared to random changes, without compromising response quality. Further investigation reveals that this dimension is more closely related to the phrasing of prompts rather than the inherent difficulty of the task or instructions. This discovery also suggests explanations for why LLMs sometimes fail to follow clear instructions and why prompt engineering is often effective, even when the content remains largely unchanged. This work provides insight into the internal workings of LLMs' instruction-following, paving the way for reliable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17515v3">From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 This paper has been accepted for NeurIPS 2024. Code and data are available at https://github.com/ameliawong1996/From_News_to_Forecast
    </div>
    <details class="paper-abstract">
      This paper introduces a novel approach that leverages Large Language Models (LLMs) and Generative Agents to enhance time series forecasting by reasoning across both text and time series data. With language as a medium, our method adaptively integrates social events into forecasting models, aligning news content with time series fluctuations to provide richer insights. Specifically, we utilize LLM-based agents to iteratively filter out irrelevant news and employ human-like reasoning to evaluate predictions. This enables the model to analyze complex events, such as unexpected incidents and shifts in social behavior, and continuously refine the selection logic of news and the robustness of the agent's output. By integrating selected news events with time series data, we fine-tune a pre-trained LLM to predict sequences of digits in time series. The results demonstrate significant improvements in forecasting accuracy, suggesting a potential paradigm shift in time series forecasting through the effective utilization of unstructured news data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00418v3">LLMs for Targeted Sentiment in News Headlines: Exploring the Descriptive-Prescriptive Dilemma</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Presented at 14th Workshop on Computational Approaches to Subjectivity, Sentiment, & Social Media Analysis (WASSA) at ACL 2024
    </div>
    <details class="paper-abstract">
      News headlines often evoke sentiment by intentionally portraying entities in particular ways, making targeted sentiment analysis (TSA) of headlines a worthwhile but difficult task. Due to its subjectivity, creating TSA datasets can involve various annotation paradigms, from descriptive to prescriptive, either encouraging or limiting subjectivity. LLMs are a good fit for TSA due to their broad linguistic and world knowledge and in-context learning abilities, yet their performance depends on prompt design. In this paper, we compare the accuracy of state-of-the-art LLMs and fine-tuned encoder models for TSA of news headlines using descriptive and prescriptive datasets across several languages. Exploring the descriptive--prescriptive continuum, we analyze how performance is affected by prompt prescriptiveness, ranging from plain zero-shot to elaborate few-shot prompts. Finally, we evaluate the ability of LLMs to quantify uncertainty via calibration error and comparison to human label variation. We find that LLMs outperform fine-tuned encoders on descriptive datasets, while calibration and F1-score generally improve with increased prescriptiveness, yet the optimal level varies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08706v2">L3Cube-IndicQuest: A Benchmark Question Answering Dataset for Evaluating Knowledge of LLMs in Indic Context</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 Accepted at PACLIC 38 (2024)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant progress in incorporating Indic languages within multilingual models. However, it is crucial to quantitatively assess whether these languages perform comparably to globally dominant ones, such as English. Currently, there is a lack of benchmark datasets specifically designed to evaluate the regional knowledge of LLMs in various Indic languages. In this paper, we present the L3Cube-IndicQuest, a gold-standard factual question-answering benchmark dataset designed to evaluate how well multilingual LLMs capture regional knowledge across various Indic languages. The dataset contains 200 question-answer pairs, each for English and 19 Indic languages, covering five domains specific to the Indic region. We aim for this dataset to serve as a benchmark, providing ground truth for evaluating the performance of LLMs in understanding and representing knowledge relevant to the Indian context. The IndicQuest can be used for both reference-based evaluation and LLM-as-a-judge evaluation. The dataset is shared publicly at https://github.com/l3cube-pune/indic-nlp .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13185v5">Chain of Ideas: Revolutionizing Research Via Novel Idea Development with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 10 pages,5 figures, conference
    </div>
    <details class="paper-abstract">
      Effective research ideation is a critical step for scientific research. However, the exponential increase in scientific literature makes it challenging for researchers to stay current with recent advances and identify meaningful research directions. Recent developments in large language models~(LLMs) suggest a promising avenue for automating the generation of novel research ideas. However, existing methods for idea generation either trivially prompt LLMs or directly expose LLMs to extensive literature without indicating useful information. Inspired by the research process of human researchers, we propose a Chain-of-Ideas~(CoI) agent, an LLM-based agent that organizes relevant literature in a chain structure to effectively mirror the progressive development in a research domain. This organization facilitates LLMs to capture the current advancements in research, thereby enhancing their ideation capabilities. Furthermore, we propose Idea Arena, an evaluation protocol that can comprehensively evaluate idea generation methods from different perspectives, aligning closely with the preferences of human researchers. Experimental results indicate that the CoI agent consistently outperforms other methods and shows comparable quality as humans in research idea generation. Moreover, our CoI agent is budget-friendly, with a minimum cost of \$0.50 to generate a candidate idea and its corresponding experimental design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22809v1">Causality-Enhanced Behavior Sequence Modeling in LLMs for Personalized Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      Recent advancements in recommender systems have focused on leveraging Large Language Models (LLMs) to improve user preference modeling, yielding promising outcomes. However, current LLM-based approaches struggle to fully leverage user behavior sequences, resulting in suboptimal preference modeling for personalized recommendations. In this study, we propose a novel Counterfactual Fine-Tuning (CFT) method to address this issue by explicitly emphasizing the role of behavior sequences when generating recommendations. Specifically, we employ counterfactual reasoning to identify the causal effects of behavior sequences on model output and introduce a task that directly fits the ground-truth labels based on these effects, achieving the goal of explicit emphasis. Additionally, we develop a token-level weighting mechanism to adjust the emphasis strength for different item tokens, reflecting the diminishing influence of behavior sequences from earlier to later tokens during predicting an item. Extensive experiments on real-world datasets demonstrate that CFT effectively improves behavior sequence modeling. Our codes are available at https://github.com/itsmeyjt/CFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07476v3">VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 ZC, SL, HZ, YX, and XL contributed equally to this project. Code: https://github.com/DAMO-NLP-SG/VideoLLaMA2
    </div>
    <details class="paper-abstract">
      In this paper, we present the VideoLLaMA 2, a set of Video Large Language Models (Video-LLMs) designed to enhance spatial-temporal modeling and audio understanding in video and audio-oriented tasks. Building upon its predecessor, VideoLLaMA 2 incorporates a tailor-made Spatial-Temporal Convolution (STC) connector, which effectively captures the intricate spatial and temporal dynamics of video data. Additionally, we integrate an Audio Branch into the model through joint training, thereby enriching the multimodal understanding capabilities of the model by seamlessly incorporating audio cues. Comprehensive evaluations on multiple-choice video question answering (MC-VQA), open-ended video question answering (OE-VQA), and video captioning (VC) tasks demonstrate that VideoLLaMA 2 consistently achieves competitive results among open-source models and even gets close to some proprietary models on several benchmarks. Furthermore, VideoLLaMA 2 exhibits reasonable improvements in audio-only and audio-video question-answering (AQA & OE-AVQA) benchmarks over existing models. These advancements underline VideoLLaMA 2's superior performance in multimodal comprehension, setting a new standard for intelligent video analysis systems. All models are public to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19759v3">Balancing Cost and Effectiveness of Synthetic Data Generation Strategies for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 NeurIPS '24 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are applied to more use cases, creating high quality, task-specific datasets for fine-tuning becomes a bottleneck for model improvement. Using high quality human data has been the most common approach to unlock model performance, but is prohibitively expensive in many scenarios. Several alternative methods have also emerged, such as generating synthetic or hybrid data, but the effectiveness of these approaches remain unclear, especially in resource-constrained scenarios and tasks that are not easily verified. To investigate this, we group various synthetic data generation strategies into three representative categories -- Answer Augmentation, Question Rephrase and New Question -- and study the performance of student LLMs trained under various constraints, namely seed instruction set size and query budget. We demonstrate that these strategies are not equally effective across settings. Notably, the optimal data generation strategy depends strongly on the ratio between the available teacher query budget and the size of the seed instruction set. When this ratio is low, generating new answers to existing questions proves most effective, but as this ratio increases, generating new questions becomes optimal. Across all tasks, we find that choice of augmentation method and other design choices matter substantially more in low to mid data regimes than in high data regimes. We provide a practical framework for selecting the appropriate augmentation method across settings, taking into account additional factors such as the scalability of each method, the importance of verifying synthetic data, and the use of different LLMs for synthetic data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05025v1">LLMs as Research Tools: A Large Scale Survey of Researchers' Usage and Perceptions</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 30 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has led many researchers to consider their usage for scientific work. Some have found benefits using LLMs to augment or automate aspects of their research pipeline, while others have urged caution due to risks and ethical concerns. Yet little work has sought to quantify and characterize how researchers use LLMs and why. We present the first large-scale survey of 816 verified research article authors to understand how the research community leverages and perceives LLMs as research tools. We examine participants' self-reported LLM usage, finding that 81% of researchers have already incorporated LLMs into different aspects of their research workflow. We also find that traditionally disadvantaged groups in academia (non-White, junior, and non-native English speaking researchers) report higher LLM usage and perceived benefits, suggesting potential for improved research equity. However, women, non-binary, and senior researchers have greater ethical concerns, potentially hindering adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00843v1">The Graph's Apprentice: Teaching an LLM Low Level Knowledge for Circuit Quality Estimation</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      Logic synthesis is a crucial phase in the circuit design process, responsible for transforming hardware description language (HDL) designs into optimized netlists. However, traditional logic synthesis methods are computationally intensive, restricting their iterative use in refining chip designs. Recent advancements in large language models (LLMs), particularly those fine-tuned on programming languages, present a promising alternative. In this paper, we introduce VeriDistill, the first end-to-end machine learning model that directly processes raw Verilog code to predict circuit quality-of-result metrics. Our model employs a novel knowledge distillation method, transferring low-level circuit insights via graphs into the predictor based on LLM. Experiments show VeriDistill outperforms state-of-the-art baselines on large-scale Verilog datasets and demonstrates robust performance when evaluated on out-of-distribution datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22662v1">$\textbf{EMOS}$: $\textbf{E}$mbodiment-aware Heterogeneous $\textbf{M}$ulti-robot $\textbf{O}$perating $\textbf{S}$ystem with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-30
      | 💬 10 pages of main content, 3 pages of references, 5 pages of appendix, 7 figures in total
    </div>
    <details class="paper-abstract">
      Heterogeneous multi-robot systems (HMRS) have emerged as a powerful approach for tackling complex tasks that single robots cannot manage alone. Current large-language-model-based multi-agent systems (LLM-based MAS) have shown success in areas like software development and operating systems, but applying these systems to robot control presents unique challenges. In particular, the capabilities of each agent in a multi-robot system are inherently tied to the physical composition of the robots, rather than predefined roles. To address this issue, we introduce a novel multi-agent framework designed to enable effective collaboration among heterogeneous robots with varying embodiments and capabilities, along with a new benchmark named Habitat-MAS. One of our key designs is $\textit{Robot Resume}$: Instead of adopting human-designed role play, we propose a self-prompted approach, where agents comprehend robot URDF files and call robot kinematics tools to generate descriptions of their physics capabilities to guide their behavior in task planning and action execution. The Habitat-MAS benchmark is designed to assess how a multi-agent framework handles tasks that require embodiment-aware reasoning, which includes 1) manipulation, 2) perception, 3) navigation, and 4) comprehensive multi-floor object rearrangement. The experimental results indicate that the robot's resume and the hierarchical design of our multi-agent system are essential for the effective operation of the heterogeneous multi-robot system within this intricate problem context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22660v1">Linguistics Theory Meets LLM: Code-Switched Text Generation via Equivalence Constrained Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      Code-switching, the phenomenon of alternating between two or more languages in a single conversation, presents unique challenges for Natural Language Processing (NLP). Most existing research focuses on either syntactic constraints or neural generation, with few efforts to integrate linguistic theory with large language models (LLMs) for generating natural code-switched text. In this paper, we introduce EZSwitch, a novel framework that combines Equivalence Constraint Theory (ECT) with LLMs to produce linguistically valid and fluent code-switched text. We evaluate our method using both human judgments and automatic metrics, demonstrating a significant improvement in the quality of generated code-switching sentences compared to baseline LLMs. To address the lack of suitable evaluation metrics, we conduct a comprehensive correlation study of various automatic metrics against human scores, revealing that current metrics often fail to capture the nuanced fluency of code-switched text. Additionally, we create CSPref, a human preference dataset based on human ratings and analyze model performance across ``hard`` and ``easy`` examples. Our findings indicate that incorporating linguistic constraints into LLMs leads to more robust and human-aligned generation, paving the way for scalable code-switching text generation across diverse language pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15155v3">MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      Foundation models are becoming valuable tools in medicine. Yet despite their promise, the best way to leverage Large Language Models (LLMs) in complex medical tasks remains an open question. We introduce a novel multi-agent framework, named Medical Decision-making Agents (MDAgents) that helps address this gap by automatically assigning a collaboration structure to a team of LLMs. The assigned solo or group collaboration structure is tailored to the medical task at hand, emulating real-world medical decision-making processes adapted to tasks of varying complexities. We evaluate our framework and baseline methods using state-of-the-art LLMs across a suite of real-world medical knowledge and medical diagnosis benchmarks, including a comparison of LLMs' medical complexity classification against human physicians. MDAgents achieved the best performance in seven out of ten benchmarks on tasks requiring an understanding of medical knowledge and multi-modal reasoning, showing a significant improvement of up to 4.2% (p < 0.05) compared to previous methods' best performances. Ablation studies reveal that MDAgents effectively determines medical complexity to optimize for efficiency and accuracy across diverse medical tasks. Notably, the combination of moderator review and external medical knowledge in group collaboration resulted in an average accuracy improvement of 11.8%. Our code can be found at https://github.com/mitmedialab/MDAgents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13296v3">The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs: An Exhaustive Review of Technologies, Research, Best Practices, Applied Research Challenges and Opportunities</a></div>
    <div class="paper-meta">
      📅 2024-10-30
    </div>
    <details class="paper-abstract">
      This report examines the fine-tuning of Large Language Models (LLMs), integrating theoretical insights with practical applications. It outlines the historical evolution of LLMs from traditional Natural Language Processing (NLP) models to their pivotal role in AI. A comparison of fine-tuning methodologies, including supervised, unsupervised, and instruction-based approaches, highlights their applicability to different tasks. The report introduces a structured seven-stage pipeline for fine-tuning LLMs, spanning data preparation, model initialization, hyperparameter tuning, and model deployment. Emphasis is placed on managing imbalanced datasets and optimization techniques. Parameter-efficient methods like Low-Rank Adaptation (LoRA) and Half Fine-Tuning are explored for balancing computational efficiency with performance. Advanced techniques such as memory fine-tuning, Mixture of Experts (MoE), and Mixture of Agents (MoA) are discussed for leveraging specialized networks and multi-agent collaboration. The report also examines novel approaches like Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO), which align LLMs with human preferences, alongside pruning and routing optimizations to improve efficiency. Further sections cover validation frameworks, post-deployment monitoring, and inference optimization, with attention to deploying LLMs on distributed and cloud-based platforms. Emerging areas such as multimodal LLMs, fine-tuning for audio and speech, and challenges related to scalability, privacy, and accountability are also addressed. This report offers actionable insights for researchers and practitioners navigating LLM fine-tuning in an evolving landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17837v3">Enabling Generative Design Tools with LLM Agents for Mechanical Computation Devices: A Case Study</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 38 pages, 12 figures
    </div>
    <details class="paper-abstract">
      In the field of Human-Computer Interaction (HCI), interactive devices with embedded mechanical computation are gaining attention. The rise of these cutting-edge devices has created a need for specialized design tools that democratize the prototyping process. While current tools streamline prototyping through parametric design and simulation, they often come with a steep learning curve and may not fully support creative ideation. In this study, we use fluidic computation interfaces as a case study to explore how design tools for such devices can be augmented by Large Language Model agents (LLMs). Integrated with LLMs, the Generative Design Tool (GDT) better understands the capabilities and limitations of new technologies, proposes diverse and practical applications, and suggests designs that are technically and contextually appropriate. Additionally, it generates design parameters for visualizing results and producing fabrication-ready support files. This paper details the GDT's framework, implementation, and performance while addressing its potential and challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17503v3">Code Repair with LLMs gives an Exploration-Exploitation Tradeoff</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Iteratively improving and repairing source code with large language models (LLMs), known as refinement, has emerged as a popular way of generating programs that would be too complex to construct in one shot. Given a bank of test cases, together with a candidate program, an LLM can improve that program by being prompted with failed test cases. But it remains an open question how to best iteratively refine code, with prior work employing simple greedy or breadth-first strategies. We show here that refinement exposes an explore-exploit tradeoff: exploit by refining the program that passes the most test cases, or explore by refining a lesser considered program. We frame this as an arm-acquiring bandit problem, which we solve with Thompson Sampling. The resulting LLM-based program synthesis algorithm is broadly applicable: Across loop invariant synthesis, visual reasoning puzzles, and competition programming problems, we find that our new method can solve more problems using fewer language model calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22480v1">Scaling LLM Inference with Optimized Sample Compute Allocation</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Sampling is a basic operation in many inference-time algorithms of large language models (LLMs). To scale up inference efficiently with a limited compute, it is crucial to find an optimal allocation for sample compute budgets: Which sampling configurations (model, temperature, language, etc.) do we use? How many samples do we generate in each configuration? We formulate these choices as a learning problem and propose OSCA, an algorithm that Optimizes Sample Compute Allocation by finding an optimal mix of different inference configurations. Our experiments show that with our learned mixed allocation, we can achieve accuracy better than the best single configuration with 128x less compute on code generation and 25x less compute on 4 reasoning tasks. OSCA is also shown to be effective in agentic workflows beyond single-turn tasks, achieving a better accuracy on SWE-Bench with 3x less compute than the default configuration. Our code and generations are released at https://github.com/LeiLiLab/OSCA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01801v4">Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 ICLR 2024
    </div>
    <details class="paper-abstract">
      In this study, we introduce adaptive KV cache compression, a plug-and-play method that reduces the memory footprint of generative inference for Large Language Models (LLMs). Different from the conventional KV cache that retains key and value vectors for all context tokens, we conduct targeted profiling to discern the intrinsic structure of attention modules. Based on the recognized structure, we then construct the KV cache in an adaptive manner: evicting long-range contexts on attention heads emphasizing local contexts, discarding non-special tokens on attention heads centered on special tokens, and only employing the standard KV cache for attention heads that broadly attend to all tokens. Moreover, with the lightweight attention profiling used to guide the construction of the adaptive KV cache, FastGen can be deployed without resource-intensive fine-tuning or re-training. In our experiments across various asks, FastGen demonstrates substantial reduction on GPU memory consumption with negligible generation quality loss. We will release our code and the compatible CUDA kernel for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22318v1">Online Detecting LLM-Generated Texts via Sequential Hypothesis Testing by Betting</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Developing algorithms to differentiate between machine-generated texts and human-written texts has garnered substantial attention in recent years. Existing methods in this direction typically concern an offline setting where a dataset containing a mix of real and machine-generated texts is given upfront, and the task is to determine whether each sample in the dataset is from a large language model (LLM) or a human. However, in many practical scenarios, sources such as news websites, social media accounts, or on other forums publish content in a streaming fashion. Therefore, in this online scenario, how to quickly and accurately determine whether the source is an LLM with strong statistical guarantees is crucial for these media or platforms to function effectively and prevent the spread of misinformation and other potential misuse of LLMs. To tackle the problem of online detection, we develop an algorithm based on the techniques of sequential hypothesis testing by betting that not only builds upon and complements existing offline detection techniques but also enjoys statistical guarantees, which include a controlled false positive rate and the expected time to correctly identify a source as an LLM. Experiments were conducted to demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22304v1">Flow-DPO: Improving LLM Mathematical Reasoning through Online Multi-Agent Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 5 pages, 4 figures, 1 table
    </div>
    <details class="paper-abstract">
      Mathematical reasoning is a crucial capability for Large Language Models (LLMs), yet generating detailed and accurate reasoning traces remains a significant challenge. This paper introduces a novel approach to produce high-quality reasoning traces for LLM fine-tuning using online learning \textbf{Flows}. Our method employs an incremental output production Flow, where component LLMs collaboratively construct solutions through iterative communication. We train the Flow using online Direct Preference Optimization (DPO) learning with rollouts, generating DPO pairs for each training example and updating models in real-time. We directly compare the quality of reasoning traces generated by our method with those produced through direct model inference, demonstrating the effectiveness of our approach in improving LLM performance in mathematical reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22293v1">Fine-Tuning LLMs for Code Mutation: A New Era of Cyber Threats</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have significantly improved their capabilities in natural language processing and code synthesis, enabling more complex applications across different fields. This paper explores the application of LLMs in the context of code mutation, a process where the structure of program code is altered without changing its functionality. Traditionally, code mutation has been employed to increase software robustness in mission-critical applications. Additionally, mutation engines have been exploited by malware developers to evade the signature-based detection methods employed by malware detection systems. Existing code mutation engines, often used by such threat actors, typically result in only limited variations in the malware, which can still be identified through static code analysis. However, the agility demonstrated by an LLM-based code synthesizer could significantly change this threat landscape by allowing for more complex code mutations that are not easily detected using static analysis. One can increase variations of codes synthesized by a pre-trained LLM through fine-tuning and retraining. This process is what we refer to as code mutation training. In this paper, we propose a novel definition of code mutation training tailored for pre-trained LLM-based code synthesizers and demonstrate this training on a lightweight pre-trained model. Our approach involves restructuring (i.e., mutating) code at the subroutine level, which allows for more manageable mutations while maintaining the semantic integrity verified through unit testing. Our experimental results illustrate the effectiveness of our approach in improving code mutation capabilities of LLM-based program synthesizers in producing varied and functionally correct code solutions, showcasing their potential to transform the landscape of code mutation and the threats associated with it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19274v2">Ripple: Accelerating LLM Inference on Smartphones with Correlation-Aware Neuron Management</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across various domains, yet deploying them on mobile devices remains an arduous challenge due to their extensive computational and memory demands. While lightweight LLMs have been developed to fit mobile environments, they suffer from degraded model accuracy. In contrast, sparsity-based techniques minimize DRAM usage by selectively transferring only relevant neurons to DRAM while retaining the full model in external storage, such as flash. However, such approaches are critically limited by numerous I/O operations, particularly on smartphones with severe IOPS constraints. In this paper, we propose Ripple, a novel approach that accelerates LLM inference on smartphones by optimizing neuron placement in flash memory. Ripple leverages the concept of Neuron Co-Activation, where neurons frequently activated together are linked to facilitate continuous read access and optimize data transfer efficiency. Our approach incorporates a two-stage solution: an offline stage that reorganizes neuron placement based on co-activation patterns, and an online stage that employs tailored data access and caching strategies to align well with hardware characteristics. Evaluations conducted on a variety of smartphones and LLMs demonstrate that Ripple achieves up to 5.93x improvements in I/O latency compared to the state-of-the-art. As the first solution to optimize storage placement under sparsity, Ripple explores a new optimization space at the intersection of sparsity-driven algorithm and storage-level system co-design in LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01489v2">Agentless: Demystifying LLM-based Software Engineering Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have significantly advanced the automation of software development tasks, including code synthesis, program repair, and test generation. More recently, researchers and industry practitioners have developed various autonomous LLM agents to perform end-to-end software development tasks. These agents are equipped with the ability to use tools, run commands, observe feedback from the environment, and plan for future actions. However, the complexity of these agent-based approaches, together with the limited abilities of current LLMs, raises the following question: Do we really have to employ complex autonomous software agents? To attempt to answer this question, we build Agentless -- an agentless approach to automatically solve software development problems. Compared to the verbose and complex setup of agent-based approaches, Agentless employs a simplistic three-phase process of localization, repair, and patch validation, without letting the LLM decide future actions or operate with complex tools. Our results on the popular SWE-bench Lite benchmark show that surprisingly the simplistic Agentless is able to achieve both the highest performance (32.00%, 96 correct fixes) and low cost ($0.70) compared with all existing open-source software agents! Furthermore, we manually classified the problems in SWE-bench Lite and found problems with exact ground truth patch or insufficient/misleading issue descriptions. As such, we construct SWE-bench Lite-S by excluding such problematic issues to perform more rigorous evaluation and comparison. Our work highlights the current overlooked potential of a simple, interpretable technique in autonomous software development. We hope Agentless will help reset the baseline, starting point, and horizon for autonomous software agents, and inspire future work along this crucial direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08910v1">Automated Feedback in Math Education: A Comparative Analysis of LLMs for Open-Ended Responses</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 12 pages including references, 4 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The effectiveness of feedback in enhancing learning outcomes is well documented within Educational Data Mining (EDM). Various prior research has explored methodologies to enhance the effectiveness of feedback. Recent developments in Large Language Models (LLMs) have extended their utility in enhancing automated feedback systems. This study aims to explore the potential of LLMs in facilitating automated feedback in math education. We examine the effectiveness of LLMs in evaluating student responses by comparing 3 different models: Llama, SBERT-Canberra, and GPT4 model. The evaluation requires the model to provide both a quantitative score and qualitative feedback on the student's responses to open-ended math problems. We employ Mistral, a version of Llama catered to math, and fine-tune this model for evaluating student responses by leveraging a dataset of student responses and teacher-written feedback for middle-school math problems. A similar approach was taken for training the SBERT model as well, while the GPT4 model used a zero-shot learning approach. We evaluate the model's performance in scoring accuracy and the quality of feedback by utilizing judgments from 2 teachers. The teachers utilized a shared rubric in assessing the accuracy and relevance of the generated feedback. We conduct both quantitative and qualitative analyses of the model performance. By offering a detailed comparison of these methods, this study aims to further the ongoing development of automated feedback systems and outlines potential future directions for leveraging generative LLMs to create more personalized learning experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22225v1">CaStL: Constraints as Specifications through LLM Translation for Long-Horizon Task and Motion Planning</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable ability in long-horizon Task and Motion Planning (TAMP) by translating clear and straightforward natural language problems into formal specifications such as the Planning Domain Definition Language (PDDL). However, real-world problems are often ambiguous and involve many complex constraints. In this paper, we introduce Constraints as Specifications through LLMs (CaStL), a framework that identifies constraints such as goal conditions, action ordering, and action blocking from natural language in multiple stages. CaStL translates these constraints into PDDL and Python scripts, which are solved using an custom PDDL solver. Tested across three PDDL domains, CaStL significantly improves constraint handling and planning success rates from natural language specification in complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22177v1">Analyzing Multimodal Interaction Strategies for LLM-Assisted Manipulation of 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 under review
    </div>
    <details class="paper-abstract">
      As more applications of large language models (LLMs) for 3D content for immersive environments emerge, it is crucial to study user behaviour to identify interaction patterns and potential barriers to guide the future design of immersive content creation and editing systems which involve LLMs. In an empirical user study with 12 participants, we combine quantitative usage data with post-experience questionnaire feedback to reveal common interaction patterns and key barriers in LLM-assisted 3D scene editing systems. We identify opportunities for improving natural language interfaces in 3D design tools and propose design recommendations for future LLM-integrated 3D content creation systems. Through an empirical study, we demonstrate that LLM-assisted interactive systems can be used productively in immersive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22153v1">Benchmarking LLM Guardrails in Handling Multilingual Toxicity</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      With the ubiquity of Large Language Models (LLMs), guardrails have become crucial to detect and defend against toxic content. However, with the increasing pervasiveness of LLMs in multilingual scenarios, their effectiveness in handling multilingual toxic inputs remains unclear. In this work, we introduce a comprehensive multilingual test suite, spanning seven datasets and over ten languages, to benchmark the performance of state-of-the-art guardrails. We also investigates the resilience of guardrails against recent jailbreaking techniques, and assess the impact of in-context safety policies and language resource availability on guardrails' performance. Our findings show that existing guardrails are still ineffective at handling multilingual toxicity and lack robustness against jailbreaking prompts. This work aims to identify the limitations of guardrails and to build a more reliable and trustworthy LLMs in multilingual scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22143v1">AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) are typically aligned, they remain vulnerable to jailbreaking through either carefully crafted prompts in natural language or, interestingly, gibberish adversarial suffixes. However, gibberish tokens have received relatively less attention despite their success in attacking aligned LLMs. Recent work, AmpleGCG~\citep{liao2024amplegcg}, demonstrates that a generative model can quickly produce numerous customizable gibberish adversarial suffixes for any harmful query, exposing a range of alignment gaps in out-of-distribution (OOD) language spaces. To bring more attention to this area, we introduce AmpleGCG-Plus, an enhanced version that achieves better performance in fewer attempts. Through a series of exploratory experiments, we identify several training strategies to improve the learning of gibberish suffixes. Our results, verified under a strict evaluation setting, show that it outperforms AmpleGCG on both open-weight and closed-source models, achieving increases in attack success rate (ASR) of up to 17\% in the white-box setting against Llama-2-7B-chat, and more than tripling ASR in the black-box setting against GPT-4. Notably, AmpleGCG-Plus jailbreaks the newer GPT-4o series of models at similar rates to GPT-4, and, uncovers vulnerabilities against the recently proposed circuit breakers defense. We publicly release AmpleGCG-Plus along with our collected training datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11208v2">Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 Accepted at NeurIPS 2024, camera ready version. Code and data are available at https://github.com/lancopku/agent-backdoor-attacks
    </div>
    <details class="paper-abstract">
      Driven by the rapid development of Large Language Models (LLMs), LLM-based agents have been developed to handle various real-world applications, including finance, healthcare, and shopping, etc. It is crucial to ensure the reliability and security of LLM-based agents during applications. However, the safety issues of LLM-based agents are currently under-explored. In this work, we take the first step to investigate one of the typical safety threats, backdoor attack, to LLM-based agents. We first formulate a general framework of agent backdoor attacks, then we present a thorough analysis of different forms of agent backdoor attacks. Specifically, compared with traditional backdoor attacks on LLMs that are only able to manipulate the user inputs and model outputs, agent backdoor attacks exhibit more diverse and covert forms: (1) From the perspective of the final attacking outcomes, the agent backdoor attacker can not only choose to manipulate the final output distribution, but also introduce the malicious behavior in an intermediate reasoning step only, while keeping the final output correct. (2) Furthermore, the former category can be divided into two subcategories based on trigger locations, in which the backdoor trigger can either be hidden in the user query or appear in an intermediate observation returned by the external environment. We implement the above variations of agent backdoor attacks on two typical agent tasks including web shopping and tool utilization. Extensive experiments show that LLM-based agents suffer severely from backdoor attacks and such backdoor vulnerability cannot be easily mitigated by current textual backdoor defense algorithms. This indicates an urgent need for further research on the development of targeted defenses against backdoor attacks on LLM-based agents. Warning: This paper may contain biased content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22118v1">The Impact of Inference Acceleration Strategies on Bias of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Last few years have seen unprecedented advances in capabilities of Large Language Models (LLMs). These advancements promise to deeply benefit a vast array of application domains. However, due to their immense size, performing inference with LLMs is both costly and slow. Consequently, a plethora of recent work has proposed strategies to enhance inference efficiency, e.g., quantization, pruning, and caching. These acceleration strategies reduce the inference cost and latency, often by several factors, while maintaining much of the predictive performance measured via common benchmarks. In this work, we explore another critical aspect of LLM performance: demographic bias in model generations due to inference acceleration optimizations. Using a wide range of metrics, we probe bias in model outputs from a number of angles. Analysis of outputs before and after inference acceleration shows significant change in bias. Worryingly, these bias effects are complex and unpredictable. A combination of an acceleration strategy and bias type may show little bias change in one model but may lead to a large effect in another. Our results highlight a need for in-depth and case-by-case evaluation of model bias after it has been modified to accelerate inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22071v1">Distinguishing Ignorance from Error in LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are susceptible to hallucinations-outputs that are ungrounded, factually incorrect, or inconsistent with prior generations. We focus on close-book Question Answering (CBQA), where previous work has not fully addressed the distinction between two possible kinds of hallucinations, namely, whether the model (1) does not hold the correct answer in its parameters or (2) answers incorrectly despite having the required knowledge. We argue that distinguishing these cases is crucial for detecting and mitigating hallucinations. Specifically, case (2) may be mitigated by intervening in the model's internal computation, as the knowledge resides within the model's parameters. In contrast, in case (1) there is no parametric knowledge to leverage for mitigation, so it should be addressed by resorting to an external knowledge source or abstaining. To help distinguish between the two cases, we introduce Wrong Answer despite having Correct Knowledge (WACK), an approach for constructing model-specific datasets for the second hallucination type. Our probing experiments indicate that the two kinds of hallucinations are represented differently in the model's inner states. Next, we show that datasets constructed using WACK exhibit variations across models, demonstrating that even when models share knowledge of certain facts, they still vary in the specific examples that lead to hallucinations. Finally, we show that training a probe on our WACK datasets leads to better hallucination detection of case (2) hallucinations than using the common generic one-size-fits-all datasets. The code is available at https://github.com/technion-cs-nlp/hallucination-mitigation .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15938v3">RuleR: Improving LLM Controllability by Rule-based Data Recycling</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Despite the remarkable advancement of Large language models (LLMs), they still lack delicate controllability under sophisticated constraints, which is critical to enhancing their response quality and the user experience. While conditional supervised fine-tuning (SFT) can potentially improve LLM controllability, curating new SFT data to fulfill the constraints usually relies on human experts or proprietary LLMs, which is time-consuming and expensive. To bridge this gap, we propose Rule-based Data Recycling (RuleR), a human/LLM-free data augmentation method incorporating multiple constraints into the original SFT data. Instead of creating new responses from scratch, RuleR integrates linguistic or formatting rules into the original instructions and modifies the responses to fulfill the rule-defined constraints. Training on the "recycled" data consolidates LLMs capability to generate constrained outputs. Extensive experiments demonstrate RuleR's effectiveness in improving LLM controllability while maintaining general instruction-following performance. RuleR's code is released on https://github.com/tianyi-lab/RuleR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09066v3">CodeCloak: A Method for Evaluating and Mitigating Code Leakage by LLM Code Assistants</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      LLM-based code assistants are becoming increasingly popular among developers. These tools help developers improve their coding efficiency and reduce errors by providing real-time suggestions based on the developer's codebase. While beneficial, the use of these tools can inadvertently expose the developer's proprietary code to the code assistant service provider during the development process. In this work, we propose a method to mitigate the risk of code leakage when using LLM-based code assistants. CodeCloak is a novel deep reinforcement learning agent that manipulates the prompts before sending them to the code assistant service. CodeCloak aims to achieve the following two contradictory goals: (i) minimizing code leakage, while (ii) preserving relevant and useful suggestions for the developer. Our evaluation, employing StarCoder and Code Llama, LLM-based code assistants models, demonstrates CodeCloak's effectiveness on a diverse set of code repositories of varying sizes, as well as its transferability across different models. We also designed a method for reconstructing the developer's original codebase from code segments sent to the code assistant service (i.e., prompts) during the development process, to thoroughly analyze code leakage risks and evaluate the effectiveness of CodeCloak under practical development scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00034v1">Is Our Chatbot Telling Lies? Assessing Correctness of an LLM-based Dutch Support Chatbot</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 10 pages + 2 pages references, 4 figures
    </div>
    <details class="paper-abstract">
      Companies support their customers using live chats and chatbots to gain their loyalty. AFAS is a Dutch company aiming to leverage the opportunity large language models (LLMs) offer to answer customer queries with minimal to no input from its customer support team. Adding to its complexity, it is unclear what makes a response correct, and that too in Dutch. Further, with minimal data available for training, the challenge is to identify whether an answer generated by a large language model is correct and do it on the fly. This study is the first to define the correctness of a response based on how the support team at AFAS makes decisions. It leverages literature on natural language generation and automated answer grading systems to automate the decision-making of the customer support team. We investigated questions requiring a binary response (e.g., Would it be possible to adjust tax rates manually?) or instructions (e.g., How would I adjust tax rate manually?) to test how close our automated approach reaches support rating. Our approach can identify wrong messages in 55\% of the cases. This work shows the viability of automatically assessing when our chatbot tell lies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21965v1">SG-Bench: Evaluating LLM Safety Generalization Across Diverse Tasks and Prompt Types</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 Accepted by NeurIPS2024 (Dataset and Benchmark Track)
    </div>
    <details class="paper-abstract">
      Ensuring the safety of large language model (LLM) applications is essential for developing trustworthy artificial intelligence. Current LLM safety benchmarks have two limitations. First, they focus solely on either discriminative or generative evaluation paradigms while ignoring their interconnection. Second, they rely on standardized inputs, overlooking the effects of widespread prompting techniques, such as system prompts, few-shot demonstrations, and chain-of-thought prompting. To overcome these issues, we developed SG-Bench, a novel benchmark to assess the generalization of LLM safety across various tasks and prompt types. This benchmark integrates both generative and discriminative evaluation tasks and includes extended data to examine the impact of prompt engineering and jailbreak on LLM safety. Our assessment of 3 advanced proprietary LLMs and 10 open-source LLMs with the benchmark reveals that most LLMs perform worse on discriminative tasks than generative ones, and are highly susceptible to prompts, indicating poor generalization in safety alignment. We also explain these findings quantitatively and qualitatively to provide insights for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00456v2">QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 21 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We introduce QuaRot, a new Quantization scheme based on Rotations, which is able to quantize LLMs end-to-end, including all weights, activations, and KV cache in 4 bits. QuaRot rotates LLMs in a way that removes outliers from the hidden state without changing the output, making quantization easier. This computational invariance is applied to the hidden state (residual) of the LLM, as well as to the activations of the feed-forward components, aspects of the attention mechanism, and to the KV cache. The result is a quantized model where all matrix multiplications are performed in 4 bits, without any channels identified for retention in higher precision. Our 4-bit quantized LLaMa2-70B model has losses of at most 0.47 WikiText-2 perplexity and retains 99% of the zero-shot performance. We also show that QuaRot can provide lossless 6 and 8 bit LLaMa2 models without any calibration data using round-to-nearest quantization. Code is available at: https://github.com/spcl/QuaRot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20941v2">Instruction-Tuned LLMs Succeed in Document-Level MT Without Fine-Tuning -- But BLEU Turns a Blind Eye</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have excelled in various NLP tasks, including machine translation (MT), yet most studies focus on sentence-level translation. This work investigates the inherent capability of instruction-tuned LLMs for document-level translation (docMT). Unlike prior approaches that require specialized techniques, we evaluate LLMs by directly prompting them to translate entire documents in a single pass. Our results show that this method improves translation quality compared to translating sentences separately, even without document-level fine-tuning. However, this advantage is not reflected in BLEU scores, which often favor sentence-based translations. We propose using the LLM-as-a-judge paradigm for evaluation, where GPT-4 is used to assess document coherence, accuracy, and fluency in a more nuanced way than n-gram-based metrics. Overall, our work demonstrates that instruction-tuned LLMs can effectively leverage document context for translation. However, we caution against using BLEU scores for evaluating docMT, as they often provide misleading outcomes, failing to capture the quality of document-level translation. Code and data are available at https://github.com/EIT-NLP/BLEUless_DocMT
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21842v1">Diffusion as Reasoning: Enhancing Object Goal Navigation with LLM-Biased Diffusion Model</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      The Object Goal Navigation (ObjectNav) task requires the agent to navigate to a specified target in an unseen environment. Since the environment layout is unknown, the agent needs to perform semantic reasoning to infer the potential location of the target, based on its accumulated memory of the environment during the navigation process. Diffusion models have been shown to be able to learn the distribution relationships between features in RGB images, and thus generate new realistic images.In this work, we propose a new approach to solving the ObjectNav task, by training a diffusion model to learn the statistical distribution patterns of objects in semantic maps, and using the map of the explored regions during navigation as the condition to generate the map of the unknown regions, thereby realizing the semantic reasoning of the target object, i.e., diffusion as reasoning (DAR). Meanwhile, we propose the global target bias and local LLM bias methods, where the former can constrain the diffusion model to generate the target object more effectively, and the latter utilizes the common sense knowledge extracted from the LLM to improve the generalization of the reasoning process. Based on the generated map in the unknown region, the agent sets the predicted location of the target as the goal and moves towards it. Experiments on Gibson and MP3D show the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21819v1">Self-Preference Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Automated evaluation leveraging large language models (LLMs), commonly referred to as LLM evaluators or LLM-as-a-judge, has been widely used in measuring the performance of dialogue systems. However, the self-preference bias in LLMs has posed significant risks, including promoting specific styles or policies intrinsic to the LLMs. Despite the importance of this issue, there is a lack of established methods to measure the self-preference bias quantitatively, and its underlying causes are poorly understood. In this paper, we introduce a novel quantitative metric to measure the self-preference bias. Our experimental results demonstrate that GPT-4 exhibits a significant degree of self-preference bias. To explore the causes, we hypothesize that LLMs may favor outputs that are more familiar to them, as indicated by lower perplexity. We analyze the relationship between LLM evaluations and the perplexities of outputs. Our findings reveal that LLMs assign significantly higher evaluations to outputs with lower perplexity than human evaluators, regardless of whether the outputs were self-generated. This suggests that the essence of the bias lies in perplexity and that the self-preference bias exists because LLMs prefer texts more familiar to them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14753v2">Collaboratively adding new knowledge to an LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      We address the question of how to successively add new knowledge to an LLM whilst retaining previously-added knowledge. We consider two settings, semi-cooperative and fully-cooperative. Overall, LoRA performs better in most cases than full-fine tuning of all parameters when both new knowledge acquisition and retention of old, including recent, knowledge are taken into account. In the semi-cooperative setting, where datasets are not available after training, MOE mixing, model merging, and LoRA-based orthogonal subspace sequential learning, using a small weight on the orthogonality term, perform well. In the fully-cooperative setting where datasets remain available, joint training and sequential training with replay are both effective approaches with LoRA training generally preferable to full fine-tuning. The codes needed to reproduce the results are provided in an open source repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04411v2">Waterfall: Framework for Robust and Scalable Text Watermarking and Provenance for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 Accepted to EMNLP 2024 Main Conference
    </div>
    <details class="paper-abstract">
      Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation. In this paper, we propose Waterfall, the first training-free framework for robust and scalable text watermarking applicable across multiple text types (e.g., articles, code) and languages supportable by LLMs, for general text and LLM data provenance. Waterfall comprises several key innovations, such as being the first to use LLM as paraphrasers for watermarking along with a novel combination of techniques that are surprisingly effective in achieving robust verifiability and scalability. We empirically demonstrate that Waterfall achieves significantly better scalability, robust verifiability, and computational efficiency compared to SOTA article-text watermarking methods, and showed how it could be directly applied to the watermarking of code. We also demonstrated that Waterfall can be used for LLM data provenance, where the watermarks of LLM training data can be detected in LLM output, allowing for detection of unauthorized use of data for LLM training and potentially enabling model-centric watermarking of open-sourced LLMs which has been a limitation of existing LLM watermarking works. Our code is available at https://github.com/aoi3142/Waterfall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21779v1">Leveraging LLMs for Hypothetical Deduction in Logical Inference: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable potential across a wide array of reasoning tasks, including logical reasoning. Although massive efforts have been made to empower the logical reasoning ability of LLMs via external logical symbolic solvers, crucial challenges of the poor generalization ability to questions with different features and inevitable question information loss of symbolic solver-driven approaches remain unresolved. To mitigate these issues, we introduce LINA, a LLM-driven neuro-symbolic approach for faithful logical reasoning. By enabling an LLM to autonomously perform the transition from propositional logic extraction to sophisticated logical reasoning, LINA not only bolsters the resilience of the reasoning process but also eliminates the dependency on external solvers. Additionally, through its adoption of a hypothetical-deductive reasoning paradigm, LINA effectively circumvents the expansive search space challenge that plagues traditional forward reasoning methods. Empirical evaluations demonstrate that LINA substantially outperforms both established propositional logic frameworks and conventional prompting techniques across a spectrum of five logical reasoning tasks. Specifically, LINA achieves an improvement of 24.34% over LINC on the FOLIO dataset, while also surpassing prompting strategies like CoT and CoT-SC by up to 24.02%. Our code is available at https://github.com/wufeiwuwoshihua/nshy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15741v3">Ladder: A Model-Agnostic Framework Boosting LLM-based Machine Translation to the Next Level</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 EMNLP 2024 Main. Data and code are available at https://github.com/fzp0424/MT-Ladder
    </div>
    <details class="paper-abstract">
      General-purpose Large Language Models (LLMs) like GPT-4 have achieved remarkable advancements in machine translation (MT) by leveraging extensive web content. On the other hand, translation-specific LLMs are built by pre-training on domain-specific monolingual corpora and fine-tuning with human-annotated translation data. Despite the superior performance, these methods either demand an unprecedented scale of computing and data or substantial human editing and annotation efforts. In this paper, we develop MT-Ladder, a novel model-agnostic and cost-effective tool to refine the performance of general LLMs for MT. MT-Ladder is trained on pseudo-refinement triplets which can be easily obtained from existing LLMs without additional human cost. During training, we propose a hierarchical fine-tuning strategy with an easy-to-hard schema, improving MT-Ladder's refining performance progressively. The trained MT-Ladder can be seamlessly integrated with any general-purpose LLMs to boost their translation performance. By utilizing Gemma-2B/7B as the backbone, MT-Ladder-2B can elevate raw translations to the level of top-tier open-source models (e.g., refining BigTranslate-13B with +6.91 BLEU and +3.52 COMET for XX-En), and MT-Ladder-7B can further enhance model performance to be on par with the state-of-the-art GPT-4. Extensive ablation and analysis corroborate the effectiveness of MT-Ladder in diverse settings. Our code is available at https://github.com/fzp0424/MT-Ladder
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21716v1">A Bayesian Approach to Harnessing the Power of LLMs in Authorship Attribution</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Authorship attribution aims to identify the origin or author of a document. Traditional approaches have heavily relied on manual features and fail to capture long-range correlations, limiting their effectiveness. Recent advancements leverage text embeddings from pre-trained language models, which require significant fine-tuning on labeled data, posing challenges in data dependency and limited interpretability. Large Language Models (LLMs), with their deep reasoning capabilities and ability to maintain long-range textual associations, offer a promising alternative. This study explores the potential of pre-trained LLMs in one-shot authorship attribution, specifically utilizing Bayesian approaches and probability outputs of LLMs. Our methodology calculates the probability that a text entails previous writings of an author, reflecting a more nuanced understanding of authorship. By utilizing only pre-trained models such as Llama-3-70B, our results on the IMDb and blog datasets show an impressive 85\% accuracy in one-shot authorship classification across ten authors. Our findings set new baselines for one-shot authorship analysis using LLMs and expand the application scope of these models in forensic linguistics. This work also includes extensive ablation studies to validate our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21695v1">CFSafety: Comprehensive Fine-grained Safety Assessment for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) rapidly evolve, they bring significant conveniences to our work and daily lives, but also introduce considerable safety risks. These models can generate texts with social biases or unethical content, and under specific adversarial instructions, may even incite illegal activities. Therefore, rigorous safety assessments of LLMs are crucial. In this work, we introduce a safety assessment benchmark, CFSafety, which integrates 5 classic safety scenarios and 5 types of instruction attacks, totaling 10 categories of safety questions, to form a test set with 25k prompts. This test set was used to evaluate the natural language generation (NLG) capabilities of LLMs, employing a combination of simple moral judgment and a 1-5 safety rating scale for scoring. Using this benchmark, we tested eight popular LLMs, including the GPT series. The results indicate that while GPT-4 demonstrated superior safety performance, the safety effectiveness of LLMs, including this model, still requires improvement. The data and code associated with this study are available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19317v2">Jump Starting Bandits with LLM-Generated Prior Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      We present substantial evidence demonstrating the benefits of integrating Large Language Models (LLMs) with a Contextual Multi-Armed Bandit framework. Contextual bandits have been widely used in recommendation systems to generate personalized suggestions based on user-specific contexts. We show that LLMs, pre-trained on extensive corpora rich in human knowledge and preferences, can simulate human behaviours well enough to jump-start contextual multi-armed bandits to reduce online learning regret. We propose an initialization algorithm for contextual bandits by prompting LLMs to produce a pre-training dataset of approximate human preferences for the bandit. This significantly reduces online learning regret and data-gathering costs for training such models. Our approach is validated empirically through two sets of experiments with different bandit setups: one which utilizes LLMs to serve as an oracle and a real-world experiment utilizing data from a conjoint survey experiment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03131v3">AIME: AI System Optimization via Multiple LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2024-10-29
      | 💬 21 pages, 10 Figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Text-based AI system optimization typically involves a feedback loop scheme where a single LLM generates an evaluation in natural language of the current output to improve the next iteration's output. However, in this work, we empirically demonstrate that for a practical and complex task (code generation) with multiple criteria to evaluate, utilizing only one LLM evaluator tends to let errors in generated code go undetected, thus leading to incorrect evaluations and ultimately suboptimal test case performance. Motivated by this failure case, we assume there exists an optimal evaluation policy that samples an evaluation between response and ground truth. We then theoretically prove that a linear combination of multiple evaluators can approximate this optimal policy. From this insight, we propose AI system optimization via Multiple LLM Evaluators (AIME). AIME is an evaluation protocol that utilizes multiple LLMs that each independently generate an evaluation on separate criteria and then combine them via concatenation. We provide an extensive empirical study showing AIME outperforming baseline methods in code generation tasks, with up to $62\%$ higher error detection rate and up to $16\%$ higher success rate than a single LLM evaluation protocol on LeetCodeHard and HumanEval datasets. We also show that the selection of the number of evaluators and which criteria to utilize is non-trivial as it can impact pact success rate by up to $12\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02352v2">Pelican: Correcting Hallucination in Vision-LLMs via Claim Decomposition and Program of Thought Verification</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      Large Visual Language Models (LVLMs) struggle with hallucinations in visual instruction following task(s), limiting their trustworthiness and real-world applicability. We propose Pelican -- a novel framework designed to detect and mitigate hallucinations through claim verification. Pelican first decomposes the visual claim into a chain of sub-claims based on first-order predicates. These sub-claims consist of (predicate, question) pairs and can be conceptualized as nodes of a computational graph. We then use Program-of-Thought prompting to generate Python code for answering these questions through flexible composition of external tools. Pelican improves over prior work by introducing (1) intermediate variables for precise grounding of object instances, and (2) shared computation for answering the sub-question to enable adaptive corrections and inconsistency identification. We finally use reasoning abilities of LLMs to verify the correctness of the claim by considering the consistency and confidence of the (question, answer) pairs from each sub-claim. Our experiments reveal a drop in hallucination rate by ~ 8% - 32% across various baseline LVLMs and a 27% drop compared to approaches proposed for hallucination mitigation on MMHal-Bench. Results on two other benchmarks further corroborate our results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15518v2">Eagle: Efficient Training-Free Router for Multi-LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) with varying capabilities and costs has created a need for efficient model selection in AI systems. LLM routers address this need by dynamically choosing the most suitable model for a given query based on task requirements and budget constraints. However, existing routers face challenges in scalability and real-time adaptation, particularly in high-volume online environments. We present Eagle, a novel LLM routing approach that combines global and local ELO ranking modules to overcome these limitations. By evaluating both general and specialized LLM abilities, Eagle provides a scalable, training-free solution that enhances model selection quality while reducing computational overhead. Our experiments across multiple datasets show Eagle consistently outperforms baseline methods, with improvements of up to 23.52 percent in Area Under Curve (AUC) scores. Moreover, Eagle demonstrates remarkable efficiency, requiring only 1/20 of baseline methods' time for initialization and 100 to 200 times faster incremental updates in online scenarios, making it well-suited for dynamic, high-volume online serving environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.20266v1">Limitations of the LLM-as-a-Judge Approach for Evaluating LLM Outputs in Expert Knowledge Tasks</a></div>
    <div class="paper-meta">
      📅 2024-10-29
    </div>
    <details class="paper-abstract">
      The potential of using Large Language Models (LLMs) themselves to evaluate LLM outputs offers a promising method for assessing model performance across various contexts. Previous research indicates that LLM-as-a-judge exhibits a strong correlation with human judges in the context of general instruction following. However, for instructions that require specialized knowledge, the validity of using LLMs as judges remains uncertain. In our study, we applied a mixed-methods approach, conducting pairwise comparisons in which both subject matter experts (SMEs) and LLMs evaluated outputs from domain-specific tasks. We focused on two distinct fields: dietetics, with registered dietitian experts, and mental health, with clinical psychologist experts. Our results showed that SMEs agreed with LLM judges 68% of the time in the dietetics domain and 64% in mental health when evaluating overall preference. Additionally, the results indicated variations in SME-LLM agreement across domain-specific aspect questions. Our findings emphasize the importance of keeping human experts in the evaluation process, as LLMs alone may not provide the depth of understanding required for complex, knowledge specific tasks. We also explore the implications of LLM evaluations across different domains and discuss how these insights can inform the design of evaluation workflows that ensure better alignment between human experts and LLMs in interactive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21479v1">TransformLLM: Adapting Large Language Models via LLM-Transformed Reading Comprehension Text</a></div>
    <div class="paper-meta">
      📅 2024-10-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in highly-specialized domains, however challenges are still present in aspects of accuracy and costs. These limitations restrict the usage of existing models in domain-specific tasks. While fine-tuning pre-trained models have shown promising results, this process can be computationally expensive and require massive datasets of the specialized application in hand. In this work, we bridge that gap. We have developed Phi-2-Legal and Mistral-Legal-7B, which are language models specifically designed for legal applications. These models are based on Phi-2 and Mistral-7B-v0.1, and have gone through continued pre-training with over 500 million tokens of legal texts. Our innovative approach significantly improves capabilities in legal tasks by using Large Language Models (LLMs) to convert raw training data into reading comprehension text. Our legal LLMs have demonstrated superior performance in legal benchmarks, even outperforming models trained on much larger datasets with more resources. This work emphasizes the effectiveness of continued pre-training on domain-specific texts, while using affordable LLMs for data conversion, which gives these models domain expertise while retaining general language understanding capabilities. While this work uses the legal domain as a test case, our method can be scaled and applied to any pre-training dataset, resulting in significant improvements across different tasks. These findings underscore the potential of domain-adaptive pre-training and reading comprehension for the development of highly effective domain-specific language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21474v1">Estimating Causal Effects of Text Interventions Leveraging LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-28
    </div>
    <details class="paper-abstract">
      Quantifying the effect of textual interventions in social systems, such as reducing anger in social media posts to see its impact on engagement, poses significant challenges. Direct interventions on real-world systems are often infeasible, necessitating reliance on observational data. Traditional causal inference methods, typically designed for binary or discrete treatments, are inadequate for handling the complex, high-dimensional nature of textual data. This paper addresses these challenges by proposing a novel approach, CausalDANN, to estimate causal effects using text transformations facilitated by large language models (LLMs). Unlike existing methods, our approach accommodates arbitrary textual interventions and leverages text-level classifiers with domain adaptation ability to produce robust effect estimates against domain shifts, even when only the control group is observed. This flexibility in handling various text interventions is a key advancement in causal estimation for textual data, offering opportunities to better understand human behaviors and develop effective policies within social systems.
    </details>
</div>
