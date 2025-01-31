# llm - 2024_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19715v1">Assessing LLMs in Malicious Code Deobfuscation of Real-world Malware Campaigns</a></div>
    <div class="paper-meta">
      📅 2024-04-30
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into various pipelines is increasingly widespread, effectively automating many manual tasks and often surpassing human capabilities. Cybersecurity researchers and practitioners have recognised this potential. Thus, they are actively exploring its applications, given the vast volume of heterogeneous data that requires processing to identify anomalies, potential bypasses, attacks, and fraudulent incidents. On top of this, LLMs' advanced capabilities in generating functional code, comprehending code context, and summarising its operations can also be leveraged for reverse engineering and malware deobfuscation. To this end, we delve into the deobfuscation capabilities of state-of-the-art LLMs. Beyond merely discussing a hypothetical scenario, we evaluate four LLMs with real-world malicious scripts used in the notorious Emotet malware campaign. Our results indicate that while not absolutely accurate yet, some LLMs can efficiently deobfuscate such payloads. Thus, fine-tuning LLMs for this task can be a viable potential for future AI-powered threat intelligence pipelines in the fight against obfuscated malware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18081v2">ComposerX: Multi-Agent Symbolic Music Composition with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-30
    </div>
    <details class="paper-abstract">
      Music composition represents the creative side of humanity, and itself is a complex task that requires abilities to understand and generate information with long dependency and harmony constraints. While demonstrating impressive capabilities in STEM subjects, current LLMs easily fail in this task, generating ill-written music even when equipped with modern techniques like In-Context-Learning and Chain-of-Thoughts. To further explore and enhance LLMs' potential in music composition by leveraging their reasoning ability and the large knowledge base in music history and theory, we propose ComposerX, an agent-based symbolic music generation framework. We find that applying a multi-agent approach significantly improves the music composition quality of GPT-4. The results demonstrate that ComposerX is capable of producing coherent polyphonic music compositions with captivating melodies, while adhering to user instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18286v2">Comparing LLM prompting with Cross-lingual transfer performance on Indigenous and Low-resource Brazilian Languages</a></div>
    <div class="paper-meta">
      📅 2024-04-30
      | 💬 Accepted to the Americas NLP Workshop at NAACL 2024 (https://turing.iimas.unam.mx/americasnlp/2024_workshop.html)
    </div>
    <details class="paper-abstract">
      Large Language Models are transforming NLP for a variety of tasks. However, how LLMs perform NLP tasks for low-resource languages (LRLs) is less explored. In line with the goals of the AmericasNLP workshop, we focus on 12 LRLs from Brazil, 2 LRLs from Africa and 2 high-resource languages (HRLs) (e.g., English and Brazilian Portuguese). Our results indicate that the LLMs perform worse for the part of speech (POS) labeling of LRLs in comparison to HRLs. We explain the reasons behind this failure and provide an error analysis through examples observed in our data set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19409v1">Countering Reward Over-optimization in LLM with Demonstration-Guided Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-04-30
    </div>
    <details class="paper-abstract">
      While Reinforcement Learning (RL) has been proven essential for tuning large language models (LLMs), it can lead to reward over-optimization (ROO). Existing approaches address ROO by adding KL regularization, requiring computationally expensive hyperparameter tuning. Additionally, KL regularization focuses solely on regularizing the language policy, neglecting a potential source of regularization: the reward function itself. Inspired by demonstration-guided RL, we here introduce the Reward Calibration from Demonstration (RCfD), which leverages human demonstrations and a reward model to recalibrate the reward objective. Formally, given a prompt, the RCfD objective minimizes the distance between the demonstrations' and LLM's rewards rather than directly maximizing the reward function. This objective shift avoids incentivizing the LLM to exploit the reward model and promotes more natural and diverse language generation. We show the effectiveness of RCfD on three language tasks, which achieves comparable performance to carefully tuned baselines while mitigating ROO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12994v2">Rethinking the Evaluation of Dialogue Systems: Effects of User Feedback on Crowdworkers and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-29
      | 💬 Accepted at SIGIR 2024 long paper track
    </div>
    <details class="paper-abstract">
      In ad-hoc retrieval, evaluation relies heavily on user actions, including implicit feedback. In a conversational setting such signals are usually unavailable due to the nature of the interactions, and, instead, the evaluation often relies on crowdsourced evaluation labels. The role of user feedback in annotators' assessment of turns in a conversational perception has been little studied. We focus on how the evaluation of task-oriented dialogue systems (TDSs), is affected by considering user feedback, explicit or implicit, as provided through the follow-up utterance of a turn being evaluated. We explore and compare two methodologies for assessing TDSs: one includes the user's follow-up utterance and one without. We use both crowdworkers and large language models (LLMs) as annotators to assess system responses across four aspects: relevance, usefulness, interestingness, and explanation quality. Our findings indicate that there is a distinct difference in ratings assigned by both annotator groups in the two setups, indicating user feedback does influence system evaluation. Workers are more susceptible to user feedback on usefulness and interestingness compared to LLMs on interestingness and relevance. User feedback leads to a more personalized assessment of usefulness by workers, aligning closely with the user's explicit feedback. Additionally, in cases of ambiguous or complex user requests, user feedback improves agreement among crowdworkers. These findings emphasize the significance of user feedback in refining system evaluations and suggest the potential for automated feedback integration in future research. We publicly release the annotated data to foster research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18919v1">TheaterGen: Character Management with LLM for Consistent Multi-turn Image Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-29
    </div>
    <details class="paper-abstract">
      Recent advances in diffusion models can generate high-quality and stunning images from text. However, multi-turn image generation, which is of high demand in real-world scenarios, still faces challenges in maintaining semantic consistency between images and texts, as well as contextual consistency of the same subject across multiple interactive turns. To address this issue, we introduce TheaterGen, a training-free framework that integrates large language models (LLMs) and text-to-image (T2I) models to provide the capability of multi-turn image generation. Within this framework, LLMs, acting as a "Screenwriter", engage in multi-turn interaction, generating and managing a standardized prompt book that encompasses prompts and layout designs for each character in the target image. Based on these, Theatergen generate a list of character images and extract guidance information, akin to the "Rehearsal". Subsequently, through incorporating the prompt book and guidance information into the reverse denoising process of T2I diffusion models, Theatergen generate the final image, as conducting the "Final Performance". With the effective management of prompt books and character images, TheaterGen significantly improves semantic and contextual consistency in synthesized images. Furthermore, we introduce a dedicated benchmark, CMIGBench (Consistent Multi-turn Image Generation Benchmark) with 8000 multi-turn instructions. Different from previous multi-turn benchmarks, CMIGBench does not define characters in advance. Both the tasks of story generation and multi-turn editing are included on CMIGBench for comprehensive evaluation. Extensive experimental results show that TheaterGen outperforms state-of-the-art methods significantly. It raises the performance bar of the cutting-edge Mini DALLE 3 model by 21% in average character-character similarity and 19% in average text-image similarity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18864v1">Performance-Aligned LLMs for Generating Fast Code</a></div>
    <div class="paper-meta">
      📅 2024-04-29
    </div>
    <details class="paper-abstract">
      Optimizing scientific software is a difficult task because codebases are often large and complex, and performance can depend upon several factors including the algorithm, its implementation, and hardware among others. Causes of poor performance can originate from disparate sources and be difficult to diagnose. Recent years have seen a multitude of work that use large language models (LLMs) to assist in software development tasks. However, these tools are trained to model the distribution of code as text, and are not specifically designed to understand performance aspects of code. In this work, we introduce a reinforcement learning based methodology to align the outputs of code LLMs with performance. This allows us to build upon the current code modeling capabilities of LLMs and extend them to generate better performing code. We demonstrate that our fine-tuned model improves the expected speedup of generated code over base models for a set of benchmark tasks from 0.9 to 1.6 for serial code and 1.9 to 4.5 for OpenMP code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18832v1">It's Difficult to be Neutral -- Human and LLM-based Sentiment Annotation of Patient Comments</a></div>
    <div class="paper-meta">
      📅 2024-04-29
    </div>
    <details class="paper-abstract">
      Sentiment analysis is an important tool for aggregating patient voices, in order to provide targeted improvements in healthcare services. A prerequisite for this is the availability of in-domain data annotated for sentiment. This article documents an effort to add sentiment annotations to free-text comments in patient surveys collected by the Norwegian Institute of Public Health (NIPH). However, annotation can be a time-consuming and resource-intensive process, particularly when it requires domain expertise. We therefore also evaluate a possible alternative to human annotation, using large language models (LLMs) as annotators. We perform an extensive evaluation of the approach for two openly available pretrained LLMs for Norwegian, experimenting with different configurations of prompts and in-context learning, comparing their performance to human annotators. We find that even for zero-shot runs, models perform well above the baseline for binary sentiment, but still cannot compete with human annotators on the full dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18978v1">Towards Generalizable Agents in Text-Based Educational Environments: A Study of Integrating RL with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-29
      | 💬 Accepted as a full paper at EDM 2024: The 17th International Conference on Educational Data Mining, 14-17 of July 2024, Atlanta
    </div>
    <details class="paper-abstract">
      There has been a growing interest in developing learner models to enhance learning and teaching experiences in educational environments. However, existing works have primarily focused on structured environments relying on meticulously crafted representations of tasks, thereby limiting the agent's ability to generalize skills across tasks. In this paper, we aim to enhance the generalization capabilities of agents in open-ended text-based learning environments by integrating Reinforcement Learning (RL) with Large Language Models (LLMs). We investigate three types of agents: (i) RL-based agents that utilize natural language for state and action representations to find the best interaction strategy, (ii) LLM-based agents that leverage the model's general knowledge and reasoning through prompting, and (iii) hybrid LLM-assisted RL agents that combine these two strategies to improve agents' performance and generalization. To support the development and evaluation of these agents, we introduce PharmaSimText, a novel benchmark derived from the PharmaSim virtual pharmacy environment designed for practicing diagnostic conversations. Our results show that RL-based agents excel in task completion but lack in asking quality diagnostic questions. In contrast, LLM-based agents perform better in asking diagnostic questions but fall short of completing the task. Finally, hybrid LLM-assisted RL agents enable us to overcome these limitations, highlighting the potential of combining RL and LLMs to develop high-performing agents for open-ended learning environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.02046v6">Recommender Systems in the Era of Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-04-29
      | 💬 Accepted by IEEE TKDE
    </div>
    <details class="paper-abstract">
      With the prosperity of e-commerce and web applications, Recommender Systems (RecSys) have become an important component of our daily life, providing personalized suggestions that cater to user preferences. While Deep Neural Networks (DNNs) have made significant advancements in enhancing recommender systems by modeling user-item interactions and incorporating textual side information, DNN-based methods still face limitations, such as difficulties in understanding users' interests and capturing textual side information, inabilities in generalizing to various recommendation scenarios and reasoning on their predictions, etc. Meanwhile, the emergence of Large Language Models (LLMs), such as ChatGPT and GPT4, has revolutionized the fields of Natural Language Processing (NLP) and Artificial Intelligence (AI), due to their remarkable abilities in fundamental responsibilities of language understanding and generation, as well as impressive generalization and reasoning capabilities. As a result, recent studies have attempted to harness the power of LLMs to enhance recommender systems. Given the rapid evolution of this research direction in recommender systems, there is a pressing need for a systematic overview that summarizes existing LLM-empowered recommender systems, to provide researchers in relevant fields with an in-depth understanding. Therefore, in this paper, we conduct a comprehensive review of LLM-empowered recommender systems from various aspects including Pre-training, Fine-tuning, and Prompting. More specifically, we first introduce representative methods to harness the power of LLMs (as a feature encoder) for learning representations of users and items. Then, we review recent techniques of LLMs for enhancing recommender systems from three paradigms, namely pre-training, fine-tuning, and prompting. Finally, we comprehensively discuss future directions in this emerging field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18496v1">AI-powered Code Review with LLMs: Early Results</a></div>
    <div class="paper-meta">
      📅 2024-04-29
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel approach to improving software quality and efficiency through a Large Language Model (LLM)-based model designed to review code and identify potential issues. Our proposed LLM-based AI agent model is trained on large code repositories. This training includes code reviews, bug reports, and documentation of best practices. It aims to detect code smells, identify potential bugs, provide suggestions for improvement, and optimize the code. Unlike traditional static code analysis tools, our LLM-based AI agent has the ability to predict future potential risks in the code. This supports a dual goal of improving code quality and enhancing developer education by encouraging a deeper understanding of best practices and efficient coding techniques. Furthermore, we explore the model's effectiveness in suggesting improvements that significantly reduce post-release bugs and enhance code review processes, as evidenced by an analysis of developer sentiment toward LLM feedback. For future work, we aim to assess the accuracy and efficiency of LLM-generated documentation updates in comparison to manual methods. This will involve an empirical study focusing on manually conducted code reviews to identify code smells and bugs, alongside an evaluation of best practice documentation, augmented by insights from developer discussions and code reviews. Our goal is to not only refine the accuracy of our LLM-based tool but also to underscore its potential in streamlining the software development lifecycle through proactive code improvement and education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18460v1">Ethical Reasoning and Moral Value Alignment of LLMs Depend on the Language we Prompt them in</a></div>
    <div class="paper-meta">
      📅 2024-04-29
    </div>
    <details class="paper-abstract">
      Ethical reasoning is a crucial skill for Large Language Models (LLMs). However, moral values are not universal, but rather influenced by language and culture. This paper explores how three prominent LLMs -- GPT-4, ChatGPT, and Llama2-70B-Chat -- perform ethical reasoning in different languages and if their moral judgement depend on the language in which they are prompted. We extend the study of ethical reasoning of LLMs by Rao et al. (2023) to a multilingual setup following their framework of probing LLMs with ethical dilemmas and policies from three branches of normative ethics: deontology, virtue, and consequentialism. We experiment with six languages: English, Spanish, Russian, Chinese, Hindi, and Swahili. We find that GPT-4 is the most consistent and unbiased ethical reasoner across languages, while ChatGPT and Llama2-70B-Chat show significant moral value bias when we move to languages other than English. Interestingly, the nature of this bias significantly vary across languages for all LLMs, including GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00732v1">LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report</a></div>
    <div class="paper-meta">
      📅 2024-04-29
    </div>
    <details class="paper-abstract">
      Low Rank Adaptation (LoRA) has emerged as one of the most widely adopted methods for Parameter Efficient Fine-Tuning (PEFT) of Large Language Models (LLMs). LoRA reduces the number of trainable parameters and memory usage while achieving comparable performance to full fine-tuning. We aim to assess the viability of training and serving LLMs fine-tuned with LoRA in real-world applications. First, we measure the quality of LLMs fine-tuned with quantized low rank adapters across 10 base models and 31 tasks for a total of 310 models. We find that 4-bit LoRA fine-tuned models outperform base models by 34 points and GPT-4 by 10 points on average. Second, we investigate the most effective base models for fine-tuning and assess the correlative and predictive capacities of task complexity heuristics in forecasting the outcomes of fine-tuning. Finally, we evaluate the latency and concurrency capabilities of LoRAX, an open-source Multi-LoRA inference server that facilitates the deployment of multiple LoRA fine-tuned models on a single GPU using shared base model weights and dynamic adapter loading. LoRAX powers LoRA Land, a web application that hosts 25 LoRA fine-tuned Mistral-7B LLMs on a single NVIDIA A100 GPU with 80GB memory. LoRA Land highlights the quality and cost-effectiveness of employing multiple specialized LLMs over a single, general-purpose LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18371v1">QANA: LLM-based Question Generation and Network Analysis for Zero-shot Key Point Analysis and Beyond</a></div>
    <div class="paper-meta">
      📅 2024-04-29
      | 💬 Under review as a conference paper at COLM 2024
    </div>
    <details class="paper-abstract">
      The proliferation of social media has led to information overload and increased interest in opinion mining. We propose "Question-Answering Network Analysis" (QANA), a novel opinion mining framework that utilizes Large Language Models (LLMs) to generate questions from users' comments, constructs a bipartite graph based on the comments' answerability to the questions, and applies centrality measures to examine the importance of opinions. We investigate the impact of question generation styles, LLM selections, and the choice of embedding model on the quality of the constructed QA networks by comparing them with annotated Key Point Analysis datasets. QANA achieves comparable performance to previous state-of-the-art supervised models in a zero-shot manner for Key Point Matching task, also reducing the computational cost from quadratic to linear. For Key Point Generation, questions with high PageRank or degree centrality align well with manually annotated key points. Notably, QANA enables analysts to assess the importance of key points from various aspects according to their selection of centrality measure. QANA's primary contribution lies in its flexibility to extract key points from a wide range of perspectives, which enhances the quality and impartiality of opinion mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18057v1">Efficient LLM Inference with Kcache</a></div>
    <div class="paper-meta">
      📅 2024-04-28
      | 💬 Technical Report, 8 pages
    </div>
    <details class="paper-abstract">
      Large Language Models(LLMs) have had a profound impact on AI applications, particularly in the domains of long-text comprehension and generation. KV Cache technology is one of the most widely used techniques in the industry. It ensures efficient sequence generation by caching previously computed KV states. However, it also introduces significant memory overhead. We discovered that KV Cache is not necessary and proposed a novel KCache technique to alleviate the memory bottleneck issue during the LLMs inference process. KCache can be used directly for inference without any training process, Our evaluations show that KCache improves the throughput of popular LLMs by 40% with the baseline, while keeping accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01551v1">Leveraging Prompts in LLMs to Overcome Imbalances in Complex Educational Text Data</a></div>
    <div class="paper-meta">
      📅 2024-04-28
      | 💬 17 pages, 5 figures, 3 tables, 2 appendices
    </div>
    <details class="paper-abstract">
      In this paper, we explore the potential of Large Language Models (LLMs) with assertions to mitigate imbalances in educational datasets. Traditional models often fall short in such contexts, particularly due to the complexity and nuanced nature of the data. This issue is especially prominent in the education sector, where cognitive engagement levels among students show significant variation in their open responses. To test our hypothesis, we utilized an existing technology for assertion-based prompt engineering through an 'Iterative - ICL PE Design Process' comparing traditional Machine Learning (ML) models against LLMs augmented with assertions (N=135). Further, we conduct a sensitivity analysis on a subset (n=27), examining the variance in model performance concerning classification metrics and cognitive engagement levels in each iteration. Our findings reveal that LLMs with assertions significantly outperform traditional ML models, particularly in cognitive engagement levels with minority representation, registering up to a 32% increase in F1-score. Additionally, our sensitivity study indicates that incorporating targeted assertions into the LLM tested on the subset enhances its performance by 11.94%. This improvement primarily addresses errors stemming from the model's limitations in understanding context and resolving lexical ambiguities in student responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18021v1">CRISPR-GPT: An LLM Agent for Automated Design of Gene-Editing Experiments</a></div>
    <div class="paper-meta">
      📅 2024-04-27
    </div>
    <details class="paper-abstract">
      The introduction of genome engineering technology has transformed biomedical research, making it possible to make precise changes to genetic information. However, creating an efficient gene-editing system requires a deep understanding of CRISPR technology, and the complex experimental systems under investigation. While Large Language Models (LLMs) have shown promise in various tasks, they often lack specific knowledge and struggle to accurately solve biological design problems. In this work, we introduce CRISPR-GPT, an LLM agent augmented with domain knowledge and external tools to automate and enhance the design process of CRISPR-based gene-editing experiments. CRISPR-GPT leverages the reasoning ability of LLMs to facilitate the process of selecting CRISPR systems, designing guide RNAs, recommending cellular delivery methods, drafting protocols, and designing validation experiments to confirm editing outcomes. We showcase the potential of CRISPR-GPT for assisting non-expert researchers with gene-editing experiments from scratch and validate the agent's effectiveness in a real-world use case. Furthermore, we explore the ethical and regulatory considerations associated with automated gene-editing design, highlighting the need for responsible and transparent use of these tools. Our work aims to bridge the gap between beginner biological researchers and CRISPR genome engineering techniques, and demonstrate the potential of LLM agents in facilitating complex biological discovery tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08058v2">CHAI: Clustered Head Attention for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-04-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with hundreds of billions of parameters have transformed the field of machine learning. However, serving these models at inference time is both compute and memory intensive, where a single request can require multiple GPUs and tens of Gigabytes of memory. Multi-Head Attention is one of the key components of LLMs, which can account for over 50% of LLMs memory and compute requirement. We observe that there is a high amount of redundancy across heads on which tokens they pay attention to. Based on this insight, we propose Clustered Head Attention (CHAI). CHAI combines heads with a high amount of correlation for self-attention at runtime, thus reducing both memory and compute. In our experiments, we show that CHAI is able to reduce the memory requirements for storing K,V cache by up to 21.4% and inference time latency by up to 1.73x without any fine-tuning required. CHAI achieves this with a maximum 3.2% deviation in accuracy across 3 different models (i.e. OPT-66B, LLAMA-7B, LLAMA-33B) and 5 different evaluation datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17842v1">Using LLMs in Software Requirements Specifications: An Empirical Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-04-27
      | 💬 Accepted to RE@Next! at the IEEE International Requirements Engineering Conference 2024 at Reykjavik, Iceland
    </div>
    <details class="paper-abstract">
      The creation of a Software Requirements Specification (SRS) document is important for any software development project. Given the recent prowess of Large Language Models (LLMs) in answering natural language queries and generating sophisticated textual outputs, our study explores their capability to produce accurate, coherent, and structured drafts of these documents to accelerate the software development lifecycle. We assess the performance of GPT-4 and CodeLlama in drafting an SRS for a university club management system and compare it against human benchmarks using eight distinct criteria. Our results suggest that LLMs can match the output quality of an entry-level software engineer to generate an SRS, delivering complete and consistent drafts. We also evaluate the capabilities of LLMs to identify and rectify problems in a given requirements document. Our experiments indicate that GPT-4 is capable of identifying issues and giving constructive feedback for rectifying them, while CodeLlama's results for validation were not as encouraging. We repeated the generation exercise for four distinct use cases to study the time saved by employing LLMs for SRS generation. The experiment demonstrates that LLMs may facilitate a significant reduction in development time for entry-level software engineers. Hence, we conclude that the LLMs can be gainfully used by software engineers to increase productivity by saving time and effort in generating, validating and rectifying software requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17833v1">Testing and Understanding Erroneous Planning in LLM Agents through Synthesized User Inputs</a></div>
    <div class="paper-meta">
      📅 2024-04-27
    </div>
    <details class="paper-abstract">
      Agents based on large language models (LLMs) have demonstrated effectiveness in solving a wide range of tasks by integrating LLMs with key modules such as planning, memory, and tool usage. Increasingly, customers are adopting LLM agents across a variety of commercial applications critical to reliability, including support for mental well-being, chemical synthesis, and software development. Nevertheless, our observations and daily use of LLM agents indicate that they are prone to making erroneous plans, especially when the tasks are complex and require long-term planning. In this paper, we propose PDoctor, a novel and automated approach to testing LLM agents and understanding their erroneous planning. As the first work in this direction, we formulate the detection of erroneous planning as a constraint satisfiability problem: an LLM agent's plan is considered erroneous if its execution violates the constraints derived from the user inputs. To this end, PDoctor first defines a domain-specific language (DSL) for user queries and synthesizes varying inputs with the assistance of the Z3 constraint solver. These synthesized inputs are natural language paragraphs that specify the requirements for completing a series of tasks. Then, PDoctor derives constraints from these requirements to form a testing oracle. We evaluate PDoctor with three mainstream agent frameworks and two powerful LLMs (GPT-3.5 and GPT-4). The results show that PDoctor can effectively detect diverse errors in agent planning and provide insights and error characteristics that are valuable to both agent developers and users. We conclude by discussing potential alternative designs and directions to extend PDoctor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17790v1">Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-04-27
    </div>
    <details class="paper-abstract">
      Cross-lingual continual pre-training of large language models (LLMs) initially trained on English corpus allows us to leverage the vast amount of English language resources and reduce the pre-training cost. In this study, we constructed Swallow, an LLM with enhanced Japanese capability, by extending the vocabulary of Llama 2 to include Japanese characters and conducting continual pre-training on a large Japanese web corpus. Experimental results confirmed that the performance on Japanese tasks drastically improved through continual pre-training, and the performance monotonically increased with the amount of training data up to 100B tokens. Consequently, Swallow achieved superior performance compared to other LLMs that were trained from scratch in English and Japanese. An analysis of the effects of continual pre-training revealed that it was particularly effective for Japanese question answering tasks. Furthermore, to elucidate effective methodologies for cross-lingual continual pre-training from English to Japanese, we investigated the impact of vocabulary expansion and the effectiveness of incorporating parallel corpora. The results showed that the efficiency gained through vocabulary expansion had no negative impact on performance, except for the summarization task, and that the combined use of parallel corpora enhanced translation ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17778v1">MRScore: Evaluating Radiology Report Generation with LLM-based Reward System</a></div>
    <div class="paper-meta">
      📅 2024-04-27
    </div>
    <details class="paper-abstract">
      In recent years, automated radiology report generation has experienced significant growth. This paper introduces MRScore, an automatic evaluation metric tailored for radiology report generation by leveraging Large Language Models (LLMs). Conventional NLG (natural language generation) metrics like BLEU are inadequate for accurately assessing the generated radiology reports, as systematically demonstrated by our observations within this paper. To address this challenge, we collaborated with radiologists to develop a framework that guides LLMs for radiology report evaluation, ensuring alignment with human analysis. Our framework includes two key components: i) utilizing GPT to generate large amounts of training data, i.e., reports with different qualities, and ii) pairing GPT-generated reports as accepted and rejected samples and training LLMs to produce MRScore as the model reward. Our experiments demonstrate MRScore's higher correlation with human judgments and superior performance in model selection compared to traditional metrics. Our code and datasets will be available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16811v2">Make Your LLM Fully Utilize the Context</a></div>
    <div class="paper-meta">
      📅 2024-04-26
      | 💬 19 pages, 7 figures, 3 tables, 9 examples
    </div>
    <details class="paper-abstract">
      While many contemporary large language models (LLMs) can process lengthy input, they still struggle to fully utilize information within the long context, known as the lost-in-the-middle challenge. We hypothesize that it stems from insufficient explicit supervision during the long-context training, which fails to emphasize that any position in a long context can hold crucial information. Based on this intuition, our study presents information-intensive (IN2) training, a purely data-driven solution to overcome lost-in-the-middle. Specifically, IN2 training leverages a synthesized long-context question-answer dataset, where the answer requires (1) fine-grained information awareness on a short segment (~128 tokens) within a synthesized long context (4K-32K tokens), and (2) the integration and reasoning of information from two or more short segments. Through applying this information-intensive training on Mistral-7B, we present FILM-7B (FILl-in-the-Middle). To thoroughly assess the ability of FILM-7B for utilizing long contexts, we design three probing tasks that encompass various context styles (document, code, and structured-data context) and information retrieval patterns (forward, backward, and bi-directional retrieval). The probing results demonstrate that FILM-7B can robustly retrieve information from different positions in its 32K context window. Beyond these probing tasks, FILM-7B significantly improves the performance on real-world long-context tasks (e.g., 23.5->26.9 F1 score on NarrativeQA), while maintaining a comparable performance on short-context tasks (e.g., 59.3->59.2 accuracy on MMLU). Github Link: https://github.com/microsoft/FILM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17283v1">Reinforcement Retrieval Leveraging Fine-grained Feedback for Fact Checking News Claims with Black-Box LLM</a></div>
    <div class="paper-meta">
      📅 2024-04-26
      | 💬 Accepted by COLING 2024
    </div>
    <details class="paper-abstract">
      Retrieval-augmented language models have exhibited promising performance across various areas of natural language processing (NLP), including fact-critical tasks. However, due to the black-box nature of advanced large language models (LLMs) and the non-retrieval-oriented supervision signal of specific tasks, the training of retrieval model faces significant challenges under the setting of black-box LLM. We propose an approach leveraging Fine-grained Feedback with Reinforcement Retrieval (FFRR) to enhance fact-checking on news claims by using black-box LLM. FFRR adopts a two-level strategy to gather fine-grained feedback from the LLM, which serves as a reward for optimizing the retrieval policy, by rating the retrieved documents based on the non-retrieval ground truth of the task. We evaluate our model on two public datasets for real-world news claim verification, and the results demonstrate that FFRR achieves significant improvements over strong LLM-enabled and non-LLM baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17196v1">Human-Imperceptible Retrieval Poisoning Attacks in LLM-Powered Applications</a></div>
    <div class="paper-meta">
      📅 2024-04-26
    </div>
    <details class="paper-abstract">
      Presently, with the assistance of advanced LLM application development frameworks, more and more LLM-powered applications can effortlessly augment the LLMs' knowledge with external content using the retrieval augmented generation (RAG) technique. However, these frameworks' designs do not have sufficient consideration of the risk of external content, thereby allowing attackers to undermine the applications developed with these frameworks. In this paper, we reveal a new threat to LLM-powered applications, termed retrieval poisoning, where attackers can guide the application to yield malicious responses during the RAG process. Specifically, through the analysis of LLM application frameworks, attackers can craft documents visually indistinguishable from benign ones. Despite the documents providing correct information, once they are used as reference sources for RAG, the application is misled into generating incorrect responses. Our preliminary experiments indicate that attackers can mislead LLMs with an 88.33\% success rate, and achieve a 66.67\% success rate in the real-world application, demonstrating the potential impact of retrieval poisoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15848v2">Detecting Conceptual Abstraction in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-25
      | 💬 Paper accepted at the LREC-COLING 2024 Conference (Paper ID: 1968) https://lrec-coling-2024.org/list-of-accepted-papers/
    </div>
    <details class="paper-abstract">
      We present a novel approach to detecting noun abstraction within a large language model (LLM). Starting from a psychologically motivated set of noun pairs in taxonomic relationships, we instantiate surface patterns indicating hypernymy and analyze the attention matrices produced by BERT. We compare the results to two sets of counterfactuals and show that we can detect hypernymy in the abstraction mechanism, which cannot solely be related to the distributional similarity of noun pairs. Our findings are a first step towards the explainability of conceptual abstraction in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17025v1">How Does Conversation Length Impact User's Satisfaction? A Case Study of Length-Controlled Conversations with LLM-Powered Chatbots</a></div>
    <div class="paper-meta">
      📅 2024-04-25
    </div>
    <details class="paper-abstract">
      Users can discuss a wide range of topics with large language models (LLMs), but they do not always prefer solving problems or getting information through lengthy conversations. This raises an intriguing HCI question: How does instructing LLMs to engage in longer or shorter conversations affect conversation quality? In this paper, we developed two Slack chatbots using GPT-4 with the ability to vary conversation lengths and conducted a user study. Participants asked the chatbots both highly and less conversable questions, engaging in dialogues with 0, 3, 5, and 7 conversational turns. We found that the conversation quality does not differ drastically across different conditions, while participants had mixed reactions. Our study demonstrates LLMs' ability to change conversation length and the potential benefits for users resulting from such changes, but we caution that changes in text form may not necessarily imply changes in quality or content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16921v1">A Short Survey of Human Mobility Prediction in Epidemic Modeling from Transformers to LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-25
    </div>
    <details class="paper-abstract">
      This paper provides a comprehensive survey of recent advancements in leveraging machine learning techniques, particularly Transformer models, for predicting human mobility patterns during epidemics. Understanding how people move during epidemics is essential for modeling the spread of diseases and devising effective response strategies. Forecasting population movement is crucial for informing epidemiological models and facilitating effective response planning in public health emergencies. Predicting mobility patterns can enable authorities to better anticipate the geographical and temporal spread of diseases, allocate resources more efficiently, and implement targeted interventions. We review a range of approaches utilizing both pretrained language models like BERT and Large Language Models (LLMs) tailored specifically for mobility prediction tasks. These models have demonstrated significant potential in capturing complex spatio-temporal dependencies and contextual patterns in textual data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00010v1">Hybrid Heterogeneous Clusters Can Lower the Energy Consumption of LLM Inference Workloads</a></div>
    <div class="paper-meta">
      📅 2024-04-25
    </div>
    <details class="paper-abstract">
      Both the training and use of Large Language Models (LLMs) require large amounts of energy. Their increasing popularity, therefore, raises critical concerns regarding the energy efficiency and sustainability of data centers that host them. This paper addresses the challenge of reducing energy consumption in data centers running LLMs. We propose a hybrid data center model that uses a cost-based scheduling framework to dynamically allocate LLM tasks across hardware accelerators that differ in their energy efficiencies and computational capabilities. Specifically, our workload-aware strategy determines whether tasks are processed on energy-efficient processors or high-performance GPUs based on the number of input and output tokens in a query. Our analysis of a representative LLM dataset, finds that this hybrid strategy can reduce CPU+GPU energy consumption by 7.5% compared to a workload-unaware baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06209v2">Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-25
      | 💬 Project page: https://tsb0601.github.io/mmvp_blog/
    </div>
    <details class="paper-abstract">
      Is vision good enough for language? Recent advancements in multimodal models primarily stem from the powerful reasoning abilities of large language models (LLMs). However, the visual component typically depends only on the instance-level contrastive language-image pre-training (CLIP). Our research reveals that the visual capabilities in recent multimodal LLMs (MLLMs) still exhibit systematic shortcomings. To understand the roots of these errors, we explore the gap between the visual embedding space of CLIP and vision-only self-supervised learning. We identify ''CLIP-blind pairs'' - images that CLIP perceives as similar despite their clear visual differences. With these pairs, we construct the Multimodal Visual Patterns (MMVP) benchmark. MMVP exposes areas where state-of-the-art systems, including GPT-4V, struggle with straightforward questions across nine basic visual patterns, often providing incorrect answers and hallucinated explanations. We further evaluate various CLIP-based vision-and-language models and found a notable correlation between visual patterns that challenge CLIP models and those problematic for multimodal LLMs. As an initial effort to address these issues, we propose a Mixture of Features (MoF) approach, demonstrating that integrating vision self-supervised learning features with MLLMs can significantly enhance their visual grounding capabilities. Together, our research suggests visual representation learning remains an open challenge, and accurate visual grounding is crucial for future successful multimodal systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02329v1">Digital ASIC Design with Ongoing LLMs: Strategies and Prospects</a></div>
    <div class="paper-meta">
      📅 2024-04-25
      | 💬 8 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      The escalating complexity of modern digital systems has imposed significant challenges on integrated circuit (IC) design, necessitating tools that can simplify the IC design flow. The advent of Large Language Models (LLMs) has been seen as a promising development, with the potential to automate the generation of Hardware Description Language (HDL) code, thereby streamlining digital IC design. However, the practical application of LLMs in this area faces substantial hurdles. Notably, current LLMs often generate HDL code with small but critical syntax errors and struggle to accurately convey the high-level semantics of circuit designs. These issues significantly undermine the utility of LLMs for IC design, leading to misinterpretations and inefficiencies. In response to these challenges, this paper presents targeted strategies to harness the capabilities of LLMs for digital ASIC design. We outline approaches that improve the reliability and accuracy of HDL code generation by LLMs. As a practical demonstration of these strategies, we detail the development of a simple three-phase Pulse Width Modulation (PWM) generator. This project, part of the "Efabless AI-Generated Open-Source Chip Design Challenge," successfully passed the Design Rule Check (DRC) and was fabricated, showcasing the potential of LLMs to enhance digital ASIC design. This work underscores the feasibility and benefits of integrating LLMs into the IC design process, offering a novel approach to overcoming the complexities of modern digital systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08291v2">CleanAgent: Automating Data Standardization with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-04-25
    </div>
    <details class="paper-abstract">
      Data standardization is a crucial part in data science life cycle. While tools like Pandas offer robust functionalities, their complexity and the manual effort required for customizing code to diverse column types pose significant challenges. Although large language models (LLMs) like ChatGPT have shown promise in automating this process through natural language understanding and code generation, it still demands expert-level programming knowledge and continuous interaction for prompt refinement. To solve these challenges, our key idea is to propose a Python library with declarative, unified APIs for standardizing column types, simplifying the code generation of LLM with concise API calls. We first propose Dataprep.Clean which is written as a component of the Dataprep Library, offers a significant reduction in complexity by enabling the standardization of specific column types with a single line of code. Then we introduce the CleanAgent framework integrating Dataprep.Clean and LLM-based agents to automate the data standardization process. With CleanAgent, data scientists need only provide their requirements once, allowing for a hands-free, automatic standardization process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16297v1">When Fuzzing Meets LLMs: Challenges and Opportunities</a></div>
    <div class="paper-meta">
      📅 2024-04-25
    </div>
    <details class="paper-abstract">
      Fuzzing, a widely-used technique for bug detection, has seen advancements through Large Language Models (LLMs). Despite their potential, LLMs face specific challenges in fuzzing. In this paper, we identified five major challenges of LLM-assisted fuzzing. To support our findings, we revisited the most recent papers from top-tier conferences, confirming that these challenges are widespread. As a remedy, we propose some actionable recommendations to help improve applying LLM in Fuzzing and conduct preliminary evaluations on DBMS fuzzing. The results demonstrate that our recommendations effectively address the identified challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16294v1">LLM-Based Section Identifiers Excel on Open Source but Stumble in Real World Applications</a></div>
    <div class="paper-meta">
      📅 2024-04-25
      | 💬 To appear in NAACL 2024 at the 6th Clinical Natural Language Processing Workshop
    </div>
    <details class="paper-abstract">
      Electronic health records (EHR) even though a boon for healthcare practitioners, are growing convoluted and longer every day. Sifting around these lengthy EHRs is taxing and becomes a cumbersome part of physician-patient interaction. Several approaches have been proposed to help alleviate this prevalent issue either via summarization or sectioning, however, only a few approaches have truly been helpful in the past. With the rise of automated methods, machine learning (ML) has shown promise in solving the task of identifying relevant sections in EHR. However, most ML methods rely on labeled data which is difficult to get in healthcare. Large language models (LLMs) on the other hand, have performed impressive feats in natural language processing (NLP), that too in a zero-shot manner, i.e. without any labeled data. To that end, we propose using LLMs to identify relevant section headers. We find that GPT-4 can effectively solve the task on both zero and few-shot settings as well as segment dramatically better than state-of-the-art methods. Additionally, we also annotate a much harder real world dataset and find that GPT-4 struggles to perform well, alluding to further research and harder benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17605v1">Autonomous LLM-driven research from data to human-verifiable research papers</a></div>
    <div class="paper-meta">
      📅 2024-04-24
    </div>
    <details class="paper-abstract">
      As AI promises to accelerate scientific discovery, it remains unclear whether fully AI-driven research is possible and whether it can adhere to key scientific values, such as transparency, traceability and verifiability. Mimicking human scientific practices, we built data-to-paper, an automation platform that guides interacting LLM agents through a complete stepwise research process, while programmatically back-tracing information flow and allowing human oversight and interactions. In autopilot mode, provided with annotated data alone, data-to-paper raised hypotheses, designed research plans, wrote and debugged analysis codes, generated and interpreted results, and created complete and information-traceable research papers. Even though research novelty was relatively limited, the process demonstrated autonomous generation of de novo quantitative insights from data. For simple research goals, a fully-autonomous cycle can create manuscripts which recapitulate peer-reviewed publications without major errors in about 80-90%, yet as goal complexity increases, human co-piloting becomes critical for assuring accuracy. Beyond the process itself, created manuscripts too are inherently verifiable, as information-tracing allows to programmatically chain results, methods and data. Our work thereby demonstrates a potential for AI-driven acceleration of scientific discovery while enhancing, rather than jeopardizing, traceability, transparency and verifiability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16164v1">Towards a Holistic Evaluation of LLMs on Factual Knowledge Recall</a></div>
    <div class="paper-meta">
      📅 2024-04-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance on a variety of NLP tasks, and are being rapidly adopted in a wide range of use cases. It is therefore of vital importance to holistically evaluate the factuality of their generated outputs, as hallucinations remain a challenging issue. In this work, we focus on assessing LLMs' ability to recall factual knowledge learned from pretraining, and the factors that affect this ability. To that end, we construct FACT-BENCH, a representative benchmark covering 20 domains, 134 property types, 3 answer types, and different knowledge popularity levels. We benchmark 31 models from 10 model families and provide a holistic assessment of their strengths and weaknesses. We observe that instruction-tuning hurts knowledge recall, as pretraining-only models consistently outperform their instruction-tuned counterparts, and positive effects of model scaling, as larger models outperform smaller ones for all model families. However, the best performance from GPT-4 still represents a large gap with the upper-bound. We additionally study the role of in-context exemplars using counterfactual demonstrations, which lead to significant degradation of factual knowledge recall for large models. By further decoupling model known and unknown knowledge, we find the degradation is attributed to exemplars that contradict a model's known knowledge, as well as the number of such exemplars. Lastly, we fine-tune LLaMA-7B in different settings of known and unknown knowledge. In particular, fine-tuning on a model's known knowledge is beneficial, and consistently outperforms fine-tuning on unknown and mixed knowledge. We will make our benchmark publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15298v2">Together We Go Further: LLMs and IDE Static Analysis for Extract Method Refactoring</a></div>
    <div class="paper-meta">
      📅 2024-04-24
    </div>
    <details class="paper-abstract">
      Long methods that encapsulate multiple responsibilities within a single method are challenging to maintain. Choosing which statements to extract into new methods has been the target of many research tools. Despite steady improvements, these tools often fail to generate refactorings that align with developers' preferences and acceptance criteria. Given that Large Language Models (LLMs) have been trained on large code corpora, if we harness their familiarity with the way developers form functions, we could suggest refactorings that developers are likely to accept. In this paper, we advance the science and practice of refactoring by synergistically combining the insights of LLMs with the power of IDEs to perform Extract Method (EM). Our formative study on 1752 EM scenarios revealed that LLMs are very effective for giving expert suggestions, yet they are unreliable: up to 76.3% of the suggestions are hallucinations. We designed a novel approach that removes hallucinations from the candidates suggested by LLMs, then further enhances and ranks suggestions based on static analysis techniques from program slicing, and finally leverages the IDE to execute refactorings correctly. We implemented this approach in an IntelliJ IDEA plugin called EM-Assist. We empirically evaluated EM-Assist on a diverse corpus that replicates 1752 actual refactorings from open-source projects. We found that EM-Assist outperforms previous state of the art tools: EM-Assist suggests the developerperformed refactoring in 53.4% of cases, improving over the recall rate of 39.4% for previous best-in-class tools. Furthermore, we conducted firehouse surveys with 16 industrial developers and suggested refactorings on their recent commits. 81.3% of them agreed with the recommendations provided by EM-Assist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16115v1">Online Personalizing White-box LLMs Generation with Neural Bandits</a></div>
    <div class="paper-meta">
      📅 2024-04-24
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      The advent of personalized content generation by LLMs presents a novel challenge: how to efficiently adapt text to meet individual preferences without the unsustainable demand of creating a unique model for each user. This study introduces an innovative online method that employs neural bandit algorithms to dynamically optimize soft instruction embeddings based on user feedback, enhancing the personalization of open-ended text generation by white-box LLMs. Through rigorous experimentation on various tasks, we demonstrate significant performance improvements over baseline strategies. NeuralTS, in particular, leads to substantial enhancements in personalized news headline generation, achieving up to a 62.9% improvement in terms of best ROUGE scores and up to 2.76% increase in LLM-agent evaluation against the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15869v1">Semantic Routing for Enhanced Performance of LLM-Assisted Intent-Based 5G Core Network Management and Orchestration</a></div>
    <div class="paper-meta">
      📅 2024-04-24
      | 💬 Submitted to IEEE GlobeCom 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly emerging in Artificial Intelligence (AI) applications, especially in the fields of natural language processing and generative AI. Not limited to text generation applications, these models inherently possess the opportunity to leverage prompt engineering, where the inputs of such models can be appropriately structured to articulate a model's purpose explicitly. A prominent example of this is intent-based networking, an emerging approach for automating and maintaining network operations and management. This paper presents semantic routing to achieve enhanced performance in LLM-assisted intent-based management and orchestration of 5G core networks. This work establishes an end-to-end intent extraction framework and presents a diverse dataset of sample user intents accompanied by a thorough analysis of the effects of encoders and quantization on overall system performance. The results show that using a semantic router improves the accuracy and efficiency of the LLM deployment compared to stand-alone LLMs with prompting architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15845v1">Exploring LLM Prompting Strategies for Joint Essay Scoring and Feedback Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-24
      | 💬 Accepted to BEA Workshop 2024
    </div>
    <details class="paper-abstract">
      Individual feedback can help students improve their essay writing skills. However, the manual effort required to provide such feedback limits individualization in practice. Automatically-generated essay feedback may serve as an alternative to guide students at their own pace, convenience, and desired frequency. Large language models (LLMs) have demonstrated strong performance in generating coherent and contextually relevant text. Yet, their ability to provide helpful essay feedback is unclear. This work explores several prompting strategies for LLM-based zero-shot and few-shot generation of essay feedback. Inspired by Chain-of-Thought prompting, we study how and to what extent automated essay scoring (AES) can benefit the quality of generated feedback. We evaluate both the AES performance that LLMs can achieve with prompting only and the helpfulness of the generated essay feedback. Our results suggest that tackling AES and feedback generation jointly improves AES performance. However, while our manual evaluation emphasizes the quality of the generated essay feedback, the impact of essay scoring on the generated feedback remains low ultimately.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15804v1">GeckOpt: LLM System Efficiency via Intent-Based Tool Selection</a></div>
    <div class="paper-meta">
      📅 2024-04-24
      | 💬 GLSVLSI 2024
    </div>
    <details class="paper-abstract">
      In this preliminary study, we investigate a GPT-driven intent-based reasoning approach to streamline tool selection for large language models (LLMs) aimed at system efficiency. By identifying the intent behind user prompts at runtime, we narrow down the API toolset required for task execution, reducing token consumption by up to 24.6\%. Early results on a real-world, massively parallel Copilot platform with over 100 GPT-4-Turbo nodes show cost reductions and potential towards improving LLM-based system efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01799v2">Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward</a></div>
    <div class="paper-meta">
      📅 2024-04-24
      | 💬 Accepted at IJCAI '24 (Survey Track), Updated TGI results
    </div>
    <details class="paper-abstract">
      Despite the impressive performance of LLMs, their widespread adoption faces challenges due to substantial computational and memory requirements during inference. Recent advancements in model compression and system-level optimization methods aim to enhance LLM inference. This survey offers an overview of these methods, emphasizing recent developments. Through experiments on LLaMA(/2)-7B, we evaluate various compression techniques, providing practical insights for efficient LLM deployment in a unified setting. The empirical analysis on LLaMA(/2)-7B highlights the effectiveness of these methods. Drawing from survey insights, we identify current limitations and discuss potential future directions to improve LLM inference efficiency. We release the codebase to reproduce the results presented in this paper at https://github.com/nyunAI/Faster-LLM-Survey
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15604v1">Hybrid LLM/Rule-based Approaches to Business Insights Generation from Structured Data</a></div>
    <div class="paper-meta">
      📅 2024-04-24
    </div>
    <details class="paper-abstract">
      In the field of business data analysis, the ability to extract actionable insights from vast and varied datasets is essential for informed decision-making and maintaining a competitive edge. Traditional rule-based systems, while reliable, often fall short when faced with the complexity and dynamism of modern business data. Conversely, Artificial Intelligence (AI) models, particularly Large Language Models (LLMs), offer significant potential in pattern recognition and predictive analytics but can lack the precision necessary for specific business applications. This paper explores the efficacy of hybrid approaches that integrate the robustness of rule-based systems with the adaptive power of LLMs in generating actionable business insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.13788v5">Can LLM-Generated Misinformation Be Detected?</a></div>
    <div class="paper-meta">
      📅 2024-04-23
      | 💬 Accepted to Proceedings of ICLR 2024. 9 pages for main paper, 40 pages including appendix. The code, results, dataset for this paper and more resources on "LLMs Meet Misinformation" have been released on the project website: https://llm-misinformation.github.io/
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has made a transformative impact. However, the potential that LLMs such as ChatGPT can be exploited to generate misinformation has posed a serious concern to online safety and public trust. A fundamental research question is: will LLM-generated misinformation cause more harm than human-written misinformation? We propose to tackle this question from the perspective of detection difficulty. We first build a taxonomy of LLM-generated misinformation. Then we categorize and validate the potential real-world methods for generating misinformation with LLMs. Then, through extensive empirical investigation, we discover that LLM-generated misinformation can be harder to detect for humans and detectors compared to human-written misinformation with the same semantics, which suggests it can have more deceptive styles and potentially cause more harm. We also discuss the implications of our discovery on combating misinformation in the age of LLMs and the countermeasures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00708v1">Interactive Analysis of LLMs using Meaningful Counterfactuals</a></div>
    <div class="paper-meta">
      📅 2024-04-23
    </div>
    <details class="paper-abstract">
      Counterfactual examples are useful for exploring the decision boundaries of machine learning models and determining feature attributions. How can we apply counterfactual-based methods to analyze and explain LLMs? We identify the following key challenges. First, the generated textual counterfactuals should be meaningful and readable to users and thus can be mentally compared to draw conclusions. Second, to make the solution scalable to long-form text, users should be equipped with tools to create batches of counterfactuals from perturbations at various granularity levels and interactively analyze the results. In this paper, we tackle the above challenges and contribute 1) a novel algorithm for generating batches of complete and meaningful textual counterfactuals by removing and replacing text segments in different granularities, and 2) LLM Analyzer, an interactive visualization tool to help users understand an LLM's behaviors by interactively inspecting and aggregating meaningful counterfactuals. We evaluate the proposed algorithm by the grammatical correctness of its generated counterfactuals using 1,000 samples from medical, legal, finance, education, and news datasets. In our experiments, 97.2% of the counterfactuals are grammatically correct. Through a use case, user studies, and feedback from experts, we demonstrate the usefulness and usability of the proposed interactive visualization tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00152v2">On-the-fly Definition Augmentation of LLMs for Biomedical NER</a></div>
    <div class="paper-meta">
      📅 2024-04-23
      | 💬 To appear at NAACL 2024 (Main)
    </div>
    <details class="paper-abstract">
      Despite their general capabilities, LLMs still struggle on biomedical NER tasks, which are difficult due to the presence of specialized terminology and lack of training data. In this work we set out to improve LLM performance on biomedical NER in limited data settings via a new knowledge augmentation approach which incorporates definitions of relevant concepts on-the-fly. During this process, to provide a test bed for knowledge augmentation, we perform a comprehensive exploration of prompting strategies. Our experiments show that definition augmentation is useful for both open source and closed LLMs. For example, it leads to a relative improvement of 15\% (on average) in GPT-4 performance (F1) across all (six) of our test datasets. We conduct extensive ablations and analyses to demonstrate that our performance improvements stem from adding relevant definitional knowledge. We find that careful prompting strategies also improve LLM performance, allowing them to outperform fine-tuned language models in few-shot settings. To facilitate future research in this direction, we release our code at https://github.com/allenai/beacon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15149v1">Bias patterns in the application of LLMs for clinical decision support: A comprehensive study</a></div>
    <div class="paper-meta">
      📅 2024-04-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful candidates to inform clinical decision-making processes. While these models play an increasingly prominent role in shaping the digital landscape, two growing concerns emerge in healthcare applications: 1) to what extent do LLMs exhibit social bias based on patients' protected attributes (like race), and 2) how do design choices (like architecture design and prompting strategies) influence the observed biases? To answer these questions rigorously, we evaluated eight popular LLMs across three question-answering (QA) datasets using clinical vignettes (patient descriptions) standardized for bias evaluations. We employ red-teaming strategies to analyze how demographics affect LLM outputs, comparing both general-purpose and clinically-trained models. Our extensive experiments reveal various disparities (some significant) across protected groups. We also observe several counter-intuitive patterns such as larger models not being necessarily less biased and fined-tuned models on medical data not being necessarily better than the general-purpose models. Furthermore, our study demonstrates the impact of prompt design on bias patterns and shows that specific phrasing can influence bias patterns and reflection-type approaches (like Chain of Thought) can reduce biased outcomes effectively. Consistent with prior studies, we call on additional evaluations, scrutiny, and enhancement of LLMs used in clinical decision support applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16420v2">Predicting Sustainable Development Goals Using Course Descriptions -- from LLMs to Conventional Foundation Models</a></div>
    <div class="paper-meta">
      📅 2024-04-23
      | 💬 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      We present our work on predicting United Nations sustainable development goals (SDG) for university courses. We use an LLM named PaLM 2 to generate training data given a noisy human-authored course description input as input. We use this data to train several different smaller language models to predict SDGs for university courses. This work contributes to better university level adaptation of SDGs. The best performing model in our experiments was BART with an F1-score of 0.786.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16733v4">LLMs for Science: Usage for Code Generation and Data Analysis</a></div>
    <div class="paper-meta">
      📅 2024-04-23
      | 💬 Accepted (ICSSP Special Issue in Journal of Software Evolution and Process); In Print
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been touted to enable increased productivity in many areas of today's work life. Scientific research as an area of work is no exception: the potential of LLM-based tools to assist in the daily work of scientists has become a highly discussed topic across disciplines. However, we are only at the very onset of this subject of study. It is still unclear how the potential of LLMs will materialise in research practice. With this study, we give first empirical evidence on the use of LLMs in the research process. We have investigated a set of use cases for LLM-based tools in scientific research, and conducted a first study to assess to which degree current tools are helpful. In this paper we report specifically on use cases related to software engineering, such as generating application code and developing scripts for data analytics. While we studied seemingly simple use cases, results across tools differ significantly. Our results highlight the promise of LLM-based tools in general, yet we also observe various issues, particularly regarding the integrity of the output these tools provide.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13558v2">LASER: Tuning-Free LLM-Driven Attention Control for Efficient Text-conditioned Image-to-Animation</a></div>
    <div class="paper-meta">
      📅 2024-04-23
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Revolutionary advancements in text-to-image models have unlocked new dimensions for sophisticated content creation, e.g., text-conditioned image editing, allowing us to edit the diverse images that convey highly complex visual concepts according to the textual guidance. Despite being promising, existing methods focus on texture- or non-rigid-based visual manipulation, which struggles to produce the fine-grained animation of smooth text-conditioned image morphing without fine-tuning, i.e., due to their highly unstructured latent space. In this paper, we introduce a tuning-free LLM-driven attention control framework, encapsulated by the progressive process of LLM planning, prompt-Aware editing, StablE animation geneRation, abbreviated as LASER. LASER employs a large language model (LLM) to refine coarse descriptions into detailed prompts, guiding pre-trained text-to-image models for subsequent image generation. We manipulate the model's spatial features and self-attention mechanisms to maintain animation integrity and enable seamless morphing directly from text prompts, eliminating the need for additional fine-tuning or annotations. Our meticulous control over spatial features and self-attention ensures structural consistency in the images. This paper presents a novel framework integrating LLMs with text-to-image models to create high-quality animations from a single text input. We also propose a Text-conditioned Image-to-Animation Benchmark to validate the effectiveness and efficacy of LASER. Extensive experiments demonstrate that LASER produces impressive, consistent, and efficient results in animation generation, positioning it as a powerful tool for advanced digital content creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14779v1">Med42 -- Evaluating Fine-Tuning Strategies for Medical LLMs: Full-Parameter vs. Parameter-Efficient Approaches</a></div>
    <div class="paper-meta">
      📅 2024-04-23
      | 💬 Published at AAAI 2024 Spring Symposium - Clinical Foundation Models
    </div>
    <details class="paper-abstract">
      This study presents a comprehensive analysis and comparison of two predominant fine-tuning methodologies - full-parameter fine-tuning and parameter-efficient tuning - within the context of medical Large Language Models (LLMs). We developed and refined a series of LLMs, based on the Llama-2 architecture, specifically designed to enhance medical knowledge retrieval, reasoning, and question-answering capabilities. Our experiments systematically evaluate the effectiveness of these tuning strategies across various well-known medical benchmarks. Notably, our medical LLM Med42 showed an accuracy level of 72% on the US Medical Licensing Examination (USMLE) datasets, setting a new standard in performance for openly available medical LLMs. Through this comparative analysis, we aim to identify the most effective and efficient method for fine-tuning LLMs in the medical domain, thereby contributing significantly to the advancement of AI-driven healthcare applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14618v1">Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing</a></div>
    <div class="paper-meta">
      📅 2024-04-22
      | 💬 Accepted to ICLR 2024 (main conference)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in most NLP tasks but also require expensive cloud servers for deployment due to their size, while smaller models that can be deployed on lower cost (e.g., edge) devices, tend to lag behind in terms of response quality. Therefore in this work we propose a hybrid inference approach which combines their respective strengths to save cost and maintain quality. Our approach uses a router that assigns queries to the small or large model based on the predicted query difficulty and the desired quality level. The desired quality level can be tuned dynamically at test time to seamlessly trade quality for cost as per the scenario requirements. In experiments our approach allows us to make up to 40% fewer calls to the large model, with no drop in response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14544v1">WangLab at MEDIQA-CORR 2024: Optimized LLM-based Programs for Medical Error Detection and Correction</a></div>
    <div class="paper-meta">
      📅 2024-04-22
    </div>
    <details class="paper-abstract">
      Medical errors in clinical text pose significant risks to patient safety. The MEDIQA-CORR 2024 shared task focuses on detecting and correcting these errors across three subtasks: identifying the presence of an error, extracting the erroneous sentence, and generating a corrected sentence. In this paper, we present our approach that achieved top performance in all three subtasks. For the MS dataset, which contains subtle errors, we developed a retrieval-based system leveraging external medical question-answering datasets. For the UW dataset, reflecting more realistic clinical notes, we created a pipeline of modules to detect, localize, and correct errors. Both approaches utilized the DSPy framework for optimizing prompts and few-shot examples in large language model (LLM) based programs. Our results demonstrate the effectiveness of LLM based programs for medical error correction. However, our approach has limitations in addressing the full diversity of potential errors in medical documentation. We discuss the implications of our work and highlight future research directions to advance the robustness and applicability of medical error detection and correction systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01855v2">Where to Move Next: Zero-shot Generalization of LLMs for Next POI Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-04-22
    </div>
    <details class="paper-abstract">
      Next Point-of-interest (POI) recommendation provides valuable suggestions for users to explore their surrounding environment. Existing studies rely on building recommendation models from large-scale users' check-in data, which is task-specific and needs extensive computational resources. Recently, the pretrained large language models (LLMs) have achieved significant advancements in various NLP tasks and have also been investigated for recommendation scenarios. However, the generalization abilities of LLMs still are unexplored to address the next POI recommendations, where users' geographical movement patterns should be extracted. Although there are studies that leverage LLMs for next-item recommendations, they fail to consider the geographical influence and sequential transitions. Hence, they cannot effectively solve the next POI recommendation task. To this end, we design novel prompting strategies and conduct empirical studies to assess the capability of LLMs, e.g., ChatGPT, for predicting a user's next check-in. Specifically, we consider several essential factors in human movement behaviors, including user geographical preference, spatial distance, and sequential transitions, and formulate the recommendation task as a ranking problem. Through extensive experiments on two widely used real-world datasets, we derive several key findings. Empirical evaluations demonstrate that LLMs have promising zero-shot recommendation abilities and can provide accurate and reasonable predictions. We also reveal that LLMs cannot accurately comprehend geographical context information and are sensitive to the order of presentation of candidate POIs, which shows the limitations of LLMs and necessitates further research on robust human mobility reasoning mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14043v1">LLMs Know What They Need: Leveraging a Missing Information Guided Framework to Empower Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-22
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) demonstrates great value in alleviating outdated knowledge or hallucination by supplying LLMs with updated and relevant knowledge. However, there are still several difficulties for RAG in understanding complex multi-hop query and retrieving relevant documents, which require LLMs to perform reasoning and retrieve step by step. Inspired by human's reasoning process in which they gradually search for the required information, it is natural to ask whether the LLMs could notice the missing information in each reasoning step. In this work, we first experimentally verified the ability of LLMs to extract information as well as to know the missing. Based on the above discovery, we propose a Missing Information Guided Retrieve-Extraction-Solving paradigm (MIGRES), where we leverage the identification of missing information to generate a targeted query that steers the subsequent knowledge retrieval. Besides, we design a sentence-level re-ranking filtering approach to filter the irrelevant content out from document, along with the information extraction capability of LLMs to extract useful information from cleaned-up documents, which in turn to bolster the overall efficacy of RAG. Extensive experiments conducted on multiple public datasets reveal the superiority of the proposed MIGRES method, and analytical experiments demonstrate the effectiveness of our proposed modules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13957v1">How Well Can LLMs Echo Us? Evaluating AI Chatbots' Role-Play Ability with ECHO</a></div>
    <div class="paper-meta">
      📅 2024-04-22
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      The role-play ability of Large Language Models (LLMs) has emerged as a popular research direction. However, existing studies focus on imitating well-known public figures or fictional characters, overlooking the potential for simulating ordinary individuals. Such an oversight limits the potential for advancements in digital human clones and non-player characters in video games. To bridge this gap, we introduce ECHO, an evaluative framework inspired by the Turing test. This framework engages the acquaintances of the target individuals to distinguish between human and machine-generated responses. Notably, our framework focuses on emulating average individuals rather than historical or fictional figures, presenting a unique advantage to apply the Turing Test. We evaluated three role-playing LLMs using ECHO, with GPT-3.5 and GPT-4 serving as foundational models, alongside the online application GPTs from OpenAI. Our results demonstrate that GPT-4 more effectively deceives human evaluators, and GPTs achieves a leading success rate of 48.3%. Furthermore, we investigated whether LLMs could discern between human-generated and machine-generated texts. While GPT-4 can identify differences, it could not determine which texts were human-produced. Our code and results of reproducing the role-playing LLMs are made publicly available via https://github.com/CUHK-ARISE/ECHO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13925v1">MARIO Eval: Evaluate Your Math LLM with your Math LLM--A mathematical dataset evaluation toolkit</a></div>
    <div class="paper-meta">
      📅 2024-04-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been explored in a variety of reasoning tasks including solving of mathematical problems. Each math dataset typically includes its own specially designed evaluation script, which, while suitable for its intended use, lacks generalizability across different datasets. Consequently, updates and adaptations to these evaluation tools tend to occur without being systematically reported, leading to inconsistencies and obstacles to fair comparison across studies. To bridge this gap, we introduce a comprehensive mathematical evaluation toolkit that not only utilizes a python computer algebra system (CAS) for its numerical accuracy, but also integrates an optional LLM, known for its considerable natural language processing capabilities. To validate the effectiveness of our toolkit, we manually annotated two distinct datasets. Our experiments demonstrate that the toolkit yields more robust evaluation results compared to prior works, even without an LLM. Furthermore, when an LLM is incorporated, there is a notable enhancement. The code for our method will be made available at \url{https://github.com/MARIO-Math-Reasoning/math_evaluation}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12636v2">Multi-Objective Fine-Tuning for Enhanced Program Repair with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities on a broad spectrum of downstream tasks. Within the realm of software engineering, specialized tasks on code, such as program repair, present unique challenges, necessitating fine-tuning to unlock state-of-the-art performance. Fine-tuning approaches proposed in the literature for LLMs on program repair tasks are however generally overlooking the need to reason about the logic behind code changes, beyond syntactic patterns in the data. High-performing fine-tuning experiments also usually come at very high computational costs. With MORepair, we propose a novel perspective on the learning focus of LLM fine-tuning for program repair: we not only adapt the LLM parameters to the syntactic nuances of the task of code transformation (objective 1), but we also specifically fine-tune the LLM with respect to the logical reason behind the code change in the training data (objective 2). Such a multi-objective fine-tuning will instruct LLMs to generate high-quality patches. We apply MORepair to fine-tune four open-source LLMs with different sizes and architectures. Experimental results on C++ and Java repair benchmarks show that the implemented fine-tuning effectively boosts LLM repair performance by 7.6% to 10% in Top-10 repair suggestions. We further show that our fine-tuning strategy yields superior performance compared to the incumbent state-of-the-art in fine-tuned models for program repair, Fine-tune-CoT and RepairLLaMA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13855v1">Understanding the role of FFNs in driving multilingual behaviour in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-22
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Multilingualism in Large Language Models (LLMs) is an yet under-explored area. In this paper, we conduct an in-depth analysis of the multilingual capabilities of a family of a Large Language Model, examining its architecture, activation patterns, and processing mechanisms across languages. We introduce novel metrics to probe the model's multilingual behaviour at different layers and shed light on the impact of architectural choices on multilingual processing. Our findings reveal different patterns of multilinugal processing in the sublayers of Feed-Forward Networks of the models. Furthermore, we uncover the phenomenon of "over-layerization" in certain model configurations, where increasing layer depth without corresponding adjustments to other parameters may degrade model performance. Through comparisons within and across languages, we demonstrate the interplay between model architecture, layer depth, and multilingual processing capabilities of LLMs trained on multiple languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13813v1">From LLM to NMT: Advancing Low-Resource Machine Translation with Claude</a></div>
    <div class="paper-meta">
      📅 2024-04-22
      | 💬 17 pages, 15 figures
    </div>
    <details class="paper-abstract">
      We show that Claude 3 Opus, a large language model (LLM) released by Anthropic in March 2024, exhibits stronger machine translation competence than other LLMs. Though we find evidence of data contamination with Claude on FLORES-200, we curate new benchmarks that corroborate the effectiveness of Claude for low-resource machine translation into English. We find that Claude has remarkable \textit{resource efficiency} -- the degree to which the quality of the translation model depends on a language pair's resource level. Finally, we show that advancements in LLM translation can be compressed into traditional neural machine translation (NMT) models. Using Claude to generate synthetic data, we demonstrate that knowledge distillation advances the state-of-the-art in Yoruba-English translation, meeting or surpassing strong baselines like NLLB-54B and Google Translate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15157v1">FASTTRACK: Fast and Accurate Fact Tracing for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-22
    </div>
    <details class="paper-abstract">
      Fact tracing seeks to identify specific training examples that serve as the knowledge source for a given query. Existing approaches to fact tracing rely on assessing the similarity between each training sample and the query along a certain dimension, such as lexical similarity, gradient, or embedding space. However, these methods fall short of effectively distinguishing between samples that are merely relevant and those that actually provide supportive evidence for the information sought by the query. This limitation often results in suboptimal effectiveness. Moreover, these approaches necessitate the examination of the similarity of individual training points for each query, imposing significant computational demands and creating a substantial barrier for practical applications. This paper introduces FASTTRACK, a novel approach that harnesses the capabilities of Large Language Models (LLMs) to validate supportive evidence for queries and at the same time clusters the training database towards a reduced extent for LLMs to trace facts. Our experiments show that FASTTRACK substantially outperforms existing methods in both accuracy and efficiency, achieving more than 100\% improvement in F1 score over the state-of-the-art methods while being X33 faster than \texttt{TracIn}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16873v1">AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-21
      | 💬 32 pages, 9 figures, 7 tables
    </div>
    <details class="paper-abstract">
      While recently Large Language Models (LLMs) have achieved remarkable successes, they are vulnerable to certain jailbreaking attacks that lead to generation of inappropriate or harmful content. Manual red-teaming requires finding adversarial prompts that cause such jailbreaking, e.g. by appending a suffix to a given instruction, which is inefficient and time-consuming. On the other hand, automatic adversarial prompt generation often leads to semantically meaningless attacks that can easily be detected by perplexity-based filters, may require gradient information from the TargetLLM, or do not scale well due to time-consuming discrete optimization processes over the token space. In this paper, we present a novel method that uses another LLM, called the AdvPrompter, to generate human-readable adversarial prompts in seconds, $\sim800\times$ faster than existing optimization-based approaches. We train the AdvPrompter using a novel algorithm that does not require access to the gradients of the TargetLLM. This process alternates between two steps: (1) generating high-quality target adversarial suffixes by optimizing the AdvPrompter predictions, and (2) low-rank fine-tuning of the AdvPrompter with the generated adversarial suffixes. The trained AdvPrompter generates suffixes that veil the input instruction without changing its meaning, such that the TargetLLM is lured to give a harmful response. Experimental results on popular open source TargetLLMs show state-of-the-art results on the AdvBench dataset, that also transfer to closed-source black-box LLM APIs. Further, we demonstrate that by fine-tuning on a synthetic dataset generated by AdvPrompter, LLMs can be made more robust against jailbreaking attacks while maintaining performance, i.e. high MMLU scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13710v1">SVGEditBench: A Benchmark Dataset for Quantitative Assessment of LLM's SVG Editing Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-04-21
      | 💬 Accepted to Workshop on Graphic Design Understanding and Generation (GDUG), a CVPR2024 workshop. Dataset: https://github.com/mti-lab/SVGEditBench
    </div>
    <details class="paper-abstract">
      Text-to-image models have shown progress in recent years. Along with this progress, generating vector graphics from text has also advanced. SVG is a popular format for vector graphics, and SVG represents a scene with XML text. Therefore, Large Language Models can directly process SVG code. Taking this into account, we focused on editing SVG with LLMs. For quantitative evaluation of LLMs' ability to edit SVG, we propose SVGEditBench. SVGEditBench is a benchmark for assessing the LLMs' ability to edit SVG code. We also show the GPT-4 and GPT-3.5 results when evaluated on the proposed benchmark. In the experiments, GPT-4 showed superior performance to GPT-3.5 both quantitatively and qualitatively. The dataset is available at https://github.com/mti-lab/SVGEditBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13571v1">Test-Time Training on Graphs with Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-04-21
    </div>
    <details class="paper-abstract">
      Graph Neural Networks have demonstrated great success in various fields of multimedia. However, the distribution shift between the training and test data challenges the effectiveness of GNNs. To mitigate this challenge, Test-Time Training (TTT) has been proposed as a promising approach. Traditional TTT methods require a demanding unsupervised training strategy to capture the information from test to benefit the main task. Inspired by the great annotation ability of Large Language Models (LLMs) on Text-Attributed Graphs (TAGs), we propose to enhance the test-time training on graphs with LLMs as annotators. In this paper, we design a novel Test-Time Training pipeline, LLMTTT, which conducts the test-time adaptation under the annotations by LLMs on a carefully-selected node set. Specifically, LLMTTT introduces a hybrid active node selection strategy that considers not only node diversity and representativeness, but also prediction signals from the pre-trained model. Given annotations from LLMs, a two-stage training strategy is designed to tailor the test-time model with the limited and noisy labels. A theoretical analysis ensures the validity of our method and extensive experiments demonstrate that the proposed LLMTTT can achieve a significant performance improvement compared to existing Out-of-Distribution (OOD) generalization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09329v2">Large Language Models are as persuasive as humans, but how? About the cognitive effort and moral-emotional language of LLM arguments</a></div>
    <div class="paper-meta">
      📅 2024-04-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are already as persuasive as humans. However, we know very little about how they do it. This paper investigates the persuasion strategies of LLMs, comparing them with human-generated arguments. Using a dataset of 1,251 participants in an experiment, we analyze the persuasion strategies of LLM-generated and human-generated arguments using measures of cognitive effort (lexical and grammatical complexity) and moral-emotional language (sentiment and moral analysis). The study reveals that LLMs produce arguments that require higher cognitive effort, exhibiting more complex grammatical and lexical structures than human counterparts. Additionally, LLMs demonstrate a significant propensity to engage more deeply with moral language, utilizing both positive and negative moral foundations more frequently than humans. In contrast with previous research, no significant difference was found in the emotional content produced by LLMs and humans. These findings contribute to the discourse on AI and persuasion, highlighting the dual potential of LLMs to both enhance and undermine informational integrity through communication strategies for digital persuasion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06658v1">ProteinEngine: Empower LLM with Domain Knowledge for Protein Engineering</a></div>
    <div class="paper-meta">
      📅 2024-04-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have garnered considerable attention for their proficiency in tackling intricate tasks, particularly leveraging their capacities for zero-shot and in-context learning. However, their utility has been predominantly restricted to general tasks due to an absence of domain-specific knowledge. This constraint becomes particularly pertinent in the realm of protein engineering, where specialized expertise is required for tasks such as protein function prediction, protein evolution analysis, and protein design, with a level of specialization that existing LLMs cannot furnish. In response to this challenge, we introduce \textsc{ProteinEngine}, a human-centered platform aimed at amplifying the capabilities of LLMs in protein engineering by seamlessly integrating a comprehensive range of relevant tools, packages, and software via API calls. Uniquely, \textsc{ProteinEngine} assigns three distinct roles to LLMs, facilitating efficient task delegation, specialized task resolution, and effective communication of results. This design fosters high extensibility and promotes the smooth incorporation of new algorithms, models, and features for future development. Extensive user studies, involving participants from both the AI and protein engineering communities across academia and industry, consistently validate the superiority of \textsc{ProteinEngine} in augmenting the reliability and precision of deep learning in protein engineering tasks. Consequently, our findings highlight the potential of \textsc{ProteinEngine} to bride the disconnected tools for future research in the protein engineering domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13343v1">UnibucLLM: Harnessing LLMs for Automated Prediction of Item Difficulty and Response Time for Multiple-Choice Questions</a></div>
    <div class="paper-meta">
      📅 2024-04-20
      | 💬 Accepted at BEA 2024 (NAACL Workshop)
    </div>
    <details class="paper-abstract">
      This work explores a novel data augmentation method based on Large Language Models (LLMs) for predicting item difficulty and response time of retired USMLE Multiple-Choice Questions (MCQs) in the BEA 2024 Shared Task. Our approach is based on augmenting the dataset with answers from zero-shot LLMs (Falcon, Meditron, Mistral) and employing transformer-based models based on six alternative feature combinations. The results suggest that predicting the difficulty of questions is more challenging. Notably, our top performing methods consistently include the question text, and benefit from the variability of LLM answers, highlighting the potential of LLMs for improving automated assessment in medical licensing exams. We make our code available https://github.com/ana-rogoz/BEA-2024.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01830v2">PiCO: Peer Review in LLMs based on the Consistency Optimization</a></div>
    <div class="paper-meta">
      📅 2024-04-20
    </div>
    <details class="paper-abstract">
      Existing large language models (LLMs) evaluation methods typically focus on testing the performance on some closed-environment and domain-specific benchmarks with human annotations. In this paper, we explore a novel unsupervised evaluation direction, utilizing peer-review mechanisms to measure LLMs automatically. In this setting, both open-source and closed-source LLMs lie in the same environment, capable of answering unlabeled questions and evaluating each other, where each LLM's response score is jointly determined by other anonymous ones. To obtain the ability hierarchy among these models, we assign each LLM a learnable capability parameter to adjust the final ranking. We formalize it as a constrained optimization problem, intending to maximize the consistency of each LLM's capabilities and scores. The key assumption behind is that high-level LLM can evaluate others' answers more accurately than low-level ones, while higher-level LLM can also achieve higher response scores. Moreover, we propose three metrics called PEN, CIN, and LIS to evaluate the gap in aligning human rankings. We perform experiments on multiple datasets with these metrics, validating the effectiveness of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13208v1">The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions</a></div>
    <div class="paper-meta">
      📅 2024-04-19
    </div>
    <details class="paper-abstract">
      Today's LLMs are susceptible to prompt injections, jailbreaks, and other attacks that allow adversaries to overwrite a model's original instructions with their own malicious prompts. In this work, we argue that one of the primary vulnerabilities underlying these attacks is that LLMs often consider system prompts (e.g., text from an application developer) to be the same priority as text from untrusted users and third parties. To address this, we propose an instruction hierarchy that explicitly defines how models should behave when instructions of different priorities conflict. We then propose a data generation method to demonstrate this hierarchical instruction following behavior, which teaches LLMs to selectively ignore lower-privileged instructions. We apply this method to GPT-3.5, showing that it drastically increases robustness -- even for attack types not seen during training -- while imposing minimal degradations on standard capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13149v1">Beyond Self-Consistency: Ensemble Reasoning Boosts Consistency and Accuracy of LLMs in Cancer Staging</a></div>
    <div class="paper-meta">
      📅 2024-04-19
      | 💬 accepted to the 22nd International Conference on Artificial Intelligence in Medicine (AIME'24)
    </div>
    <details class="paper-abstract">
      Advances in large language models (LLMs) have encouraged their adoption in the healthcare domain where vital clinical information is often contained in unstructured notes. Cancer staging status is available in clinical reports, but it requires natural language processing to extract the status from the unstructured text. With the advance in clinical-oriented LLMs, it is promising to extract such status without extensive efforts in training the algorithms. Prompting approaches of the pre-trained LLMs that elicit a model's reasoning process, such as chain-of-thought, may help to improve the trustworthiness of the generated responses. Using self-consistency further improves model performance, but often results in inconsistent generations across the multiple reasoning paths. In this study, we propose an ensemble reasoning approach with the aim of improving the consistency of the model generations. Using an open access clinical large language model to determine the pathologic cancer stage from real-world pathology reports, we show that the ensemble reasoning approach is able to improve both the consistency and performance of the LLM in determining cancer stage, thereby demonstrating the potential to use these models in clinical or other domains where reliability and trustworthiness are critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13033v1">Sample Design Engineering: An Empirical Study of What Makes Good Downstream Fine-Tuning Samples for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-19
      | 💬 23 pages, 12 figures, 14 tables
    </div>
    <details class="paper-abstract">
      In the burgeoning field of Large Language Models (LLMs) like ChatGPT and LLaMA, Prompt Engineering (PE) is renowned for boosting zero-shot or in-context learning (ICL) through prompt modifications. Yet, the realm of the sample design for downstream fine-tuning, crucial for task-specific LLM adaptation, is largely unexplored. This paper introduces Sample Design Engineering (SDE), a methodical approach to enhancing LLMs' post-tuning performance by refining input, output, and reasoning designs. We conduct a series of in-domain (ID) and out-of-domain (OOD) experiments to assess the impact of various design options on LLMs' downstream performance, revealing several intriguing patterns that hold consistently across different LLMs. Based on these insights, we propose an integrated SDE strategy, combining the most effective options, and validate its consistent superiority over heuristic sample designs in complex downstream tasks like multi-aspect sentiment analysis, event extraction, and nested entity recognition. Additionally, analyses of LLMs' inherent prompt/output perplexity, zero-shot, and ICL abilities illustrate that good PE strategies may not always translate to good SDE strategies. Code available at https://github.com/beyondguo/LLM-Tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13028v1">When Life gives you LLMs, make LLM-ADE: Large Language Models with Adaptive Data Engineering</a></div>
    <div class="paper-meta">
      📅 2024-04-19
      | 💬 6 pages, 3 tables and 3 figures
    </div>
    <details class="paper-abstract">
      This paper presents the LLM-ADE framework, a novel methodology for continued pre-training of large language models (LLMs) that addresses the challenges of catastrophic forgetting and double descent. LLM-ADE employs dynamic architectural adjustments, including selective block freezing and expansion, tailored to specific datasets. This strategy enhances model adaptability to new data while preserving previously acquired knowledge. We demonstrate LLM-ADE's effectiveness on the TinyLlama model across various general knowledge benchmarks, showing significant performance improvements without the drawbacks of traditional continuous training methods. This approach promises a more versatile and robust way to keep LLMs current and efficient in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.15517v2">A Dataflow Compiler for Efficient LLM Inference using Custom Microscaling Formats</a></div>
    <div class="paper-meta">
      📅 2024-04-19
    </div>
    <details class="paper-abstract">
      Model quantization represents both parameters (weights) and intermediate values (activations) in a more compact format, thereby directly reducing both computational and memory cost in hardware. The quantization of recent large language models (LLMs) faces challenges to achieve competitive memory density compared to other models such as convolutional neural networks, since values in LLMs require larger dynamic ranges. Current hardware can expedite computation for LLMs using compact numerical formats such as low-bitwidth integers or floating-point numbers. Each has advantages: integer operations simplify circuit design, whereas floating-point calculations can enhance accuracy when a wider dynamic range is required. In this work, we seek an efficient data format that combines the best of both worlds: Microscaling (MX) formats. MX formats are efficient data formats that achieve both large dynamic ranges and high memory density. In this paper, we propose a compiler named MASE for exploring mixed-precision MX formats on dataflow hardware accelerators for LLM inference. Our main contributions are twofold. First, we propose a novel orchestration abstraction to explore both software and hardware optimizations with new data formats. Second, MASE achieves LLM inference at an average precision of 4-bits, with minimal to no accuracy degradation. To our knowledge, MASE represents the first effort to harness fine-grain multi-precision MX formats in the design of LLM hardware accelerators. Over a range of LLMs and datasets, MASE achieves an average improvement of 24% in $\Delta$ accuracy with an overhead of only 3% in energy efficiency compared to designs using 8-bit fixed-point numbers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12772v1">Generating Test Scenarios from NL Requirements using Retrieval-Augmented LLMs: An Industrial Study</a></div>
    <div class="paper-meta">
      📅 2024-04-19
    </div>
    <details class="paper-abstract">
      Test scenarios are specific instances of test cases that describe actions to validate a particular software functionality. By outlining the conditions under which the software operates and the expected outcomes, test scenarios ensure that the software functionality is tested in an integrated manner. Test scenarios are crucial for systematically testing an application under various conditions, including edge cases, to identify potential issues and guarantee overall performance and reliability. Specifying test scenarios is tedious and requires a deep understanding of software functionality and the underlying domain. It further demands substantial effort and investment from already time- and budget-constrained requirements engineers and testing teams. This paper presents an automated approach (RAGTAG) for test scenario generation using Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs). RAG allows the integration of specific domain knowledge with LLMs' generation capabilities. We evaluate RAGTAG on two industrial projects from Austrian Post with bilingual requirements in German and English. Our results from an interview survey conducted with four experts on five dimensions -- relevance, coverage, correctness, coherence and feasibility, affirm the potential of RAGTAG in automating test scenario generation. Specifically, our results indicate that, despite the difficult task of analyzing bilingual requirements, RAGTAG is able to produce scenarios that are well-aligned with the underlying requirements and provide coverage of different aspects of the intended functionality. The generated scenarios are easily understandable to experts and feasible for testing in the project environment. The overall correctness is deemed satisfactory; however, gaps in capturing exact action sequences and domain nuances remain, underscoring the need for domain expertise when applying LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15127v3">Evaluation of LLM Chatbots for OSINT-based Cyber Threat Awareness</a></div>
    <div class="paper-meta">
      📅 2024-04-19
    </div>
    <details class="paper-abstract">
      Knowledge sharing about emerging threats is crucial in the rapidly advancing field of cybersecurity and forms the foundation of Cyber Threat Intelligence (CTI). In this context, Large Language Models are becoming increasingly significant in the field of cybersecurity, presenting a wide range of opportunities. This study surveys the performance of ChatGPT, GPT4all, Dolly, Stanford Alpaca, Alpaca-LoRA, Falcon, and Vicuna chatbots in binary classification and Named Entity Recognition (NER) tasks performed using Open Source INTelligence (OSINT). We utilize well-established data collected in previous research from Twitter to assess the competitiveness of these chatbots when compared to specialized models trained for those tasks. In binary classification experiments, Chatbot GPT-4 as a commercial model achieved an acceptable F1 score of 0.94, and the open-source GPT4all model achieved an F1 score of 0.90. However, concerning cybersecurity entity recognition, all evaluated chatbots have limitations and are less effective. This study demonstrates the capability of chatbots for OSINT binary classification and shows that they require further improvement in NER to effectively replace specially trained models. Our results shed light on the limitations of the LLM chatbots when compared to specialized models, and can help researchers improve chatbots technology with the objective to reduce the required effort to integrate machine learning in OSINT-based CTI tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.14724v3">A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2024-04-19
    </div>
    <details class="paper-abstract">
      The powerful ability to understand, follow, and generate complex language emerging from large language models (LLMs) makes LLM-generated text flood many areas of our daily lives at an incredible speed and is widely accepted by humans. As LLMs continue to expand, there is an imperative need to develop detectors that can detect LLM-generated text. This is crucial to mitigate potential misuse of LLMs and safeguard realms like artistic expression and social networks from harmful influence of LLM-generated content. The LLM-generated text detection aims to discern if a piece of text was produced by an LLM, which is essentially a binary classification task. The detector techniques have witnessed notable advancements recently, propelled by innovations in watermarking techniques, statistics-based detectors, neural-base detectors, and human-assisted methods. In this survey, we collate recent research breakthroughs in this area and underscore the pressing need to bolster detector research. We also delve into prevalent datasets, elucidating their limitations and developmental requirements. Furthermore, we analyze various LLM-generated text detection paradigms, shedding light on challenges like out-of-distribution problems, potential attacks, real-world data issues and the lack of effective evaluation framework. Conclusively, we highlight interesting directions for future research in LLM-generated text detection to advance the implementation of responsible artificial intelligence (AI). Our aim with this survey is to provide a clear and comprehensive introduction for newcomers while also offering seasoned researchers a valuable update in the field of LLM-generated text detection. The useful resources are publicly available at: https://github.com/NLP2CT/LLM-generated-Text-Detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15883v2">HyKGE: A Hypothesis Knowledge Graph Enhanced Framework for Accurate and Reliable Medical LLMs Responses</a></div>
    <div class="paper-meta">
      📅 2024-04-19
      | 💬 version 2
    </div>
    <details class="paper-abstract">
      In this paper, we investigate the retrieval-augmented generation (RAG) based on Knowledge Graphs (KGs) to improve the accuracy and reliability of Large Language Models (LLMs). Recent approaches suffer from insufficient and repetitive knowledge retrieval, tedious and time-consuming query parsing, and monotonous knowledge utilization. To this end, we develop a Hypothesis Knowledge Graph Enhanced (HyKGE) framework, which leverages LLMs' powerful reasoning capacity to compensate for the incompleteness of user queries, optimizes the interaction process with LLMs, and provides diverse retrieved knowledge. Specifically, HyKGE explores the zero-shot capability and the rich knowledge of LLMs with Hypothesis Outputs to extend feasible exploration directions in the KGs, as well as the carefully curated prompt to enhance the density and efficiency of LLMs' responses. Furthermore, we introduce the HO Fragment Granularity-aware Rerank Module to filter out noise while ensuring the balance between diversity and relevance in retrieved knowledge. Experiments on two Chinese medical multiple-choice question datasets and one Chinese open-domain medical Q&A dataset with two LLM turbos demonstrate the superiority of HyKGE in terms of accuracy and explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04997v2">Adapting LLMs for Efficient Context Processing through Soft Prompt Compression</a></div>
    <div class="paper-meta">
      📅 2024-04-18
      | 💬 This paper has been accepted by the 2024 International Conference on Image Processing and Computer Applications (IPCA 2024)
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has inaugurated a transformative epoch in natural language processing, fostering unprecedented proficiency in text generation, comprehension, and contextual scrutiny. Nevertheless, effectively handling extensive contexts, crucial for myriad applications, poses a formidable obstacle owing to the intrinsic constraints of the models' context window sizes and the computational burdens entailed by their operations. This investigation presents an innovative framework that strategically tailors LLMs for streamlined context processing by harnessing the synergies among natural language summarization, soft prompt compression, and augmented utility preservation mechanisms. Our methodology, dubbed SoftPromptComp, amalgamates natural language prompts extracted from summarization methodologies with dynamically generated soft prompts to forge a concise yet semantically robust depiction of protracted contexts. This depiction undergoes further refinement via a weighting mechanism optimizing information retention and utility for subsequent tasks. We substantiate that our framework markedly diminishes computational overhead and enhances LLMs' efficacy across various benchmarks, while upholding or even augmenting the caliber of the produced content. By amalgamating soft prompt compression with sophisticated summarization, SoftPromptComp confronts the dual challenges of managing lengthy contexts and ensuring model scalability. Our findings point towards a propitious trajectory for augmenting LLMs' applicability and efficiency, rendering them more versatile and pragmatic for real-world applications. This research enriches the ongoing discourse on optimizing language models, providing insights into the potency of soft prompts and summarization techniques as pivotal instruments for the forthcoming generation of NLP solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05434v2">Cost-Performance Optimization for Processing Low-Resource Language Tasks Using Commercial LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive zero/few-shot inference and generation quality for high-resource languages (HRLs). A few of them have been trained on low-resource languages (LRLs) and give decent performance. Owing to the prohibitive costs of training LLMs, they are usually used as a network service, with the client charged by the count of input and output tokens. The number of tokens strongly depends on the script and language, as well as the LLM's subword vocabulary. We show that LRLs are at a pricing disadvantage, because the well-known LLMs produce more tokens for LRLs than HRLs. This is because most currently popular LLMs are optimized for HRL vocabularies. Our objective is to level the playing field: reduce the cost of processing LRLs in contemporary LLMs while ensuring that predictive and generative qualities are not compromised. As means to reduce the number of tokens processed by the LLM, we consider code-mixing, translation, and transliteration of LRLs to HRLs. We perform an extensive study using the IndicXTREME classification and six generative tasks dataset, covering 15 Indic and 3 other languages, while using GPT-4 (one of the costliest LLM services released so far) as a commercial LLM. We observe and analyze interesting patterns involving token count, cost, and quality across a multitude of languages and tasks. We show that choosing the best policy to interact with the LLM can reduce cost by 90% while giving better or comparable performance compared to communicating with the LLM in the original LRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12272v1">Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences</a></div>
    <div class="paper-meta">
      📅 2024-04-18
      | 💬 16 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Due to the cumbersome nature of human evaluation and limitations of code-based evaluation, Large Language Models (LLMs) are increasingly being used to assist humans in evaluating LLM outputs. Yet LLM-generated evaluators simply inherit all the problems of the LLMs they evaluate, requiring further human validation. We present a mixed-initiative approach to ``validate the validators'' -- aligning LLM-generated evaluation functions (be it prompts or code) with human requirements. Our interface, EvalGen, provides automated assistance to users in generating evaluation criteria and implementing assertions. While generating candidate implementations (Python functions, LLM grader prompts), EvalGen asks humans to grade a subset of LLM outputs; this feedback is used to select implementations that better align with user grades. A qualitative study finds overall support for EvalGen but underscores the subjectivity and iterative process of alignment. In particular, we identify a phenomenon we dub \emph{criteria drift}: users need criteria to grade outputs, but grading outputs helps users define criteria. What is more, some criteria appears \emph{dependent} on the specific LLM outputs observed (rather than independent criteria that can be defined \emph{a priori}), raising serious questions for approaches that assume the independence of evaluation from observation of model outputs. We present our interface and implementation details, a comparison of our algorithm with a baseline approach, and implications for the design of future LLM evaluation assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06598v2">CigaR: Cost-efficient Program Repair with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-18
    </div>
    <details class="paper-abstract">
      Large language models (LLM) have proven to be effective at automated program repair (APR). However, using LLMs can be costly, with companies invoicing users by the number of tokens. In this paper, we propose CigaR, the first LLM-based APR tool that focuses on minimizing the repair cost. CigaR works in two major steps: generating a first plausible patch and multiplying plausible patches. CigaR optimizes the prompts and the prompt setting to maximize the information given to LLMs using the smallest possible number of tokens. Our experiments on 429 bugs from the widely used Defects4J and HumanEval-Java datasets shows that CigaR reduces the token cost by 73%. On average, CigaR spends 127k tokens per bug while the baseline uses 467k tokens per bug. On the subset of bugs that are fixed by both, CigaR spends 20k per bug while the baseline uses 608k tokens, a cost saving of 96%. Our extensive experiments show that CigaR is a cost-effective LLM-based program repair tool that uses a low number of tokens to automatically generate patches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12192v1">Aligning Actions and Walking to LLM-Generated Textual Descriptions</a></div>
    <div class="paper-meta">
      📅 2024-04-18
      | 💬 Accepted at 2nd Workshop on Learning with Few or without Annotated Face, Body and Gesture Data
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various domains, including data augmentation and synthetic data generation. This work explores the use of LLMs to generate rich textual descriptions for motion sequences, encompassing both actions and walking patterns. We leverage the expressive power of LLMs to align motion representations with high-level linguistic cues, addressing two distinct tasks: action recognition and retrieval of walking sequences based on appearance attributes. For action recognition, we employ LLMs to generate textual descriptions of actions in the BABEL-60 dataset, facilitating the alignment of motion sequences with linguistic representations. In the domain of gait analysis, we investigate the impact of appearance attributes on walking patterns by generating textual descriptions of motion sequences from the DenseGait dataset using LLMs. These descriptions capture subtle variations in walking styles influenced by factors such as clothing choices and footwear. Our approach demonstrates the potential of LLMs in augmenting structured motion attributes and aligning multi-modal representations. The findings contribute to the advancement of comprehensive motion understanding and open up new avenues for leveraging LLMs in multi-modal alignment and data augmentation for motion analysis. We make the code publicly available at https://github.com/Radu1999/WalkAndText
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12010v1">ParaFusion: A Large-Scale LLM-Driven English Paraphrase Dataset Infused with High-Quality Lexical and Syntactic Diversity</a></div>
    <div class="paper-meta">
      📅 2024-04-18
    </div>
    <details class="paper-abstract">
      Paraphrase generation is a pivotal task in natural language processing (NLP). Existing datasets in the domain lack syntactic and lexical diversity, resulting in paraphrases that closely resemble the source sentences. Moreover, these datasets often contain hate speech and noise, and may unintentionally include non-English language sentences. This research introduces ParaFusion, a large-scale, high-quality English paraphrase dataset developed using Large Language Models (LLM) to address these challenges. ParaFusion augments existing datasets with high-quality data, significantly enhancing both lexical and syntactic diversity while maintaining close semantic similarity. It also mitigates the presence of hate speech and reduces noise, ensuring a cleaner and more focused English dataset. Results show that ParaFusion offers at least a 25% improvement in both syntactic and lexical diversity, measured across several metrics for each data source. The paper also aims to set a gold standard for paraphrase evaluation as it contains one of the most comprehensive evaluation strategies to date. The results underscore the potential of ParaFusion as a valuable resource for improving NLP applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11943v1">AgentCoord: Visually Exploring Coordination Strategy for LLM-based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2024-04-18
    </div>
    <details class="paper-abstract">
      The potential of automatic task-solving through Large Language Model (LLM)-based multi-agent collaboration has recently garnered widespread attention from both the research community and industry. While utilizing natural language to coordinate multiple agents presents a promising avenue for democratizing agent technology for general users, designing coordination strategies remains challenging with existing coordination frameworks. This difficulty stems from the inherent ambiguity of natural language for specifying the collaboration process and the significant cognitive effort required to extract crucial information (e.g. agent relationship, task dependency, result correspondence) from a vast amount of text-form content during exploration. In this work, we present a visual exploration framework to facilitate the design of coordination strategies in multi-agent collaboration. We first establish a structured representation for LLM-based multi-agent coordination strategy to regularize the ambiguity of natural language. Based on this structure, we devise a three-stage generation method that leverages LLMs to convert a user's general goal into an executable initial coordination strategy. Users can further intervene at any stage of the generation process, utilizing LLMs and a set of interactions to explore alternative strategies. Whenever a satisfactory strategy is identified, users can commence the collaboration and examine the visually enhanced execution result. We develop AgentCoord, a prototype interactive system, and conduct a formal user study to demonstrate the feasibility and effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11865v1">From Image to Video, what do we need in multimodal LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-04-18
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have demonstrated profound capabilities in understanding multimodal information, covering from Image LLMs to the more complex Video LLMs. Numerous studies have illustrated their exceptional cross-modal comprehension. Recently, integrating video foundation models with large language models to build a comprehensive video understanding system has been proposed to overcome the limitations of specific pre-defined vision tasks. However, the current advancements in Video LLMs tend to overlook the foundational contributions of Image LLMs, often opting for more complicated structures and a wide variety of multimodal data for pre-training. This approach significantly increases the costs associated with these methods.In response to these challenges, this work introduces an efficient method that strategically leverages the priors of Image LLMs, facilitating a resource-efficient transition from Image to Video LLMs. We propose RED-VILLM, a Resource-Efficient Development pipeline for Video LLMs from Image LLMs, which utilizes a temporal adaptation plug-and-play structure within the image fusion module of Image LLMs. This adaptation extends their understanding capabilities to include temporal information, enabling the development of Video LLMs that not only surpass baseline performances but also do so with minimal instructional data and training resources. Our approach highlights the potential for a more cost-effective and scalable advancement in multimodal models, effectively building upon the foundational work of Image LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11531v1">Pack of LLMs: Model Fusion at Test-Time via Perplexity Optimization</a></div>
    <div class="paper-meta">
      📅 2024-04-17
    </div>
    <details class="paper-abstract">
      Fusing knowledge from multiple Large Language Models (LLMs) can combine their diverse strengths to achieve improved performance on a given task. However, current fusion approaches either rely on learning-based fusers that do not generalize to new LLMs, or do not take into account how well each LLM understands the input. In this work, we study LLM fusion at test-time, which enables leveraging knowledge from arbitrary user-specified LLMs during inference. We introduce Pack of LLMs (PackLLM), an effective method for test-time fusion that leverages each LLM's expertise, given an input prompt. PackLLM performs model fusion by solving an optimization problem for determining each LLM's importance, so that perplexity over the input prompt is minimized. First, our simple PackLLM-sim variant validates that perplexity is a good indicator for measuring each LLM's expertise. Second, our PackLLM-opt variant approximately solves the perplexity minimization problem via a greedy algorithm. The derived importance weights are used to combine the LLMs during inference. We conduct experiments with over 100 total LLMs on a diverse set of tasks. Experimental results show that (i) perplexity is a reliable measure for LLM fusion, (ii) PackLLM outperforms test-time fusion baselines by 1.89% accuracy points, and (iii) PackLLM can leverage new LLMs to improve performance over learning-based fusion approaches by 3.92-11.94% accuracy points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.15964v5">ChatCAD+: Towards a Universal and Reliable Interactive CAD using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-17
      | 💬 Authors Zihao Zhao, Sheng Wang, Jinchen Gu, Yitao Zhu contributed equally to this work and should be considered co-first authors
    </div>
    <details class="paper-abstract">
      The integration of Computer-Aided Diagnosis (CAD) with Large Language Models (LLMs) presents a promising frontier in clinical applications, notably in automating diagnostic processes akin to those performed by radiologists and providing consultations similar to a virtual family doctor. Despite the promising potential of this integration, current works face at least two limitations: (1) From the perspective of a radiologist, existing studies typically have a restricted scope of applicable imaging domains, failing to meet the diagnostic needs of different patients. Also, the insufficient diagnostic capability of LLMs further undermine the quality and reliability of the generated medical reports. (2) Current LLMs lack the requisite depth in medical expertise, rendering them less effective as virtual family doctors due to the potential unreliability of the advice provided during patient consultations. To address these limitations, we introduce ChatCAD+, to be universal and reliable. Specifically, it is featured by two main modules: (1) Reliable Report Generation and (2) Reliable Interaction. The Reliable Report Generation module is capable of interpreting medical images from diverse domains and generate high-quality medical reports via our proposed hierarchical in-context learning. Concurrently, the interaction module leverages up-to-date information from reputable medical websites to provide reliable medical advice. Together, these designed modules synergize to closely align with the expertise of human medical professionals, offering enhanced consistency and reliability for interpretation and advice. The source code is available at https://github.com/zhaozh10/ChatCAD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15379v1">CS1-LLM: Integrating LLMs into CS1 Instruction</a></div>
    <div class="paper-meta">
      📅 2024-04-17
      | 💬 to be published in Proceedings of the 29th ACM conference on innovation and technology in computer science education (ITiCSE)
    </div>
    <details class="paper-abstract">
      The recent, widespread availability of Large Language Models (LLMs) like ChatGPT and GitHub Copilot may impact introductory programming courses (CS1) both in terms of what should be taught and how to teach it. Indeed, recent research has shown that LLMs are capable of solving the majority of the assignments and exams we previously used in CS1. In addition, professional software engineers are often using these tools, raising the question of whether we should be training our students in their use as well. This experience report describes a CS1 course at a large research-intensive university that fully embraces the use of LLMs from the beginning of the course. To incorporate the LLMs, the course was intentionally altered to reduce emphasis on syntax and writing code from scratch. Instead, the course now emphasizes skills needed to successfully produce software with an LLM. This includes explaining code, testing code, and decomposing large problems into small functions that are solvable by an LLM. In addition to frequent, formative assessments of these skills, students were given three large, open-ended projects in three separate domains (data science, image processing, and game design) that allowed them to showcase their creativity in topics of their choosing. In an end-of-term survey, students reported that they appreciated learning with the assistance of the LLM and that they interacted with the LLM in a variety of ways when writing code. We provide lessons learned for instructors who may wish to incorporate LLMs into their course.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11338v1">LLMs for Cyber Security: New Opportunities</a></div>
    <div class="paper-meta">
      📅 2024-04-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are a class of powerful and versatile models that are beneficial to many industries. With the emergence of LLMs, we take a fresh look at cyber security, specifically exploring and summarizing the potential of LLMs in addressing challenging problems in the security and safety domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11209v1">Prompt-Guided Generation of Structured Chest X-Ray Report Using a Pre-trained LLM</a></div>
    <div class="paper-meta">
      📅 2024-04-17
      | 💬 Accepted by IEEE Conference on Multimedia Expo 2024
    </div>
    <details class="paper-abstract">
      Medical report generation automates radiology descriptions from images, easing the burden on physicians and minimizing errors. However, current methods lack structured outputs and physician interactivity for clear, clinically relevant reports. Our method introduces a prompt-guided approach to generate structured chest X-ray reports using a pre-trained large language model (LLM). First, we identify anatomical regions in chest X-rays to generate focused sentences that center on key visual elements, thereby establishing a structured report foundation with anatomy-based sentences. We also convert the detected anatomy into textual prompts conveying anatomical comprehension to the LLM. Additionally, the clinical context prompts guide the LLM to emphasize interactivity and clinical requirements. By integrating anatomy-focused sentences and anatomy/clinical prompts, the pre-trained LLM can generate structured chest X-ray reports tailored to prompted anatomical regions and clinical contexts. We evaluate using language generation and clinical effectiveness metrics, demonstrating strong performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08144v2">LLM Agents can Autonomously Exploit One-day Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2024-04-17
    </div>
    <details class="paper-abstract">
      LLMs have becoming increasingly powerful, both in their benign and malicious uses. With the increase in capabilities, researchers have been increasingly interested in their ability to exploit cybersecurity vulnerabilities. In particular, recent work has conducted preliminary studies on the ability of LLM agents to autonomously hack websites. However, these studies are limited to simple vulnerabilities. In this work, we show that LLM agents can autonomously exploit one-day vulnerabilities in real-world systems. To show this, we collected a dataset of 15 one-day vulnerabilities that include ones categorized as critical severity in the CVE description. When given the CVE description, GPT-4 is capable of exploiting 87% of these vulnerabilities compared to 0% for every other model we test (GPT-3.5, open-source LLMs) and open-source vulnerability scanners (ZAP and Metasploit). Fortunately, our GPT-4 agent requires the CVE description for high performance: without the description, GPT-4 can exploit only 7% of the vulnerabilities. Our findings raise questions around the widespread deployment of highly capable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13081v1">SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-17
      | 💬 Accepted at ICLR 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant advancements in various natural language processing tasks, including question answering (QA) tasks. While incorporating new information with the retrieval of relevant passages is a promising way to improve QA with LLMs, the existing methods often require additional fine-tuning which becomes infeasible with recent LLMs. Augmenting retrieved passages via prompting has the potential to address this limitation, but this direction has been limitedly explored. To this end, we design a simple yet effective framework to enhance open-domain QA (ODQA) with LLMs, based on the summarized retrieval (SuRe). SuRe helps LLMs predict more accurate answers for a given question, which are well-supported by the summarized retrieval that could be viewed as an explicit rationale extracted from the retrieved passages. Specifically, SuRe first constructs summaries of the retrieved passages for each of the multiple answer candidates. Then, SuRe confirms the most plausible answer from the candidate set by evaluating the validity and ranking of the generated summaries. Experimental results on diverse ODQA benchmarks demonstrate the superiority of SuRe, with improvements of up to 4.6% in exact match (EM) and 4.0% in F1 score over standard prompting approaches. SuRe also can be integrated with a broad range of retrieval methods and LLMs. Finally, the generated summaries from SuRe show additional advantages to measure the importance of retrieved passages and serve as more preferred rationales by models and humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10960v1">Uncertainty-Based Abstention in LLMs Improves Safety and Reduces Hallucinations</a></div>
    <div class="paper-meta">
      📅 2024-04-16
    </div>
    <details class="paper-abstract">
      A major barrier towards the practical deployment of large language models (LLMs) is their lack of reliability. Three situations where this is particularly apparent are correctness, hallucinations when given unanswerable questions, and safety. In all three cases, models should ideally abstain from responding, much like humans, whose ability to understand uncertainty makes us refrain from answering questions we don't know. Inspired by analogous approaches in classification, this study explores the feasibility and efficacy of abstaining while uncertain in the context of LLMs within the domain of question-answering. We investigate two kinds of uncertainties, statistical uncertainty metrics and a distinct verbalized measure, termed as In-Dialogue Uncertainty (InDU). Using these uncertainty measures combined with models with and without Reinforcement Learning with Human Feedback (RLHF), we show that in all three situations, abstention based on the right kind of uncertainty measure can boost the reliability of LLMs. By sacrificing only a few highly uncertain samples we can improve correctness by 2% to 8%, avoid 50% hallucinations via correctly identifying unanswerable questions and increase safety by 70% up to 99% with almost no additional computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.03740v2">A Survey on Prompting Techniques in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-16
      | 💬 10 pages, 4 Figures
    </div>
    <details class="paper-abstract">
      Autoregressive Large Language Models have transformed the landscape of Natural Language Processing. Pre-train and prompt paradigm has replaced the conventional approach of pre-training and fine-tuning for many downstream NLP tasks. This shift has been possible largely due to LLMs and innovative prompting techniques. LLMs have shown great promise for a variety of downstream tasks owing to their vast parameters and huge datasets that they are pre-trained on. However, in order to fully realize their potential, their outputs must be guided towards the desired outcomes. Prompting, in which a specific input or instruction is provided to guide the LLMs toward the intended output, has become a tool for achieving this goal. In this paper, we discuss the various prompting techniques that have been applied to fully harness the power of LLMs. We present a taxonomy of existing literature on prompting techniques and provide a concise survey based on this taxonomy. Further, we identify some open problems in the realm of prompting in autoregressive LLMs which could serve as a direction for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10933v1">LLMem: Estimating GPU Memory Usage for Fine-Tuning Pre-Trained LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-16
      | 💬 9 pages, 9 figures, accepted to IJCAI 2024
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained large language models (LLMs) with limited hardware presents challenges due to GPU memory constraints. Various distributed fine-tuning methods have been proposed to alleviate memory constraints on GPU. However, determining the most effective method for achieving rapid fine-tuning while preventing GPU out-of-memory issues in a given environment remains unclear. To address this challenge, we introduce LLMem, a solution that estimates the GPU memory consumption when applying distributed fine-tuning methods across multiple GPUs and identifies the optimal method. We conduct GPU memory usage estimation prior to fine-tuning, leveraging the fundamental structure of transformer-based decoder models and the memory usage distribution of each method. Experimental results show that LLMem accurately estimates peak GPU memory usage on a single GPU, with error rates of up to 1.6%. Additionally, it shows an average error rate of 3.0% when applying distributed fine-tuning methods to LLMs with more than a billion parameters on multi-GPU setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01132v4">Automated Evaluation of Classroom Instructional Support with LLMs and BoWs: Connecting Global Predictions to Specific Feedback</a></div>
    <div class="paper-meta">
      📅 2024-04-16
    </div>
    <details class="paper-abstract">
      With the aim to provide teachers with more specific, frequent, and actionable feedback about their teaching, we explore how Large Language Models (LLMs) can be used to estimate ``Instructional Support'' domain scores of the CLassroom Assessment Scoring System (CLASS), a widely used observation protocol. We design a machine learning architecture that uses either zero-shot prompting of Meta's Llama2, and/or a classic Bag of Words (BoW) model, to classify individual utterances of teachers' speech (transcribed automatically using OpenAI's Whisper) for the presence of Instructional Support. Then, these utterance-level judgments are aggregated over a 15-min observation session to estimate a global CLASS score. Experiments on two CLASS-coded datasets of toddler and pre-kindergarten classrooms indicate that (1) automatic CLASS Instructional Support estimation accuracy using the proposed method (Pearson $R$ up to $0.48$) approaches human inter-rater reliability (up to $R=0.55$); (2) LLMs generally yield slightly greater accuracy than BoW for this task, though the best models often combined features extracted from both LLM and BoW; and (3) for classifying individual utterances, there is still room for improvement of automated methods compared to human-level judgments. Finally, (4) we illustrate how the model's outputs can be visualized at the utterance level to provide teachers with explainable feedback on which utterances were most positively or negatively correlated with specific CLASS dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10896v1">From a Lossless (~1.5:1) Compression Algorithm for Llama2 7B Weights to Variable Precision, Variable Range, Compressed Numeric Data Types for CNNs and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-16
    </div>
    <details class="paper-abstract">
      This paper starts with a simple lossless ~1.5:1 compression algorithm for the weights of the Large Language Model (LLM) Llama2 7B [1] that can be implemented in ~200 LUTs in AMD FPGAs, processing over 800 million bfloat16 numbers per second. This framework is then extended to variable precision, variable range, compressed numerical data types that are a user defined super set of both floats and posits [2]. The paper then discusses a simple hardware implementation of such format based on ANS (Asymmetrical Numeral Systems) [3] that acts as a bridge between this flexible data format and a computational engine while, at the same time, achieving bandwidth reduction. An example of a token factory using weight compression and sharing is also given.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.14972v3">Scaling Down to Scale Up: A Cost-Benefit Analysis of Replacing OpenAI's LLM with Open Source SLMs in Production</a></div>
    <div class="paper-meta">
      📅 2024-04-16
      | 💬 Updated title, Revised content
    </div>
    <details class="paper-abstract">
      Many companies use large language models (LLMs) offered as a service, like OpenAI's GPT-4, to create AI-enabled product experiences. Along with the benefits of ease-of-use and shortened time-to-solution, this reliance on proprietary services has downsides in model control, performance reliability, uptime predictability, and cost. At the same time, a flurry of open-source small language models (SLMs) has been made available for commercial use. However, their readiness to replace existing capabilities remains unclear, and a systematic approach to holistically evaluate these SLMs is not readily available. This paper presents a systematic evaluation methodology and a characterization of modern open-source SLMs and their trade-offs when replacing proprietary LLMs for a real-world product feature. We have designed SLaM, an open-source automated analysis tool that enables the quantitative and qualitative testing of product features utilizing arbitrary SLMs. Using SLaM, we examine the quality and performance characteristics of modern SLMs relative to an existing customer-facing implementation using the OpenAI GPT-4 API. Across 9 SLMs and their 29 variants, we observe that SLMs provide competitive results, significant performance consistency improvements, and a cost reduction of 5x~29x when compared to GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10757v1">Deep Learning and LLM-based Methods Applied to Stellar Lightcurve Classification</a></div>
    <div class="paper-meta">
      📅 2024-04-16
      | 💬 35 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Light curves serve as a valuable source of information on stellar formation and evolution. With the rapid advancement of machine learning techniques, it can be effectively processed to extract astronomical patterns and information. In this study, we present a comprehensive evaluation of deep-learning and large language model (LLM) based models for the automatic classification of variable star light curves, based on large datasets from the Kepler and K2 missions. Special emphasis is placed on Cepheids, RR Lyrae, and eclipsing binaries, examining the influence of observational cadence and phase distribution on classification precision. Employing AutoDL optimization, we achieve striking performance with the 1D-Convolution+BiLSTM architecture and the Swin Transformer, hitting accuracies of 94\% and 99\% correspondingly, with the latter demonstrating a notable 83\% accuracy in discerning the elusive Type II Cepheids-comprising merely 0.02\% of the total dataset.We unveil StarWhisper LightCurve (LC), an innovative Series comprising three LLM-based models: LLM, multimodal large language model (MLLM), and Large Audio Language Model (LALM). Each model is fine-tuned with strategic prompt engineering and customized training methods to explore the emergent abilities of these models for astronomical data. Remarkably, StarWhisper LC Series exhibit high accuracies around 90\%, significantly reducing the need for explicit feature engineering, thereby paving the way for streamlined parallel data processing and the progression of multifaceted multimodal models in astronomical applications. The study furnishes two detailed catalogs illustrating the impacts of phase and sampling intervals on deep learning classification accuracy, showing that a substantial decrease of up to 14\% in observation duration and 21\% in sampling points can be realized without compromising accuracy by more than 10\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10678v1">Automating REST API Postman Test Cases Using LLM</a></div>
    <div class="paper-meta">
      📅 2024-04-16
    </div>
    <details class="paper-abstract">
      In the contemporary landscape of technological advancements, the automation of manual processes is crucial, compelling the demand for huge datasets to effectively train and test machines. This research paper is dedicated to the exploration and implementation of an automated approach to generate test cases specifically using Large Language Models. The methodology integrates the use of Open AI to enhance the efficiency and effectiveness of test case generation for training and evaluating Large Language Models. This formalized approach with LLMs simplifies the testing process, making it more efficient and comprehensive. Leveraging natural language understanding, LLMs can intelligently formulate test cases that cover a broad range of REST API properties, ensuring comprehensive testing. The model that is developed during the research is trained using manually collected postman test cases or instances for various Rest APIs. LLMs enhance the creation of Postman test cases by automating the generation of varied and intricate test scenarios. Postman test cases offer streamlined automation, collaboration, and dynamic data handling, providing a user-friendly and efficient approach to API testing compared to traditional test cases. Thus, the model developed not only conforms to current technological standards but also holds the promise of evolving into an idea of substantial importance in future technological advancements.
    </details>
</div>
