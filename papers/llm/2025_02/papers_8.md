# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12372v3">Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information. In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve strong performances on various benchmark datasets without finetuning and expensive self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18649v2">LeDex: Training LLMs to Better Self-Debug and Explain Code</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 This paper is accepted by The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)
    </div>
    <details class="paper-abstract">
      In the domain of code generation, self-debugging is crucial. It allows LLMs to refine their generated code based on execution feedback. This is particularly important because generating correct solutions in one attempt proves challenging for complex tasks. Prior works on self-debugging mostly focus on prompting methods by providing LLMs with few-shot examples, which work poorly on small open-sourced LLMs. In this work, we propose LeDex, a training framework that significantly improves the self-debugging capability of LLMs. Intuitively, we observe that a chain of explanations on the wrong code followed by code refinement helps LLMs better analyze the wrong code and do refinement. We thus propose an automated pipeline to collect a high-quality dataset for code explanation and refinement by generating a number of explanations and refinement trajectories from the LLM itself or a larger teacher model and filtering via execution verification. We perform supervised fine-tuning (SFT) and further reinforcement learning (RL) on both success and failure trajectories with a novel reward design considering code explanation and refinement quality. SFT improves the pass@1 by up to 15.92% and pass@10 by 9.30% over four benchmarks. RL training brings additional up to 3.54% improvement on pass@1 and 2.55% improvement on pass@10. The trained LLMs show iterative refinement ability and can keep refining code continuously. Lastly, our human evaluation shows that the LLMs trained with our framework generate more useful code explanations and help developers better understand bugs in source code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09819v1">A Solver-Aided Hierarchical Language for LLM-Driven CAD Design</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been enormously successful in solving a wide variety of structured and unstructured generative tasks, but they struggle to generate procedural geometry in Computer Aided Design (CAD). These difficulties arise from an inability to do spatial reasoning and the necessity to guide a model through complex, long range planning to generate complex geometry. We enable generative CAD Design with LLMs through the introduction of a solver-aided, hierarchical domain specific language (DSL) called AIDL, which offloads the spatial reasoning requirements to a geometric constraint solver. Additionally, we show that in the few-shot regime, AIDL outperforms even a language with in-training data (OpenSCAD), both in terms of generating visual results closer to the prompt and creating objects that are easier to post-process and reason about.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09801v1">Unit Testing Past vs. Present: Examining LLMs' Impact on Defect Detection and Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs), such as ChatGPT and GitHub Copilot, into software engineering workflows has shown potential to enhance productivity, particularly in software testing. This paper investigates whether LLM support improves defect detection effectiveness during unit testing. Building on prior studies comparing manual and tool-supported testing, we replicated and extended an experiment where participants wrote unit tests for a Java-based system with seeded defects within a time-boxed session, supported by LLMs. Comparing LLM supported and manual testing, results show that LLM support significantly increases the number of unit tests generated, defect detection rates, and overall testing efficiency. These findings highlight the potential of LLMs to improve testing and defect detection outcomes, providing empirical insights into their practical application in software testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09797v1">A Survey on LLM-based News Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      News recommender systems play a critical role in mitigating the information overload problem. In recent years, due to the successful applications of large language model technologies, researchers have utilized Discriminative Large Language Models (DLLMs) or Generative Large Language Models (GLLMs) to improve the performance of news recommender systems. Although several recent surveys review significant challenges for deep learning-based news recommender systems, such as fairness, privacy-preserving, and responsibility, there is a lack of a systematic survey on Large Language Model (LLM)-based news recommender systems. In order to review different core methodologies and explore potential issues systematically, we categorize DLLM-based and GLLM-based news recommender systems under the umbrella of LLM-based news recommender systems. In this survey, we first overview the development of deep learning-based news recommender systems. Then, we review LLM-based news recommender systems based on three aspects: news-oriented modeling, user-oriented modeling, and prediction-oriented modeling. Next, we examine the challenges from various perspectives, including datasets, benchmarking tools, and methodologies. Furthermore, we conduct extensive experiments to analyze how large language model technologies affect the performance of different news recommender systems. Finally, we comprehensively explore the future directions for LLM-based news recommendations in the era of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09778v1">Prompt and circumstance: A word-by-word LLM prompting approach to interlinear glossing for low-resource languages</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Partly automated creation of interlinear glossed text (IGT) has the potential to assist in linguistic documentation. We argue that LLMs can make this process more accessible to linguists because of their capacity to follow natural-language instructions. We investigate the effectiveness of a retrieval-based LLM prompting approach to glossing, applied to the seven languages from the SIGMORPHON 2023 shared task. Our system beats the BERT-based shared task baseline for every language in the morpheme-level score category, and we show that a simple 3-best oracle has higher word-level scores than the challenge winner (a tuned sequence model) in five languages. In a case study on Tsez, we ask the LLM to automatically create and follow linguistic instructions, reducing errors on a confusing grammatical feature. Our results thus demonstrate the potential contributions which LLMs can make in interactive systems for glossing, both in making suggestions to human annotators and following directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09766v1">LLM-Generated Microservice Implementations from RESTful API Definitions</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The growing need for scalable, maintainable, and fast-deploying systems has made microservice architecture widely popular in software development. This paper presents a system that uses Large Language Models (LLMs) to automate the API-first development of RESTful microservices. This system assists in creating OpenAPI specification, generating server code from it, and refining the code through a feedback loop that analyzes execution logs and error messages. By focusing on the API-first methodology, this system ensures that microservices are designed with well-defined interfaces, promoting consistency and reliability across the development life-cycle. The integration of log analysis enables the LLM to detect and address issues efficiently, reducing the number of iterations required to produce functional and robust services. This process automates the generation of microservices and also simplifies the debugging and refinement phases, allowing developers to focus on higher-level design and integration tasks. This system has the potential to benefit software developers, architects, and organizations to speed up software development cycles and reducing manual effort. To assess the potential of the system, we conducted surveys with six industry practitioners. After surveying practitioners, the system demonstrated notable advantages in enhancing development speed, automating repetitive tasks, and simplifying the prototyping process. While experienced developers appreciated its efficiency for specific tasks, some expressed concerns about its limitations in handling advanced customizations and larger scale projects. The code is publicly available at https://github.com/sirbh/code-gen
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00735v2">`Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09749v1">Vote-Tree-Planner: Optimizing Execution Order in LLM-based Task Planning Pipeline via Voting</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted to RSS24-W: TaskSpec
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into closed-loop robotic task planning has become increasingly popular within embodied artificial intelligence. Previous efforts mainly focused on leveraging the strong reasoning abilities of LLMs to enhance task planning performance while often overlooking task planning efficiency and executability due to repetitive queries to LLMs. This paper addresses the synergy between LLMs and task planning systems, aiming to minimize redundancy while enhancing planning effectiveness. Specifically, building upon Prog-Prompt and the high-level concept of Tree-Planner, we propose Vote-Tree-Planner. This sampling strategy utilizes votes to guide plan traversal during the decision-making process. Our approach is motivated by a straightforward observation: assigning weights to agents during decision-making enables the evaluation of critical paths before execution. With this simple vote-tree construction, our method further improves the success rate and reduces the number of queries to LLMs. The experimental results highlight that our Vote-Tree-Planner demonstrates greater stability and shows a higher average success rate and goal condition recall on the unseen dataset compared with previous baseline methods. These findings underscore the potential of the Vote-Tree-Planner to enhance planning accuracy, reliability, and efficiency in LLM-based planning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09720v1">NestQuant: Nested Lattice Quantization for Matrix Products and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) has emerged as a critical technique for efficient deployment of large language models (LLMs). This work proposes NestQuant, a novel PTQ scheme for weights and activations that is based on self-similar nested lattices. Recent work have mathematically shown such quantizers to be information-theoretically optimal for low-precision matrix multiplication. We implement a practical low-complexity version of NestQuant based on Gosset lattice, making it a drop-in quantizer for any matrix multiplication step (e.g., in self-attention, MLP etc). For example, NestQuant quantizes weights, KV-cache, and activations of Llama-3-8B to 4 bits, achieving perplexity of 6.6 on wikitext2. This represents more than 55% reduction in perplexity gap with respect to unquantized model (perplexity of 6.14) compared to state-of-the-art Meta's SpinQuant (perplexity 7.3). Comparisons on various LLM evaluation benchmarks also show a reduction in performance degradation induced by quantization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09673v1">Are Smarter LLMs Safer? Exploring Safety-Reasoning Trade-offs in Prompting and Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success across various NLP benchmarks. However, excelling in complex tasks that require nuanced reasoning and precise decision-making demands more than raw language proficiency--LLMs must reason, i.e., think logically, draw from past experiences, and synthesize information to reach conclusions and take action. To enhance reasoning abilities, approaches such as prompting and fine-tuning have been widely explored. While these methods have led to clear improvements in reasoning, their impact on LLM safety remains less understood. In this work, we investigate the interplay between reasoning and safety in LLMs. We highlight the latent safety risks that arise as reasoning capabilities improve, shedding light on previously overlooked vulnerabilities. At the same time, we explore how reasoning itself can be leveraged to enhance safety, uncovering potential mitigation strategies. By examining both the risks and opportunities in reasoning-driven LLM safety, our study provides valuable insights for developing models that are not only more capable but also more trustworthy in real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09606v1">Human-LLM Coevolution: Evidence from Academic Writing</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      With a statistical analysis of arXiv paper abstracts, we report a marked drop in the frequency of several words previously identified as overused by ChatGPT, such as "delve", starting soon after they were pointed out in early 2024. The frequency of certain other words favored by ChatGPT, such as "significant", has instead kept increasing. These phenomena suggest that some authors of academic papers have adapted their use of large language models (LLMs), for example, by selecting outputs or applying modifications to the LLM-generated content. Such coevolution and cooperation of humans and LLMs thus introduce additional challenges to the detection of machine-generated text in real-world scenarios. Estimating the impact of LLMs on academic writing by examining word frequency remains feasible, and more attention should be paid to words that were already frequently employed, including those that have decreased in frequency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09597v1">Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted at ICLR 2025 as oral presentation. Code and data at: https://prefeval.github.io/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as chatbots, yet their ability to personalize responses to user preferences remains limited. We introduce PrefEval, a benchmark for evaluating LLMs' ability to infer, memorize and adhere to user preferences in a long-context conversational setting. PrefEval comprises 3,000 manually curated user preference and query pairs spanning 20 topics. PrefEval contains user personalization or preference information in both explicit and implicit forms, and evaluates LLM performance using a generation and a classification task. With PrefEval, we evaluated the aforementioned preference following capabilities of 10 open-source and proprietary LLMs in multi-session conversations with varying context lengths up to 100k tokens. We benchmark with various prompting, iterative feedback, and retrieval-augmented generation methods. Our benchmarking effort reveals that state-of-the-art LLMs face significant challenges in proactively following users' preferences during conversations. In particular, in zero-shot settings, preference following accuracy falls below 10% at merely 10 turns (~3k tokens) across most evaluated models. Even with advanced prompting and retrieval methods, preference following still deteriorates in long-context conversations. Furthermore, we show that fine-tuning on PrefEval significantly improves performance. We believe PrefEval serves as a valuable resource for measuring, understanding, and enhancing LLMs' preference following abilities, paving the way for personalized conversational agents. Our code and dataset are available at https://prefeval.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05925v2">Hello Again! LLM-powered Personalized Agent for Long-term Dialogue</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Open-domain dialogue systems have seen remarkable advancements with the development of large language models (LLMs). Nonetheless, most existing dialogue systems predominantly focus on brief single-session interactions, neglecting the real-world demands for long-term companionship and personalized interactions with chatbots. Crucial to addressing this real-world need are event summary and persona management, which enable reasoning for appropriate long-term dialogue responses. Recent progress in the human-like cognitive and reasoning capabilities of LLMs suggests that LLM-based agents could significantly enhance automated perception, decision-making, and problem-solving. In response to this potential, we introduce a model-agnostic framework, the Long-term Dialogue Agent (LD-Agent), which incorporates three independently tunable modules dedicated to event perception, persona extraction, and response generation. For the event memory module, long and short-term memory banks are employed to separately focus on historical and ongoing sessions, while a topic-based retrieval mechanism is introduced to enhance the accuracy of memory retrieval. Furthermore, the persona module conducts dynamic persona modeling for both users and agents. The integration of retrieved memories and extracted personas is subsequently fed into the generator to induce appropriate responses. The effectiveness, generality, and cross-domain capabilities of LD-Agent are empirically demonstrated across various illustrative benchmarks, models, and tasks. The code is released at https://github.com/leolee99/LD-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06773v2">Evaluating Zero-Shot Long-Context LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      This study evaluates the effectiveness of zero-shot compression techniques on large language models (LLMs) under long-context. We identify the tendency for computational errors to increase under long-context when employing certain compression methods. We propose a hypothesis to explain the varied behavior of different LLM compression techniques and explore remedies to mitigate the performance decline observed in some techniques under long-context. This is a course report for COS 598D Machine Learning and Systems by Prof. Kai Li at Princeton University. Due to limited computational resources, our experiments were conducted only on LLaMA-2-7B-32K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09532v1">Mind the Gap! Choice Independence in Using Multilingual LLMs for Persuasive Co-Writing Tasks in Different Languages</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Recent advances in generative AI have precipitated a proliferation of novel writing assistants. These systems typically rely on multilingual large language models (LLMs), providing globalized workers the ability to revise or create diverse forms of content in different languages. However, there is substantial evidence indicating that the performance of multilingual LLMs varies between languages. Users who employ writing assistance for multiple languages are therefore susceptible to disparate output quality. Importantly, recent research has shown that people tend to generalize algorithmic errors across independent tasks, violating the behavioral axiom of choice independence. In this paper, we analyze whether user utilization of novel writing assistants in a charity advertisement writing task is affected by the AI's performance in a second language. Furthermore, we quantify the extent to which these patterns translate into the persuasiveness of generated charity advertisements, as well as the role of peoples' beliefs about LLM utilization in their donation choices. Our results provide evidence that writers who engage with an LLM-based writing assistant violate choice independence, as prior exposure to a Spanish LLM reduces subsequent utilization of an English LLM. While these patterns do not affect the aggregate persuasiveness of the generated advertisements, people's beliefs about the source of an advertisement (human versus AI) do. In particular, Spanish-speaking female participants who believed that they read an AI-generated advertisement strongly adjusted their donation behavior downwards. Furthermore, people are generally not able to adequately differentiate between human-generated and LLM-generated ads. Our work has important implications for the design, development, integration, and adoption of multilingual LLMs as assistive agents -- particularly in writing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05331v2">Fine-Tuned LLMs are "Time Capsules" for Tracking Societal Bias Through Books</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 9 pages (excluding references), accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Books, while often rich in cultural insights, can also mirror societal biases of their eras - biases that Large Language Models (LLMs) may learn and perpetuate during training. We introduce a novel method to trace and quantify these biases using fine-tuned LLMs. We develop BookPAGE, a corpus comprising 593 fictional books across seven decades (1950-2019), to track bias evolution. By fine-tuning LLMs on books from each decade and using targeted prompts, we examine shifts in biases related to gender, sexual orientation, race, and religion. Our findings indicate that LLMs trained on decade-specific books manifest biases reflective of their times, with both gradual trends and notable shifts. For example, model responses showed a progressive increase in the portrayal of women in leadership roles (from 8% to 22%) from the 1950s to 2010s, with a significant uptick in the 1990s (from 4% to 12%), possibly aligning with third-wave feminism. Same-sex relationship references increased markedly from the 1980s to 2000s (from 0% to 10%), mirroring growing LGBTQ+ visibility. Concerningly, negative portrayals of Islam rose sharply in the 2000s (26% to 38%), likely reflecting post-9/11 sentiments. Importantly, we demonstrate that these biases stem mainly from the books' content and not the models' architecture or initial training. Our study offers a new perspective on societal bias trends by bridging AI, literary studies, and social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09497v1">Improve LLM-based Automatic Essay Scoring with Linguistic Features</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 To be published in the workshop Innovation and Responsibility in AI-Supported Education (iRaise) at the 2025 Conference on Artificial Intelligence (AAAI)
    </div>
    <details class="paper-abstract">
      Automatic Essay Scoring (AES) assigns scores to student essays, reducing the grading workload for instructors. Developing a scoring system capable of handling essays across diverse prompts is challenging due to the flexibility and diverse nature of the writing task. Existing methods typically fall into two categories: supervised feature-based approaches and large language model (LLM)-based methods. Supervised feature-based approaches often achieve higher performance but require resource-intensive training. In contrast, LLM-based methods are computationally efficient during inference but tend to suffer from lower performance. This paper combines these approaches by incorporating linguistic features into LLM-based scoring. Experimental results show that this hybrid method outperforms baseline models for both in-domain and out-of-domain writing prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00326v9">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 19 pages, 12 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09419v1">On multi-token prediction for efficient LLM inference</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      We systematically investigate multi-token prediction (MTP) capabilities within LLMs pre-trained for next-token prediction (NTP). We first show that such models inherently possess MTP capabilities via numerical marginalization over intermediate token probabilities, though performance is data-dependent and improves with model scale. Furthermore, we explore the challenges of integrating MTP heads into frozen LLMs and find that their hidden layers are strongly specialized for NTP, making adaptation non-trivial. Finally, we show that while joint training of MTP heads with the backbone improves performance, it cannot fully overcome this barrier, prompting further research in this direction. Our findings provide a deeper understanding of MTP applied to pretrained LLMs, informing strategies for accelerating inference through parallel token prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02280v2">The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 NAACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable capabilities on not just language tasks, but also various tasks that are not linguistic in nature, such as logical reasoning and social inference. In the human brain, neuroscience has identified a core language system that selectively and causally supports language processing. We here ask whether similar specialization for language emerges in LLMs. We identify language-selective units within 18 popular LLMs, using the same localization approach that is used in neuroscience. We then establish the causal role of these units by demonstrating that ablating LLM language-selective units -- but not random units -- leads to drastic deficits in language tasks. Correspondingly, language-selective LLM units are more aligned to brain recordings from the human language system than random units. Finally, we investigate whether our localization method extends to other cognitive domains: while we find specialized networks in some LLMs for reasoning and social capabilities, there are substantial differences among models. These findings provide functional and causal evidence for specialization in large language models, and highlight parallels with the functional organization in the brain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04708v2">Exploring Hierarchical Molecular Graph Representation in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 9 pages, 4 tables, 1 figure, paper under review
    </div>
    <details class="paper-abstract">
      Following the milestones in large language models (LLMs) and multimodal models, we have seen a surge in applying LLMs to biochemical tasks. Leveraging graph features and molecular text representations, LLMs can tackle various tasks, such as predicting chemical reaction outcomes and describing molecular properties. However, most current work overlooks the *multi-level nature* of the graph modality, even though different chemistry tasks may benefit from different feature levels. In this work, we first study the effect of feature granularity and reveal that even reducing all GNN-generated feature tokens to a single one does not significantly impact model performance. We then investigate the effect of various graph feature levels and demonstrate that both the quality of LLM-generated molecules and model performance across different tasks depend on different graph feature levels. Therefore, we conclude with two key insights: (1) current molecular-related multimodal LLMs lack a comprehensive understanding of graph features, and (2) static processing is not sufficient for hierarchical graph feature. We share our findings in detail, with the hope of paving the way for the community to develop more advanced multimodal LLMs for incorporating molecular graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09334v1">ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 MLSys 2025
    </div>
    <details class="paper-abstract">
      Recent developments in large language models (LLMs) have demonstrated their remarkable proficiency in a range of tasks. Compared to in-house homogeneous GPU clusters, deploying LLMs in cloud environments with diverse types of GPUs is crucial for addressing the GPU shortage problem and being more cost-effective. However, the diversity of network environments and various GPU types on the cloud bring difficulties to achieving high-performance serving. In this work, we propose ThunderServe, a high-performance and cost-efficient LLM serving system for heterogeneous cloud environments. We introduce a novel scheduling algorithm, which optimizes the deployment plan of LLM serving to accommodate the heterogeneous resource and network bandwidth conditions in cloud environments. Furthermore, we propose a lightweight re-scheduling mechanism, designed to adapt to fluctuating online conditions (e.g., node failures, workload shifts) without the need for costly restarts of ongoing services. Empirical results in both heterogeneous cloud and homogeneous in-house environments reveal that ThunderServe delivers up to a 2.1$\times$ and on average a $1.7\times$ increase in throughput and achieves up to a 2.5$\times$ and on average a $1.5\times$ reduction in latency deadlines compared with state-of-the-art systems given the same price budget, suggesting opting for cloud services provides a more cost-efficient solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09331v1">Beyond English: The Impact of Prompt Translation Strategies across Languages and Tasks in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted for NAACL findings 2025
    </div>
    <details class="paper-abstract">
      Despite advances in the multilingual capabilities of Large Language Models (LLMs) across diverse tasks, English remains the dominant language for LLM research and development. So, when working with a different language, this has led to the widespread practice of pre-translation, i.e., translating the task prompt into English before inference. Selective pre-translation, a more surgical approach, focuses on translating specific prompt components. However, its current use is sporagic and lacks a systematic research foundation. Consequently, the optimal pre-translation strategy for various multilingual settings and tasks remains unclear. In this work, we aim to uncover the optimal setup for pre-translation by systematically assessing its use. Specifically, we view the prompt as a modular entity, composed of four functional parts: instruction, context, examples, and output, either of which could be translated or not. We evaluate pre-translation strategies across 35 languages covering both low and high-resource languages, on various tasks including Question Answering (QA), Natural Language Inference (NLI), Named Entity Recognition (NER), and Abstractive Summarization. Our experiments show the impact of factors as similarity to English, translation quality and the size of pre-trained data, on the model performance with pre-translation. We suggest practical guidelines for choosing optimal strategies in various multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09328v1">Copilot Arena: A Platform for Code LLM Evaluation in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Evaluating in-the-wild coding capabilities of large language models (LLMs) is a challenging endeavor with no clear solution. We introduce Copilot Arena, a platform to collect user preferences for code generation through native integration into a developer's working environment. Copilot Arena comprises a novel interface for comparing pairs of model outputs, a sampling strategy optimized to reduce latency, and a prompting scheme to enable code completion functionality. Copilot Arena has served over 4.5 million suggestions from 10 models and collected over 11k pairwise judgements. Our results highlight the importance of model evaluations in integrated settings. We find that model rankings from Copilot Arena differ from those of existing evaluations, which we attribute to the more realistic distribution of data and tasks contained in Copilot Arena. We also identify novel insights into human preferences on code such as an observed consistency in user preference across programming languages yet significant variation in preference due to task category. We open-source Copilot Arena and release data to enable human-centric evaluations and improve understanding of coding assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09316v1">A Judge-free LLM Open-ended Generation Benchmark Based on the Distributional Hypothesis</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Evaluating the open-ended text generation of large language models (LLMs) is challenging because of the lack of a clear ground truth and the high cost of human or LLM-based assessments. We propose a novel benchmark that evaluates LLMs using n-gram statistics and rules, without relying on human judgement or LLM-as-a-judge approaches. Using 50 question and reference answer sets, we introduce three new metrics based on n-grams and rules: Fluency, Truthfulness, and Helpfulness. Our benchmark strongly correlates with GPT-4o-based evaluations while requiring significantly fewer computational resources, demonstrating its effectiveness as a scalable alternative for assessing LLMs' open-ended generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09284v1">SparQLe: Speech Queries to Text Translation Through LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      With the growing influence of Large Language Models (LLMs), there is increasing interest in integrating speech representations with them to enable more seamless multi-modal processing and speech understanding. This study introduces a novel approach that leverages self-supervised speech representations in combination with instruction-tuned LLMs for speech-to-text translation. The proposed approach leverages a modality adapter to align extracted speech features with instruction-tuned LLMs using English-language data. Our experiments demonstrate that this method effectively preserves the semantic content of the input speech and serves as an effective bridge between self-supervised speech models and instruction-tuned LLMs, offering a promising solution for various speech understanding applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07115v2">Online Scheduling for LLM Inference with KV Cache Constraints</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference, where a trained model generates text one word at a time in response to user prompts, is a computationally intensive process requiring efficient scheduling to optimize latency and resource utilization. A key challenge in LLM inference is the management of the Key-Value (KV) cache, which reduces redundant computations but introduces memory constraints. In this work, we model LLM inference with KV cache constraints theoretically and propose novel batching and scheduling algorithms that minimize inference latency while effectively managing the KV cache's memory. We analyze both semi-online and fully online scheduling models, and our results are threefold. First, we provide a polynomial-time algorithm that achieves exact optimality in terms of average latency in the semi-online prompt arrival model. Second, in the fully online case with a stochastic prompt arrival, we introduce an efficient online scheduling algorithm with constant regret. Third, we prove that no algorithm (deterministic or randomized) can achieve a constant competitive ratio in fully online adversarial settings. Our empirical evaluations on a public LLM inference dataset, using the Llama-70B model on A100 GPUs, show that our approach significantly outperforms benchmark algorithms used currently in practice, achieving lower latency while reducing energy consumption. Overall, our results offer a path toward more sustainable and cost-effective LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09211v1">Visual Graph Question Answering with ASP and LLMs for Language Parsing</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 In Proceedings ICLP 2024, arXiv:2502.08453. This work was partially funded from the Bosch Center for AI
    </div>
    <details class="paper-abstract">
      Visual Question Answering (VQA) is a challenging problem that requires to process multimodal input. Answer-Set Programming (ASP) has shown great potential in this regard to add interpretability and explainability to modular VQA architectures. In this work, we address the problem of how to integrate ASP with modules for vision and natural language processing to solve a new and demanding VQA variant that is concerned with images of graphs (not graphs in symbolic form). Images containing graph-based structures are an ubiquitous and popular form of visualisation. Here, we deal with the particular problem of graphs inspired by transit networks, and we introduce a novel dataset that amends an existing one by adding images of graphs that resemble metro lines. Our modular neuro-symbolic approach combines optical graph recognition for graph parsing, a pretrained optical character recognition neural network for parsing labels, Large Language Models (LLMs) for language processing, and ASP for reasoning. This method serves as a first baseline and achieves an overall average accuracy of 73% on the dataset. Our evaluation provides further evidence of the potential of modular neuro-symbolic systems, in particular with pretrained models that do not involve any further training and logic programming for reasoning, to solve complex VQA tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09209v1">On LLM-generated Logic Programs and their Inference Execution Methods</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 In Proceedings ICLP 2024, arXiv:2502.08453
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained on petabytes of data are highly compressed repositories of a significant proportion of the knowledge accumulated and distilled so far. In this paper we study techniques to elicit this knowledge in the form of several classes of logic programs, including propositional Horn clauses, Dual Horn clauses, relational triplets and Definite Clause Grammars. Exposing this knowledge as logic programs enables sound reasoning methods that can verify alignment of LLM outputs to their intended uses and extend their inference capabilities. We study new execution methods for the generated programs, including soft-unification of abducible facts against LLM-generated content stored in a vector database as well as GPU-based acceleration of minimal model computation that supports inference with large LLM-generated programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09204v1">Logical Lease Litigation: Prolog and LLMs for Rental Law Compliance in New York</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 In Proceedings ICLP 2024, arXiv:2502.08453
    </div>
    <details class="paper-abstract">
      Legal cases require careful logical reasoning following the laws, whereas interactions with non- technical users must be in natural language. As an application combining logical reasoning using Prolog and natural language processing using large language models (LLMs), this paper presents a novel approach and system, LogicLease, to automate the analysis of landlord-tenant legal cases in the state of New York. LogicLease determines compliance with relevant legal requirements by analyzing case descriptions and citing all relevant laws. It leverages LLMs for information extraction and Prolog for legal reasoning. By separating information extraction from legal reasoning, LogicLease achieves greater transparency and control over the legal logic applied to each case. We evaluate the accuracy, efficiency, and robustness of LogicLease through a series of tests, achieving 100% accuracy and an average processing time of 2.57 seconds. LogicLease presents advantages over state-of-the-art LLM- based legal analysis systems by providing clear, step-by-step reasoning, citing specific laws, and distinguishing itself by its ability to avoid hallucinations - a common issue in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09192v1">Thinking beyond the anthropomorphic paradigm benefits LLM research</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Anthropomorphism, or the attribution of human traits to technology, is an automatic and unconscious response that occurs even in those with advanced technical expertise. In this position paper, we analyze hundreds of thousands of computer science research articles from the past decade and present empirical evidence of the prevalence and growth of anthropomorphic terminology in research on large language models (LLMs). This terminology reflects deeper anthropomorphic conceptualizations which shape how we think about and conduct LLM research. We argue these conceptualizations may be limiting, and that challenging them opens up new pathways for understanding and improving LLMs beyond human analogies. To illustrate this, we identify and analyze five core anthropomorphic assumptions shaping prominent methodologies across the LLM development lifecycle, from the assumption that models must use natural language for reasoning tasks to the assumption that model capabilities should be evaluated through human-centric benchmarks. For each assumption, we demonstrate how non-anthropomorphic alternatives can open new directions for research and development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09834v3">LLMs Meet Library Evolution: Evaluating Deprecated API Usage in LLM-based Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted by ICSE'25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), pre-trained or fine-tuned on large code corpora, have shown effectiveness in generating code completions. However, in LLM-based code completion, LLMs may struggle to use correct and up-to-date Application Programming Interfaces (APIs) due to the rapid and continuous evolution of libraries. While existing studies have highlighted issues with predicting incorrect APIs, the specific problem of deprecated API usage in LLM-based code completion has not been thoroughly investigated. To address this gap, we conducted the first evaluation study on deprecated API usage in LLM-based code completion. This study involved seven advanced LLMs, 145 API mappings from eight popular Python libraries, and 28,125 completion prompts. The study results reveal the status quo (i.e., API usage plausibility and deprecated usage rate) of deprecated API and replacing API usage in LLM-based code completion from the perspectives of model, prompt, and library, and indicate the root causes behind. Based on these findings, we propose two lightweight fixing approaches, REPLACEAPI and INSERTPROMPT, which can serve as baseline approaches for future research on mitigating deprecated API usage in LLM-based completion. Additionally, we provide implications for future research on integrating library evolution with LLM-driven software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09175v1">FLAME: Flexible LLM-Assisted Moderation Engine</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09156v1">Improving TCM Question Answering through Tree-Organized Self-Reflective Retrieval with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Objectives: Large language models (LLMs) can harness medical knowledge for intelligent question answering (Q&A), promising support for auxiliary diagnosis and medical talent cultivation. However, there is a deficiency of highly efficient retrieval-augmented generation (RAG) frameworks within the domain of Traditional Chinese Medicine (TCM). Our purpose is to observe the effect of the Tree-Organized Self-Reflective Retrieval (TOSRR) framework on LLMs in TCM Q&A tasks. Materials and Methods: We introduce the novel approach of knowledge organization, constructing a tree structure knowledge base with hierarchy. At inference time, our self-reflection framework retrieves from this knowledge base, integrating information across chapters. Questions from the TCM Medical Licensing Examination (MLE) and the college Classics Course Exam (CCE) were randomly selected as benchmark datasets. Results: By coupling with GPT-4, the framework can improve the best performance on the TCM MLE benchmark by 19.85% in absolute accuracy, and improve recall accuracy from 27% to 38% on CCE datasets. In manual evaluation, the framework improves a total of 18.52 points across dimensions of safety, consistency, explainability, compliance, and coherence. Conclusion: The TOSRR framework can effectively improve LLM's capability in Q&A tasks of TCM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09142v1">LLM-Driven Augmented Reality Puppeteer: Controller-Free Voice-Commanded Robot Teleoperation</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Accepted as conference proceeding in International Conference on Human-Computer Interaction 2025 (HCI International 2025)
    </div>
    <details class="paper-abstract">
      The integration of robotics and augmented reality (AR) presents transformative opportunities for advancing human-robot interaction (HRI) by improving usability, intuitiveness, and accessibility. This work introduces a controller-free, LLM-driven voice-commanded AR puppeteering system, enabling users to teleoperate a robot by manipulating its virtual counterpart in real time. By leveraging natural language processing (NLP) and AR technologies, our system -- prototyped using Meta Quest 3 -- eliminates the need for physical controllers, enhancing ease of use while minimizing potential safety risks associated with direct robot operation. A preliminary user demonstration successfully validated the system's functionality, demonstrating its potential for safer, more intuitive, and immersive robotic control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09101v1">Bridging the Gap Between LLMs and Human Intentions: Progresses and Challenges in Instruction Understanding, Intention Reasoning, and Reliable Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional capabilities in understanding and generation. However, when interacting with human instructions in real-world scenarios, LLMs still face significant challenges, particularly in accurately capturing and comprehending human instructions and intentions. This paper focuses on three challenges in LLM-based text generation tasks: instruction understanding, intention reasoning, and reliable generation. Regarding human complex instruction, LLMs have deficiencies in understanding long contexts and instructions in multi-round conversations. For intention reasoning, LLMs may have inconsistent command reasoning, difficulty reasoning about commands containing incorrect information, difficulty understanding user ambiguous language commands, and a weak understanding of user intention in commands. Besides, In terms of reliable generation, LLMs may have unstable generated content and unethical generation. To this end, we classify and analyze the performance of LLMs in challenging scenarios and conduct a comprehensive evaluation of existing solutions. Furthermore, we introduce benchmarks and categorize them based on the aforementioned three core challenges. Finally, we explore potential directions for future research to enhance the reliability and adaptability of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06931v2">Non-Prehensile Tool-Object Manipulation by Integrating LLM-Based Planning and Manoeuvrability-Driven Controls</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The ability to wield tools was once considered exclusive to human intelligence, but it's now known that many other animals, like crows, possess this capability. Yet, robotic systems still fall short of matching biological dexterity. In this paper, we investigate the use of Large Language Models (LLMs), tool affordances, and object manoeuvrability for non-prehensile tool-based manipulation tasks. Our novel method leverages LLMs based on scene information and natural language instructions to enable symbolic task planning for tool-object manipulation. This approach allows the system to convert the human language sentence into a sequence of feasible motion functions. We have developed a novel manoeuvrability-driven controller using a new tool affordance model derived from visual feedback. This controller helps guide the robot's tool utilization and manipulation actions, even within confined areas, using a stepping incremental approach. The proposed methodology is evaluated with experiments to prove its effectiveness under various manipulation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09082v1">CoSER: Coordinating LLM-Based Persona Simulation of Established Roles</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Role-playing language agents (RPLAs) have emerged as promising applications of large language models (LLMs). However, simulating established characters presents a challenging task for RPLAs, due to the lack of authentic character datasets and nuanced evaluation methods using such data. In this paper, we present CoSER, a collection of a high-quality dataset, open models, and an evaluation protocol towards effective RPLAs of established characters. The CoSER dataset covers 17,966 characters from 771 renowned books. It provides authentic dialogues with real-world intricacies, as well as diverse data types such as conversation setups, character experiences and internal thoughts. Drawing from acting methodology, we introduce given-circumstance acting for training and evaluating role-playing LLMs, where LLMs sequentially portray multiple characters in book scenes. Using our dataset, we develop CoSER 8B and CoSER 70B, i.e., advanced open role-playing LLMs built on LLaMA-3.1 models. Extensive experiments demonstrate the value of the CoSER dataset for RPLA training, evaluation and retrieval. Moreover, CoSER 70B exhibits state-of-the-art performance surpassing or matching GPT-4o on our evaluation and three existing benchmarks, i.e., achieving 75.80% and 93.47% accuracy on the InCharacter and LifeChoice benchmarks respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10913v2">ASHABot: An LLM-Powered Chatbot to Support the Informational Needs of Community Health Workers</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Community health workers (CHWs) provide last-mile healthcare services but face challenges due to limited medical knowledge and training. This paper describes the design, deployment, and evaluation of ASHABot, an LLM-powered, experts-in-the-loop, WhatsApp-based chatbot to address the information needs of CHWs in India. Through interviews with CHWs and their supervisors and log analysis, we examine factors affecting their engagement with ASHABot, and ASHABot's role in addressing CHWs' informational needs. We found that ASHABot provided a private channel for CHWs to ask rudimentary and sensitive questions they hesitated to ask supervisors. CHWs trusted the information they received on ASHABot and treated it as an authoritative resource. CHWs' supervisors expanded their knowledge by contributing answers to questions ASHABot failed to answer, but were concerned about demands on their workload and increased accountability. We emphasize positioning LLMs as supplemental fallible resources within the community healthcare ecosystem, instead of as replacements for supervisor support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09061v1">CRANE: Reasoning with constrained LLM generation</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Code generation, symbolic math reasoning, and other tasks require LLMs to produce outputs that are both syntactically and semantically correct. Constrained LLM generation is a promising direction to enforce adherence to formal grammar, but prior works have empirically observed that strict enforcement of formal constraints often diminishes the reasoning capabilities of LLMs. In this work, we first provide a theoretical explanation for why constraining LLM outputs to very restrictive grammars that only allow syntactically valid final answers reduces the reasoning capabilities of the model. Second, we demonstrate that by augmenting the output grammar with carefully designed additional rules, it is always possible to preserve the reasoning capabilities of the LLM while ensuring syntactic and semantic correctness in its outputs. Building on these theoretical insights, we propose a reasoning-augmented constrained decoding algorithm, CRANE, which effectively balances the correctness of constrained generation with the flexibility of unconstrained generation. Experiments on multiple open-source LLMs and benchmarks show that CRANE significantly outperforms both state-of-the-art constrained decoding strategies and standard unconstrained decoding, showing up to 10% points accuracy improvement over baselines on challenging symbolic reasoning benchmarks GSM-symbolic and FOLIO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09056v1">An Open Recipe: Adapting Language-Specific LLMs to a Reasoning Model in One Day via Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      This paper investigates data selection and model merging methodologies aimed at incorporating advanced reasoning capabilities such as those of DeepSeek R1 into language-specific large language models (LLMs), with a particular focus on the Thai LLM. Our goal is to enhance the reasoning capabilities of language-specific LLMs while maintaining their target language abilities. DeepSeek R1 excels in reasoning but primarily benefits high-resource languages such as English and Chinese. However, low-resource languages remain underserved due to the dominance of English-centric training data and model optimizations, which limit performance in these languages. This limitation results in unreliable code-switching and diminished effectiveness on tasks in low-resource languages. Meanwhile, local and regional LLM initiatives have attempted to bridge this gap by developing language-specific LLMs that focus on improving local linguistic fidelity. We demonstrate that, with only publicly available datasets and a computational budget of $120, it is possible to enhance the reasoning capabilities of language-specific LLMs to match the level of DeepSeek R1, without compromising their performance on target language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09054v1">Cost-Saving LLM Cascades with Early Abstention</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 6 pages, 1 figure
    </div>
    <details class="paper-abstract">
      LLM cascades are based on the idea that processing all queries with the largest and most expensive LLMs is inefficient. Instead, cascades deploy small LLMs to answer the majority of queries, limiting the use of large and expensive LLMs to only the most difficult queries. This approach can significantly reduce costs without impacting performance. However, risk-sensitive domains such as finance or medicine place an additional premium on avoiding model errors. Recognizing that even the most expensive models may make mistakes, applications in these domains benefit from allowing LLM systems to completely abstain from answering a query when the chance of making a mistake is significant. However, giving a cascade the ability to abstain poses an immediate design question for LLM cascades: should abstention only be allowed at the final model or also at earlier models? Since the error patterns of small and large models are correlated, the latter strategy may further reduce inference costs by letting inexpensive models anticipate abstention decisions by expensive models, thereby obviating the need to run the expensive models. We investigate the benefits of "early abstention" in LLM cascades and find that it reduces the overall test loss by 2.2% on average across six benchmarks (GSM8K, MedMCQA, MMLU, TriviaQA, TruthfulQA, and XSum). These gains result from a more effective use of abstention, which trades a 4.1% average increase in the overall abstention rate for a 13.0% reduction in cost and a 5.0% reduction in error rate. Our findings demonstrate that it is possible to leverage correlations between the error patterns of different language models to drive performance improvements for LLM systems with abstention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01694v2">Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      This paper tackles the problem of video question answering (VideoQA), a task that often requires multi-step reasoning and a profound understanding of spatial-temporal dynamics. While large video-language models perform well on benchmarks, they often lack explainability and spatial-temporal grounding. In this paper, we propose Agent-of-Thoughts Distillation (AoTD), a method that enhances models by incorporating automatically generated Chain-of-Thoughts (CoTs) into the instruction-tuning process. Specifically, we leverage an agent-based system to decompose complex questions into sub-tasks, and address them with specialized vision models, the intermediate results are then treated as reasoning chains. We also introduce a verification mechanism using a large language model (LLM) to ensure the reliability of generated CoTs. Extensive experiments demonstrate that AoTD improves the performance on multiple-choice and open-ended benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00511v2">Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Preliminary work
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated remarkable reasoning capabilities. However, single-shot inference often yields unreliable results for complex reasoning tasks, leading researchers to explore multiple reasoning paths through methods such as perplexity and self-consistency. In this paper, we present the first theoretical error decomposition analysis of these techniques, breaking down their error into estimation error and model error. Our analysis reveals a fundamental trade-off: perplexity methods suffer from substantial model error due to the absence of a proper consistency function, while self-consistency exhibits high estimation error due to a slow error convergence rate. To overcome these limitations, we propose Reasoning-Pruning Perplexity Consistency (RPC). This approach combines Perplexity Consistency, which seamlessly integrates LLM perplexity with self-consistency, and Reasoning Pruning, which eliminates low-probability reasoning paths to effectively prevent the degeneration of estimation error reduction. Theoretical analysis demonstrates that RPC not only accelerates the convergence rate of estimation error to an exponential level but also holds strong potential for further reducing model error. Extensive empirical evaluations on seven benchmark datasets confirm that RPC can significantly improve reasoning performance, sample efficiency, and confidence reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06635v2">Steel-LLM:From Scratch to Open Source -- A Personal Journey in Building a Chinese-Centric LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      Steel-LLM is a Chinese-centric language model developed from scratch with the goal of creating a high-quality, open-source model despite limited computational resources. Launched in March 2024, the project aimed to train a 1-billion-parameter model on a large-scale dataset, prioritizing transparency and the sharing of practical insights to assist others in the community. The training process primarily focused on Chinese data, with a small proportion of English data included, addressing gaps in existing open-source LLMs by providing a more detailed and practical account of the model-building journey. Steel-LLM has demonstrated competitive performance on benchmarks such as CEVAL and CMMLU, outperforming early models from larger institutions. This paper provides a comprehensive summary of the project's key contributions, including data collection, model design, training methodologies, and the challenges encountered along the way, offering a valuable resource for researchers and practitioners looking to develop their own LLMs. The model checkpoints and training script are available at https://github.com/zhanshijinwat/Steel-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06572v2">LawGPT: Knowledge-Guided Data Generation and Its Application to Legal LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), both proprietary and open-source, have demonstrated remarkable capabilities across various natural language processing tasks. However, they face significant limitations in legal reasoning tasks. Proprietary models introduce data privacy risks and high inference costs, while open-source models underperform due to insufficient legal domain training data. To address these limitations, we study data generation for legal reasoning to improve the legal reasoning performance of open-source LLMs with the help of proprietary LLMs. This is challenging due to the lack of legal knowledge in proprietary LLMs and the difficulty in verifying the generated data. We propose KgDG, a knowledge-guided data generation framework for legal reasoning. Our framework enables leveraging legal knowledge to enhance generation diversity and introduces a refinement and verification process to ensure the quality of generated data. Moreover, we expand the generated dataset to further enhance the LLM reasoning capabilities. Using KgDG, we create a synthetic legal reasoning dataset containing 50K high-quality examples. Our trained model LawGPT outperforms existing legal-specific LLMs and achieves performance comparable to proprietary LLMs, demonstrating the effectiveness of KgDG and LawGPT. Our code and resources is publicly available at https://github.com/LAMDASZ-ML/Knowledge-Guide-Data-Generation .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09017v1">Diversity Enhances an LLM's Performance in RAG and Long-context Task</a></div>
    <div class="paper-meta">
      📅 2025-02-13
    </div>
    <details class="paper-abstract">
      The rapid advancements in large language models (LLMs) have highlighted the challenge of context window limitations, primarily due to the quadratic time complexity of the self-attention mechanism (\(O(N^2)\), where \(N\) denotes the context window length). This constraint impacts tasks such as retrieval-augmented generation (RAG) in question answering (Q\&A) and long context summarization. A common approach involves selecting content with the highest similarity to the query; however, this often leads to redundancy and the exclusion of diverse yet relevant information. Building on principles from Maximal Marginal Relevance (MMR) and Farthest Point Sampling (FPS), we integrate diversity into the content selection process. Our findings reveal that incorporating diversity substantially increases the recall of selecting relevant sentences or chunks before LLM-based Q\&A and summarization. These results highlight the importance of maintaining diversity in future LLM applications to further improve summarization and Q\&A outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05006v3">COAST: Enhancing the Code Debugging Ability of LLMs through Communicative Agent Based Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Code debugging is a vital stage of software development, essential for ensuring the reliability and performance of Large Language Models (LLMs) in the code generation task. Human debugging typically follows a multi-stage process, which includes Bug Localization, Bug Identification, Code Repair, and Code Recognition. However, existing code debugging benchmarks predominantly focus on the Code Repair stage, which offers only a limited perspective on evaluating the debugging capabilities of LLMs. In this paper, we introduce DEBUGEVAL, a comprehensive benchmark for evaluating the debugging abilities of LLMs by emulating the multi-stage human debugging process. Through evaluating on DEBUGEVAL, we observe that 7B-scale models consistently underperform compared to their larger counterparts, highlighting their limitations in comprehending code semantics. In this case, we propose the COmmunicative Agent-based data SynThesis (COAST) framework, which employs a multi-agent system to generate high-quality training data for supervised fine-tuning (SFT). Experimental results demonstrate that COAST-generated data outperform human-curated and GPT-4-generated data, enabling 7B-scale LLMs to achieve debugging performance comparable to GPT-3.5. All data and codes are available at https://github.com/NEUIR/COAST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07191v2">Bag of Tricks for Inference-time Computation of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      With the advancement of large language models (LLMs), solving complex reasoning tasks has gained increasing attention. Inference-time computation methods (e.g., Best-of-N, beam search, et al.) are particularly valuable as they can enhance reasoning performance without modifying model parameters or requiring additional training. However, these techniques come with implementation challenges, and most existing methods remain at the proof-of-concept stage with limited practical adoption due to their computational complexity and varying effectiveness across different tasks. In this paper, we investigate and benchmark diverse inference-time computation strategies across reasoning tasks of varying complexity. Since most current methods rely on a proposer-verifier pipeline that first generates candidate solutions (e.g., reasoning solutions) and then selects the best one based on reward signals (e.g., RLHF rewards, process rewards), our research focuses on optimizing both candidate solution generation (e.g., instructing prompts, hyperparameters such as temperature and top-p) and reward mechanisms (e.g., self-evaluation, reward types). Through extensive experiments (more than 20,000 A100-80G GPU hours with over 1,000 experiments) across a variety of models (e.g., Llama, Qwen, and Mistral families) of various sizes, our ablation studies reveal that previously overlooked strategies can significantly enhance performance (e.g., tuning temperature can improve reasoning task performance by up to 5%). Furthermore, we establish a standardized benchmark for inference-time computation by systematically evaluating six representative methods across eight reasoning tasks. These findings provide a stronger foundation for future research. The code is available at https://github.com/usail-hkust/benchmark_inference_time_computation_LL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15512v2">Reverse Question Answering: Can an LLM Write a Question so Hard (or Bad) that it Can't Answer?</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 NAACL 2025
    </div>
    <details class="paper-abstract">
      Question answering (QA), giving correct answers to questions, is a popular task, but we test reverse question answering (RQA): for an input answer, give a question with that answer. Past work tests QA and RQA separately, but we test them jointly, comparing their difficulty, aiding benchmark design, and checking reasoning consistency. We run 16 LLMs on QA and RQA with trivia questions/answers, revealing: 1) Versus QA, LLMs are much less accurate in RQA for numerical answers, but slightly more accurate in RQA for textual answers; 2) LLMs often answer their own invalid questions from RQA accurately in QA, so RQA errors are not from knowledge gaps alone; 3) RQA errors correlate with question difficulty and inversely correlate with answer frequencies in the Dolma corpus; and 4) LLMs struggle to provide valid multi-hop questions. By finding question and answer types that lead to RQA errors, we suggest improvements for LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.09113v2">Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Recent research indicates that large language models (LLMs) are susceptible to jailbreaking attacks that can generate harmful content. This paper introduces a novel token-level attack method, Adaptive Dense-to-Sparse Constrained Optimization (ADC), which has been shown to successfully jailbreak multiple open-source LLMs. Drawing inspiration from the difficulties of discrete token optimization, our method relaxes the discrete jailbreak optimization into a continuous optimization process while gradually increasing the sparsity of the optimizing vectors. This technique effectively bridges the gap between discrete and continuous space optimization. Experimental results demonstrate that our method is more effective and efficient than state-of-the-art token-level methods. On Harmbench, our approach achieves the highest attack success rate on seven out of eight LLMs compared to the latest jailbreak methods. Trigger Warning: This paper contains model behavior that can be offensive in nature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14427v3">GraphSOS: Graph Sampling and Order Selection to Help LLMs Understand Graphs Better</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      The success of Large Language Models (LLMs) in various domains has led researchers to apply them to graph-related problems by converting graph data into natural language text. However, unlike graph data, natural language inherently has sequential order. We observe a counter-intuitive fact that when the order of nodes or edges in the natural language description of a graph is shuffled, despite describing the same graph, model performance fluctuates between high performance and random guessing. Additionally, due to LLMs' limited input context length, current methods typically randomly sample neighbors of target nodes as representatives of their neighborhood, which may not always be effective for accurate reasoning. To address these gaps, we introduce GraphSOS (Graph Sampling and Order Selection). This novel model framework features an Order Selector Module to ensure proper serialization order of the graph and a Subgraph Sampling Module to sample subgraphs with better structure for better reasoning. Furthermore, we propose Graph CoT obtained through distillation, and enhance LLM's reasoning and zero-shot learning capabilities for graph tasks through instruction tuning. Experiments on multiple datasets for node classification and graph question-answering demonstrate that GraphSOS improves LLMs' performance and generalization ability on graph tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08045v1">Break the Checkbox: Challenging Closed-Style Evaluations of Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      A large number of studies rely on closed-style multiple-choice surveys to evaluate cultural alignment in Large Language Models (LLMs). In this work, we challenge this constrained evaluation paradigm and explore more realistic, unconstrained approaches. Using the World Values Survey (WVS) and Hofstede Cultural Dimensions as case studies, we demonstrate that LLMs exhibit stronger cultural alignment in less constrained settings, where responses are not forced. Additionally, we show that even minor changes, such as reordering survey choices, lead to inconsistent outputs, exposing the limitations of closed-style evaluations. Our findings advocate for more robust and flexible evaluation frameworks that focus on specific cultural proxies, encouraging more nuanced and accurate assessments of cultural alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08037v1">Franken-Adapter: Cross-Lingual Adaptation of LLMs by Embedding Surgery</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 33 pages
    </div>
    <details class="paper-abstract">
      The capabilities of Large Language Models (LLMs) in low-resource languages lag far behind those in English, making their universal accessibility a significant challenge. To alleviate this, we present $\textit{Franken-Adapter}$, a modular language adaptation approach for decoder-only LLMs with embedding surgery. Our method begins by creating customized vocabularies for target languages and performing language adaptation through embedding tuning on multilingual data. These pre-trained embeddings are subsequently integrated with LLMs that have been instruction-tuned on English alignment data to enable zero-shot cross-lingual transfer. Our experiments on $\texttt{Gemma2}$ models with up to 27B parameters demonstrate improvements of up to 20% across 96 languages, spanning both discriminative and generative tasks, with minimal regressions ($<$1%) in English. Further in-depth analysis reveals the critical role of customizing tokenizers in enhancing language adaptation, while boosting inference efficiency. Additionally, we show the versatility of our method by achieving a 14% improvement over a math-optimized LLM across 20 languages, offering a modular solution to transfer reasoning abilities across languages post hoc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07049v2">LLMs in Software Security: A Survey of Vulnerability Detection Techniques and Insights</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 33 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are emerging as transformative tools for software vulnerability detection, addressing critical challenges in the security domain. Traditional methods, such as static and dynamic analysis, often falter due to inefficiencies, high false positive rates, and the growing complexity of modern software systems. By leveraging their ability to analyze code structures, identify patterns, and generate repair sugges- tions, LLMs, exemplified by models like GPT, BERT, and CodeBERT, present a novel and scalable approach to mitigating vulnerabilities. This paper provides a detailed survey of LLMs in vulnerability detection. It examines key aspects, including model architectures, application methods, target languages, fine-tuning strategies, datasets, and evaluation metrics. We also analyze the scope of current research problems, highlighting the strengths and weaknesses of existing approaches. Further, we address challenges such as cross-language vulnerability detection, multimodal data integration, and repository-level analysis. Based on these findings, we propose solutions for issues like dataset scalability, model interpretability, and applications in low-resource scenarios. Our contributions are threefold: (1) a systematic review of how LLMs are applied in vulnerability detection; (2) an analysis of shared patterns and differences across studies, with a unified framework for understanding the field; and (3) a summary of key challenges and future research directions. This work provides valuable insights for advancing LLM-based vulnerability detection. We also maintain and regularly update latest selected paper on https://github.com/OwenSanzas/LLM-For-Vulnerability-Detection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11278v3">Do Not Design, Learn: A Trainable Scoring Function for Uncertainty Estimation in Generative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Uncertainty estimation (UE) of generative large language models (LLMs) is crucial for evaluating the reliability of generated sequences. A significant subset of UE methods utilize token probabilities to assess uncertainty, aggregating multiple token probabilities into a single UE score using a scoring function. Existing scoring functions for probability-based UE, such as length-normalized scoring and semantic contribution-based weighting, are designed to solve certain aspects of the problem but exhibit limitations, including the inability to handle biased probabilities and complex semantic dependencies between tokens. To address these issues, in this work, we propose Learnable Response Scoring (LARS) function, a novel scoring function that leverages supervised data to capture complex dependencies between tokens and probabilities, thereby producing more reliable and calibrated response scores in computing the uncertainty of LLM generations. Our comprehensive experiments across question-answering and arithmetical reasoning tasks with various datasets demonstrate that LARS significantly outperforms existing scoring functions, achieving improvements of up to 16\% AUROC score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02890v3">Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08777v1">Zero-Shot Belief: A Hard Problem for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Submitted to ACL 2025
    </div>
    <details class="paper-abstract">
      We present two LLM-based approaches to zero-shot source-and-target belief prediction on FactBank: a unified system that identifies events, sources, and belief labels in a single pass, and a hybrid approach that uses a fine-tuned DeBERTa tagger for event detection. We show that multiple open-sourced, closed-source, and reasoning-based LLMs struggle with the task. Using the hybrid approach, we achieve new state-of-the-art results on FactBank and offer a detailed error analysis. Our approach is then tested on the Italian belief corpus ModaFact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08773v1">Universal Model Routing for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Large language models' significant advances in capabilities are accompanied by significant increases in inference costs. Model routing is a simple technique for reducing inference cost, wherein one maintains a pool of candidate LLMs, and learns to route each prompt to the smallest feasible LLM. Existing works focus on learning a router for a fixed pool of LLMs. In this paper, we consider the problem of dynamic routing, where new, previously unobserved LLMs are available at test time. We propose a new approach to this problem that relies on representing each LLM as a feature vector, derived based on predictions on a set of representative prompts. Based on this, we detail two effective strategies, relying on cluster-based routing and a learned cluster map respectively. We prove that these strategies are estimates of a theoretically optimal routing rule, and provide an excess risk bound to quantify their errors. Experiments on a range of public benchmarks show the effectiveness of the proposed strategies in routing amongst more than 30 unseen LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02362v3">Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08756v1">From PowerPoint UI Sketches to Web-Based Applications: Pattern-Driven Code Generation for GIS Dashboard Development Using Knowledge-Augmented LLMs, Context-Aware Visual Prompting, and the React Framework</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Developing web-based GIS applications, commonly known as CyberGIS dashboards, for querying and visualizing GIS data in environmental research often demands repetitive and resource-intensive efforts. While Generative AI offers automation potential for code generation, it struggles with complex scientific applications due to challenges in integrating domain knowledge, software engineering principles, and UI design best practices. This paper introduces a knowledge-augmented code generation framework that retrieves software engineering best practices, domain expertise, and advanced technology stacks from a specialized knowledge base to enhance Generative Pre-trained Transformers (GPT) for front-end development. The framework automates the creation of GIS-based web applications (e.g., dashboards, interfaces) from user-defined UI wireframes sketched in tools like PowerPoint or Adobe Illustrator. A novel Context-Aware Visual Prompting method, implemented in Python, extracts layouts and interface features from these wireframes to guide code generation. Our approach leverages Large Language Models (LLMs) to generate front-end code by integrating structured reasoning, software engineering principles, and domain knowledge, drawing inspiration from Chain-of-Thought (CoT) prompting and Retrieval-Augmented Generation (RAG). A case study demonstrates the framework's capability to generate a modular, maintainable web platform hosting multiple dashboards for visualizing environmental and energy data (e.g., time-series, shapefiles, rasters) from user-sketched wireframes. By employing a knowledge-driven approach, the framework produces scalable, industry-standard front-end code using design patterns such as Model-View-ViewModel (MVVM) and frameworks like React. This significantly reduces manual effort in design and coding, pioneering an automated and efficient method for developing smart city software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08691v1">AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Understanding human behavior and society is a central focus in social sciences, with the rise of generative social science marking a significant paradigmatic shift. By leveraging bottom-up simulations, it replaces costly and logistically challenging traditional experiments with scalable, replicable, and systematic computational approaches for studying complex social dynamics. Recent advances in large language models (LLMs) have further transformed this research paradigm, enabling the creation of human-like generative social agents and realistic simulacra of society. In this paper, we propose AgentSociety, a large-scale social simulator that integrates LLM-driven agents, a realistic societal environment, and a powerful large-scale simulation engine. Based on the proposed simulator, we generate social lives for over 10k agents, simulating their 5 million interactions both among agents and between agents and their environment. Furthermore, we explore the potential of AgentSociety as a testbed for computational social experiments, focusing on four key social issues: polarization, the spread of inflammatory messages, the effects of universal basic income policies, and the impact of external shocks such as hurricanes. These four issues serve as valuable cases for assessing AgentSociety's support for typical research methods -- such as surveys, interviews, and interventions -- as well as for investigating the patterns, causes, and underlying mechanisms of social issues. The alignment between AgentSociety's outcomes and real-world experimental results not only demonstrates its ability to capture human behaviors and their underlying mechanisms, but also underscores its potential as an important platform for social scientists and policymakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09667v1">k-LLMmeans: Summaries as Centroids for Interpretable and Scalable LLM-Based Text Clustering</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      We introduce k-LLMmeans, a novel modification of the k-means clustering algorithm that utilizes LLMs to generate textual summaries as cluster centroids, thereby capturing contextual and semantic nuances often lost when relying on purely numerical means of document embeddings. This modification preserves the properties of k-means while offering greater interpretability: the cluster centroid is represented by an LLM-generated summary, whose embedding guides cluster assignments. We also propose a mini-batch variant, enabling efficient online clustering for streaming text data and providing real-time interpretability of evolving cluster centroids. Through extensive simulations, we show that our methods outperform vanilla k-means on multiple metrics while incurring only modest LLM usage that does not scale with dataset size. Finally, We present a case study showcasing the interpretability of evolving cluster centroids in sequential text streams. As part of our evaluation, we compile a new dataset from StackExchange, offering a benchmark for text-stream clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08631v1">Ensemble based approach to quantifying uncertainty of LLM based classifications</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      The output of Large Language Models (LLMs) are a function of the internal model's parameters and the input provided into the context window. The hypothesis presented here is that under a greedy sampling strategy the variance in the LLM's output is a function of the conceptual certainty embedded in the model's parametric knowledge, as well as the lexical variance in the input. Finetuning the model results in reducing the sensitivity of the model output to the lexical input variations. This is then applied to a classification problem and a probabilistic method is proposed for estimating the certainties of the predicted classes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08599v1">SPeCtrum: A Grounded Framework for Multidimensional Identity Representation in LLM-Based Agent</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 21 pages, 8 figures, 5 tables, Accepted in NAACL2025 Main
    </div>
    <details class="paper-abstract">
      Existing methods for simulating individual identities often oversimplify human complexity, which may lead to incomplete or flattened representations. To address this, we introduce SPeCtrum, a grounded framework for constructing authentic LLM agent personas by incorporating an individual's multidimensional self-concept. SPeCtrum integrates three core components: Social Identity (S), Personal Identity (P), and Personal Life Context (C), each contributing distinct yet interconnected aspects of identity. To evaluate SPeCtrum's effectiveness in identity representation, we conducted automated and human evaluations. Automated evaluations using popular drama characters showed that Personal Life Context (C)-derived from short essays on preferences and daily routines-modeled characters' identities more effectively than Social Identity (S) and Personal Identity (P) alone and performed comparably to the full SPC combination. In contrast, human evaluations involving real-world individuals found that the full SPC combination provided a more comprehensive self-concept representation than C alone. Our findings suggest that while C alone may suffice for basic identity simulation, integrating S, P, and C enhances the authenticity and accuracy of real-world identity representation. Overall, SPeCtrum offers a structured approach for simulating individuals in LLM agents, enabling more personalized human-AI interactions and improving the realism of simulation-based behavioral studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08586v1">Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      A high volume of recent ML security literature focuses on attacks against aligned large language models (LLMs). These attacks may extract private information or coerce the model into producing harmful outputs. In real-world deployments, LLMs are often part of a larger agentic pipeline including memory systems, retrieval, web access, and API calling. Such additional components introduce vulnerabilities that make these LLM-powered agents much easier to attack than isolated LLMs, yet relatively little work focuses on the security of LLM agents. In this paper, we analyze security and privacy vulnerabilities that are unique to LLM agents. We first provide a taxonomy of attacks categorized by threat actors, objectives, entry points, attacker observability, attack strategies, and inherent vulnerabilities of agent pipelines. We then conduct a series of illustrative attacks on popular open-source and commercial agents, demonstrating the immediate practical implications of their vulnerabilities. Notably, our attacks are trivial to implement and require no understanding of machine learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05905v5">Truthful Aggregation of LLMs with an Application to Online Advertising</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      The next frontier of online advertising is revenue generation from LLM-generated content. We consider a setting where advertisers aim to influence the responses of an LLM to align with their interests, while platforms seek to maximize advertiser value and ensure user satisfaction. The challenge is that advertisers' preferences generally conflict with those of the user, and advertisers may misreport their preferences. To address this, we introduce MOSAIC, an auction mechanism that ensures that truthful reporting is a dominant strategy for advertisers and that aligns the utility of each advertiser with their contribution to social welfare. Importantly, the mechanism operates without LLM fine-tuning or access to model weights and provably converges to the output of the optimally fine-tuned LLM as computational resources increase. Additionally, it can incorporate contextual information about advertisers, which significantly improves social welfare. Through experiments with a publicly available LLM, we show that MOSAIC leads to high advertiser value and platform revenue with low computational overhead. While our motivating application is online advertising, our mechanism can be applied in any setting with monetary transfers, making it a general-purpose solution for truthfully aggregating the preferences of self-interested agents over LLM-generated replies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08550v1">LLMs can implicitly learn from mistakes in-context</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Learning from mistakes is a fundamental feature of human intelligence. Previous work has shown that Large Language Models (LLMs) can also learn from incorrect answers when provided with a comprehensive rationale detailing why an answer is wrong or how to correct it. In this work, we examine whether LLMs can learn from mistakes in mathematical reasoning tasks when these explanations are not provided. We investigate if LLMs are able to implicitly infer such rationales simply from observing both incorrect and correct answers. Surprisingly, we find that LLMs perform better, on average, when rationales are eliminated from the context and incorrect answers are simply shown alongside correct ones. This approach also substantially outperforms chain-of-thought prompting in our evaluations. We show that these results are consistent across LLMs of different sizes and varying reasoning abilities. Further, we carry out an in-depth analysis, and show that prompting with both wrong and correct answers leads to greater performance and better generalisation than introducing additional, more diverse question-answer pairs into the context. Finally, we show that new rationales generated by models that have only observed incorrect and correct answers are scored equally as highly by humans as those produced with the aid of exemplar rationales. Our results demonstrate that LLMs are indeed capable of in-context implicit learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08515v1">The Paradox of Stochasticity: Limited Creativity and Computational Decoupling in Temperature-Varied LLM Outputs of Structured Fictional Data</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      This study examines how temperature settings and model architectures affect the generation of structured fictional data (names, birthdates) across three large language models (LLMs): llama3.1:8b, deepseek-r1:8b, and mistral:latest. By systematically testing temperature values from 0.0 to 1.0 in increments of 0.1, we conducted 330 trials yielding 889 structured entities, validated for syntactic consistency. Key findings reveal that model architecture significantly influences computational efficiency, with mistral:latest and llama3.1:8b processing data 8x faster than deepseek-r1:8b. Contrary to expectations, temperature showed no correlation with processing time, challenging assumptions about stochastic sampling costs. Output diversity remained limited, as models consistently defaulted to common name archetypes (e.g., 'John Doe' and 'Jane Smith') across all temperatures, though rare names clustered at intermediate values (0.3-0.7). These results demonstrate that architectural optimizations, rather than temperature adjustments, dominate performance in structured generation tasks. The findings emphasize prioritizing model selection over hyperparameter tuning for efficiency and suggest explicit diversity constraints are necessary to mitigate default output biases in synthetic data pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08503v1">Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities?</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      In this work, we identify the "2D-Cheating" problem in 3D LLM evaluation, where these tasks might be easily solved by VLMs with rendered images of point clouds, exposing ineffective evaluation of 3D LLMs' unique 3D capabilities. We test VLM performance across multiple 3D LLM benchmarks and, using this as a reference, propose principles for better assessing genuine 3D understanding. We also advocate explicitly separating 3D abilities from 1D or 2D aspects when evaluating 3D LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08395v1">IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are helping millions of users write texts about diverse issues, and in doing so expose users to different ideas and perspectives. This creates concerns about issue bias, where an LLM tends to present just one perspective on a given issue, which in turn may influence how users think about this issue. So far, it has not been possible to measure which issue biases LLMs actually manifest in real user interactions, making it difficult to address the risks from biased LLMs. Therefore, we create IssueBench: a set of 2.49m realistic prompts for measuring issue bias in LLM writing assistance, which we construct based on 3.9k templates (e.g. "write a blog about") and 212 political issues (e.g. "AI regulation") from real user interactions. Using IssueBench, we show that issue biases are common and persistent in state-of-the-art LLMs. We also show that biases are remarkably similar across models, and that all models align more with US Democrat than Republican voter opinion on a subset of issues. IssueBench can easily be adapted to include other issues, templates, or tasks. By enabling robust and realistic measurement, we hope that IssueBench can bring a new quality of evidence to ongoing discussions about LLM biases and how to address them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08381v1">The MoE-Empowered Edge LLMs Deployment: Architecture, Challenges, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 7pages, 1 table, 6 figures
    </div>
    <details class="paper-abstract">
      The powerfulness of LLMs indicates that deploying various LLMs with different scales and architectures on end, edge, and cloud to satisfy different requirements and adaptive heterogeneous hardware is the critical way to achieve ubiquitous intelligence for 6G. However, the massive parameter scale of LLMs poses significant challenges in deploying them on edge devices due to high computational and storage demands. Considering that the sparse activation in Mixture of Experts (MoE) is effective on scalable and dynamic allocation of computational and communications resources at the edge, this paper proposes a novel MoE-empowered collaborative deployment framework for edge LLMs, denoted as CoEL. This framework fully leverages the properties of MoE architecture and encompasses four key aspects: Perception, Deployment, Compression, and Updating. Edge servers broadcast their resource status and the specific resource requirements of LLMs to their neighbors. Then, utilizing this data, two sophisticated deployment strategies are proposed for satisfying varying model scales, ensuring that each model is deployed effectively. One for deploying LLMs on a single edge device through intra-device resource collaboration, and another for a distributed deployment across multiple edge devices via inter-device resource collaboration. Furthermore, both the models and the intermediate data are compressed for reducing memory footprint by quantization and reducing the volume of intermediate data by token fusion and pruning. Finally, given the dynamic of network topology, resource status, and user requirements, the deployment strategies are regularly updated to maintain its relevance and effectiveness. This paper also delineates the challenges and potential research directions for the deployment of edge LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18652v7">$C^2$: Scalable Auto-Feedback for LLM-based Chart Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 NAACL 2025 Main (Long)
    </div>
    <details class="paper-abstract">
      Generating high-quality charts with Large Language Models (LLMs) presents significant challenges due to limited data and the high cost of scaling through human curation. $\langle \text{instruction}, \text{data}, \text{code} \rangle$ triplets are scarce and expensive to manually curate as their creation demands technical expertise. To address this scalability challenge, we introduce a reference-free automatic feedback generator, which eliminates the need for costly human intervention. Our novel framework, C$^2$, consists of (1) an automatic feedback provider (ChartAF) and (2) a diverse, reference-free dataset (ChartUIE-8K). The results are compelling: in our first experiment, 74% of respondents strongly preferred, and 10% preferred, the results after feedback. The second post-feedback experiment demonstrates that ChartAF outperform nine baselines. Moreover, ChartUIE-8K significantly improves data diversity by increasing queries, datasets, and chart types by 5982%, 1936%, and 91%, respectively, over benchmarks. Finally, a study of LLM users revealed that 94% of participants preferred ChartUIE-8K's queries, with 93% deeming them aligned with real-world use cases. Core contributions are available as open-source at chartsquared.github.io, with ample qualitative examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08353v1">Trustworthy GNNs with LLMs: A Systematic Review and Taxonomy</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Submitted to IJCAI 2025
    </div>
    <details class="paper-abstract">
      With the extensive application of Graph Neural Networks (GNNs) across various domains, their trustworthiness has emerged as a focal point of research. Some existing studies have shown that the integration of large language models (LLMs) can improve the semantic understanding and generation capabilities of GNNs, which in turn improves the trustworthiness of GNNs from various aspects. Our review introduces a taxonomy that offers researchers a clear framework for comprehending the principles and applications of different methods and helps clarify the connections and differences among various approaches. Then we systematically survey representative approaches along the four categories of our taxonomy. Through our taxonomy, researchers can understand the applicable scenarios, potential advantages, and limitations of each approach for the the trusted integration of GNNs with LLMs. Finally, we present some promising directions of work and future trends for the integration of LLMs and GNNs to improve model trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08332v1">Modification and Generated-Text Detection: Achieving Dual Detection Capabilities for the Outputs of LLM by Watermark</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      The development of large language models (LLMs) has raised concerns about potential misuse. One practical solution is to embed a watermark in the text, allowing ownership verification through watermark extraction. Existing methods primarily focus on defending against modification attacks, often neglecting other spoofing attacks. For example, attackers can alter the watermarked text to produce harmful content without compromising the presence of the watermark, which could lead to false attribution of this malicious content to the LLM. This situation poses a serious threat to the LLMs service providers and highlights the significance of achieving modification detection and generated-text detection simultaneously. Therefore, we propose a technique to detect modifications in text for unbiased watermark which is sensitive to modification. We introduce a new metric called ``discarded tokens", which measures the number of tokens not included in watermark detection. When a modification occurs, this metric changes and can serve as evidence of the modification. Additionally, we improve the watermark detection process and introduce a novel method for unbiased watermark. Our experiments demonstrate that we can achieve effective dual detection capabilities: modification detection and generated-text detection by watermark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08312v1">Word Synchronization Challenge: A Benchmark for Word Association Responses for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      This paper introduces the Word Synchronization Challenge, a novel benchmark to evaluate large language models (LLMs) in Human-Computer Interaction (HCI). This benchmark uses a dynamic game-like framework to test LLMs ability to mimic human cognitive processes through word associations. By simulating complex human interactions, it assesses how LLMs interpret and align with human thought patterns during conversational exchanges, which are essential for effective social partnerships in HCI. Initial findings highlight the influence of model sophistication on performance, offering insights into the models capabilities to engage in meaningful social interactions and adapt behaviors in human-like ways. This research advances the understanding of LLMs potential to replicate or diverge from human cognitive functions, paving the way for more nuanced and empathetic human-machine collaborations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08298v1">Improving Existing Optimization Algorithms with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into optimization has created a powerful synergy, opening exciting research opportunities. This paper investigates how LLMs can enhance existing optimization algorithms. Using their pre-trained knowledge, we demonstrate their ability to propose innovative heuristic variations and implementation strategies. To evaluate this, we applied a non-trivial optimization algorithm, Construct, Merge, Solve and Adapt (CMSA) -- a hybrid metaheuristic for combinatorial optimization problems that incorporates a heuristic in the solution construction phase. Our results show that an alternative heuristic proposed by GPT-4o outperforms the expert-designed heuristic of CMSA, with the performance gap widening on larger and denser graphs. Project URL: https://imp-opt-algo-llms.surge.sh/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03824v2">Syntriever: How to Train Your Retriever with Synthetic Data from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), Findings, Accepted
    </div>
    <details class="paper-abstract">
      LLMs have boosted progress in many AI applications. Recently, there were attempts to distill the vast knowledge of LLMs into information retrieval systems. Those distillation methods mostly use output probabilities of LLMs which are unavailable in the latest black-box LLMs. We propose Syntriever, a training framework for retrievers using synthetic data from black-box LLMs. Syntriever consists of two stages. Firstly in the distillation stage, we synthesize relevant and plausibly irrelevant passages and augmented queries using chain-of-thoughts for the given queries. LLM is asked to self-verify the synthetic data for possible hallucinations, after which retrievers are trained with a loss designed to cluster the embeddings of relevant passages. Secondly in the alignment stage, we align the retriever with the preferences of LLMs. We propose a preference modeling called partial Plackett-Luce ranking to learn LLM preferences with regularization which prevents the model from deviating excessively from that trained in the distillation stage. Experiments show that Syntriever achieves state-of-the-art performances on benchmark datasets from various domains in nDCG@$K$. The code is available at \href{https://github.com/kmswin1/Syntriever}{https://github.com/kmswin1/Syntriever}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08271v1">MoLoRec: A Generalizable and Efficient Framework for LLM-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in recent years, owing to their impressive generalization capabilities and rich world knowledge. To capitalize on the potential of using LLMs as recommender systems, mainstream approaches typically focus on two paradigms. The first paradigm designs multi-domain or multi-task instruction data for generalizable recommendation, so as to align LLMs with general recommendation areas and deal with cold-start recommendation. The second paradigm enhances domain-specific recommendation tasks with parameter-efficient fine-tuning techniques, in order to improve models under the warm recommendation scenarios. While most previous works treat these two paradigms separately, we argue that they have complementary advantages, and combining them together would be helpful. To that end, in this paper, we propose a generalizable and efficient LLM-based recommendation framework MoLoRec. Our approach starts by parameter-efficient fine-tuning a domain-general module with general recommendation instruction data, to align LLM with recommendation knowledge. Then, given users' behavior of a specific domain, we construct a domain-specific instruction dataset and apply efficient fine-tuning to the pre-trained LLM. After that, we provide approaches to integrate the above domain-general part and domain-specific part with parameters mixture. Please note that, MoLoRec is efficient with plug and play, as the domain-general module is trained only once, and any domain-specific plug-in can be efficiently merged with only domain-specific fine-tuning. Extensive experiments on multiple datasets under both warm and cold-start recommendation scenarios validate the effectiveness and generality of the proposed MoLoRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08224v1">Flow-of-Action: SOP Enhanced LLM-Based Multi-Agent System for Root Cause Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Accepted by WWW'25 Industry Track
    </div>
    <details class="paper-abstract">
      In the realm of microservices architecture, the occurrence of frequent incidents necessitates the employment of Root Cause Analysis (RCA) for swift issue resolution. It is common that a serious incident can take several domain experts hours to identify the root cause. Consequently, a contemporary trend involves harnessing Large Language Models (LLMs) as automated agents for RCA. Though the recent ReAct framework aligns well with the Site Reliability Engineers (SREs) for its thought-action-observation paradigm, its hallucinations often lead to irrelevant actions and directly affect subsequent results. Additionally, the complex and variable clues of the incident can overwhelm the model one step further. To confront these challenges, we propose Flow-of-Action, a pioneering Standard Operation Procedure (SOP) enhanced LLM-based multi-agent system. By explicitly summarizing the diagnosis steps of SREs, SOP imposes constraints on LLMs at crucial junctures, guiding the RCA process towards the correct trajectory. To facilitate the rational and effective utilization of SOPs, we design an SOP-centric framework called SOP flow. SOP flow contains a series of tools, including one for finding relevant SOPs for incidents, another for automatically generating SOPs for incidents without relevant ones, and a tool for converting SOPs into code. This significantly alleviates the hallucination issues of ReAct in RCA tasks. We also design multiple auxiliary agents to assist the main agent by removing useless noise, narrowing the search space, and informing the main agent whether the RCA procedure can stop. Compared to the ReAct method's 35.50% accuracy, our Flow-of-Action method achieves 64.01%, meeting the accuracy requirements for RCA in real-world systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07709v2">MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Open-ended learning agents must efficiently prioritize goals in vast possibility spaces, focusing on those that maximize learning progress (LP). When such autotelic exploration is achieved by LLM agents trained with online RL in high-dimensional and evolving goal spaces, a key challenge for LP prediction is modeling one's own competence, a form of metacognitive monitoring. Traditional approaches either require extensive sampling or rely on brittle expert-defined goal groupings. We introduce MAGELLAN, a metacognitive framework that lets LLM agents learn to predict their competence and LP online. By capturing semantic relationships between goals, MAGELLAN enables sample-efficient LP estimation and dynamic adaptation to evolving goal spaces through generalization. In an interactive learning environment, we show that MAGELLAN improves LP prediction efficiency and goal prioritization, being the only method allowing the agent to fully master a large and evolving goal space. These results demonstrate how augmenting LLM agents with a metacognitive ability for LP predictions can effectively scale curriculum learning to open-ended goal spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08213v1">LLM Modules: Knowledge Transfer from a Large to a Small Model using Enhanced Cross-Attention</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Code and pre-trained weights available at https://huggingface.co/kkolomeitsev/llm-modules
    </div>
    <details class="paper-abstract">
      In this work, we propose an architecture of LLM Modules that enables the transfer of knowledge from a large pre-trained model to a smaller model using an Enhanced Cross-Attention mechanism. In the proposed scheme, the Qwen2-1.5B model is frozen and its representations are passed through specially designed attention layers to the GPT-Neo-125M model, which is trained on limited computational resources. Experimental results on the Bespoke-Stratos-17k dataset demonstrate that after 15 epochs of training, the combined model generates responses comparable in quality to those obtained by distillation. We discuss the advantages of the modular approach, provide examples of input queries and comparative analysis, and outline prospects for further extension of the method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08180v1">Enhancing LLM Character-Level Manipulation via Divide and Conquer</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong generalization capabilities across a wide range of natural language processing (NLP) tasks. However, they exhibit notable weaknesses in character-level string manipulation, struggling with fundamental operations such as character deletion, insertion, and substitution. These challenges stem primarily from tokenization constraints, despite the critical role of such operations in data preprocessing and code generation. Through systematic analysis, we derive two key insights: (1) LLMs face significant difficulties in leveraging intrinsic token knowledge for character-level reasoning, and (2) atomized word structures can substantially enhance LLMs' ability to process token-level structural information. Building on these insights, we propose Character-Level Manipulation via Divide and Conquer, a novel approach designed to bridge the gap between token-level processing and character-level manipulation. Our method decomposes complex operations into explicit character-level subtasks coupled with controlled token reconstruction phases, leading to significant improvements in accuracy. Without additional training, our method significantly improves accuracies on the $\texttt{Deletion}$, $\texttt{Insertion}$, and $\texttt{Substitution}$ tasks. To support further research, we open-source our implementation and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08177v1">SycEval: Evaluating LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in educational, clinical, and professional settings, but their tendency for sycophancy -- prioritizing user agreement over independent reasoning -- poses risks to reliability. This study introduces a framework to evaluate sycophantic behavior in ChatGPT-4o, Claude-Sonnet, and Gemini-1.5-Pro across AMPS (mathematics) and MedQuad (medical advice) datasets. Sycophantic behavior was observed in 58.19% of cases, with Gemini exhibiting the highest rate (62.47%) and ChatGPT the lowest (56.71%). Progressive sycophancy, leading to correct answers, occurred in 43.52% of cases, while regressive sycophancy, leading to incorrect answers, was observed in 14.66%. Preemptive rebuttals demonstrated significantly higher sycophancy rates than in-context rebuttals (61.75% vs. 56.52%, $Z=5.87$, $p<0.001$), particularly in computational tasks, where regressive sycophancy increased significantly (preemptive: 8.13%, in-context: 3.54%, $p<0.001$). Simple rebuttals maximized progressive sycophancy ($Z=6.59$, $p<0.001$), while citation-based rebuttals exhibited the highest regressive rates ($Z=6.59$, $p<0.001$). Sycophantic behavior showed high persistence (78.5%, 95% CI: [77.2%, 79.8%]) regardless of context or model. These findings emphasize the risks and opportunities of deploying LLMs in structured and dynamic domains, offering insights into prompt programming and model optimization for safer AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20002v3">The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 This work was submitted for review on Sept. 5, 2024, and the initial version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects the up-to-date experimental results
    </div>
    <details class="paper-abstract">
      The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08145v1">Democratizing AI: Open-source Scalable LLM Training on GPU-based Supercomputers</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Training and fine-tuning large language models (LLMs) with hundreds of billions to trillions of parameters requires tens of thousands of GPUs, and a highly scalable software stack. In this work, we present a novel four-dimensional hybrid parallel algorithm implemented in a highly scalable, portable, open-source framework called AxoNN. We describe several performance optimizations in AxoNN to improve matrix multiply kernel performance, overlap non-blocking collectives with computation, and performance modeling to choose performance optimal configurations. These have resulted in unprecedented scaling and peak flop/s (bf16) for training of GPT-style transformer models on Perlmutter (620.1 Petaflop/s), Frontier (1.381 Exaflop/s) and Alps (1.423 Exaflop/s). While the abilities of LLMs improve with the number of trainable parameters, so do privacy and copyright risks caused by memorization of training data, which can cause disclosure of sensitive or private information at inference time. We highlight this side effect of scale through experiments that explore "catastrophic memorization", where models are sufficiently large to memorize training data in a single pass, and present an approach to prevent it. As part of this study, we demonstrate fine-tuning of a 405-billion parameter LLM using AxoNN on Frontier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08142v1">Bridging the Safety Gap: A Guardrail Pipeline for Trustworthy LLM Inferences</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 arXiv admin note: text overlap with arXiv:2406.10847
    </div>
    <details class="paper-abstract">
      We present Wildflare GuardRail, a guardrail pipeline designed to enhance the safety and reliability of Large Language Model (LLM) inferences by systematically addressing risks across the entire processing workflow. Wildflare GuardRail integrates several core functional modules, including Safety Detector that identifies unsafe inputs and detects hallucinations in model outputs while generating root-cause explanations, Grounding that contextualizes user queries with information retrieved from vector databases, Customizer that adjusts outputs in real time using lightweight, rule-based wrappers, and Repairer that corrects erroneous LLM outputs using hallucination explanations provided by Safety Detector. Results show that our unsafe content detection model in Safety Detector achieves comparable performance with OpenAI API, though trained on a small dataset constructed with several public datasets. Meanwhile, the lightweight wrappers can address malicious URLs in model outputs in 1.06s per query with 100% accuracy without costly model calls. Moreover, the hallucination fixing model demonstrates effectiveness in reducing hallucinations with an accuracy of 80.7%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08141v1">LowRA: Accurate and Efficient LoRA Fine-Tuning of LLMs under 2 Bits</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) is increasingly costly as models scale to hundreds of billions of parameters, and even parameter-efficient fine-tuning (PEFT) methods like LoRA remain resource-intensive. We introduce LowRA, the first framework to enable LoRA fine-tuning below 2 bits per parameter with minimal performance loss. LowRA optimizes fine-grained quantization - mapping, threshold selection, and precision assignment - while leveraging efficient CUDA kernels for scalable deployment. Extensive evaluations across 4 LLMs and 4 datasets show that LowRA achieves a superior performance-precision trade-off above 2 bits and remains accurate down to 1.15 bits, reducing memory usage by up to 50%. Our results highlight the potential of ultra-low-bit LoRA fine-tuning for resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14654v2">MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) have demonstrated significant advancements, particularly in their ability to serve as agents thereby surpassing their traditional role as chatbots. These agents can leverage their planning and tool utilization capabilities to address tasks specified at a high level. However, a standardized dataset to benchmark the agent capabilities of LLMs in medical applications is currently lacking, making the evaluation of LLMs on complex tasks in interactive healthcare environments challenging. To address this gap, we introduce MedAgentBench, a broad evaluation suite designed to assess the agent capabilities of large language models within medical records contexts. MedAgentBench encompasses 300 patient-specific clinically-derived tasks from 10 categories written by human physicians, realistic profiles of 100 patients with over 700,000 data elements, a FHIR-compliant interactive environment, and an accompanying codebase. The environment uses the standard APIs and communication infrastructure used in modern EMR systems, so it can be easily migrated into live EMR systems. MedAgentBench presents an unsaturated agent-oriented benchmark that current state-of-the-art LLMs exhibit some ability to succeed at. The best model (Claude 3.5 Sonnet v2) achieves a success rate of 69.67%. However, there is still substantial space for improvement which gives the community a next direction to optimize. Furthermore, there is significant variation in performance across task categories. MedAgentBench establishes this and is publicly available at https://github.com/stanfordmlgroup/MedAgentBench , offering a valuable framework for model developers to track progress and drive continuous improvements in the agent capabilities of large language models within the medical domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08127v1">Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Ongoing work, 13 pages, 2 figures, 3 Tables
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown strong general reasoning abilities, yet their effectiveness in financial reasoning remains underexplored. In this study, we comprehensively evaluate 16 powerful reasoning and general LLMs on three complex financial tasks involving financial text, tabular data, and equations, assessing numerical reasoning, tabular interpretation, financial terminology comprehension, long-context processing, and equation-based problem solving. Our results show that while better datasets and pretraining improve financial reasoning, general enhancements like CoT fine-tuning do not always yield consistent gains. Moreover, all reasoning strategies face challenges in improving performance on long-context and multi-table tasks. To address these limitations, we develop a financial reasoning-enhanced model based on Llama-3.1-8B-Instruct, by CoT fine-tuning and reinforcement learning with domain-specific reasoning paths. Even with simple fine-tuning with one financial dataset, our model achieves a consistent 10% performance improvement across tasks, surpassing all 8B models and even Llama3-70B-Instruct and Llama3.1-70B-Instruct on average. Our results highlight the need for domain-specific adaptations in financial tasks, emphasizing future directions such as multi-table reasoning, long-context processing, and financial terminology comprehension. All our datasets, models, and codes are publicly available. Furthermore, we introduce a leaderboard for benchmarking future datasets and models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04961v2">Demystifying Domain-adaptive Post-training for Financial LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Domain-adaptive post-training of large language models (LLMs) has emerged as a promising approach for specialized domains such as medicine and finance. However, significant challenges remain in identifying optimal adaptation criteria and training strategies across varying data and model configurations. To address these challenges, we introduce FINDAP, a systematic and fine-grained investigation into domain adaptive post-training of LLMs for the finance domain. Our approach consists of four key components: FinCap, which defines the core capabilities required for the target domain; FinRec, an effective training recipe that jointly optimizes continual pre-training and instruction-following, along with a novel preference data distillation method leveraging process signals from a generative reward model; FinTrain, a curated set of training datasets supporting FinRec; and FinEval, a comprehensive evaluation suite aligned with FinCap. The resulting model, Llama-Fin, achieves state-of-the-art performance across a wide range of financial tasks. Our analysis also highlights how each post-training stage contributes to distinct capabilities, uncovering specific challenges and effective solutions, providing valuable insights for domain adaptation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17630v2">Uncertainty Quantification and Decomposition for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 WWW 2025
    </div>
    <details class="paper-abstract">
      Despite the widespread adoption of large language models (LLMs) for recommendation, we demonstrate that LLMs often exhibit uncertainty in their recommendations. To ensure the trustworthy use of LLMs in generating recommendations, we emphasize the importance of assessing the reliability of recommendations generated by LLMs. We start by introducing a novel framework for estimating the predictive uncertainty to quantitatively measure the reliability of LLM-based recommendations. We further propose to decompose the predictive uncertainty into recommendation uncertainty and prompt uncertainty, enabling in-depth analyses of the primary source of uncertainty. Through extensive experiments, we (1) demonstrate predictive uncertainty effectively indicates the reliability of LLM-based recommendations, (2) investigate the origins of uncertainty with decomposed uncertainty measures, and (3) propose uncertainty-aware prompting for a lower predictive uncertainty and enhanced recommendation. Our source code and model weights are available at https://github.com/WonbinKweon/UNC_LLM_REC_WWW2025
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08109v1">HuDEx: Integrating Hallucination Detection and Explainability for Enhancing the Reliability of LLM responses</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have shown promising improvements, often surpassing existing methods across a wide range of downstream tasks in natural language processing. However, these models still face challenges, which may hinder their practical applicability. For example, the phenomenon of hallucination is known to compromise the reliability of LLMs, especially in fields that demand high factual precision. Current benchmarks primarily focus on hallucination detection and factuality evaluation but do not extend beyond identification. This paper proposes an explanation enhanced hallucination-detection model, coined as HuDEx, aimed at enhancing the reliability of LLM-generated responses by both detecting hallucinations and providing detailed explanations. The proposed model provides a novel approach to integrate detection with explanations, and enable both users and the LLM itself to understand and reduce errors. Our measurement results demonstrate that the proposed model surpasses larger LLMs, such as Llama3 70B and GPT-4, in hallucination detection accuracy, while maintaining reliable explanations. Furthermore, the proposed model performs well in both zero-shot and other test environments, showcasing its adaptability across diverse benchmark datasets. The proposed approach further enhances the hallucination detection research by introducing a novel approach to integrating interpretability with hallucination detection, which further enhances the performance and reliability of evaluating hallucinations in language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18794v2">Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Accepted by ICRA 2025
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) tasks require an agent to follow textual instructions to navigate through 3D environments. Traditional approaches use supervised learning methods, relying heavily on domain-specific datasets to train VLN models. Recent methods try to utilize closed-source large language models (LLMs) like GPT-4 to solve VLN tasks in zero-shot manners, but face challenges related to expensive token costs and potential data breaches in real-world applications. In this work, we introduce Open-Nav, a novel study that explores open-source LLMs for zero-shot VLN in the continuous environment. Open-Nav employs a spatial-temporal chain-of-thought (CoT) reasoning approach to break down tasks into instruction comprehension, progress estimation, and decision-making. It enhances scene perceptions with fine-grained object and spatial knowledge to improve LLM's reasoning in navigation. Our extensive experiments in both simulated and real-world environments demonstrate that Open-Nav achieves competitive performance compared to using closed-source LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07143v1">Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Accurate and efficient diagnosis in online medical consultations remains a challenge for current large language models. These models often rely on single-turn interactions and lack the ability to refine their predictions through follow-up questions. Additionally, their responses frequently contain complex medical terminology, making them less accessible to non-medical users and creating barriers to effective communication. In this paper, we introduce Ask Patients with Patience (APP), the first multi-turn dialogue that enables LLMs to iteratively refine diagnoses based on grounded reasoning. By integrating medical guidelines and entropy minimization, APP improves both diagnostic accuracy and efficiency. Furthermore, it features human-centric communication that bridges the gap between user comprehension and medical terminology, significantly enhancing user accessibility and engagement. We evaluated APP using a subset of the ReMeDi dataset, comparing it with single-turn and traditional multi-turn LLM baselines. APP achieved higher similarity scores in diagnosis predictions, demonstrating better alignment with ground truth diagnoses. Entropy analysis showed that APP reduces diagnostic uncertainty more rapidly across iterations, increasing confidence in its predictions. APP also excels in user accessibility and empathy, further bridging the gap between complex medical language and user understanding. Code will be released at: https://github.com/SuperMedIntel/AskPatients.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01618v3">A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved significant performance gains via scaling up model sizes and/or data. However, recent evidence suggests diminishing returns from such approaches, motivating scaling the computation spent at inference time. Existing inference-time scaling methods, usually with reward models, cast the task as a search problem, which tends to be vulnerable to reward hacking as a consequence of approximation errors in reward models. In this paper, we instead cast inference-time scaling as a probabilistic inference task and leverage sampling-based techniques to explore the typical set of the state distribution of a state-space model with an approximate likelihood, rather than optimize for its mode directly. We propose a novel inference-time scaling approach by adapting particle-based Monte Carlo methods to this task. Our empirical evaluation demonstrates that our methods have a 4-16x better scaling rate over our deterministic search counterparts on various challenging mathematical reasoning tasks. Using our approach, we show that Qwen2.5-Math-1.5B-Instruct can surpass GPT-4o accuracy in only 4 rollouts, while Qwen2.5-Math-7B-Instruct scales to o1 level accuracy in only 32 rollouts. Our work not only presents an effective method to inference-time scaling, but also connects the rich literature in probabilistic inference with inference-time scaling of LLMs to develop more robust algorithms in future work. Code, videos, and further information available at https://probabilistic-inference-scaling.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.24155v2">Blind Spot Navigation in LLM Reasoning with Thought Space Explorer</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated their potential in handling complex reasoning tasks, which are usually achieved by constructing a thought chain to guide the model to solve the problem with multi-step thinking. However, existing methods often remain confined to previously explored solution spaces and thus overlook the critical blind spot within LLMs' cognitive range. To address these issues, we design the Thought Space Explorer (TSE), a novel framework to expand and optimize thought structures to guide LLMs to explore their blind spots of thinking. By generating new reasoning steps and branches based on the original thought structure with various designed strategies, TSE broadens the thought space and alleviates the impact of blind spots for LLM reasoning. Experimental results on multiple levels of reasoning tasks demonstrate the efficacy of TSE. We also conduct extensive analysis to understand how structured and expansive thought can contribute to unleashing the potential of LLM reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07987v1">Universal Adversarial Attack on Aligned Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07982v1">Deep Semantic Graph Learning via LLM based Node Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Graph learning has attracted significant attention due to its widespread real-world applications. Current mainstream approaches rely on text node features and obtain initial node embeddings through shallow embedding learning using GNNs, which shows limitations in capturing deep textual semantics. Recent advances in Large Language Models (LLMs) have demonstrated superior capabilities in understanding text semantics, transforming traditional text feature processing. This paper proposes a novel framework that combines Graph Transformer architecture with LLM-enhanced node features. Specifically, we leverage LLMs to generate rich semantic representations of text nodes, which are then processed by a multi-head self-attention mechanism in the Graph Transformer to capture both local and global graph structural information. Our model utilizes the Transformer's attention mechanism to dynamically aggregate neighborhood information while preserving the semantic richness provided by LLM embeddings. Experimental results demonstrate that the LLM-enhanced node features significantly improve the performance of graph learning models on node classification tasks. This approach shows promising results across multiple graph learning tasks, offering a practical direction for combining graph networks with language models.
    </details>
</div>
