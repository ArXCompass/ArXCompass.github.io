# llm - 2024_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01736v1">Can LLMs predict the convergence of Stochastic Gradient Descent?</a></div>
    <div class="paper-meta">
      📅 2024-08-03
      | 💬 9 pages. Accepted to 1st ICML Workshop on In-Context Learning at ICML 2024
    </div>
    <details class="paper-abstract">
      Large-language models are notoriously famous for their impressive performance across a wide range of tasks. One surprising example of such impressive performance is a recently identified capacity of LLMs to understand the governing principles of dynamical systems satisfying the Markovian property. In this paper, we seek to explore this direction further by studying the dynamics of stochastic gradient descent in convex and non-convex optimization. By leveraging the theoretical link between the SGD and Markov chains, we show a remarkable zero-shot performance of LLMs in predicting the local minima to which SGD converges for previously unseen starting points. On a more general level, we inquire about the possibility of using LLMs to perform zero-shot randomized trials for larger deep learning models used in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01703v1">WaitGPT: Monitoring and Steering Conversational LLM Agent in Data Analysis with On-the-Fly Code Visualization</a></div>
    <div class="paper-meta">
      📅 2024-08-03
      | 💬 Accepted in the 37th Annual ACM Symposium on User Interface Software and Technology (UIST'24)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) support data analysis through conversational user interfaces, as exemplified in OpenAI's ChatGPT (formally known as Advanced Data Analysis or Code Interpreter). Essentially, LLMs produce code for accomplishing diverse analysis tasks. However, presenting raw code can obscure the logic and hinder user verification. To empower users with enhanced comprehension and augmented control over analysis conducted by LLMs, we propose a novel approach to transform LLM-generated code into an interactive visual representation. In the approach, users are provided with a clear, step-by-step visualization of the LLM-generated code in real time, allowing them to understand, verify, and modify individual data operations in the analysis. Our design decisions are informed by a formative study (N=8) probing into user practice and challenges. We further developed a prototype named WaitGPT and conducted a user study (N=12) to evaluate its usability and effectiveness. The findings from the user study reveal that WaitGPT facilitates monitoring and steering of data analysis performed by LLMs, enabling participants to enhance error detection and increase their overall confidence in the results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16732v2">PyBench: Evaluating LLM Agent on various real-world coding tasks</a></div>
    <div class="paper-meta">
      📅 2024-08-03
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      The LLM Agent, equipped with a code interpreter, is capable of automatically solving real-world coding tasks, such as data analysis and image editing. However, existing benchmarks primarily focus on either simplistic tasks, such as completing a few lines of code, or on extremely complex and specific tasks at the repository level, neither of which are representative of various daily coding tasks. To address this gap, we introduce \textbf{PyBench}, a benchmark encompassing five main categories of real-world tasks, covering more than 10 types of files. Given a high-level user query and related files, the LLM Agent needs to reason and execute Python code via a code interpreter for a few turns before making a formal response to fulfill the user's requirements. Successfully addressing tasks in PyBench demands a robust understanding of various Python packages, superior reasoning capabilities, and the ability to incorporate feedback from executed code. Our evaluations indicate that current open-source LLMs are struggling with these tasks. Hence, we conduct analysis and experiments on four kinds of datasets proving that comprehensive abilities are needed for PyBench. Our fine-tuned 8B size model: \textbf{PyLlama3} achieves an exciting performance on PyBench which surpasses many 33B and 70B size models. Our Benchmark, Training Dataset, and Model are available at: {https://github.com/Mercury7353/PyBench}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01623v1">Dialog Flow Induction for Constrainable LLM-Based Chatbots</a></div>
    <div class="paper-meta">
      📅 2024-08-03
      | 💬 Accepted at SIGDIAL 2024
    </div>
    <details class="paper-abstract">
      LLM-driven dialog systems are used in a diverse set of applications, ranging from healthcare to customer service. However, given their generalization capability, it is difficult to ensure that these chatbots stay within the boundaries of the specialized domains, potentially resulting in inaccurate information and irrelevant responses. This paper introduces an unsupervised approach for automatically inducing domain-specific dialog flows that can be used to constrain LLM-based chatbots. We introduce two variants of dialog flow based on the availability of in-domain conversation instances. Through human and automatic evaluation over various dialog domains, we demonstrate that our high-quality data-guided dialog flows achieve better domain coverage, thereby overcoming the need for extensive manual crafting of such flows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10230v1">A General-Purpose Device for Interaction with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-02
    </div>
    <details class="paper-abstract">
      This paper investigates integrating large language models (LLMs) with advanced hardware, focusing on developing a general-purpose device designed for enhanced interaction with LLMs. Initially, we analyze the current landscape, where virtual assistants and LLMs are reshaping human-technology interactions, highlighting pivotal advancements and setting the stage for a new era of intelligent hardware. Despite substantial progress in LLM technology, a significant gap exists in hardware development, particularly concerning scalability, efficiency, affordability, and multimodal capabilities. This disparity presents both challenges and opportunities, underscoring the need for hardware that is not only powerful but also versatile and capable of managing the sophisticated demands of modern computation. Our proposed device addresses these needs by emphasizing scalability, multimodal data processing, enhanced user interaction, and privacy considerations, offering a comprehensive platform for LLM integration in various applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04645v1">Evaluating the Impact of Advanced LLM Techniques on AI-Lecture Tutors for a Robotics Course</a></div>
    <div class="paper-meta">
      📅 2024-08-02
      | 💬 The article is an extended version of a paper presented at the International Workshop on AI in Education and Educational Research (AIEER) at ECAI-2024 (27th European Conference on Artificial Intelligence)
    </div>
    <details class="paper-abstract">
      This study evaluates the performance of Large Language Models (LLMs) as an Artificial Intelligence-based tutor for a university course. In particular, different advanced techniques are utilized, such as prompt engineering, Retrieval-Augmented-Generation (RAG), and fine-tuning. We assessed the different models and applied techniques using common similarity metrics like BLEU-4, ROUGE, and BERTScore, complemented by a small human evaluation of helpfulness and trustworthiness. Our findings indicate that RAG combined with prompt engineering significantly enhances model responses and produces better factual answers. In the context of education, RAG appears as an ideal technique as it is based on enriching the input of the model with additional information and material which usually is already present for a university course. Fine-tuning, on the other hand, can produce quite small, still strong expert models, but poses the danger of overfitting. Our study further asks how we measure performance of LLMs and how well current measurements represent correctness or relevance? We find high correlation on similarity metrics and a bias of most of these metrics towards shorter responses. Overall, our research points to both the potential and challenges of integrating LLMs in educational settings, suggesting a need for balanced training approaches and advanced evaluation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01423v1">Prompt Recursive Search: A Living Framework with Adaptive Growth in LLM Auto-Prompting</a></div>
    <div class="paper-meta">
      📅 2024-08-02
      | 💬 8 pages,4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit remarkable proficiency in addressing a diverse array of tasks within the Natural Language Processing (NLP) domain, with various prompt design strategies significantly augmenting their capabilities. However, these prompts, while beneficial, each possess inherent limitations. The primary prompt design methodologies are twofold: The first, exemplified by the Chain of Thought (CoT), involves manually crafting prompts specific to individual datasets, hence termed Expert-Designed Prompts (EDPs). Once these prompts are established, they are unalterable, and their effectiveness is capped by the expertise of the human designers. When applied to LLMs, the static nature of EDPs results in a uniform approach to both simple and complex problems within the same dataset, leading to the inefficient use of tokens for straightforward issues. The second method involves prompts autonomously generated by the LLM, known as LLM-Derived Prompts (LDPs), which provide tailored solutions to specific problems, mitigating the limitations of EDPs. However, LDPs may encounter a decline in performance when tackling complex problems due to the potential for error accumulation during the solution planning process. To address these challenges, we have conceived a novel Prompt Recursive Search (PRS) framework that leverages the LLM to generate solutions specific to the problem, thereby conserving tokens. The framework incorporates an assessment of problem complexity and an adjustable structure, ensuring a reduction in the likelihood of errors. We have substantiated the efficacy of PRS framework through extensive experiments using LLMs with different numbers of parameters across a spectrum of datasets in various domains. Compared to the CoT method, the PRS method has increased the accuracy on the BBH dataset by 8% using Llama3-7B model, achieving a 22% improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01420v1">Mission Impossible: A Statistical Perspective on Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on a deluge of text data with limited quality control. As a result, LLMs can exhibit unintended or even harmful behaviours, such as leaking information, fake news or hate speech. Countermeasures, commonly referred to as preference alignment, include fine-tuning the pretrained LLMs with carefully crafted text examples of desired behaviour. Even then, empirical evidence shows preference aligned LLMs can be enticed to harmful behaviour. This so called jailbreaking of LLMs is typically achieved by adversarially modifying the input prompt to the LLM. Our paper provides theoretical insights into the phenomenon of preference alignment and jailbreaking from a statistical perspective. Under our framework, we first show that pretrained LLMs will mimic harmful behaviour if present in the training corpus. Under that same framework, we then introduce a statistical notion of alignment, and lower-bound the jailbreaking probability, showing that it is unpreventable under reasonable assumptions. Based on our insights, we propose an alteration to the currently prevalent alignment strategy RLHF. Specifically, we introduce a simple modification to the RLHF objective, we call E-RLHF, that aims to increase the likelihood of safe responses. E-RLHF brings no additional training cost, and is compatible with other methods. Empirically, we demonstrate that E-RLHF outperforms RLHF on all alignment problems put forward by the AdvBench and HarmBench project without sacrificing model performance as measured by the MT-Bench project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01354v1">MCGMark: An Encodable and Robust Online Watermark for LLM-Generated Malicious Code</a></div>
    <div class="paper-meta">
      📅 2024-08-02
    </div>
    <details class="paper-abstract">
      With the advent of large language models (LLMs), numerous software service providers (SSPs) are dedicated to developing LLMs customized for code generation tasks, such as CodeLlama and Copilot. However, these LLMs can be leveraged by attackers to create malicious software, which may pose potential threats to the software ecosystem. For example, they can automate the creation of advanced phishing malware. To address this issue, we first conduct an empirical study and design a prompt dataset, MCGTest, which involves approximately 400 person-hours of work and consists of 406 malicious code generation tasks. Utilizing this dataset, we propose MCGMark, the first robust, code structure-aware, and encodable watermarking approach to trace LLM-generated code. We embed encodable information by controlling the token selection and ensuring the output quality based on probabilistic outliers. Additionally, we enhance the robustness of the watermark by considering the structural features of malicious code, preventing the embedding of the watermark in easily modified positions, such as comments. We validate the effectiveness and robustness of MCGMark on the DeepSeek-Coder. MCGMark achieves an embedding success rate of 88.9% within a maximum output limit of 400 tokens. Furthermore, it also demonstrates strong robustness and has minimal impact on the quality of the output code. Our approach assists SSPs in tracing and holding responsible parties accountable for malicious code generated by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01346v1">Prompt Refinement or Fine-tuning? Best Practices for using LLMs in Computational Social Science Tasks</a></div>
    <div class="paper-meta">
      📅 2024-08-02
      | 💬 5 pages, 1 table
    </div>
    <details class="paper-abstract">
      Large Language Models are expressive tools that enable complex tasks of text understanding within Computational Social Science. Their versatility, while beneficial, poses a barrier for establishing standardized best practices within the field. To bring clarity on the values of different strategies, we present an overview of the performance of modern LLM-based classification methods on a benchmark of 23 social knowledge tasks. Our results point to three best practices: select models with larger vocabulary and pre-training corpora; avoid simple zero-shot in favor of AI-enhanced prompting; fine-tune on task-specific data, and consider more complex forms instruction-tuning on multiple datasets only when only training data is more abundant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01323v1">FANNO: Augmenting High-Quality Instruction Data with Open-Sourced LLMs Only</a></div>
    <div class="paper-meta">
      📅 2024-08-02
    </div>
    <details class="paper-abstract">
      Instruction fine-tuning stands as a crucial advancement in leveraging large language models (LLMs) for enhanced task performance. However, the annotation of instruction datasets has traditionally been expensive and laborious, often relying on manual annotations or costly API calls of proprietary LLMs. To address these challenges, we introduce FANNO, a fully autonomous, open-sourced framework that revolutionizes the annotation process without the need for pre-existing annotated data. Utilizing a Mistral-7b-instruct model, FANNO efficiently produces diverse and high-quality datasets through a structured process involving document pre-screening, instruction generation, and response generation. Experiments on Open LLM Leaderboard and AlpacaEval benchmark show that the FANNO can generate high-quality data with diversity and complexity for free, comparable to human-annotated or cleaned datasets like Alpaca-GPT4-Cleaned.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02659v2">LLMs Plagiarize: Ensuring Responsible Sourcing of Large Language Model Training Data Through Knowledge Graph Comparison</a></div>
    <div class="paper-meta">
      📅 2024-08-02
    </div>
    <details class="paper-abstract">
      In light of recent legal allegations brought by publishers, newspapers, and other creators of copyrighted corpora against large language model developers who use their copyrighted materials for training or fine-tuning purposes, we propose a novel system, a variant of a plagiarism detection system, that assesses whether a knowledge source has been used in the training or fine-tuning of a large language model. Unlike current methods, we utilize an approach that uses Resource Description Framework (RDF) triples to create knowledge graphs from both a source document and an LLM continuation of that document. These graphs are then analyzed with respect to content using cosine similarity and with respect to structure using a normalized version of graph edit distance that shows the degree of isomorphism. Unlike traditional plagiarism systems that focus on content matching and keyword identification between a source and a target corpus, our approach enables a broader and more accurate evaluation of similarity between a source document and LLM continuation by focusing on relationships between ideas and their organization with regards to others. Additionally, our approach does not require access to LLM metrics like perplexity that may be unavailable in closed large language model "black-box" systems, as well as the training corpus. We thus assess whether an LLM has "plagiarized" a corpus in its continuation through similarity measures. A prototype of our system will be found on a hyperlinked GitHub repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19630v2">LLMs' Understanding of Natural Language Revealed</a></div>
    <div class="paper-meta">
      📅 2024-08-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are the result of a massive experiment in bottom-up, data-driven reverse engineering of language at scale. Despite their utility in a number of downstream NLP tasks, ample research has shown that LLMs are incapable of performing reasoning in tasks that require quantification over and the manipulation of symbolic variables (e.g., planning and problem solving); see for example [25][26]. In this document, however, we will focus on testing LLMs for their language understanding capabilities, their supposed forte. As we will show here, the language understanding capabilities of LLMs have been widely exaggerated. While LLMs have proven to generate human-like coherent language (since that's how they were designed), their language understanding capabilities have not been properly tested. In particular, we believe that the language understanding capabilities of LLMs should be tested by performing an operation that is the opposite of 'text generation' and specifically by giving the LLM snippets of text as input and then querying what the LLM "understood". As we show here, when doing so it will become apparent that LLMs do not truly understand language, beyond very superficial inferences that are essentially the byproduct of the memorization of massive amounts of ingested text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01168v1">Misinforming LLMs: vulnerabilities, challenges and opportunities</a></div>
    <div class="paper-meta">
      📅 2024-08-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant advances in natural language processing, but their underlying mechanisms are often misunderstood. Despite exhibiting coherent answers and apparent reasoning behaviors, LLMs rely on statistical patterns in word embeddings rather than true cognitive processes. This leads to vulnerabilities such as "hallucination" and misinformation. The paper argues that current LLM architectures are inherently untrustworthy due to their reliance on correlations of sequential patterns of word embedding vectors. However, ongoing research into combining generative transformer-based models with fact bases and logic programming languages may lead to the development of trustworthy LLMs capable of generating statements based on given truth and explaining their self-reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01122v1">CFBench: A Comprehensive Constraints-Following Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-02
      | 💬 15 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The adeptness of Large Language Models (LLMs) in comprehending and following natural language instructions is critical for their deployment in sophisticated real-world applications. Existing evaluations mainly focus on fragmented constraints or narrow scenarios, but they overlook the comprehensiveness and authenticity of constraints from the user's perspective. To bridge this gap, we propose CFBench, a large-scale Comprehensive Constraints Following Benchmark for LLMs, featuring 1,000 curated samples that cover more than 200 real-life scenarios and over 50 NLP tasks. CFBench meticulously compiles constraints from real-world instructions and constructs an innovative systematic framework for constraint types, which includes 10 primary categories and over 25 subcategories, and ensures each constraint is seamlessly integrated within the instructions. To make certain that the evaluation of LLM outputs aligns with user perceptions, we propose an advanced methodology that integrates multi-dimensional assessment criteria with requirement prioritization, covering various perspectives of constraints, instructions, and requirement fulfillment. Evaluating current leading LLMs on CFBench reveals substantial room for improvement in constraints following, and we further investigate influencing factors and enhancement strategies. The data and code are publicly available at https://github.com/PKU-Baichuan-MLSystemLab/CFBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01055v1">LLM as Runtime Error Handler: A Promising Pathway to Adaptive Self-Healing of Software Systems</a></div>
    <div class="paper-meta">
      📅 2024-08-02
    </div>
    <details class="paper-abstract">
      Unanticipated runtime errors, lacking predefined handlers, can abruptly terminate execution and lead to severe consequences, such as data loss or system crashes. Despite extensive efforts to identify potential errors during the development phase, such unanticipated errors remain a challenge to to be entirely eliminated, making the runtime mitigation measurements still indispensable to minimize their impact. Automated self-healing techniques, such as reusing existing handlers, have been investigated to reduce the loss coming through with the execution termination. However, the usability of existing methods is retained by their predefined heuristic rules and they fail to handle diverse runtime errors adaptively. Recently, the advent of Large Language Models (LLMs) has opened new avenues for addressing this problem. Inspired by their remarkable capabilities in understanding and generating code, we propose to deal with the runtime errors in a real-time manner using LLMs. Specifically, we propose Healer, the first LLM-assisted self-healing framework for handling runtime errors. When an unhandled runtime error occurs, Healer will be activated to generate a piece of error-handling code with the help of its internal LLM and the code will be executed inside the runtime environment owned by the framework to obtain a rectified program state from which the program should continue its execution. Our exploratory study evaluates the performance of Healer using four different code benchmarks and three state-of-the-art LLMs, GPT-3.5, GPT-4, and CodeQwen-7B. Results show that, without the need for any fine-tuning, GPT-4 can successfully help programs recover from 72.8% of runtime errors, highlighting the potential of LLMs in handling runtime errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01008v1">Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-02
      | 💬 LA-UR-24-28177
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing (NLP) tasks, such as question-answering, sentiment analysis, text summarization, and machine translation. However, the ever-growing complexity of LLMs demands immense computational resources, hindering the broader research and application of these models. To address this, various parameter-efficient fine-tuning strategies, such as Low-Rank Approximation (LoRA) and Adapters, have been developed. Despite their potential, these methods often face limitations in compressibility. Specifically, LoRA struggles to scale effectively with the increasing number of trainable parameters in modern large scale LLMs. Additionally, Low-Rank Economic Tensor-Train Adaptation (LoRETTA), which utilizes tensor train decomposition, has not yet achieved the level of compression necessary for fine-tuning very large scale models with limited resources. This paper introduces Tensor Train Low-Rank Approximation (TT-LoRA), a novel parameter-efficient fine-tuning (PEFT) approach that extends LoRETTA with optimized tensor train (TT) decomposition integration. By eliminating Adapters and traditional LoRA-based structures, TT-LoRA achieves greater model compression without compromising downstream task performance, along with reduced inference latency and computational overhead. We conduct an exhaustive parameter search to establish benchmarks that highlight the trade-off between model compression and performance. Our results demonstrate significant compression of LLMs while maintaining comparable performance to larger models, facilitating their deployment on resource-constraint platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00948v1">Leveraging Large Language Models (LLMs) for Traffic Management at Urban Intersections: The Case of Mixed Traffic Scenarios</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      Urban traffic management faces significant challenges due to the dynamic environments, and traditional algorithms fail to quickly adapt to this environment in real-time and predict possible conflicts. This study explores the ability of a Large Language Model (LLM), specifically, GPT-4o-mini to improve traffic management at urban intersections. We recruited GPT-4o-mini to analyze, predict position, detect and resolve the conflicts at an intersection in real-time for various basic scenarios. The key findings of this study to investigate whether LLMs can logically reason and understand the scenarios to enhance the traffic efficiency and safety by providing real-time analysis. The study highlights the potential of LLMs in urban traffic management creating more intelligent and more adaptive systems. Results showed the GPT-4o-mini was effectively able to detect and resolve conflicts in heavy traffic, congestion, and mixed-speed conditions. The complex scenario of multiple intersections with obstacles and pedestrians saw successful conflict management as well. Results show that the integration of LLMs promises to improve the effectiveness of traffic control for safer and more efficient urban intersection management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01168v2">How Ethical Should AI Be? How AI Alignment Shapes the Risk Preferences of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      This study examines the risk preferences of Large Language Models (LLMs) and how aligning them with human ethical standards affects their economic decision-making. Analyzing 30 LLMs reveals a range of inherent risk profiles, from risk-averse to risk-seeking. We find that aligning LLMs with human values, focusing on harmlessness, helpfulness, and honesty, shifts them towards risk aversion. While some alignment improves investment forecast accuracy, excessive alignment leads to overly cautious predictions, potentially resulting in severe underinvestment. Our findings highlight the need for a nuanced approach that balances ethical alignment with the specific requirements of economic domains when using LLMs in finance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04643v1">Risks, Causes, and Mitigations of Widespread Deployments of Large Language Models (LLMs): A Survey</a></div>
    <div class="paper-meta">
      📅 2024-08-01
      | 💬 Accepted to 2nd International Conference on Artificial Intelligence, Blockchain, and Internet of Things (AIBThings-2024), September 07-08, 2024, Michigan, USA
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs), such as ChatGPT and LLaMA, have significantly transformed Natural Language Processing (NLP) with their outstanding abilities in text generation, summarization, and classification. Nevertheless, their widespread adoption introduces numerous challenges, including issues related to academic integrity, copyright, environmental impacts, and ethical considerations such as data bias, fairness, and privacy. The rapid evolution of LLMs also raises concerns regarding the reliability and generalizability of their evaluations. This paper offers a comprehensive survey of the literature on these subjects, systematically gathered and synthesized from Google Scholar. Our study provides an in-depth analysis of the risks associated with specific LLMs, identifying sub-risks, their causes, and potential solutions. Furthermore, we explore the broader challenges related to LLMs, detailing their causes and proposing mitigation strategies. Through this literature analysis, our survey aims to deepen the understanding of the implications and complexities surrounding these powerful models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10886v2">SLIP: Securing LLMs IP Using Weights Decomposition</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently seen widespread adoption, in both academia and industry. As these models grow, they become valuable intellectual property (IP), reflecting enormous investments by their owners. Moreover, the high cost of cloud-based deployment has driven interest towards deployment to edge devices, yet this risks exposing valuable parameters to theft and unauthorized use. Current methods to protect models' IP on the edge have limitations in terms of practicality, loss in accuracy, or suitability to requirements. In this paper, we introduce a novel hybrid inference algorithm, named SLIP, designed to protect edge-deployed models from theft. SLIP is the first hybrid protocol that is both practical for real-world applications and provably secure, while having zero accuracy degradation and minimal impact on latency. It involves partitioning the model between two computing resources, one secure but expensive, and another cost-effective but vulnerable. This is achieved through matrix decomposition, ensuring that the secure resource retains a maximally sensitive portion of the model's IP while performing a minimal amount of computations, and vice versa for the vulnerable resource. Importantly, the protocol includes security guarantees that prevent attackers from exploiting the partition to infer the secured information. Finally, we present experimental results that show the robustness and effectiveness of our method, positioning it as a compelling solution for protecting LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20240v2">Social and Ethical Risks Posed by General-Purpose LLMs for Settling Newcomers in Canada</a></div>
    <div class="paper-meta">
      📅 2024-08-01
      | 💬 26 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The non-profit settlement sector in Canada supports newcomers in achieving successful integration. This sector faces increasing operational pressures amidst rising immigration targets, which highlights a need for enhanced efficiency and innovation, potentially through reliable AI solutions. The ad-hoc use of general-purpose generative AI, such as ChatGPT, might become a common practice among newcomers and service providers to address this need. However, these tools are not tailored for the settlement domain and can have detrimental implications for immigrants and refugees. We explore the risks that these tools might pose on newcomers to first, warn against the unguarded use of generative AI, and second, to incentivize further research and development in creating AI literacy programs as well as customized LLMs that are aligned with the preferences of the impacted communities. Crucially, such technologies should be designed to integrate seamlessly into the existing workflow of the settlement sector, ensuring human oversight, trustworthiness, and accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00878v1">Multi-Aspect Reviewed-Item Retrieval via LLM Query Decomposition and Aspect Fusion</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      While user-generated product reviews often contain large quantities of information, their utility in addressing natural language product queries has been limited, with a key challenge being the need to aggregate information from multiple low-level sources (reviews) to a higher item level during retrieval. Existing methods for reviewed-item retrieval (RIR) typically take a late fusion (LF) approach which computes query-item scores by simply averaging the top-K query-review similarity scores for an item. However, we demonstrate that for multi-aspect queries and multi-aspect items, LF is highly sensitive to the distribution of aspects covered by reviews in terms of aspect frequency and the degree of aspect separation across reviews. To address these LF failures, we propose several novel aspect fusion (AF) strategies which include Large Language Model (LLM) query extraction and generative reranking. Our experiments show that for imbalanced review corpora, AF can improve over LF by a MAP@10 increase from 0.36 to 0.52, while achieving equivalent performance for balanced review corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00741v1">DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      The rapid evolution and widespread adoption of generative large language models (LLMs) have made them a pivotal workload in various applications. Today, LLM inference clusters receive a large number of queries with strict Service Level Objectives (SLOs). To achieve the desired performance, these models execute on power-hungry GPUs causing the inference clusters to consume large amount of energy and, consequently, result in excessive carbon emissions. Fortunately, we find that there is a great opportunity to exploit the heterogeneity in inference compute properties and fluctuations in inference workloads, to significantly improve energy-efficiency. However, such a diverse and dynamic environment creates a large search-space where different system configurations (e.g., number of instances, model parallelism, and GPU frequency) translate into different energy-performance trade-offs. To address these challenges, we propose DynamoLLM, the first energy-management framework for LLM inference environments. DynamoLLM automatically and dynamically reconfigures the inference cluster to optimize for energy and cost of LLM serving under the service's performance SLOs. We show that at a service-level, DynamoLLM conserves 53% energy and 38% operational carbon emissions, and reduces 61% cost to the customer, while meeting the latency SLOs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13692v2">Prover-Verifier Games improve legibility of LLM outputs</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00818v1">Y Social: an LLM-powered Social Media Digital Twin</a></div>
    <div class="paper-meta">
      📅 2024-08-01
      | 💬 29 pages, 5 figures
    </div>
    <details class="paper-abstract">
      In this paper we introduce Y, a new-generation digital twin designed to replicate an online social media platform. Digital twins are virtual replicas of physical systems that allow for advanced analyses and experimentation. In the case of social media, a digital twin such as Y provides a powerful tool for researchers to simulate and understand complex online interactions. {\tt Y} leverages state-of-the-art Large Language Models (LLMs) to replicate sophisticated agent behaviors, enabling accurate simulations of user interactions, content dissemination, and network dynamics. By integrating these aspects, Y offers valuable insights into user engagement, information spread, and the impact of platform policies. Moreover, the integration of LLMs allows Y to generate nuanced textual content and predict user responses, facilitating the study of emergent phenomena in online environments. To better characterize the proposed digital twin, in this paper we describe the rationale behind its implementation, provide examples of the analyses that can be performed on the data it enables to be generated, and discuss its relevance for multidisciplinary research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10251v3">The Impact of Quantization on Retrieval-Augmented Generation: An Analysis of Small LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-01
      | 💬 Accepted to the IR-RAG Workshop at SIGIR 2024
    </div>
    <details class="paper-abstract">
      Post-training quantization reduces the computational demand of Large Language Models (LLMs) but can weaken some of their capabilities. Since LLM abilities emerge with scale, smaller LLMs are more sensitive to quantization. In this paper, we explore how quantization affects smaller LLMs' ability to perform retrieval-augmented generation (RAG), specifically in longer contexts. We chose personalization for evaluation because it is a challenging domain to perform using RAG as it requires long-context reasoning over multiple documents. We compare the original FP16 and the quantized INT4 performance of multiple 7B and 8B LLMs on two tasks while progressively increasing the number of retrieved documents to test how quantized models fare against longer contexts. To better understand the effect of retrieval, we evaluate three retrieval models in our experiments. Our findings reveal that if a 7B LLM performs the task well, quantization does not impair its performance and long-context reasoning capabilities. We conclude that it is possible to utilize RAG with quantized smaller LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07822v1">Exploration of LLMs, EEG, and behavioral data to measure and support attention and sleep</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      We explore the application of large language models (LLMs), pre-trained models with massive textual data for detecting and improving these altered states. We investigate the use of LLMs to estimate attention states, sleep stages, and sleep quality and generate sleep improvement suggestions and adaptive guided imagery scripts based on electroencephalogram (EEG) and physical activity data (e.g. waveforms, power spectrogram images, numerical features). Our results show that LLMs can estimate sleep quality based on human textual behavioral features and provide personalized sleep improvement suggestions and guided imagery scripts; however detecting attention, sleep stages, and sleep quality based on EEG and activity data requires further training data and domain-specific knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11733v2">How Are LLMs Mitigating Stereotyping Harms? Learning from Search Engine Studies</a></div>
    <div class="paper-meta">
      📅 2024-08-01
      | 💬 Accepted at AAAI/ACM AI, Ethics, and Society
    </div>
    <details class="paper-abstract">
      With the widespread availability of LLMs since the release of ChatGPT and increased public scrutiny, commercial model development appears to have focused their efforts on 'safety' training concerning legal liabilities at the expense of social impact evaluation. This mimics a similar trend which we could observe for search engine autocompletion some years prior. We draw on scholarship from NLP and search engine auditing and present a novel evaluation task in the style of autocompletion prompts to assess stereotyping in LLMs. We assess LLMs by using four metrics, namely refusal rates, toxicity, sentiment and regard, with and without safety system prompts. Our findings indicate an improvement to stereotyping outputs with the system prompt, but overall a lack of attention by LLMs under study to certain harms classified as toxic, particularly for prompts about peoples/ethnicities and sexual orientation. Mentions of intersectional identities trigger a disproportionate amount of stereotyping. Finally, we discuss the implications of these findings about stereotyping harms in light of the coming intermingling of LLMs and search and the choice of stereotyping mitigation policy to adopt. We address model builders, academics, NLP practitioners and policy makers, calling for accountability and awareness concerning stereotyping harms, be it for training data curation, leader board design and usage, or social impact measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00539v1">Intermittent Semi-working Mask: A New Masking Paradigm for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      Multi-turn dialogues are a key interaction method between humans and Large Language Models (LLMs), as conversations extend over multiple rounds, keeping LLMs' high generation quality and low latency is a challenge. Mainstream LLMs can be grouped into two categories based on masking strategy: causal LLM and prefix LLM. Several works have demonstrated that prefix LLMs tend to outperform causal ones in scenarios that heavily depend on historical context such as multi-turn dialogues or in-context learning, thanks to their bidirectional attention on prefix sequences. However, prefix LLMs have an inherent inefficient training problem in multi-turn dialogue datasets. In addition, the attention mechanism of prefix LLM makes it unable to reuse Key-Value Cache (KV Cache) across dialogue rounds to reduce generation latency. In this paper, we propose a novel masking scheme called Intermittent Semi-working Mask (ISM) to address these problems. Specifically, we apply alternate bidirectional and unidirectional attention on queries and answers in the dialogue history. In this way, ISM is able to maintain the high quality of prefix LLM and low generation latency of causal LLM, simultaneously. Extensive experiments illustrate that our ISM achieves significant performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00462v1">Designing Efficient LLM Accelerators for Edge Devices</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      The increase in open-source availability of Large Language Models (LLMs) has enabled users to deploy them on more and more resource-constrained edge devices to reduce reliance on network connections and provide more privacy. However, the high computation and memory demands of LLMs make their execution on resource-constrained edge devices challenging and inefficient. To address this issue, designing new and efficient edge accelerators for LLM inference is crucial. FPGA-based accelerators are ideal for LLM acceleration due to their reconfigurability, as they enable model-specific optimizations and higher performance per watt. However, creating and integrating FPGA-based accelerators for LLMs (particularly on edge devices) has proven challenging, mainly due to the limited hardware design flows for LLMs in existing FPGA platforms. To tackle this issue, in this paper we first propose a new design platform, named SECDA-LLM, that utilizes the SECDA methodology to streamline the process of designing, integrating, and deploying efficient FPGA-based LLM accelerators for the llama.cpp inference framework. We then demonstrate, through a case study, the potential benefits of SECDA-LLM by creating a new MatMul accelerator that supports block floating point quantized operations for LLMs. Our initial accelerator design, deployed on the PYNQ-Z1 board, reduces latency 1.7 seconds per token or ~2 seconds per word) by 11x over the dual-core Arm NEON-based CPU execution for the TinyLlama model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00352v1">Autonomous LLM-Enhanced Adversarial Attack for Text-to-Motion</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      Human motion generation driven by deep generative models has enabled compelling applications, but the ability of text-to-motion (T2M) models to produce realistic motions from text prompts raises security concerns if exploited maliciously. Despite growing interest in T2M, few methods focus on safeguarding these models against adversarial attacks, with existing work on text-to-image models proving insufficient for the unique motion domain. In the paper, we propose ALERT-Motion, an autonomous framework leveraging large language models (LLMs) to craft targeted adversarial attacks against black-box T2M models. Unlike prior methods modifying prompts through predefined rules, ALERT-Motion uses LLMs' knowledge of human motion to autonomously generate subtle yet powerful adversarial text descriptions. It comprises two key modules: an adaptive dispatching module that constructs an LLM-based agent to iteratively refine and search for adversarial prompts; and a multimodal information contrastive module that extracts semantically relevant motion information to guide the agent's search. Through this LLM-driven approach, ALERT-Motion crafts adversarial prompts querying victim models to produce outputs closely matching targeted motions, while avoiding obvious perturbations. Evaluations across popular T2M models demonstrate ALERT-Motion's superiority over previous methods, achieving higher attack success rates with stealthier adversarial prompts. This pioneering work on T2M adversarial attacks highlights the urgency of developing defensive measures as motion generation technology advances, urging further research into safe and responsible deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.11409v4">LLMs as Hackers: Autonomous Linux Privilege Escalation Attacks</a></div>
    <div class="paper-meta">
      📅 2024-08-01
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Penetration testing, an essential component of software security testing, allows organizations to identify and remediate vulnerabilities in their systems, thus bolstering their defense mechanisms against cyberattacks. One recent advancement in the realm of penetration testing is the utilization of Language Models (LLMs). We explore the intersection of LLMs and penetration testing to gain insight into their capabilities and challenges in the context of privilege escalation. We introduce a fully automated privilege-escalation tool designed for evaluating the efficacy of LLMs for (ethical) hacking, executing benchmarks using multiple LLMs, and investigating their respective results. Our results show that GPT-4-turbo is well suited to exploit vulnerabilities (33-83% of vulnerabilities). GPT-3.5-turbo can abuse 16-50% of vulnerabilities, while local models, such as Llama3, can only exploit between 0 and 33% of the vulnerabilities. We analyze the impact of different context sizes, in-context learning, optional high-level guidance mechanisms, and memory management techniques. We discuss challenging areas for LLMs, including maintaining focus during testing, coping with errors, and finally comparing LLMs with human hackers. The current version of the LLM-guided privilege-escalation prototype can be found at https://github.com/ipa-labs/hackingBuddyGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00214v1">Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization: A Case Study of Power Control</a></div>
    <div class="paper-meta">
      📅 2024-08-01
    </div>
    <details class="paper-abstract">
      Large language model (LLM) has recently been considered a promising technique for many fields. This work explores LLM-based wireless network optimization via in-context learning. To showcase the potential of LLM technologies, we consider the base station (BS) power control as a case study, a fundamental but crucial technique that is widely investigated in wireless networks. Different from existing machine learning (ML) methods, our proposed in-context learning algorithm relies on LLM's inference capabilities. It avoids the complexity of tedious model training and hyper-parameter fine-tuning, which is a well-known bottleneck of many ML algorithms. Specifically, the proposed algorithm first describes the target task via formatted natural language, and then designs the in-context learning framework and demonstration examples. After that, it considers two cases, namely discrete-state and continuous-state problems, and proposes state-based and ranking-based methods to select appropriate examples for these two cases, respectively. Finally, the simulations demonstrate that the proposed algorithm can achieve comparable performance as conventional deep reinforcement learning (DRL) techniques without dedicated model training or fine-tuning. Such an efficient and low-complexity approach has great potential for future wireless network optimization.
    </details>
</div>
