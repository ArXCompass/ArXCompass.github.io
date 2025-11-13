# llm - 2024_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00122v1">A Course Shared Task on Evaluating LLM Output for Clinical Questions</a></div>
    <div class="paper-meta">
      📅 2024-07-31
      | 💬 accepted at the sixth Workshop on Teaching NLP (co-located with ACL 2024)
    </div>
    <details class="paper-abstract">
      This paper presents a shared task that we organized at the Foundations of Language Technology (FoLT) course in 2023/2024 at the Technical University of Darmstadt, which focuses on evaluating the output of Large Language Models (LLMs) in generating harmful answers to health-related clinical questions. We describe the task design considerations and report the feedback we received from the students. We expect the task and the findings reported in this paper to be relevant for instructors teaching natural language processing (NLP) and designing course assignments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20999v2">MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-07-31
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated remarkable capabilities in a wide range of tasks. Typically, an LLM is pre-trained on large corpora and subsequently fine-tuned on task-specific datasets. However, during fine-tuning, LLMs may forget the knowledge acquired in the pre-training stage, leading to a decline in general capabilities. To address this issue, we propose a new fine-tuning algorithm termed Momentum-Filtered Optimizer (MoFO). The key idea of MoFO is to iteratively select and update the model parameters with the largest momentum magnitudes. Compared to full-parameter training, MoFO achieves similar fine-tuning performance while keeping parameters closer to the pre-trained model, thereby mitigating knowledge forgetting. Unlike most existing methods for forgetting mitigation, MoFO combines the following two advantages. First, MoFO does not require access to pre-training data. This makes MoFO particularly suitable for fine-tuning scenarios where pre-training data is unavailable, such as fine-tuning checkpoint-only open-source LLMs. Second, MoFO does not alter the original loss function. This could avoid impairing the model performance on the fine-tuning tasks. We validate MoFO through rigorous convergence analysis and extensive experiments, demonstrating its superiority over existing methods in mitigating forgetting and enhancing fine-tuning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21778v1">Tulip Agent -- Enabling LLM-Based Agents to Solve Tasks Using Large Tool Libraries</a></div>
    <div class="paper-meta">
      📅 2024-07-31
      | 💬 19 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We introduce tulip agent, an architecture for autonomous LLM-based agents with Create, Read, Update, and Delete access to a tool library containing a potentially large number of tools. In contrast to state-of-the-art implementations, tulip agent does not encode the descriptions of all available tools in the system prompt, which counts against the model's context window, or embed the entire prompt for retrieving suitable tools. Instead, the tulip agent can recursively search for suitable tools in its extensible tool library, implemented exemplarily as a vector store. The tulip agent architecture significantly reduces inference costs, allows using even large tool libraries, and enables the agent to adapt and extend its set of tools. We evaluate the architecture with several ablation studies in a mathematics context and demonstrate its generalizability with an application to robotics. A reference implementation and the benchmark are available at github.com/HRI-EU/tulip_agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21647v1">Human interaction classifier for LLM based chatbot</a></div>
    <div class="paper-meta">
      📅 2024-07-31
      | 💬 16 pages, 13 figures
    </div>
    <details class="paper-abstract">
      This study investigates different approaches to classify human interactions in an artificial intelligence-based environment, specifically for Applus+ IDIADA's intelligent agent AIDA. The main objective is to develop a classifier that accurately identifies the type of interaction received (Conversation, Services, or Document Translation) to direct requests to the appropriate channel and provide a more specialized and efficient service. Various models are compared, including LLM-based classifiers, KNN using Titan and Cohere embeddings, SVM, and artificial neural networks. Results show that SVM and ANN models with Cohere embeddings achieve the best overall performance, with superior F1 scores and faster execution times compared to LLM-based approaches. The study concludes that the SVM model with Cohere embeddings is the most suitable option for classifying human interactions in the AIDA environment, offering an optimal balance between accuracy and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21593v1">LLM-for-X: Application-agnostic Integration of Large Language Models to Support Personal Writing Workflows</a></div>
    <div class="paper-meta">
      📅 2024-07-31
    </div>
    <details class="paper-abstract">
      To enhance productivity and to streamline workflows, there is a growing trend to embed large language model (LLM) functionality into applications, from browser-based web apps to native apps that run on personal computers. Here, we introduce LLM-for-X, a system-wide shortcut layer that seamlessly augments any application with LLM services through a lightweight popup dialog. Our native layer seamlessly connects front-end applications to popular LLM backends, such as ChatGPT and Gemini, using their uniform chat front-ends as the programming interface or their custom API calls. We demonstrate the benefits of LLM-for-X across a wide variety of applications, including Microsoft Office, VSCode, and Adobe Acrobat as well as popular web apps such as Overleaf. In our evaluation, we compared LLM-for-X with ChatGPT's web interface in a series of tasks, showing that our approach can provide users with quick, efficient, and easy-to-use LLM assistance without context switching to support writing and reading tasks that is agnostic of the specific application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21579v1">A Performance Study of LLM-Generated Code on Leetcode</a></div>
    <div class="paper-meta">
      📅 2024-07-31
    </div>
    <details class="paper-abstract">
      This study evaluates the efficiency of code generation by Large Language Models (LLMs) and measures their performance against human-crafted solutions using a dataset from Leetcode. We compare 18 LLMs, considering factors such as model temperature and success rate, and their impact on code performance. This research introduces a novel method for measuring and comparing the speed of LLM-generated code, revealing that LLMs produce code with comparable performance, irrespective of the adopted LLM. We also find that LLMs are capable of generating code that is, on average, more efficient than the code written by humans. The paper further discusses the use of Leetcode as a benchmarking dataset, the limitations imposed by potential data contamination, and the platform's measurement reliability. We believe that our findings contribute to a better understanding of LLM capabilities in code generation and set the stage for future optimizations in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.20501v3">Neural Retrievers are Biased Towards LLM-Generated Content</a></div>
    <div class="paper-meta">
      📅 2024-07-31
      | 💬 KDD 2024
    </div>
    <details class="paper-abstract">
      Recently, the emergence of large language models (LLMs) has revolutionized the paradigm of information retrieval (IR) applications, especially in web search, by generating vast amounts of human-like texts on the Internet. As a result, IR systems in the LLM era are facing a new challenge: the indexed documents are now not only written by human beings but also automatically generated by the LLMs. How these LLM-generated documents influence the IR systems is a pressing and still unexplored question. In this work, we conduct a quantitative evaluation of IR models in scenarios where both human-written and LLM-generated texts are involved. Surprisingly, our findings indicate that neural retrieval models tend to rank LLM-generated documents higher. We refer to this category of biases in neural retrievers towards the LLM-generated content as the \textbf{source bias}. Moreover, we discover that this bias is not confined to the first-stage neural retrievers, but extends to the second-stage neural re-rankers. Then, in-depth analyses from the perspective of text compression indicate that LLM-generated texts exhibit more focused semantics with less noise, making it easier for neural retrieval models to semantic match. To mitigate the source bias, we also propose a plug-and-play debiased constraint for the optimization objective, and experimental results show its effectiveness. Finally, we discuss the potential severe concerns stemming from the observed source bias and hope our findings can serve as a critical wake-up call to the IR community and beyond. To facilitate future explorations of IR in the LLM era, the constructed two new benchmarks are available at https://github.com/KID-22/Source-Bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21553v1">CXSimulator: A User Behavior Simulation using LLM Embeddings for Web-Marketing Campaign Assessment</a></div>
    <div class="paper-meta">
      📅 2024-07-31
      | 💬 5 pages, 2 figures, 1 table, the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24)
    </div>
    <details class="paper-abstract">
      This paper presents the Customer Experience (CX) Simulator, a novel framework designed to assess the effects of untested web-marketing campaigns through user behavior simulations. The proposed framework leverages large language models (LLMs) to represent various events in a user's behavioral history, such as viewing an item, applying a coupon, or purchasing an item, as semantic embedding vectors. We train a model to predict transitions between events from their LLM embeddings, which can even generalize to unseen events by learning from diverse training data. In web-marketing applications, we leverage this transition prediction model to simulate how users might react differently when new campaigns or products are presented to them. This allows us to eliminate the need for costly online testing and enhance the marketers' abilities to reveal insights. Our numerical evaluation and user study, utilizing BigQuery Public Datasets from the Google Merchandise Store, demonstrate the effectiveness of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12689v2">Can LLMs Understand Computer Networks? Towards a Virtual System Administrator</a></div>
    <div class="paper-meta">
      📅 2024-07-31
    </div>
    <details class="paper-abstract">
      Recent advancements in Artificial Intelligence, and particularly Large Language Models (LLMs), offer promising prospects for aiding system administrators in managing the complexity of modern networks. However, despite this potential, a significant gap exists in the literature regarding the extent to which LLMs can understand computer networks. Without empirical evidence, system administrators might rely on these models without assurance of their efficacy in performing network-related tasks accurately. In this paper, we are the first to conduct an exhaustive study on LLMs' comprehension of computer networks. We formulate several research questions to determine whether LLMs can provide correct answers when supplied with a network topology and questions on it. To assess them, we developed a thorough framework for evaluating LLMs' capabilities in various network-related tasks. We evaluate our framework on multiple computer networks employing proprietary (e.g., GPT4) and open-source (e.g., Llama2) models. Our findings in general purpose LLMs using a zero-shot scenario demonstrate promising results, with the best model achieving an average accuracy of 79.3%. Proprietary LLMs achieve noteworthy results in small and medium networks, while challenges persist in comprehending complex network topologies, particularly for open-source models. Moreover, we provide insight into how prompt engineering can enhance the accuracy of some tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21531v1">Can LLMs "Reason" in Music? An Evaluation of LLMs' Capability of Music Understanding and Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-31
      | 💬 Accepted by ISMIR2024
    </div>
    <details class="paper-abstract">
      Symbolic Music, akin to language, can be encoded in discrete symbols. Recent research has extended the application of large language models (LLMs) such as GPT-4 and Llama2 to the symbolic music domain including understanding and generation. Yet scant research explores the details of how these LLMs perform on advanced music understanding and conditioned generation, especially from the multi-step reasoning perspective, which is a critical aspect in the conditioned, editable, and interactive human-computer co-creation process. This study conducts a thorough investigation of LLMs' capability and limitations in symbolic music processing. We identify that current LLMs exhibit poor performance in song-level multi-step music reasoning, and typically fail to leverage learned music knowledge when addressing complex musical tasks. An analysis of LLMs' responses highlights distinctly their pros and cons. Our findings suggest achieving advanced musical capability is not intrinsically obtained by LLMs, and future research should focus more on bridging the gap between music knowledge and reasoning, to improve the co-creation experience for musicians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11359v3">Prompt, Plan, Perform: LLM-based Humanoid Control via Quantized Imitation Learning</a></div>
    <div class="paper-meta">
      📅 2024-07-31
    </div>
    <details class="paper-abstract">
      In recent years, reinforcement learning and imitation learning have shown great potential for controlling humanoid robots' motion. However, these methods typically create simulation environments and rewards for specific tasks, resulting in the requirements of multiple policies and limited capabilities for tackling complex and unknown tasks. To overcome these issues, we present a novel approach that combines adversarial imitation learning with large language models (LLMs). This innovative method enables the agent to learn reusable skills with a single policy and solve zero-shot tasks under the guidance of LLMs. In particular, we utilize the LLM as a strategic planner for applying previously learned skills to novel tasks through the comprehension of task-specific prompts. This empowers the robot to perform the specified actions in a sequence. To improve our model, we incorporate codebook-based vector quantization, allowing the agent to generate suitable actions in response to unseen textual commands from LLMs. Furthermore, we design general reward functions that consider the distinct motion features of humanoid robots, ensuring the agent imitates the motion data while maintaining goal orientation without additional guiding direction approaches or policies. To the best of our knowledge, this is the first framework that controls humanoid robots using a single learning policy network and LLM as a planner. Extensive experiments demonstrate that our method exhibits efficient and adaptive ability in complicated motion tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2407.20529v1">Can LLMs be Fooled? Investigating Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-31
      | 💬 14 pages, 1 figure. arXiv admin note: text overlap with arXiv:2403.12503
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has garnered significant popularity and wielded immense power across various domains within Natural Language Processing (NLP). While their capabilities are undeniably impressive, it is crucial to identify and scrutinize their vulnerabilities especially when those vulnerabilities can have costly consequences. One such LLM, trained to provide a concise summarization from medical documents could unequivocally leak personal patient data when prompted surreptitiously. This is just one of many unfortunate examples that have been unveiled and further research is necessary to comprehend the underlying reasons behind such vulnerabilities. In this study, we delve into multiple sections of vulnerabilities which are model-based, training-time, inference-time vulnerabilities, and discuss mitigation strategies including "Model Editing" which aims at modifying LLMs behavior, and "Chroma Teaming" which incorporates synergy of multiple teaming strategies to enhance LLMs' resilience. This paper will synthesize the findings from each vulnerability section and propose new directions of research and development. By understanding the focal points of current vulnerabilities, we can better anticipate and mitigate future risks, paving the road for more robust and secure LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11514v3">LLM in a flash: Efficient Large Language Model Inference with Limited Memory</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 ACL 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are central to modern natural language processing, delivering exceptional performance in various tasks. However, their substantial computational and memory requirements present challenges, especially for devices with limited DRAM capacity. This paper tackles the challenge of efficiently running LLMs that exceed the available DRAM capacity by storing the model parameters in flash memory, but bringing them on demand to DRAM. Our method involves constructing an inference cost model that takes into account the characteristics of flash memory, guiding us to optimize in two critical areas: reducing the volume of data transferred from flash and reading data in larger, more contiguous chunks. Within this hardware-informed framework, we introduce two principal techniques. First, "windowing" strategically reduces data transfer by reusing previously activated neurons, and second, "row-column bundling", tailored to the sequential data access strengths of flash memory, increases the size of data chunks read from flash memory. These methods collectively enable running models up to twice the size of the available DRAM, with a 4-5x and 20-25x increase in inference speed compared to naive loading approaches in CPU and GPU, respectively. Our integration of sparsity awareness, context-adaptive loading, and a hardware-oriented design paves the way for effective inference of LLMs on devices with limited memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04640v1">LLMs for Enhanced Agricultural Meteorological Recommendations</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Agricultural meteorological recommendations are crucial for enhancing crop productivity and sustainability by providing farmers with actionable insights based on weather forecasts, soil conditions, and crop-specific data. This paper presents a novel approach that leverages large language models (LLMs) and prompt engineering to improve the accuracy and relevance of these recommendations. We designed a multi-round prompt framework to iteratively refine recommendations using updated data and feedback, implemented on ChatGPT, Claude2, and GPT-4. Our method was evaluated against baseline models and a Chain-of-Thought (CoT) approach using manually collected datasets. The results demonstrate significant improvements in accuracy and contextual relevance, with our approach achieving up to 90\% accuracy and high GPT-4 scores. Additional validation through real-world pilot studies further confirmed the practical benefits of our method, highlighting its potential to transform agricultural practices and decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20990v1">From Feature Importance to Natural Language Explanations Using LLMs with RAG</a></div>
    <div class="paper-meta">
      📅 2024-07-30
    </div>
    <details class="paper-abstract">
      As machine learning becomes increasingly integral to autonomous decision-making processes involving human interaction, the necessity of comprehending the model's outputs through conversational means increases. Most recently, foundation models are being explored for their potential as post hoc explainers, providing a pathway to elucidate the decision-making mechanisms of predictive models. In this work, we introduce traceable question-answering, leveraging an external knowledge repository to inform the responses of Large Language Models (LLMs) to user queries within a scene understanding task. This knowledge repository comprises contextual details regarding the model's output, containing high-level features, feature importance, and alternative probabilities. We employ subtractive counterfactual reasoning to compute feature importance, a method that entails analysing output variations resulting from decomposing semantic features. Furthermore, to maintain a seamless conversational flow, we integrate four key characteristics - social, causal, selective, and contrastive - drawn from social science research on human explanations into a single-shot prompt, guiding the response generation process. Our evaluation demonstrates that explanations generated by the LLMs encompassed these elements, indicating its potential to bridge the gap between complex model outputs and natural language expressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20970v1">Large Language Models (LLMs) for Semantic Communication in Edge-based IoT Networks</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 6pages, 3 figures, Magazine
    </div>
    <details class="paper-abstract">
      With the advent of Fifth Generation (5G) and Sixth Generation (6G) communication technologies, as well as the Internet of Things (IoT), semantic communication is gaining attention among researchers as current communication technologies are approaching Shannon's limit. On the other hand, Large Language Models (LLMs) can understand and generate human-like text, based on extensive training on diverse datasets with billions of parameters. Considering the recent near-source computational technologies like Edge, in this article, we give an overview of a framework along with its modules, where LLMs can be used under the umbrella of semantic communication at the network edge for efficient communication in IoT networks. Finally, we discuss a few applications and analyze the challenges and opportunities to develop such systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20859v1">Breaking Agents: Compromising Autonomous LLM Agents Through Malfunction Amplification</a></div>
    <div class="paper-meta">
      📅 2024-07-30
    </div>
    <details class="paper-abstract">
      Recently, autonomous agents built on large language models (LLMs) have experienced significant development and are being deployed in real-world applications. These agents can extend the base LLM's capabilities in multiple ways. For example, a well-built agent using GPT-3.5-Turbo as its core can outperform the more advanced GPT-4 model by leveraging external components. More importantly, the usage of tools enables these systems to perform actions in the real world, moving from merely generating text to actively interacting with their environment. Given the agents' practical applications and their ability to execute consequential actions, it is crucial to assess potential vulnerabilities. Such autonomous systems can cause more severe damage than a standalone language model if compromised. While some existing research has explored harmful actions by LLM agents, our study approaches the vulnerability from a different perspective. We introduce a new type of attack that causes malfunctions by misleading the agent into executing repetitive or irrelevant actions. We conduct comprehensive evaluations using various attack methods, surfaces, and properties to pinpoint areas of susceptibility. Our experiments reveal that these attacks can induce failure rates exceeding 80\% in multiple scenarios. Through attacks on implemented and deployable agents in multi-agent scenarios, we accentuate the realistic risks associated with these vulnerabilities. To mitigate such attacks, we propose self-examination detection methods. However, our findings indicate these attacks are difficult to detect effectively using LLMs alone, highlighting the substantial risks associated with this vulnerability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20840v1">Large Language Model (LLM)-enabled Graphs in Dynamic Networking</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 10 pages, 6 figures, published to IEEE NETWORK
    </div>
    <details class="paper-abstract">
      Recent advances in generative artificial intelligence (AI), and particularly the integration of large language models (LLMs), have had considerable impact on multiple domains. Meanwhile, enhancing dynamic network performance is a crucial element in promoting technological advancement and meeting the growing demands of users in many applications areas involving networks. In this article, we explore an integration of LLMs and graphs in dynamic networks, focusing on potential applications and a practical study. Specifically, we first review essential technologies and applications of LLM-enabled graphs, followed by an exploration of their advantages in dynamic networking. Subsequently, we introduce and analyze LLM-enabled graphs and their applications in dynamic networks from the perspective of LLMs as different roles. On this basis, we propose a novel framework of LLM-enabled graphs for networking optimization, and then present a case study on UAV networking, concentrating on optimizing UAV trajectory and communication resource allocation to validate the effectiveness of the proposed framework. Finally, we outline several potential future extensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19087v2">Evaluating the Capability of LLMs in Identifying Compilation Errors in Configurable Systems</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 Accepted at NIER track of the Brazilian Symposium on Software Engineering (SBES 2024), 7 Pages
    </div>
    <details class="paper-abstract">
      Compilation is an important process in developing configurable systems, such as Linux. However, identifying compilation errors in configurable systems is not straightforward because traditional compilers are not variability-aware. Previous approaches that detect some of these compilation errors often rely on advanced techniques that require significant effort from programmers. This study evaluates the efficacy of Large Language Models (LLMs), specifically ChatGPT4, Le Chat Mistral and Gemini Advanced 1.5, in identifying compilation errors in configurable systems. Initially, we evaluate 50 small products in C++, Java, and C languages, followed by 30 small configurable systems in C, covering 17 different types of compilation errors. ChatGPT4 successfully identified most compilation errors in individual products and in configurable systems, while Le Chat Mistral and Gemini Advanced 1.5 detected some of them. LLMs have shown potential in assisting developers in identifying compilation errors in configurable systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20729v1">Adapting Safe-for-Work Classifier for Malaysian Language Text: Enhancing Alignment in LLM-Ops Framework</a></div>
    <div class="paper-meta">
      📅 2024-07-30
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into operational workflows (LLM-Ops), there is a pressing need for effective guardrails to ensure safe and aligned interactions, including the ability to detect potentially unsafe or inappropriate content across languages. However, existing safe-for-work classifiers are primarily focused on English text. To address this gap for the Malaysian language, we present a novel safe-for-work text classifier tailored specifically for Malaysian language content. By curating and annotating a first-of-its-kind dataset of Malaysian text spanning multiple content categories, we trained a classification model capable of identifying potentially unsafe material using state-of-the-art natural language processing techniques. This work represents an important step in enabling safer interactions and content filtering to mitigate potential risks and ensure responsible deployment of LLMs. To maximize accessibility and promote further research towards enhancing alignment in LLM-Ops for the Malaysian context, the model is publicly released at https://huggingface.co/malaysia-ai/malaysian-sfw-classifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03452v3">Large Language Models (LLMs) as Agents for Augmented Democracy</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 24 pages main manuscript with 4 figures. 13 pages of supplementary material
    </div>
    <details class="paper-abstract">
      We explore an augmented democracy system built on off-the-shelf LLMs fine-tuned to augment data on citizen's preferences elicited over policies extracted from the government programs of the two main candidates of Brazil's 2022 presidential election. We use a train-test cross-validation setup to estimate the accuracy with which the LLMs predict both: a subject's individual political choices and the aggregate preferences of the full sample of participants. At the individual level, we find that LLMs predict out of sample preferences more accurately than a "bundle rule", which would assume that citizens always vote for the proposals of the candidate aligned with their self-reported political orientation. At the population level, we show that a probabilistic sample augmented by an LLM provides a more accurate estimate of the aggregate preferences of a population than the non-augmented probabilistic sample alone. Together, these results indicates that policy preference data augmented using LLMs can capture nuances that transcend party lines and represents a promising avenue of research for data augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20588v1">Enhancing Agricultural Machinery Management through Advanced LLM Integration</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The integration of artificial intelligence into agricultural practices, specifically through Consultation on Intelligent Agricultural Machinery Management (CIAMM), has the potential to revolutionize efficiency and sustainability in farming. This paper introduces a novel approach that leverages large language models (LLMs), particularly GPT-4, combined with multi-round prompt engineering to enhance decision-making processes in agricultural machinery management. We systematically developed and refined prompts to guide the LLMs in generating precise and contextually relevant outputs. Our approach was evaluated using a manually curated dataset from various online sources, and performance was assessed with accuracy and GPT-4 Scores. Comparative experiments were conducted using LLama-2-70B, ChatGPT, and GPT-4 models, alongside baseline and state-of-the-art methods such as Chain of Thought (CoT) and Thought of Thought (ThoT). The results demonstrate that our method significantly outperforms these approaches, achieving higher accuracy and relevance in generated responses. This paper highlights the potential of advanced prompt engineering techniques in improving the robustness and applicability of AI in agricultural contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20529v1">Can LLMs be Fooled? Investigating Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 14 pages, 1 figure. arXiv admin note: text overlap with arXiv:2403.12503
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has garnered significant popularity and wielded immense power across various domains within Natural Language Processing (NLP). While their capabilities are undeniably impressive, it is crucial to identify and scrutinize their vulnerabilities especially when those vulnerabilities can have costly consequences. One such LLM, trained to provide a concise summarization from medical documents could unequivocally leak personal patient data when prompted surreptitiously. This is just one of many unfortunate examples that have been unveiled and further research is necessary to comprehend the underlying reasons behind such vulnerabilities. In this study, we delve into multiple sections of vulnerabilities which are model-based, training-time, inference-time vulnerabilities, and discuss mitigation strategies including "Model Editing" which aims at modifying LLMs behavior, and "Chroma Teaming" which incorporates synergy of multiple teaming strategies to enhance LLMs' resilience. This paper will synthesize the findings from each vulnerability section and propose new directions of research and development. By understanding the focal points of current vulnerabilities, we can better anticipate and mitigate future risks, paving the road for more robust and secure LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19594v2">Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge</a></div>
    <div class="paper-meta">
      📅 2024-07-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly surpassing human knowledge in many domains. While improving these models traditionally relies on costly human data, recent self-rewarding mechanisms (Yuan et al., 2024) have shown that LLMs can improve by judging their own responses instead of relying on human labelers. However, existing methods have primarily focused on improving model responses rather than judgment capabilities, resulting in rapid saturation during iterative training. To address this issue, we introduce a novel Meta-Rewarding step to the self-improvement process, where the model judges its own judgements and uses that feedback to refine its judgment skills. Surprisingly, this unsupervised approach improves the model's ability to judge {\em and} follow instructions, as demonstrated by a win rate improvement of Llama-3-8B-Instruct from 22.9% to 39.4% on AlpacaEval 2, and 20.6% to 29.1% on Arena-Hard. These results strongly suggest the potential for self-improving models without human supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2404.03602v2">Evaluating LLMs at Detecting Errors in LLM Responses</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 COLM 2024, 46 pages, Benchmark and code: https://github.com/psunlpgroup/ReaLMistake
    </div>
    <details class="paper-abstract">
      With Large Language Models (LLMs) being widely used across various tasks, detecting errors in their responses is increasingly crucial. However, little research has been conducted on error detection of LLM responses. Collecting error annotations on LLM responses is challenging due to the subjective nature of many NLP tasks, and thus previous research focuses on tasks of little practical value (e.g., word sorting) or limited error types (e.g., faithfulness in summarization). This work introduces ReaLMistake, the first error detection benchmark consisting of objective, realistic, and diverse errors made by LLMs. ReaLMistake contains three challenging and meaningful tasks that introduce objectively assessable errors in four categories (reasoning correctness, instruction-following, context-faithfulness, and parameterized knowledge), eliciting naturally observed and diverse errors in responses of GPT-4 and Llama 2 70B annotated by experts. We use ReaLMistake to evaluate error detectors based on 12 LLMs. Our findings show: 1) Top LLMs like GPT-4 and Claude 3 detect errors made by LLMs at very low recall, and all LLM-based error detectors perform much worse than humans. 2) Explanations by LLM-based error detectors lack reliability. 3) LLMs-based error detection is sensitive to small changes in prompts but remains challenging to improve. 4) Popular approaches to improving LLMs, including self-consistency and majority vote, do not improve the error detection performance. Our benchmark and code are provided at https://github.com/psunlpgroup/ReaLMistake.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04637v1">APE: Active Learning-based Tooling for Finding Informative Few-shot Examples for LLM-based Entity Matching</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 3 pages, Proceedings of the Fifth Workshop on Data Science with Human-in-the-Loop (DaSH 2024)
    </div>
    <details class="paper-abstract">
      Prompt engineering is an iterative procedure often requiring extensive manual effort to formulate suitable instructions for effectively directing large language models (LLMs) in specific tasks. Incorporating few-shot examples is a vital and effective approach to providing LLMs with precise instructions, leading to improved LLM performance. Nonetheless, identifying the most informative demonstrations for LLMs is labor-intensive, frequently entailing sifting through an extensive search space. In this demonstration, we showcase a human-in-the-loop tool called APE (Active Prompt Engineering) designed for refining prompts through active learning. Drawing inspiration from active learning, APE iteratively selects the most ambiguous examples for human feedback, which will be transformed into few-shot examples within the prompt. The demo recording can be found with the submission or be viewed at https://youtu.be/OwQ6MQx53-Y.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19528v3">Harnessing LLMs for Automated Video Content Analysis: An Exploratory Workflow of Short Videos on Depression</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 7 pages, 2 figures, accepted by CSCW 24
    </div>
    <details class="paper-abstract">
      Despite the growing interest in leveraging Large Language Models (LLMs) for content analysis, current studies have primarily focused on text-based content. In the present work, we explored the potential of LLMs in assisting video content analysis by conducting a case study that followed a new workflow of LLM-assisted multimodal content analysis. The workflow encompasses codebook design, prompt engineering, LLM processing, and human evaluation. We strategically crafted annotation prompts to get LLM Annotations in structured form and explanation prompts to generate LLM Explanations for a better understanding of LLM reasoning and transparency. To test LLM's video annotation capabilities, we analyzed 203 keyframes extracted from 25 YouTube short videos about depression. We compared the LLM Annotations with those of two human coders and found that LLM has higher accuracy in object and activity Annotations than emotion and genre Annotations. Moreover, we identified the potential and limitations of LLM's capabilities in annotating videos. Based on the findings, we explore opportunities and challenges for future research and improvements to the workflow. We also discuss ethical concerns surrounding future studies based on LLM-assisted video analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16251v3">Prompt Leakage effect and defense strategies for multi-turn LLM interactions</a></div>
    <div class="paper-meta">
      📅 2024-07-29
    </div>
    <details class="paper-abstract">
      Prompt leakage poses a compelling security and privacy threat in LLM applications. Leakage of system prompts may compromise intellectual property, and act as adversarial reconnaissance for an attacker. A systematic evaluation of prompt leakage threats and mitigation strategies is lacking, especially for multi-turn LLM interactions. In this paper, we systematically investigate LLM vulnerabilities against prompt leakage for 10 closed- and open-source LLMs, across four domains. We design a unique threat model which leverages the LLM sycophancy effect and elevates the average attack success rate (ASR) from 17.7% to 86.2% in a multi-turn setting. Our standardized setup further allows dissecting leakage of specific prompt contents such as task instructions and knowledge documents. We measure the mitigation effect of 7 black-box defense strategies, along with finetuning an open-source model to defend against leakage attempts. We present different combination of defenses against our threat model, including a cost analysis. Our study highlights key takeaways for building secure LLM applications and provides directions for research in multi-turn LLM interactions
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20157v1">rLLM: Relational Table Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-29
    </div>
    <details class="paper-abstract">
      We introduce rLLM (relationLLM), a PyTorch library designed for Relational Table Learning (RTL) with Large Language Models (LLMs). The core idea is to decompose state-of-the-art Graph Neural Networks, LLMs, and Table Neural Networks into standardized modules, to enable the fast construction of novel RTL-type models in a simple "combine, align, and co-train" manner. To illustrate the usage of rLLM, we introduce a simple RTL method named \textbf{BRIDGE}. Additionally, we present three novel relational tabular datasets (TML1M, TLF2K, and TACM12K) by enhancing classic datasets. We hope rLLM can serve as a useful and easy-to-use development framework for RTL-related tasks. Our code is available at: https://github.com/rllm-project/rllm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11190v2">VisionTasker: Mobile Task Automation Using Vision Based UI Understanding and LLM Task Planning</a></div>
    <div class="paper-meta">
      📅 2024-07-29
    </div>
    <details class="paper-abstract">
      Mobile task automation is an emerging field that leverages AI to streamline and optimize the execution of routine tasks on mobile devices, thereby enhancing efficiency and productivity. Traditional methods, such as Programming By Demonstration (PBD), are limited due to their dependence on predefined tasks and susceptibility to app updates. Recent advancements have utilized the view hierarchy to collect UI information and employed Large Language Models (LLM) to enhance task automation. However, view hierarchies have accessibility issues and face potential problems like missing object descriptions or misaligned structures. This paper introduces VisionTasker, a two-stage framework combining vision-based UI understanding and LLM task planning, for mobile task automation in a step-by-step manner. VisionTasker firstly converts a UI screenshot into natural language interpretations using a vision-based UI understanding approach, eliminating the need for view hierarchies. Secondly, it adopts a step-by-step task planning method, presenting one interface at a time to the LLM. The LLM then identifies relevant elements within the interface and determines the next action, enhancing accuracy and practicality. Extensive experiments show that VisionTasker outperforms previous methods, providing effective UI representations across four datasets. Additionally, in automating 147 real-world tasks on an Android smartphone, VisionTasker demonstrates advantages over humans in tasks where humans show unfamiliarity and shows significant improvements when integrated with the PBD mechanism. VisionTasker is open-source and available at https://github.com/AkimotoAyako/VisionTasker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20042v1">When to Stop? Towards Efficient Code Generation in LLMs with Excess Token Prevention</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 To appear at ISSTA 2024
    </div>
    <details class="paper-abstract">
      Code generation aims to automatically generate code snippets that meet given natural language requirements and plays an important role in software development. Although Code LLMs have shown excellent performance in this domain, their long generation time poses a signification limitation in practice use. In this paper, we first conduct an in-depth preliminary study with different Code LLMs on code generation tasks and identify a significant efficiency issue, i.e., continual generation of excess tokens. It harms the developer productivity and leads to huge computational wastes. To address it, we introduce CodeFast, an inference acceleration approach for Code LLMs on code generation. The key idea of CodeFast is to terminate the inference process in time when unnecessary excess tokens are detected. First, we propose an automatic data construction framework to obtain training data. Then, we train a unified lightweight model GenGuard applicable to multiple programming languages to predict whether to terminate inference at the current step. Finally, we enhance Code LLM with GenGuard to accelerate its inference in code generation tasks. We conduct extensive experiments with CodeFast on five representative Code LLMs across four widely used code generation datasets. Experimental results show that (1) CodeFast can significantly improve the inference speed of various Code LLMs in code generation, ranging form 34% to 452%, without compromising the quality of generated code. (2) CodeFast is stable across different parameter settings and can generalize to untrained datasets. Our code and data are available at https://github.com/DeepSoftwareAnalytics/CodeFast
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19998v1">Do LLMs Really Adapt to Domains? An Ontology Learning Perspective</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 Accepted at ISWC 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated unprecedented prowess across various natural language processing tasks in various application domains. Recent studies show that LLMs can be leveraged to perform lexical semantic tasks, such as Knowledge Base Completion (KBC) or Ontology Learning (OL). However, it has not effectively been verified whether their success is due to their ability to reason over unstructured or semi-structured data, or their effective learning of linguistic patterns and senses alone. This unresolved question is particularly crucial when dealing with domain-specific data, where the lexical senses and their meaning can completely differ from what a LLM has learned during its training stage. This paper investigates the following question: Do LLMs really adapt to domains and remain consistent in the extraction of structured knowledge, or do they only learn lexical senses instead of reasoning? To answer this question and, we devise a controlled experiment setup that uses WordNet to synthesize parallel corpora, with English and gibberish terms. We examine the differences in the outputs of LLMs for each corpus in two OL tasks: relation extraction and taxonomy discovery. Empirical results show that, while adapting to the gibberish corpora, off-the-shelf LLMs do not consistently reason over semantic relationships between concepts, and instead leverage senses and their frame. However, fine-tuning improves the performance of LLMs on lexical semantic tasks even when the domain-specific terms are arbitrary and unseen during pre-training, hinting at the applicability of pre-trained LLMs for OL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19922v1">Monetizing Currency Pair Sentiments through LLM Explainability</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 7 pages, 3 figures, AIFin@ECAI 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) play a vital role in almost every domain in today's organizations. In the context of this work, we highlight the use of LLMs for sentiment analysis (SA) and explainability. Specifically, we contribute a novel technique to leverage LLMs as a post-hoc model-independent tool for the explainability of SA. We applied our technique in the financial domain for currency-pair price predictions using open news feed data merged with market prices. Our application shows that the developed technique is not only a viable alternative to using conventional eXplainable AI but can also be fed back to enrich the input to the machine learning (ML) model to better predict future currency-pair values. We envision our results could be generalized to employing explainability as a conventional enrichment for ML input for better ML predictions in general.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04869v2">Prompting Multi-Modal Tokens to Enhance End-to-End Autonomous Driving Imitation Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-29
    </div>
    <details class="paper-abstract">
      The utilization of Large Language Models (LLMs) within the realm of reinforcement learning, particularly as planners, has garnered a significant degree of attention in recent scholarly literature. However, a substantial proportion of existing research predominantly focuses on planning models for robotics that transmute the outputs derived from perception models into linguistic forms, thus adopting a `pure-language' strategy. In this research, we propose a hybrid End-to-End learning framework for autonomous driving by combining basic driving imitation learning with LLMs based on multi-modality prompt tokens. Instead of simply converting perception results from the separated train model into pure language input, our novelty lies in two aspects. 1) The end-to-end integration of visual and LiDAR sensory input into learnable multi-modality tokens, thereby intrinsically alleviating description bias by separated pre-trained perception models. 2) Instead of directly letting LLMs drive, this paper explores a hybrid setting of letting LLMs help the driving model correct mistakes and complicated scenarios. The results of our experiments suggest that the proposed methodology can attain driving scores of 49.21%, coupled with an impressive route completion rate of 91.34% in the offline evaluation conducted via CARLA. These performance metrics are comparable to the most advanced driving models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08422v2">On the (In)Security of LLM App Stores</a></div>
    <div class="paper-meta">
      📅 2024-07-29
    </div>
    <details class="paper-abstract">
      LLM app stores have seen rapid growth, leading to the proliferation of numerous custom LLM apps. However, this expansion raises security concerns. In this study, we propose a three-layer concern framework to identify the potential security risks of LLM apps, i.e., LLM apps with abusive potential, LLM apps with malicious intent, and LLM apps with exploitable vulnerabilities. Over five months, we collected 786,036 LLM apps from six major app stores: GPT Store, FlowGPT, Poe, Coze, Cici, and Character.AI. Our research integrates static and dynamic analysis, the development of a large-scale toxic word dictionary (i.e., ToxicDict) comprising over 31,783 entries, and automated monitoring tools to identify and mitigate threats. We uncovered that 15,146 apps had misleading descriptions, 1,366 collected sensitive personal information against their privacy policies, and 15,996 generated harmful content such as hate speech, self-harm, extremism, etc. Additionally, we evaluated the potential for LLM apps to facilitate malicious activities, finding that 616 apps could be used for malware generation, phishing, etc. Our findings highlight the urgent need for robust regulatory frameworks and enhanced enforcement mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19798v1">Teaching LLMs at Charles University: Assignments and Activities</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 6th TeachNLP workshop at ACL 2024
    </div>
    <details class="paper-abstract">
      This paper presents teaching materials, particularly assignments and ideas for classroom activities, from a new course on large language models (LLMs) taught at Charles University. The assignments include experiments with LLM inference for weather report generation and machine translation. The classroom activities include class quizzes, focused research on downstream tasks and datasets, and an interactive "best paper" session aimed at reading and comprehension of research papers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19795v1">VolDoGer: LLM-assisted Datasets for Domain Generalization in Vision-Language Tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 31 pages, 5 figures, 20 tables
    </div>
    <details class="paper-abstract">
      Domain generalizability is a crucial aspect of a deep learning model since it determines the capability of the model to perform well on data from unseen domains. However, research on the domain generalizability of deep learning models for vision-language tasks remains limited, primarily because of the lack of required datasets. To address these challenges, we propose VolDoGer: Vision-Language Dataset for Domain Generalization, a dedicated dataset designed for domain generalization that addresses three vision-language tasks: image captioning, visual question answering, and visual entailment. We constructed VolDoGer by extending LLM-based data annotation techniques to vision-language tasks, thereby alleviating the burden of recruiting human annotators. We evaluated the domain generalizability of various models, ranging from fine-tuned models to a recent multimodal large language model, through VolDoGer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.05908v2">Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 ES-FoMo Workshop at ICML 2023 / Published in TMLR
    </div>
    <details class="paper-abstract">
      This paper presents "Predictive Pipelined Decoding (PPD)," an approach that speeds up greedy decoding in Large Language Models (LLMs) while maintaining the exact same output as the original decoding. Unlike conventional strategies, PPD employs additional compute resources to parallelize the initiation of subsequent token decoding during the current token decoding. This method reduces decoding latency and reshapes the understanding of trade-offs in LLM decoding strategies. We have developed a theoretical framework that allows us to analyze the trade-off between computation and latency. Using this framework, we can analytically estimate the potential reduction in latency associated with our proposed method, achieved through the assessment of the match rate, represented as p_correct. The results demonstrate that the use of extra computational resources has the potential to accelerate LLM decoding. Additionally, we implement PPD and conduct preliminary experiments to empirically validate its efficacy, addressing potential practical overheads not covered by theoretical analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19638v1">From Pre-training Corpora to Large Language Models: What Factors Influence LLM Performance in Causal Discovery Tasks?</a></div>
    <div class="paper-meta">
      📅 2024-07-29
    </div>
    <details class="paper-abstract">
      Recent advances in artificial intelligence have seen Large Language Models (LLMs) demonstrate notable proficiency in causal discovery tasks. This study explores the factors influencing the performance of LLMs in causal discovery tasks. Utilizing open-source LLMs, we examine how the frequency of causal relations within their pre-training corpora affects their ability to accurately respond to causal discovery queries. Our findings reveal that a higher frequency of causal mentions correlates with better model performance, suggesting that extensive exposure to causal information during training enhances the models' causal discovery capabilities. Additionally, we investigate the impact of context on the validity of causal relations. Our results indicate that LLMs might exhibit divergent predictions for identical causal relations when presented in different contexts. This paper provides the first comprehensive analysis of how different factors contribute to LLM performance in causal discovery tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19617v1">AgEval: A Benchmark for Zero-Shot and Few-Shot Plant Stress Phenotyping with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-29
    </div>
    <details class="paper-abstract">
      Plant stress phenotyping traditionally relies on expert assessments and specialized models, limiting scalability in agriculture. Recent advances in multimodal large language models (LLMs) offer potential solutions to this challenge. We present AgEval, a benchmark comprising 12 diverse plant stress phenotyping tasks, to evaluate these models' capabilities. Our study assesses zero-shot and few-shot in-context learning performance of state-of-the-art models, including Claude, GPT, Gemini, and LLaVA. Results show significant performance improvements with few-shot learning, with F1 scores increasing from 46.24% to 73.37% in 8-shot identification for the best-performing model. Few-shot examples from other classes in the dataset have negligible or negative impacts, although having the exact category example helps to increase performance by 15.38%. We also quantify the consistency of model performance across different classes within each task, finding that the coefficient of variance (CV) ranges from 26.02% to 58.03% across models, implying that subject matter expertise is needed - of 'difficult' classes - to achieve reliability in performance. AgEval establishes baseline metrics for multimodal LLMs in agricultural applications, offering insights into their promise for enhancing plant stress phenotyping at scale. Benchmark and code can be accessed at: https://anonymous.4open.science/r/AgEval/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19616v1">TopicTag: Automatic Annotation of NMF Topic Models Using Chain of Thought and Prompt Tuning with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 Accepted to ACM Symposium on Document Engineering 2024 (DocEng 24), 2024
    </div>
    <details class="paper-abstract">
      Topic modeling is a technique for organizing and extracting themes from large collections of unstructured text. Non-negative matrix factorization (NMF) is a common unsupervised approach that decomposes a term frequency-inverse document frequency (TF-IDF) matrix to uncover latent topics and segment the dataset accordingly. While useful for highlighting patterns and clustering documents, NMF does not provide explicit topic labels, necessitating subject matter experts (SMEs) to assign labels manually. We present a methodology for automating topic labeling in documents clustered via NMF with automatic model determination (NMFk). By leveraging the output of NMFk and employing prompt engineering, we utilize large language models (LLMs) to generate accurate topic labels. Our case study on over 34,000 scientific abstracts on Knowledge Graphs demonstrates the effectiveness of our method in enhancing knowledge management and document organization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.00783v2">On the Roles of LLMs in Planning: Embedding LLMs into Planning Graphs</a></div>
    <div class="paper-meta">
      📅 2024-07-29
    </div>
    <details class="paper-abstract">
      Plan synthesis aims to generate a course of actions or policies to transit given initial states to goal states, provided domain models that could be designed by experts or learnt from training data or interactions with the world. Intrigued by the claims of emergent planning capabilities in large language models (LLMs), works have been proposed to investigate the planning effectiveness of LLMs, without considering any utilization of off-the-shelf planning techniques in LLMs. In this paper, we aim to further study the insight of the planning capability of LLMs by investigating the roles of LLMs in off-the-shelf planning frameworks. To do this, we investigate the effectiveness of embedding LLMs into one of the well-known planning frameworks, graph-based planning, proposing a novel LLMs-based planning framework with LLMs embedded in two levels of planning graphs, i.e., mutual constraints generation level and constraints solving level. We empirically exhibit the effectiveness of our proposed framework in various planning domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.05140v3">Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 ICML 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency in understanding and generating natural language. However, their capabilities wane in highly specialized domains underrepresented in the pretraining corpus, such as physical and biomedical sciences. This work explores how to repurpose general LLMs into effective task solvers for specialized domains. We introduce a novel, model-agnostic framework for learning custom input tags, which are parameterized as continuous vectors appended to the LLM's embedding layer, to condition the LLM. We design two types of input tags: domain tags are used to delimit specialized representations (e.g., chemical formulas) and provide domain-relevant context; function tags are used to represent specific functions (e.g., predicting molecular properties) and compress function-solving instructions. We develop a three-stage protocol to learn these tags using auxiliary data and domain knowledge. By explicitly disentangling task domains from task functions, our method enables zero-shot generalization to unseen problems through diverse combinations of the input tags. It also boosts LLM's performance in various specialized domains, such as predicting protein or chemical properties and modeling drug-target interactions, outperforming expert models tailored to these tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12068v2">Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness</a></div>
    <div class="paper-meta">
      📅 2024-07-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19526v1">Impact of Decoding Methods on Human Alignment of Conversational LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-28
    </div>
    <details class="paper-abstract">
      To be included into chatbot systems, Large language models (LLMs) must be aligned with human conversational conventions. However, being trained mainly on web-scraped data gives existing LLMs a voice closer to informational text than actual human speech. In this paper, we examine the effect of decoding methods on the alignment between LLM-generated and human conversations, including Beam Search, Top K Sampling, and Nucleus Sampling. We present new measures of alignment in substance, style, and psychometric orientation, and experiment with two conversation datasets. Our results provide subtle insights: better alignment is attributed to fewer beams in Beam Search and lower values of P in Nucleus Sampling. We also find that task-oriented and open-ended datasets perform differently in terms of alignment, indicating the significance of taking into account the context of the interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19517v1">Evaluating LLMs for Text-to-SQL Generation With Complex SQL Workload</a></div>
    <div class="paper-meta">
      📅 2024-07-28
    </div>
    <details class="paper-abstract">
      This study presents a comparative analysis of the a complex SQL benchmark, TPC-DS, with two existing text-to-SQL benchmarks, BIRD and Spider. Our findings reveal that TPC-DS queries exhibit a significantly higher level of structural complexity compared to the other two benchmarks. This underscores the need for more intricate benchmarks to simulate realistic scenarios effectively. To facilitate this comparison, we devised several measures of structural complexity and applied them across all three benchmarks. The results of this study can guide future research in the development of more sophisticated text-to-SQL benchmarks. We utilized 11 distinct Language Models (LLMs) to generate SQL queries based on the query descriptions provided by the TPC-DS benchmark. The prompt engineering process incorporated both the query description as outlined in the TPC-DS specification and the database schema of TPC-DS. Our findings indicate that the current state-of-the-art generative AI models fall short in generating accurate decision-making queries. We conducted a comparison of the generated queries with the TPC-DS gold standard queries using a series of fuzzy structure matching techniques based on query features. The results demonstrated that the accuracy of the generated queries is insufficient for practical real-world application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.12488v5">How Does Naming Affect LLMs on Code Analysis Tasks?</a></div>
    <div class="paper-meta">
      📅 2024-07-28
      | 💬 3 Table, 8 figures
    </div>
    <details class="paper-abstract">
      The Large Language Models (LLMs), such as GPT and BERT, were proposed for natural language processing (NLP) and have shown promising results as general-purpose language models. An increasing number of industry professionals and researchers are adopting LLMs for program analysis tasks. However, one significant difference between programming languages and natural languages is that a programmer has the flexibility to assign any names to variables, methods, and functions in the program, whereas a natural language writer does not. Intuitively, the quality of naming in a program affects the performance of LLMs in program analysis tasks. This paper investigates how naming affects LLMs on code analysis tasks. Specifically, we create a set of datasets with code containing nonsense or misleading names for variables, methods, and functions, respectively. We then use well-trained models (CodeBERT) to perform code analysis tasks on these datasets. The experimental results show that naming has a significant impact on the performance of code analysis tasks based on LLMs, indicating that code representation learning based on LLMs heavily relies on well-defined names in code. Additionally, we conduct a case study on some special code analysis tasks using GPT, providing further insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19354v1">The Emerged Security and Privacy of LLM Agent: A Survey with Case Studies</a></div>
    <div class="paper-meta">
      📅 2024-07-28
    </div>
    <details class="paper-abstract">
      Inspired by the rapid development of Large Language Models (LLMs), LLM agents have evolved to perform complex tasks. LLM agents are now extensively applied across various domains, handling vast amounts of data to interact with humans and execute tasks. The widespread applications of LLM agents demonstrate their significant commercial value; however, they also expose security and privacy vulnerabilities. At the current stage, comprehensive research on the security and privacy of LLM agents is highly needed. This survey aims to provide a comprehensive overview of the newly emerged privacy and security issues faced by LLM agents. We begin by introducing the fundamental knowledge of LLM agents, followed by a categorization and analysis of the threats. We then discuss the impacts of these threats on humans, environment, and other agents. Subsequently, we review existing defensive strategies, and finally explore future trends. Additionally, the survey incorporates diverse case studies to facilitate a more accessible understanding. By highlighting these critical security and privacy issues, the survey seeks to stimulate future research towards enhancing the security and privacy of LLM agents, thereby increasing their reliability and trustworthiness in future applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03602v2">Evaluating LLMs at Detecting Errors in LLM Responses</a></div>
    <div class="paper-meta">
      📅 2024-07-27
      | 💬 COLM 2024, 46 pages, Benchmark and code: https://github.com/psunlpgroup/ReaLMistake
    </div>
    <details class="paper-abstract">
      With Large Language Models (LLMs) being widely used across various tasks, detecting errors in their responses is increasingly crucial. However, little research has been conducted on error detection of LLM responses. Collecting error annotations on LLM responses is challenging due to the subjective nature of many NLP tasks, and thus previous research focuses on tasks of little practical value (e.g., word sorting) or limited error types (e.g., faithfulness in summarization). This work introduces ReaLMistake, the first error detection benchmark consisting of objective, realistic, and diverse errors made by LLMs. ReaLMistake contains three challenging and meaningful tasks that introduce objectively assessable errors in four categories (reasoning correctness, instruction-following, context-faithfulness, and parameterized knowledge), eliciting naturally observed and diverse errors in responses of GPT-4 and Llama 2 70B annotated by experts. We use ReaLMistake to evaluate error detectors based on 12 LLMs. Our findings show: 1) Top LLMs like GPT-4 and Claude 3 detect errors made by LLMs at very low recall, and all LLM-based error detectors perform much worse than humans. 2) Explanations by LLM-based error detectors lack reliability. 3) LLMs-based error detection is sensitive to small changes in prompts but remains challenging to improve. 4) Popular approaches to improving LLMs, including self-consistency and majority vote, do not improve the error detection performance. Our benchmark and code are provided at https://github.com/psunlpgroup/ReaLMistake.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19299v1">The Impact of LoRA Adapters for LLMs on Clinical NLP Classification Under Data Limitations</a></div>
    <div class="paper-meta">
      📅 2024-07-27
      | 💬 Under revisions
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) for clinical Natural Language Processing (NLP) poses significant challenges due to the domain gap and limited data availability. This study investigates the effectiveness of various adapter techniques, equivalent to Low-Rank Adaptation (LoRA), for fine-tuning LLMs in a resource-constrained hospital environment. We experimented with four structures-Adapter, Lightweight, TinyAttention, and Gated Residual Network (GRN)-as final layers for clinical notes classification. We fine-tuned biomedical pre-trained models, including CamemBERT-bio, AliBERT, and DrBERT, alongside two Transformer-based models. Our extensive experimental results indicate that i) employing adapter structures does not yield significant improvements in fine-tuning biomedical pre-trained LLMs, and ii) simpler Transformer-based models, trained from scratch, perform better under resource constraints. Among the adapter structures, GRN demonstrated superior performance with accuracy, precision, recall, and an F1 score of 0.88. Moreover, the total training time for LLMs exceeded 1000 hours, compared to under 6 hours for simpler transformer-based models, highlighting that LLMs are more suitable for environments with extensive computational resources and larger datasets. Consequently, this study demonstrates that simpler Transformer-based models can be effectively trained from scratch, providing a viable solution for clinical NLP tasks in low-resource environments with limited data availability. By identifying the GRN as the most effective adapter structure, we offer a practical approach to enhance clinical note classification without requiring extensive computational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19262v1">Understanding Memorisation in LLMs: Dynamics, Influencing Factors, and Implications</a></div>
    <div class="paper-meta">
      📅 2024-07-27
    </div>
    <details class="paper-abstract">
      Understanding whether and to what extent large language models (LLMs) have memorised training data has important implications for the reliability of their output and the privacy of their training data. In order to cleanly measure and disentangle memorisation from other phenomena (e.g. in-context learning), we create an experimental framework that is based on repeatedly exposing LLMs to random strings. Our framework allows us to better understand the dynamics, i.e., the behaviour of the model, when repeatedly exposing it to random strings. Using our framework, we make several striking observations: (a) we find consistent phases of the dynamics across families of models (Pythia, Phi and Llama2), (b) we identify factors that make some strings easier to memorise than others, and (c) we identify the role of local prefixes and global context in memorisation. We also show that sequential exposition to different random strings has a significant effect on memorisation. Our results, often surprising, have significant downstream implications in the study and usage of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19200v1">On Behalf of the Stakeholders: Trends in NLP Model Interpretability in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-27
    </div>
    <details class="paper-abstract">
      Recent advancements in NLP systems, particularly with the introduction of LLMs, have led to widespread adoption of these systems by a broad spectrum of users across various domains, impacting decision-making, the job market, society, and scientific research. This surge in usage has led to an explosion in NLP model interpretability and analysis research, accompanied by numerous technical surveys. Yet, these surveys often overlook the needs and perspectives of explanation stakeholders. In this paper, we address three fundamental questions: Why do we need interpretability, what are we interpreting, and how? By exploring these questions, we examine existing interpretability paradigms, their properties, and their relevance to different stakeholders. We further explore the practical implications of these paradigms by analyzing trends from the past decade across multiple research fields. To this end, we retrieved thousands of papers and employed an LLM to characterize them. Our analysis reveals significant disparities between NLP developers and non-developer users, as well as between research fields, underscoring the diverse needs of stakeholders. For example, explanations of internal model components are rarely used outside the NLP field. We hope this paper informs the future design, development, and application of methods that align with the objectives and requirements of various stakeholders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12801v3">Evaluation of LLMs Biases Towards Elite Universities: A Persona-Based Exploration</a></div>
    <div class="paper-meta">
      📅 2024-07-27
      | 💬 10 pages, 4 Figures
    </div>
    <details class="paper-abstract">
      This study investigates whether popular LLMs exhibit bias towards elite universities when generating personas for technology industry professionals. We employed a novel persona-based approach to compare the educational background predictions of GPT-3.5, Gemini, and Claude 3 Sonnet with actual data from LinkedIn. The study focused on various roles at Microsoft, Meta, and Google, including VP Product, Director of Engineering, and Software Engineer. We generated 432 personas across the three LLMs and analyzed the frequency of elite universities (Stanford, MIT, UC Berkeley, and Harvard) in these personas compared to LinkedIn data. Results showed that LLMs significantly overrepresented elite universities, featuring these universities 72.45% of the time, compared to only 8.56% in the actual LinkedIn data. ChatGPT 3.5 exhibited the highest bias, followed by Claude Sonnet 3, while Gemini performed best. This research highlights the need to address educational bias in LLMs and suggests strategies for mitigating such biases in AI-driven recruitment processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19126v1">Greedy Output Approximation: Towards Efficient Structured Pruning for LLMs Without Retraining</a></div>
    <div class="paper-meta">
      📅 2024-07-26
    </div>
    <details class="paper-abstract">
      To remove redundant components of large language models (LLMs) without incurring significant computational costs, this work focuses on single-shot pruning without a retraining phase. We simplify the pruning process for Transformer-based LLMs by identifying a depth-2 pruning structure that functions independently. Additionally, we propose two inference-aware pruning criteria derived from the optimization perspective of output approximation, which outperforms traditional training-aware metrics such as gradient and Hessian. We also introduce a two-step reconstruction technique to mitigate pruning errors without model retraining. Experimental results demonstrate that our approach significantly reduces computational costs and hardware requirements while maintaining superior performance across various datasets and models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.10254v2">LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins</a></div>
    <div class="paper-meta">
      📅 2024-07-26
      | 💬 To appear in the proceedings of the 7th AAAI / ACM Conference on AI, Ethics, and Society (AIES), October 2024
    </div>
    <details class="paper-abstract">
      Large language model (LLM) platforms, such as ChatGPT, have recently begun offering an app ecosystem to interface with third-party services on the internet. While these apps extend the capabilities of LLM platforms, they are developed by arbitrary third parties and thus cannot be implicitly trusted. Apps also interface with LLM platforms and users using natural language, which can have imprecise interpretations. In this paper, we propose a framework that lays a foundation for LLM platform designers to analyze and improve the security, privacy, and safety of current and future third-party integrated LLM platforms. Our framework is a formulation of an attack taxonomy that is developed by iteratively exploring how LLM platform stakeholders could leverage their capabilities and responsibilities to mount attacks against each other. As part of our iterative process, we apply our framework in the context of OpenAI's plugin (apps) ecosystem. We uncover plugins that concretely demonstrate the potential for the types of issues that we outline in our attack taxonomy. We conclude by discussing novel challenges and by providing recommendations to improve the security, privacy, and safety of present and future LLM-based computing platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19053v1">A Study of Using Multimodal LLMs for Non-Crash Functional Bug Detection in Android Apps</a></div>
    <div class="paper-meta">
      📅 2024-07-26
    </div>
    <details class="paper-abstract">
      Numerous approaches employing various strategies have been developed to test the graphical user interfaces (GUIs) of mobile apps. However, traditional GUI testing techniques, such as random and model-based testing, primarily focus on generating test sequences that excel in achieving high code coverage but often fail to act as effective test oracles for non-crash functional (NCF) bug detection. To tackle these limitations, this study empirically investigates the capability of leveraging large language models (LLMs) to be test oracles to detect NCF bugs in Android apps. Our intuition is that the training corpora of LLMs, encompassing extensive mobile app usage and bug report descriptions, enable them with the domain knowledge relevant to NCF bug detection. We conducted a comprehensive empirical study to explore the effectiveness of LLMs as test oracles for detecting NCF bugs in Android apps on 71 well-documented NCF bugs. The results demonstrated that LLMs achieve a 49% bug detection rate, outperforming existing tools for detecting NCF bugs in Android apps. Additionally, by leveraging LLMs to be test oracles, we successfully detected 24 previously unknown NCF bugs in 64 Android apps, with four of these bugs being confirmed or fixed. However, we also identified limitations of LLMs, primarily related to performance degradation, inherent randomness, and false positives. Our study highlights the potential of leveraging LLMs as test oracles for Android NCF bug detection and suggests directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18786v1">The power of Prompts: Evaluating and Mitigating Gender Bias in MT with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-26
    </div>
    <details class="paper-abstract">
      This paper studies gender bias in machine translation through the lens of Large Language Models (LLMs). Four widely-used test sets are employed to benchmark various base LLMs, comparing their translation quality and gender bias against state-of-the-art Neural Machine Translation (NMT) models for English to Catalan (En $\rightarrow$ Ca) and English to Spanish (En $\rightarrow$ Es) translation directions. Our findings reveal pervasive gender bias across all models, with base LLMs exhibiting a higher degree of bias compared to NMT models. To combat this bias, we explore prompting engineering techniques applied to an instruction-tuned LLM. We identify a prompt structure that significantly reduces gender bias by up to 12% on the WinoMT evaluation dataset compared to more straightforward prompts. These results significantly reduce the gender bias accuracy gap between LLMs and traditional NMT systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12126v2">LLMs-in-the-loop Part-1: Expert Small AI Models for Bio-Medical Text Translation</a></div>
    <div class="paper-meta">
      📅 2024-07-26
      | 💬 14 pages, 2 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Machine translation is indispensable in healthcare for enabling the global dissemination of medical knowledge across languages. However, complex medical terminology poses unique challenges to achieving adequate translation quality and accuracy. This study introduces a novel "LLMs-in-the-loop" approach to develop supervised neural machine translation models optimized specifically for medical texts. While large language models (LLMs) have demonstrated powerful capabilities, this research shows that small, specialized models trained on high-quality in-domain (mostly synthetic) data can outperform even vastly larger LLMs. Custom parallel corpora in six languages were compiled from scientific articles, synthetically generated clinical documents, and medical texts. Our LLMs-in-the-loop methodology employs synthetic data generation, rigorous evaluation, and agent orchestration to enhance performance. We developed small medical translation models using the MarianMT base model. We introduce a new medical translation test dataset to standardize evaluation in this domain. Assessed using BLEU, METEOR, ROUGE, and BERT scores on this test set, our MarianMT-based models outperform Google Translate, DeepL, and GPT-4-Turbo. Results demonstrate that our LLMs-in-the-loop approach, combined with fine-tuning high-quality, domain-specific data, enables specialized models to outperform general-purpose and some larger systems. This research, part of a broader series on expert small models, paves the way for future healthcare-related AI developments, including deidentification and bio-medical entity extraction models. Our study underscores the potential of tailored neural translation models and the LLMs-in-the-loop methodology to advance the field through improved data generation, evaluation, agent, and modeling techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00783v2">On the Roles of LLMs in Planning: Embedding LLMs into Planning Graphs</a></div>
    <div class="paper-meta">
      📅 2024-07-26
    </div>
    <details class="paper-abstract">
      Plan synthesis aims to generate a course of actions or policies to transit given initial states to goal states, provided domain models that could be designed by experts or learnt from training data or interactions with the world. Intrigued by the claims of emergent planning capabilities in large language models (LLMs), works have been proposed to investigate the planning effectiveness of LLMs, without considering any utilization of off-the-shelf planning techniques in LLMs. In this paper, we aim to further study the insight of the planning capability of LLMs by investigating the roles of LLMs in off-the-shelf planning frameworks. To do this, we investigate the effectiveness of embedding LLMs into one of the well-known planning frameworks, graph-based planning, proposing a novel LLMs-based planning framework with LLMs embedded in two levels of planning graphs, i.e., mutual constraints generation level and constraints solving level. We empirically exhibit the effectiveness of our proposed framework in various planning domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07279v1">Interactive and Automatic Generation of Primitive Custom Circuit Layout Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-26
      | 💬 6 pages, 11 figures
    </div>
    <details class="paper-abstract">
      In this study, we investigate the use of Large Language Models (LLMs) for the interactive and automated production of customs circuit layouts described in natural language. Our proposed layout automation process leverages a template-and-grid-based layout generation framework to create process-portable layout generators tailored for various custom circuits, including standard cells and high-speed mixed-signal circuits. However, rather than directly describing the layout generators in traditional programming language, we utilize natural language using LLMs to make the layout generation process more intuitive and efficient. This approach also supports interactive modifications of the layout generator code, enhancing customization capabilities. We demonstrate the effectiveness of our LLM-based layout generation method across several custom circuit examples, such as logic standard cells, a serializer and a strong arm latch, including their completeness in terms of Design Rule Check (DRC), Layout Versus Schematic (LVS) test, and post-layout performance for high-speed circuits. Our experimental results indicate that LLMs can generate a diverse range of circuit layouts with substantial customization options.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04603v3">Analyzing LLM Usage in an Advanced Computing Class in India</a></div>
    <div class="paper-meta">
      📅 2024-07-26
      | 💬 Under review: 8 pages
    </div>
    <details class="paper-abstract">
      This study examines the use of large language models (LLMs) by undergraduate and graduate students for programming assignments in advanced computing classes. Unlike existing research, which primarily focuses on introductory classes and lacks in-depth analysis of actual student-LLM interactions, our work fills this gap. We conducted a comprehensive analysis involving 411 students from a Distributed Systems class at an Indian university, where they completed three programming assignments and shared their experiences through Google Form surveys. Our findings reveal that students leveraged LLMs for a variety of tasks, including code generation, debugging, conceptual inquiries, and test case creation. They employed a spectrum of prompting strategies, ranging from basic contextual prompts to advanced techniques like chain-of-thought prompting and iterative refinement. While students generally viewed LLMs as beneficial for enhancing productivity and learning, we noted a concerning trend of over-reliance, with many students submitting entire assignment descriptions to obtain complete solutions. Given the increasing use of LLMs in the software industry, our study highlights the need to update undergraduate curricula to include training on effective prompting strategies and to raise awareness about the benefits and potential drawbacks of LLM usage in academic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18498v1">A Reliable Common-Sense Reasoning Socialbot Built Using LLMs and Goal-Directed ASP</a></div>
    <div class="paper-meta">
      📅 2024-07-26
    </div>
    <details class="paper-abstract">
      The development of large language models (LLMs), such as GPT, has enabled the construction of several socialbots, like ChatGPT, that are receiving a lot of attention for their ability to simulate a human conversation. However, the conversation is not guided by a goal and is hard to control. In addition, because LLMs rely more on pattern recognition than deductive reasoning, they can give confusing answers and have difficulty integrating multiple topics into a cohesive response. These limitations often lead the LLM to deviate from the main topic to keep the conversation interesting. We propose AutoCompanion, a socialbot that uses an LLM model to translate natural language into predicates (and vice versa) and employs commonsense reasoning based on Answer Set Programming (ASP) to hold a social conversation with a human. In particular, we rely on s(CASP), a goal-directed implementation of ASP as the backend. This paper presents the framework design and how an LLM is used to parse user messages and generate a response from the s(CASP) engine output. To validate our proposal, we describe (real) conversations in which the chatbot's goal is to keep the user entertained by talking about movies and books, and s(CASP) ensures (i) correctness of answers, (ii) coherence (and precision) during the conversation, which it dynamically regulates to achieve its specific purpose, and (iii) no deviation from the main topic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05140v3">Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains</a></div>
    <div class="paper-meta">
      📅 2024-07-26
      | 💬 ICML 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency in understanding and generating natural language. However, their capabilities wane in highly specialized domains underrepresented in the pretraining corpus, such as physical and biomedical sciences. This work explores how to repurpose general LLMs into effective task solvers for specialized domains. We introduce a novel, model-agnostic framework for learning custom input tags, which are parameterized as continuous vectors appended to the LLM's embedding layer, to condition the LLM. We design two types of input tags: domain tags are used to delimit specialized representations (e.g., chemical formulas) and provide domain-relevant context; function tags are used to represent specific functions (e.g., predicting molecular properties) and compress function-solving instructions. We develop a three-stage protocol to learn these tags using auxiliary data and domain knowledge. By explicitly disentangling task domains from task functions, our method enables zero-shot generalization to unseen problems through diverse combinations of the input tags. It also boosts LLM's performance in various specialized domains, such as predicting protein or chemical properties and modeling drug-target interactions, outperforming expert models tailored to these tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18992v1">Towards Automated Solution Recipe Generation for Industrial Asset Management with LLM</a></div>
    <div class="paper-meta">
      📅 2024-07-26
    </div>
    <details class="paper-abstract">
      This study introduces a novel approach to Industrial Asset Management (IAM) by incorporating Conditional-Based Management (CBM) principles with the latest advancements in Large Language Models (LLMs). Our research introduces an automated model-building process, traditionally reliant on intensive collaboration between data scientists and domain experts. We present two primary innovations: a taxonomy-guided prompting generation that facilitates the automatic creation of AI solution recipes and a set of LLM pipelines designed to produce a solution recipe containing a set of artifacts composed of documents, sample data, and models for IAM. These pipelines, guided by standardized principles, enable the generation of initial solution templates for heterogeneous asset classes without direct human input, reducing reliance on extensive domain knowledge and enhancing automation. We evaluate our methodology by assessing asset health and sustainability across a spectrum of ten asset classes. Our findings illustrate the potential of LLMs and taxonomy-based LLM prompting pipelines in transforming asset management, offering a blueprint for subsequent research and development initiatives to be integrated into a rapid client solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06782v4">Debating with More Persuasive LLMs Leads to More Truthful Answers</a></div>
    <div class="paper-meta">
      📅 2024-07-25
      | 💬 For code please check: https://github.com/ucl-dark/llm_debate
    </div>
    <details class="paper-abstract">
      Common methods for aligning large language models (LLMs) with desired behaviour heavily rely on human-labelled data. However, as models grow increasingly sophisticated, they will surpass human expertise, and the role of human evaluation will evolve into non-experts overseeing experts. In anticipation of this, we ask: can weaker models assess the correctness of stronger models? We investigate this question in an analogous setting, where stronger models (experts) possess the necessary information to answer questions and weaker models (non-experts) lack this information. The method we evaluate is debate, where two LLM experts each argue for a different answer, and a non-expert selects the answer. We find that debate consistently helps both non-expert models and humans answer questions, achieving 76% and 88% accuracy respectively (naive baselines obtain 48% and 60%). Furthermore, optimising expert debaters for persuasiveness in an unsupervised manner improves non-expert ability to identify the truth in debates. Our results provide encouraging empirical evidence for the viability of aligning models with debate in the absence of ground truth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17429v2">How Do Students Interact with an LLM-powered Virtual Teaching Assistant in Different Educational Settings?</a></div>
    <div class="paper-meta">
      📅 2024-07-25
      | 💬 Accepted in the Seventeenth International Conference on Educational Data Mining (EDM) Workshop: Leveraging LLMs for Next Generation Educational Technologies, July 2024
    </div>
    <details class="paper-abstract">
      Jill Watson, a virtual teaching assistant powered by LLMs, answers student questions and engages them in extended conversations on courseware provided by the instructors. In this paper, we analyze student interactions with Jill across multiple courses and colleges, focusing on the types and complexity of student questions based on Bloom's Revised Taxonomy and tool usage patterns. We find that, by supporting a wide range of cognitive demands, Jill encourages students to engage in sophisticated, higher-order cognitive questions. However, the frequency of usage varies significantly across deployments, and the types of questions asked depend on course-specific contexts. These findings pave the way for future work on AI-driven educational tools tailored to individual learning styles and course structure, potentially enhancing both the teaching and learning experience in classrooms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18370v1">Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement</a></div>
    <div class="paper-meta">
      📅 2024-07-25
    </div>
    <details class="paper-abstract">
      We present a principled approach to provide LLM-based evaluation with a rigorous guarantee of human agreement. We first propose that a reliable evaluation method should not uncritically rely on model preferences for pairwise evaluation, but rather assess the confidence of judge models and selectively decide when to trust its judgement. We then show that under this selective evaluation framework, human agreement can be provably guaranteed -- such that the model evaluation aligns with that of humans to a user-specified agreement level. As part of our framework, we also introduce Simulated Annotators, a novel confidence estimation method that significantly improves judge calibration and thus enables high coverage of evaluated instances. Finally, we propose Cascaded Selective Evaluation, where we use cheaper models as initial judges and escalate to stronger models only when necessary -- again, while still providing a provable guarantee of human agreement. Experimental results show that Cascaded Selective Evaluation guarantees strong alignment with humans, far beyond what LLM judges could achieve without selective evaluation. For example, on a subset of Chatbot Arena where GPT-4 almost never achieves 80% human agreement, our method, even while employing substantially cost-effective models such as Mistral-7B, guarantees over 80% human agreement with almost 80% test coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19708v2">Harmonic LLMs are Trustworthy</a></div>
    <div class="paper-meta">
      📅 2024-07-25
      | 💬 15 pages, 2 figures, 16 tables; added Claude-3.0, GPT-4o, Mistral-7B, Mixtral-8x7B, and more annotation for other models
    </div>
    <details class="paper-abstract">
      We introduce an intuitive method to test the robustness (stability and explainability) of any black-box LLM in real-time via its local deviation from harmoniticity, denoted as $\gamma$. To the best of our knowledge this is the first completely model-agnostic and unsupervised method of measuring the robustness of any given response from an LLM, based upon the model itself conforming to a purely mathematical standard. To show general application and immediacy of results, we measure $\gamma$ in 10 popular LLMs (ChatGPT, Claude-2.1, Claude3.0, GPT-4, GPT-4o, Smaug-72B, Mixtral-8x7B, Llama2-7B, Mistral-7B and MPT-7B) across thousands of queries in three objective domains: WebQA, ProgrammingQA, and TruthfulQA. Across all models and domains tested, human annotation confirms that $\gamma \to 0$ indicates trustworthiness, and conversely searching higher values of $\gamma$ easily exposes examples of hallucination, a fact that enables efficient adversarial prompt generation through stochastic gradient ascent in $\gamma$. The low-$\gamma$ leaders among the models in the respective domains are GPT-4o, GPT-4, and Smaug-72B, providing evidence that mid-size open-source models can win out against large commercial models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00725v2">The Larger the Better? Improved LLM Code-Generation via Budget Reallocation</a></div>
    <div class="paper-meta">
      📅 2024-07-25
      | 💬 COLM 2024
    </div>
    <details class="paper-abstract">
      It is a common belief that large language models (LLMs) are better than smaller-sized ones. However, larger models also require significantly more time and compute during inference. This begs the question: what happens when both models operate under the same budget? (e.g., compute, run-time). To address this question, we analyze code generation LLMs of various sizes and make comparisons such as running a 70B model once vs. generating five outputs from a 13B model. We consider a standard unit-test setup, which can be used to select the correct output from the smaller model. Our findings reveal that the repeated use of smaller models can yield consistent improvements, with gains of up to 15% across five tasks. On the other hand, in scenarios where unit-tests are unavailable, a ranking-based selection of candidates from the smaller model falls short of the performance of a single output from larger ones. Our results highlight the potential of using smaller models instead of larger ones, and the importance of studying approaches for ranking LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17874v1">Improving Domain-Specific ASR with LLM-Generated Contextual Descriptions</a></div>
    <div class="paper-meta">
      📅 2024-07-25
      | 💬 Accepted to INTERSPEECH 2024
    </div>
    <details class="paper-abstract">
      End-to-end automatic speech recognition (E2E ASR) systems have significantly improved speech recognition through training on extensive datasets. Despite these advancements, they still struggle to accurately recognize domain specific words, such as proper nouns and technical terminologies. To address this problem, we propose a method to utilize the state-of-the-art Whisper without modifying its architecture, preserving its generalization performance while enabling it to leverage descriptions effectively. Moreover, we propose two additional training techniques to improve the domain specific ASR: decoder fine-tuning, and context perturbation. We also propose a method to use a Large Language Model (LLM) to generate descriptions with simple metadata, when descriptions are unavailable. Our experiments demonstrate that proposed methods notably enhance domain-specific ASR accuracy on real-life datasets, with LLM-generated descriptions outperforming human-crafted ones in effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17788v1">PenHeal: A Two-Stage LLM Framework for Automated Pentesting and Optimal Remediation</a></div>
    <div class="paper-meta">
      📅 2024-07-25
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have shown significant potential in enhancing cybersecurity defenses against sophisticated threats. LLM-based penetration testing is an essential step in automating system security evaluations by identifying vulnerabilities. Remediation, the subsequent crucial step, addresses these discovered vulnerabilities. Since details about vulnerabilities, exploitation methods, and software versions offer crucial insights into system weaknesses, integrating penetration testing with vulnerability remediation into a cohesive system has become both intuitive and necessary. This paper introduces PenHeal, a two-stage LLM-based framework designed to autonomously identify and mitigate security vulnerabilities. The framework integrates two LLM-enabled components: the Pentest Module, which detects multiple vulnerabilities within a system, and the Remediation Module, which recommends optimal remediation strategies. The integration is facilitated through Counterfactual Prompting and an Instructor module that guides the LLMs using external knowledge to explore multiple potential attack paths effectively. Our experimental results demonstrate that PenHeal not only automates the identification and remediation of vulnerabilities but also significantly improves vulnerability coverage by 31%, increases the effectiveness of remediation strategies by 32%, and reduces the associated costs by 46% compared to baseline models. These outcomes highlight the transformative potential of LLMs in reshaping cybersecurity practices, offering an innovative solution to defend against cyber threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11686v3">CCoE: A Compact LLM with Collaboration of Experts</a></div>
    <div class="paper-meta">
      📅 2024-07-25
    </div>
    <details class="paper-abstract">
      In the domain of Large Language Model (LLM), LLMs demonstrate significant capabilities in natural language understanding and generation. With the growing needs of applying LLMs on various domains, it is a research question that how to efficiently train and build a model that has expertise in different domains but with a low training cost. We propose CCoE architecture, a framework of easily coupling multiple strong domain experts together to fuse into a big LLM, provides a collective way of utilizing the different domain expert LLMs. Besides, training a large collaborative of multiple expert LLMs requires a high requirements on training sources. CCoE bypasses this problem through isolating other experts and train each expert separately. The design of CCoE assembles multiple expert LLMs through the CoE (Collaboration of Experts) layer. Each CoE layer could have one or more expert LLMs. Expert LLMs have different number of layers and have been well-trained for different domain tasks. Each expert is fine-tuned to be able to achieve the comparable results with SOTA domain LLMs. We start from 5 experts in the domain of Code, Math, Law, text-to-SQL and Medical. The results indicate that our CCoE framework can easily and efficiently boost nearly 10%-20% performance on original base model in different domains but using less resources on training, as well as inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11483v2">AgentKit: Structured LLM Reasoning with Dynamic Graphs</a></div>
    <div class="paper-meta">
      📅 2024-07-24
    </div>
    <details class="paper-abstract">
      We propose an intuitive LLM prompting framework (AgentKit) for multifunctional agents. AgentKit offers a unified framework for explicitly constructing a complex "thought process" from simple natural language prompts. The basic building block in AgentKit is a node, containing a natural language prompt for a specific subtask. The user then puts together chains of nodes, like stacking LEGO pieces. The chains of nodes can be designed to explicitly enforce a naturally structured "thought process". For example, for the task of writing a paper, one may start with the thought process of 1) identify a core message, 2) identify prior research gaps, etc. The nodes in AgentKit can be designed and combined in different ways to implement multiple advanced capabilities including on-the-fly hierarchical planning, reflection, and learning from interactions. In addition, due to the modular nature and the intuitive design to simulate explicit human thought process, a basic agent could be implemented as simple as a list of prompts for the subtasks and therefore could be designed and tuned by someone without any programming experience. Quantitatively, we show that agents designed through AgentKit achieve SOTA performance on WebShop and Crafter. These advances underscore AgentKit's potential in making LLM agents effective and accessible for a wider range of applications. https://github.com/holmeswww/AgentKit
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17468v1">WildHallucinations: Evaluating Long-form Factuality in LLMs with Real-World Entity Queries</a></div>
    <div class="paper-meta">
      📅 2024-07-24
    </div>
    <details class="paper-abstract">
      While hallucinations of large language models (LLMs) prevail as a major challenge, existing evaluation benchmarks on factuality do not cover the diverse domains of knowledge that the real-world users of LLMs seek information about. To bridge this gap, we introduce WildHallucinations, a benchmark that evaluates factuality. It does so by prompting LLMs to generate information about entities mined from user-chatbot conversations in the wild. These generations are then automatically fact-checked against a systematically curated knowledge source collected from web search. Notably, half of these real-world entities do not have associated Wikipedia pages. We evaluate 118,785 generations from 15 LLMs on 7,919 entities. We find that LLMs consistently hallucinate more on entities without Wikipedia pages and exhibit varying hallucination rates across different domains. Finally, given the same base models, adding a retrieval component only slightly reduces hallucinations but does not eliminate hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17291v1">How Good (Or Bad) Are LLMs at Detecting Misleading Visualizations?</a></div>
    <div class="paper-meta">
      📅 2024-07-24
      | 💬 To be presented at IEEE VIS 2024
    </div>
    <details class="paper-abstract">
      In this study, we address the growing issue of misleading charts, a prevalent problem that undermines the integrity of information dissemination. Misleading charts can distort the viewer's perception of data, leading to misinterpretations and decisions based on false information. The development of effective automatic detection methods for misleading charts is an urgent field of research. The recent advancement of multimodal Large Language Models (LLMs) has introduced a promising direction for addressing this challenge. We explored the capabilities of these models in analyzing complex charts and assessing the impact of different prompting strategies on the models' analyses. We utilized a dataset of misleading charts collected from the internet by prior research and crafted nine distinct prompts, ranging from simple to complex, to test the ability of four different multimodal LLMs in detecting over 21 different chart issues. Through three experiments--from initial exploration to detailed analysis--we progressively gained insights into how to effectively prompt LLMs to identify misleading charts and developed strategies to address the scalability challenges encountered as we expanded our detection range from the initial five issues to 21 issues in the final experiment. Our findings reveal that multimodal LLMs possess a strong capability for chart comprehension and critical thinking in data interpretation. There is significant potential in employing multimodal LLMs to counter misleading information by supporting critical thinking and enhancing visualization literacy. This study demonstrates the applicability of LLMs in addressing the pressing concern of misleading charts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17190v1">Fusing LLMs and KGs for Formal Causal Reasoning behind Financial Risk Contagion</a></div>
    <div class="paper-meta">
      📅 2024-07-24
    </div>
    <details class="paper-abstract">
      Financial risks trend to spread from one entity to another, ultimately leading to systemic risks. The key to preventing such risks lies in understanding the causal chains behind risk contagion. Despite this, prevailing approaches primarily emphasize identifying risks, overlooking the underlying causal analysis of risk. To address such an issue, we propose a Risk Contagion Causal Reasoning model called RC2R, which uses the logical reasoning capabilities of large language models (LLMs) to dissect the causal mechanisms of risk contagion grounded in the factual and expert knowledge embedded within financial knowledge graphs (KGs). At the data level, we utilize financial KGs to construct causal instructions, empowering LLMs to perform formal causal reasoning on risk propagation and tackle the "causal parrot" problem of LLMs. In terms of model architecture, we integrate a fusion module that aligns tokens and nodes across various granularities via multi-scale contrastive learning, followed by the amalgamation of textual and graph-structured data through soft prompt with cross multi-head attention mechanisms. To quantify risk contagion, we introduce a risk pathway inference module for calculating risk scores for each node in the graph. Finally, we visualize the risk contagion pathways and their intensities using Sankey diagrams, providing detailed causal explanations. Comprehensive experiments on financial KGs and supply chain datasets demonstrate that our model outperforms several state-of-the-art models in prediction performance and out-of-distribution (OOD) generalization capabilities. We will make our dataset and code publicly accessible to encourage further research and development in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17086v1">AI-Gadget Kit: Integrating Swarm User Interfaces with LLM-driven Agents for Rich Tabletop Game Applications</a></div>
    <div class="paper-meta">
      📅 2024-07-24
    </div>
    <details class="paper-abstract">
      While Swarm User Interfaces (SUIs) have succeeded in enriching tangible interaction experiences, their limitations in autonomous action planning have hindered the potential for personalized and dynamic interaction generation in tabletop games. Based on the AI-Gadget Kit we developed, this paper explores how to integrate LLM-driven agents within tabletop games to enable SUIs to execute complex interaction tasks. After defining the design space of this kit, we elucidate the method for designing agents that can extend the meta-actions of SUIs to complex motion planning. Furthermore, we introduce an add-on prompt method that simplifies the design process for four interaction behaviors and four interaction relationships in tabletop games. Lastly, we present several application scenarios that illustrate the potential of AI-Gadget Kit to construct personalized interaction in SUI tabletop games. We expect to use our work as a case study to inspire research on multi-agent-driven SUI for other scenarios with complex interaction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10834v2">MetaLLM: A High-performant and Cost-efficient Dynamic Framework for Wrapping LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-24
    </div>
    <details class="paper-abstract">
      The rapid progress in machine learning (ML) has brought forth many large language models (LLMs) that excel in various tasks and areas. These LLMs come with different abilities and costs in terms of computation or pricing. Since the demand for each query can vary, e.g., because of the queried domain or its complexity, defaulting to one LLM in an application is not usually the best choice, whether it is the biggest, priciest, or even the one with the best average test performance. Consequently, picking the right LLM that is both accurate and cost-effective for an application remains a challenge. In this paper, we introduce MetaLLM, a framework that dynamically and intelligently routes each query to the optimal LLM (among several available LLMs) for classification tasks, achieving significantly improved accuracy and cost-effectiveness. By framing the selection problem as a multi-armed bandit, MetaLLM balances prediction accuracy and cost efficiency under uncertainty. Our experiments, conducted on popular LLM platforms such as OpenAI's GPT models, Amazon's Titan, Anthropic's Claude, and Meta's LLaMa, showcase MetaLLM's efficacy in real-world scenarios, laying the groundwork for future extensions beyond classification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.12575v3">LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2024-07-24
      | 💬 Accepted for publication in IEEE Symposium on Security and Privacy 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been suggested for use in automated vulnerability repair, but benchmarks showing they can consistently identify security-related bugs are lacking. We thus develop SecLLMHolmes, a fully automated evaluation framework that performs the most detailed investigation to date on whether LLMs can reliably identify and reason about security-related bugs. We construct a set of 228 code scenarios and analyze eight of the most capable LLMs across eight different investigative dimensions using our framework. Our evaluation shows LLMs provide non-deterministic responses, incorrect and unfaithful reasoning, and perform poorly in real-world scenarios. Most importantly, our findings reveal significant non-robustness in even the most advanced models like `PaLM2' and `GPT-4': by merely changing function or variable names, or by the addition of library functions in the source code, these models can yield incorrect answers in 26% and 17% of cases, respectively. These findings demonstrate that further LLM advances are needed before LLMs can be used as general purpose security assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12533v2">To Help or Not to Help: LLM-based Attentive Support for Human-Robot Group Interactions</a></div>
    <div class="paper-meta">
      📅 2024-07-24
      | 💬 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024
    </div>
    <details class="paper-abstract">
      How can a robot provide unobtrusive physical support within a group of humans? We present Attentive Support, a novel interaction concept for robots to support a group of humans. It combines scene perception, dialogue acquisition, situation understanding, and behavior generation with the common-sense reasoning capabilities of Large Language Models (LLMs). In addition to following user instructions, Attentive Support is capable of deciding when and how to support the humans, and when to remain silent to not disturb the group. With a diverse set of scenarios, we show and evaluate the robot's attentive behavior, which supports and helps the humans when required, while not disturbing if no help is needed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17024v1">LLM-Generated Tips Rival Expert-Created Tips in Helping Students Answer Quantum-Computing Questions</a></div>
    <div class="paper-meta">
      📅 2024-07-24
    </div>
    <details class="paper-abstract">
      Individual teaching is among the most successful ways to impart knowledge. Yet, this method is not always feasible due to large numbers of students per educator. Quantum computing serves as a prime example facing this issue, due to the hype surrounding it. Alleviating high workloads for teachers, often accompanied with individual teaching, is crucial for continuous high quality education. Therefore, leveraging Large Language Models (LLMs) such as GPT-4 to generate educational content can be valuable. We conducted two complementary studies exploring the feasibility of using GPT-4 to automatically generate tips for students. In the first one students (N=46) solved four multiple-choice quantum computing questions with either the help of expert-created or LLM-generated tips. To correct for possible biases towards LLMs, we introduced two additional conditions, making some participants believe that they were given expert-created tips, when they were given LLM-generated tips and vice versa. Our second study (N=23) aimed to directly compare the LLM-generated and expert-created tips, evaluating their quality, correctness and helpfulness, with both experienced educators and students participating. Participants in our second study found that the LLM-generated tips were significantly more helpful and pointed better towards relevant concepts than the expert-created tips, while being more prone to be giving away the answer. While participants in the first study performed significantly better in answering the quantum computing questions when given tips labeled as LLM-generated, even if they were created by an expert. This phenomenon could be a placebo effect induced by the participants' biases for LLM-generated content. Ultimately, we find that LLM-generated tips are good enough to be used instead of expert tips in the context of quantum computing basics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16974v1">SelfPiCo: Self-Guided Partial Code Execution with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-24
      | 💬 Accepted by ISSTA'24
    </div>
    <details class="paper-abstract">
      Code executability plays a vital role in software debugging and testing (e.g., detecting runtime exceptions or assertion violations). However, code execution, especially partial or arbitrary code execution, is a non-trivial task due to missing definitions and complex third-party dependencies. To make partial code (such as code snippets posted on the web or code fragments deep inside complex software projects) executable, the existing study has proposed a machine learning model to predict the undefined element types and inject the pre-defined dummy values into execution. However, the performance of their tool is limited due to its simply designed dummy values and the inability to continue learning. In this paper, we design and implement a novel framework, named SelfPiCo (Self Guided Partial Code Executor), to dynamically guide partial code execution by incorporating the open-source LLM (i.e., Code Llama) within an interactive loop. Particularly, SelfPiCo leverages few-shot in-context learning and chain-of-thought reasoning to elicit human knowledge and logical reasoning based on fine-tuning the Code Llama model. SelfPiCo continuously learns from code execution results and refines its predictions step after step. Our evaluations demonstrate that SelfPiCo can execute 72.7% and 83.3% of all lines in the open-source code and Stack Overflow snippets, outperforming the most recent state-of-the-art Lexecutor by 37.9% and 33.5%, respectively. Moreover, SelfPiCo successfully detected 18 and 33 runtime type error issues by executing the partial code from eight GitHub software projects and 43 Stack Overflow posts, demonstrating the practical usage and potential application of our framework in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02363v2">LLM as Dataset Analyst: Subpopulation Structure Discovery with Large Language Model</a></div>
    <div class="paper-meta">
      📅 2024-07-24
      | 💬 ECCV24 Camera Ready
    </div>
    <details class="paper-abstract">
      The distribution of subpopulations is an important property hidden within a dataset. Uncovering and analyzing the subpopulation distribution within datasets provides a comprehensive understanding of the datasets, standing as a powerful tool beneficial to various downstream tasks, including Dataset Subpopulation Organization, Subpopulation Shift, and Slice Discovery. Despite its importance, there has been no work that systematically explores the subpopulation distribution of datasets to our knowledge. To address the limitation and solve all the mentioned tasks in a unified way, we introduce a novel concept of subpopulation structures to represent, analyze, and utilize subpopulation distributions within datasets. To characterize the structures in an interpretable manner, we propose the Subpopulation Structure Discovery with Large Language Models (SSD-LLM) framework, which employs world knowledge and instruction-following capabilities of Large Language Models (LLMs) to linguistically analyze informative image captions and summarize the structures. Furthermore, we propose complete workflows to address downstream tasks, named Task-specific Tuning, showcasing the application of the discovered structure to a spectrum of subpopulation-related tasks, including dataset subpopulation organization, subpopulation shift, and slice discovery. Furthermore, we propose complete workflows to address downstream tasks, named Task-specific Tuning, showcasing the application of the discovered structure to a spectrum of subpopulation-related tasks, including dataset subpopulation organization, subpopulation shift, and slice discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04418v2">Enabling On-Device LLMs Personalization with Smartphone Sensing</a></div>
    <div class="paper-meta">
      📅 2024-07-24
      | 💬 5 pages, 3 figures, conference demo paper
    </div>
    <details class="paper-abstract">
      This demo presents a novel end-to-end framework that combines on-device large language models (LLMs) with smartphone sensing technologies to achieve context-aware and personalized services. The framework addresses critical limitations of current personalization solutions via cloud LLMs, such as privacy concerns, latency and cost, and limited personal information. To achieve this, we innovatively proposed deploying LLMs on smartphones with multimodal sensor data through context-aware sensing and customized prompt engineering, ensuring privacy and enhancing personalization performance. A case study involving a university student demonstrated the capability of the framework to provide tailored recommendations. In addition, we show that the framework achieves the best trade-off in privacy, performance, latency, cost, battery and energy consumption between on-device and cloud LLMs. To the best of our knowledge, this is the first framework to provide on-device LLMs personalization with smartphone sensing. Future work will incorporate more diverse sensor data and involve extensive user studies to enhance personalization. Our proposed framework has the potential to substantially improve user experiences across domains including healthcare, productivity, and entertainment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07791v2">Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 18 Pages, working in progress
    </div>
    <details class="paper-abstract">
      The rapid adoption of large language models (LLMs) in multi-agent systems has highlighted their impressive capabilities in various applications, such as collaborative problem-solving and autonomous negotiation. However, the security implications of these LLM-based multi-agent systems have not been thoroughly investigated, particularly concerning the spread of manipulated knowledge. In this paper, we investigate this critical issue by constructing a detailed threat model and a comprehensive simulation environment that mirrors real-world multi-agent deployments in a trusted platform. Subsequently, we propose a novel two-stage attack method involving Persuasiveness Injection and Manipulated Knowledge Injection to systematically explore the potential for manipulated knowledge (i.e., counterfactual and toxic knowledge) spread without explicit prompt manipulation. Our method leverages the inherent vulnerabilities of LLMs in handling world knowledge, which can be exploited by attackers to unconsciously spread fabricated information. Through extensive experiments, we demonstrate that our attack method can successfully induce LLM-based agents to spread both counterfactual and toxic knowledge without degrading their foundational capabilities during agent communication. Furthermore, we show that these manipulations can persist through popular retrieval-augmented generation frameworks, where several benign agents store and retrieve manipulated chat histories for future interactions. This persistence indicates that even after the interaction has ended, the benign agents may continue to be influenced by manipulated knowledge. Our findings reveal significant security risks in LLM-based multi-agent systems, emphasizing the imperative need for robust defenses against manipulated knowledge spread, such as introducing ``guardian'' agents and advanced fact-checking tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16805v1">TAMIGO: Empowering Teaching Assistants using LLM-assisted viva and code assessment in an Advanced Computing Class</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly transformed the educational landscape, offering new tools for students, instructors, and teaching assistants. This paper investigates the application of LLMs in assisting teaching assistants (TAs) with viva and code assessments in an advanced computing class on distributed systems in an Indian University. We develop TAMIGO, an LLM-based system for TAs to evaluate programming assignments. For viva assessment, the TAs generated questions using TAMIGO and circulated these questions to the students for answering. The TAs then used TAMIGO to generate feedback on student answers. For code assessment, the TAs selected specific code blocks from student code submissions and fed it to TAMIGO to generate feedback for these code blocks. The TAMIGO-generated feedback for student answers and code blocks was used by the TAs for further evaluation. We evaluate the quality of LLM-generated viva questions, model answers, feedback on viva answers, and feedback on student code submissions. Our results indicate that LLMs are highly effective at generating viva questions when provided with sufficient context and background information. However, the results for LLM-generated feedback on viva answers were mixed; instances of hallucination occasionally reduced the accuracy of feedback. Despite this, the feedback was consistent, constructive, comprehensive, balanced, and did not overwhelm the TAs. Similarly, for code submissions, the LLM-generated feedback was constructive, comprehensive and balanced, though there was room for improvement in aligning the feedback with the instructor-provided rubric for code evaluation. Our findings contribute to understanding the benefits and limitations of integrating LLMs into educational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09636v2">Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference</a></div>
    <div class="paper-meta">
      📅 2024-07-23
    </div>
    <details class="paper-abstract">
      Transformers have emerged as the backbone of large language models (LLMs). However, generation remains inefficient due to the need to store in memory a cache of key-value representations for past tokens, whose size scales linearly with the input sequence length and batch size. As a solution, we propose Dynamic Memory Compression (DMC), a method for online key-value cache compression at inference time. Most importantly, the model learns to apply different compression ratios in different heads and layers. We retrofit pre-trained LLMs such as Llama 2 (7B, 13B and 70B) into DMC Transformers, achieving up to 7x throughput increase during auto-regressive inference on an NVIDIA H100 GPU. DMC is applied via continued pre-training on a negligible percentage of the original data without adding any extra parameters. DMC preserves the original downstream performance with up to 4x cache compression, outperforming up-trained grouped-query attention (GQA) and key-value eviction policies (H$_2$O, TOVA). GQA and DMC can be even combined to obtain compounded gains. Hence, DMC can serve as a drop-in replacement for KV caching in existing LLMs to fit longer contexts and larger batches within any given memory budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16624v1">Semantic Change Characterization with LLMs using Rhetorics</a></div>
    <div class="paper-meta">
      📅 2024-07-23
    </div>
    <details class="paper-abstract">
      Languages continually evolve in response to societal events, resulting in new terms and shifts in meanings. These changes have significant implications for computer applications, including automatic translation and chatbots, making it essential to characterize them accurately. The recent development of LLMs has notably advanced natural language understanding, particularly in sense inference and reasoning. In this paper, we investigate the potential of LLMs in characterizing three types of semantic change: dimension, relation, and orientation. We achieve this by combining LLMs' Chain-of-Thought with rhetorical devices and conducting an experimental assessment of our approach using newly created datasets. Our results highlight the effectiveness of LLMs in capturing and analyzing semantic changes, providing valuable insights to improve computational linguistic applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16604v1">Shared Imagination: LLMs Hallucinate Alike</a></div>
    <div class="paper-meta">
      📅 2024-07-23
    </div>
    <details class="paper-abstract">
      Despite the recent proliferation of large language models (LLMs), their training recipes -- model architecture, pre-training data and optimization algorithm -- are often very similar. This naturally raises the question of the similarity among the resulting models. In this paper, we propose a novel setting, imaginary question answering (IQA), to better understand model similarity. In IQA, we ask one model to generate purely imaginary questions (e.g., on completely made-up concepts in physics) and prompt another model to answer. Surprisingly, despite the total fictionality of these questions, all models can answer each other's questions with remarkable success, suggesting a "shared imagination space" in which these models operate during such hallucinations. We conduct a series of investigations into this phenomenon and discuss implications on model homogeneity, hallucination, and computational creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16576v1">Exploring Automatic Cryptographic API Misuse Detection in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-23
    </div>
    <details class="paper-abstract">
      While the automated detection of cryptographic API misuses has progressed significantly, its precision diminishes for intricate targets due to the reliance on manually defined patterns. Large Language Models (LLMs), renowned for their contextual understanding, offer a promising avenue to address existing shortcomings. However, applying LLMs in this security-critical domain presents challenges, particularly due to the unreliability stemming from LLMs' stochastic nature and the well-known issue of hallucination. To explore the prevalence of LLMs' unreliable analysis and potential solutions, this paper introduces a systematic evaluation framework to assess LLMs in detecting cryptographic misuses, utilizing a comprehensive dataset encompassing both manually-crafted samples and real-world projects. Our in-depth analysis of 11,940 LLM-generated reports highlights that the inherent instabilities in LLMs can lead to over half of the reports being false positives. Nevertheless, we demonstrate how a constrained problem scope, coupled with LLMs' self-correction capability, significantly enhances the reliability of the detection. The optimized approach achieves a remarkable detection rate of nearly 90%, surpassing traditional methods and uncovering previously unknown misuses in established benchmarks. Moreover, we identify the failure patterns that persistently hinder LLMs' reliability, including both cryptographic knowledge deficiency and code semantics misinterpretation. Guided by these insights, we develop an LLM-based workflow to examine open-source repositories, leading to the discovery of 63 real-world cryptographic misuses. Of these, 46 have been acknowledged by the development community, with 23 currently being addressed and 6 resolved. Reflecting on developers' feedback, we offer recommendations for future research and the development of LLM-based security tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16557v1">Patched RTC: evaluating LLMs for diverse software development tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-23
    </div>
    <details class="paper-abstract">
      This paper introduces Patched Round-Trip Correctness (Patched RTC), a novel evaluation technique for Large Language Models (LLMs) applied to diverse software development tasks, particularly focusing on "outer loop" activities such as bug fixing, code review, and documentation updates. Patched RTC extends the original Round-Trip Correctness method to work with any LLM and downstream task, offering a self-evaluating framework that measures consistency and robustness of model responses without human intervention. The study demonstrates a correlation between Patched RTC scores and task-specific accuracy metrics, presenting it as an alternative to the LLM-as-Judge paradigm for open-domain task evaluation. We implement Patched RTC in an open-source framework called patchwork, allowing for transparent evaluation during inference across various patchflows. Experiments comparing GPT-3.5 and GPT-4 models across different software development tasks reveal that Patched RTC effectively distinguishes model performance and task difficulty. The paper also explores the impact of consistency prompts on improving model accuracy, suggesting that Patched RTC can guide prompt refinement and model selection for complex software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14973v3">GenCeption: Evaluate Multimodal LLMs with Unlabeled Unimodal Data</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 Significantly extended from v2. Source code: https://github.com/llcresearch/GenCeption. Leaderboard: https://huggingface.co/spaces/valbuc/GenCeption
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) are typically assessed using expensive annotated multimodal benchmarks, which often lag behind the rapidly evolving demands of MLLM evaluation. This paper outlines and validates GenCeption, a novel, annotation-free evaluation method that requires only unimodal data to measure inter-modality semantic coherence and inversely assesses MLLMs' tendency to hallucinate. This approach eliminates the need for costly data annotation, minimizes the risk of training data contamination, results in slower benchmark saturation, and avoids the illusion of emerging abilities. Inspired by the DrawCeption game, GenCeption begins with a non-textual sample and proceeds through iterative description and generation steps. The semantic drift across iterations is quantified using the GC@T metric. Based on the GenCeption method, we establish the MMECeption benchmark for evaluating Vision LLMs (VLLMs), and compare performance of several popular VLLMs and human annotators. Our empirical results validate GenCeption's effectiveness, demonstrating strong correlations with established VLLM benchmarks. VLLMs still significantly lack behind human performance and struggle especially with text-intensive tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16370v1">Evolutionary Prompt Design for LLM-Based Post-ASR Error Correction</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 in submission
    </div>
    <details class="paper-abstract">
      Building upon the strength of modern large language models (LLMs), generative error correction (GEC) has emerged as a promising paradigm that can elevate the performance of modern automatic speech recognition (ASR) systems. One representative approach is to leverage in-context learning to prompt LLMs so that a better hypothesis can be generated by the LLMs based on a carefully-designed prompt and an $N$-best list of hypotheses produced by ASR systems. However, it is yet unknown whether the existing prompts are the most effective ones for the task of post-ASR error correction. In this context, this paper first explores alternative prompts to identify an initial set of effective prompts, and then proposes to employ an evolutionary prompt optimization algorithm to refine the initial prompts. Evaluations results on the CHiME-4 subset of the Task $1$ of the SLT $2024$ GenSEC challenge show the effectiveness and potential of the proposed algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16329v1">PhenoFlow: A Human-LLM Driven Visual Analytics System for Exploring Large and Complex Stroke Datasets</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 11 pages, 5 figures, paper to appear in IEEE Transactions on Visualization and Computer Graphics (TVCG) (Proc. IEEE VIS 2024)
    </div>
    <details class="paper-abstract">
      Acute stroke demands prompt diagnosis and treatment to achieve optimal patient outcomes. However, the intricate and irregular nature of clinical data associated with acute stroke, particularly blood pressure (BP) measurements, presents substantial obstacles to effective visual analytics and decision-making. Through a year-long collaboration with experienced neurologists, we developed PhenoFlow, a visual analytics system that leverages the collaboration between human and Large Language Models (LLMs) to analyze the extensive and complex data of acute ischemic stroke patients. PhenoFlow pioneers an innovative workflow, where the LLM serves as a data wrangler while neurologists explore and supervise the output using visualizations and natural language interactions. This approach enables neurologists to focus more on decision-making with reduced cognitive load. To protect sensitive patient information, PhenoFlow only utilizes metadata to make inferences and synthesize executable codes, without accessing raw patient data. This ensures that the results are both reproducible and interpretable while maintaining patient privacy. The system incorporates a slice-and-wrap design that employs temporal folding to create an overlaid circular visualization. Combined with a linear bar graph, this design aids in exploring meaningful patterns within irregularly measured BP data. Through case studies, PhenoFlow has demonstrated its capability to support iterative analysis of extensive clinical datasets, reducing cognitive load and enabling neurologists to make well-informed decisions. Grounded in long-term collaboration with domain experts, our research demonstrates the potential of utilizing LLMs to tackle current challenges in data-driven clinical decision-making for acute ischemic stroke patients.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16286v1">A deeper look at depth pruning of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are not only resource-intensive to train but even more costly to deploy in production. Therefore, recent work has attempted to prune blocks of LLMs based on cheap proxies for estimating block importance, effectively removing 10% of blocks in well-trained LLaMa-2 and Mistral 7b models without any significant degradation of downstream metrics. In this paper, we explore different block importance metrics by considering adaptive metrics such as Shapley value in addition to static ones explored in prior work. We show that adaptive metrics exhibit a trade-off in performance between tasks i.e., improvement on one task may degrade performance on the other due to differences in the computed block influences. Furthermore, we extend this analysis from a complete block to individual self-attention and feed-forward layers, highlighting the propensity of the self-attention layers to be more amendable to pruning, even allowing removal of upto 33% of the self-attention layers without incurring any performance degradation on MMLU for Mistral 7b (significant reduction in costly maintenance of KV-cache). Finally, we look at simple performance recovery techniques to emulate the pruned layers by training lightweight additive bias or low-rank linear adapters. Performance recovery using emulated updates avoids performance degradation for the initial blocks (up to 5% absolute improvement on MMLU), which is either competitive or superior to the learning-based technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12022v2">ITERTL: An Iterative Framework for Fine-tuning LLMs for RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 There is some mistakes about the Experimental Setup in Section4.1
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated excellent performance in understanding human instructions and generating code, which has inspired researchers to explore the feasibility of generating RTL code with LLMs. However, the existing approaches to fine-tune LLMs on RTL codes typically are conducted on fixed datasets, which do not fully stimulate the capability of LLMs and require large amounts of reference data. To mitigate these issues , we introduce a simple yet effective iterative training paradigm named ITERTL. During each iteration, samples are drawn from the model trained in the previous cycle. Then these new samples are employed for training in this loop. Through this iterative approach, the distribution mismatch between the model and the training samples is reduced. Additionally, the model is thus enabled to explore a broader generative space and receive more comprehensive feedback. Theoretical analyses are conducted to investigate the mechanism of the effectiveness. Experimental results show the model trained through our proposed approach can compete with and even outperform the state-of-the-art (SOTA) open-source model with nearly 37\% reference samples, achieving remarkable 42.9\% and 62.2\% pass@1 rate on two VerilogEval evaluation datasets respectively. While using the same amount of reference samples, our method can achieved a relative improvement of 16.9\% and 12.5\% in pass@1 compared to the non-iterative method. This study facilitates the application of LLMs for generating RTL code in practical scenarios with limited data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16216v1">A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More</a></div>
    <div class="paper-meta">
      📅 2024-07-23
    </div>
    <details class="paper-abstract">
      With advancements in self-supervised learning, the availability of trillions tokens in a pre-training corpus, instruction fine-tuning, and the development of large Transformers with billions of parameters, large language models (LLMs) are now capable of generating factual and coherent responses to human queries. However, the mixed quality of training data can lead to the generation of undesired responses, presenting a significant challenge. Over the past two years, various methods have been proposed from different perspectives to enhance LLMs, particularly in aligning them with human expectation. Despite these efforts, there has not been a comprehensive survey paper that categorizes and details these approaches. In this work, we aim to address this gap by categorizing these papers into distinct topics and providing detailed explanations of each alignment method, thereby helping readers gain a thorough understanding of the current state of the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11406v3">Don't Go To Extremes: Revealing the Excessive Sensitivity and Calibration Limitations of LLMs in Implicit Hate Speech Detection</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 ACL 2024 Main Conference
    </div>
    <details class="paper-abstract">
      The fairness and trustworthiness of Large Language Models (LLMs) are receiving increasing attention. Implicit hate speech, which employs indirect language to convey hateful intentions, occupies a significant portion of practice. However, the extent to which LLMs effectively address this issue remains insufficiently examined. This paper delves into the capability of LLMs to detect implicit hate speech (Classification Task) and express confidence in their responses (Calibration Task). Our evaluation meticulously considers various prompt patterns and mainstream uncertainty estimation methods. Our findings highlight that LLMs exhibit two extremes: (1) LLMs display excessive sensitivity towards groups or topics that may cause fairness issues, resulting in misclassifying benign statements as hate speech. (2) LLMs' confidence scores for each method excessively concentrate on a fixed range, remaining unchanged regardless of the dataset's complexity. Consequently, the calibration performance is heavily reliant on primary classification accuracy. These discoveries unveil new limitations of LLMs, underscoring the need for caution when optimizing models to ensure they do not veer towards extremes. This serves as a reminder to carefully consider sensitivity and confidence in the pursuit of model fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09136v3">Rescue: Ranking LLM Responses with Partial Ordering to Improve Response Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 ACL 2024 SRW
    </div>
    <details class="paper-abstract">
      Customizing LLMs for a specific task involves separating high-quality responses from lower-quality ones. This skill can be developed using supervised fine-tuning with extensive human preference data. However, obtaining a large volume of expert-annotated data is costly for most tasks. In this paper, we explore a novel method to optimize LLMs using ranking metrics. This method trains the model to prioritize the best responses from a pool of candidates created for a particular task. Rather than a traditional full ordering, we advocate for a partial ordering, as achieving consensus on the perfect order of candidate responses can be challenging. Our partial ordering is more robust, less sensitive to noise, and can be achieved with limited human annotations or through heuristic methods. We test our system's improved response generation ability using benchmark datasets, including textual entailment and multi-document question answering. We conduct ablation studies to understand crucial factors, such as how to gather candidate responses for a specific task, determine their most suitable order, and balance supervised fine-tuning with ranking metrics. Our approach, named Rescue, offers a promising avenue for enhancing the response generation and task accuracy of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16148v1">CHIME: LLM-Assisted Hierarchical Organization of Scientific Studies for Literature Review Support</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 2024 ACL Findings
    </div>
    <details class="paper-abstract">
      Literature review requires researchers to synthesize a large amount of information and is increasingly challenging as the scientific literature expands. In this work, we investigate the potential of LLMs for producing hierarchical organizations of scientific studies to assist researchers with literature review. We define hierarchical organizations as tree structures where nodes refer to topical categories and every node is linked to the studies assigned to that category. Our naive LLM-based pipeline for hierarchy generation from a set of studies produces promising yet imperfect hierarchies, motivating us to collect CHIME, an expert-curated dataset for this task focused on biomedicine. Given the challenging and time-consuming nature of building hierarchies from scratch, we use a human-in-the-loop process in which experts correct errors (both links between categories and study assignment) in LLM-generated hierarchies. CHIME contains 2,174 LLM-generated hierarchies covering 472 topics, and expert-corrected hierarchies for a subset of 100 topics. Expert corrections allow us to quantify LLM performance, and we find that while they are quite good at generating and organizing categories, their assignment of studies to categories could be improved. We attempt to train a corrector model with human feedback which improves study assignment by 12.6 F1 points. We release our dataset and models to encourage research on developing better assistive tools for literature review.
    </details>
</div>
