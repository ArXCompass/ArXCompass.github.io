# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05529v1">PoSSUM: A Protocol for Surveying Social-media Users with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      This paper introduces PoSSUM, an open-source protocol for unobtrusive polling of social-media users via multimodal Large Language Models (LLMs). PoSSUM leverages users' real-time posts, images, and other digital traces to create silicon samples that capture information not present in the LLM's training data. To obtain representative estimates, PoSSUM employs Multilevel Regression and Post-Stratification (MrP) with structured priors to counteract the observable selection biases of social-media platforms. The protocol is validated during the 2024 U.S. Presidential Election, for which five PoSSUM polls were conducted and published on GitHub and X. In the final poll, fielded October 17-26 with a synthetic sample of 1,054 X users, PoSSUM accurately predicted the outcomes in 50 of 51 states and assigned the Republican candidate a win probability of 0.65. Notably, it also exhibited lower state-level bias than most established pollsters. These results demonstrate PoSSUM's potential as a fully automated, unobtrusive alternative to traditional survey methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05507v1">Grammar-Based Code Representation: Is It a Worthy Pursuit for LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Grammar serves as a cornerstone in programming languages and software engineering, providing frameworks to define the syntactic space and program structure. Existing research demonstrates the effectiveness of grammar-based code representations in small-scale models, showing their ability to reduce syntax errors and enhance performance. However, as language models scale to the billion level or beyond, syntax-level errors become rare, making it unclear whether grammar information still provides performance benefits. To explore this, we develop a series of billion-scale GrammarCoder models, incorporating grammar rules in the code generation process. Experiments on HumanEval (+) and MBPP (+) demonstrate a notable improvement in code generation accuracy. Further analysis shows that grammar-based representations enhance LLMs' ability to discern subtle code differences, reducing semantic errors caused by minor variations. These findings suggest that grammar-based code representations remain valuable even in billion-scale models, not only by maintaining syntax correctness but also by improving semantic differentiation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05493v1">Benchmarking LLMs in Recommendation Tasks: A Comparative Evaluation with Conventional Recommenders</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      In recent years, integrating large language models (LLMs) into recommender systems has created new opportunities for improving recommendation quality. However, a comprehensive benchmark is needed to thoroughly evaluate and compare the recommendation capabilities of LLMs with traditional recommender systems. In this paper, we introduce RecBench, which systematically investigates various item representation forms (including unique identifier, text, semantic embedding, and semantic identifier) and evaluates two primary recommendation tasks, i.e., click-through rate prediction (CTR) and sequential recommendation (SeqRec). Our extensive experiments cover up to 17 large models and are conducted across five diverse datasets from fashion, news, video, books, and music domains. Our findings indicate that LLM-based recommenders outperform conventional recommenders, achieving up to a 5% AUC improvement in the CTR scenario and up to a 170% NDCG@10 improvement in the SeqRec scenario. However, these substantial performance gains come at the expense of significantly reduced inference efficiency, rendering the LLM-as-RS paradigm impractical for real-time recommendation environments. We aim for our findings to inspire future research, including recommendation-specific model acceleration methods. We will release our code, data, configurations, and platform to enable other researchers to reproduce and build upon our experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02694v4">MeanCache: User-Centric Semantic Caching for LLM Web Services</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted at 2025 IEEE 39th International Parallel and Distributed Processing Symposium (IPDPS)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) like ChatGPT and Llama have revolutionized natural language processing and search engine dynamics. However, these models incur exceptionally high computational costs. For instance, GPT-3 consists of 175 billion parameters, where inference demands billions of floating-point operations. Caching is a natural solution to reduce LLM inference costs on repeated queries, which constitute about 31% of the total queries. However, existing caching methods are incapable of finding semantic similarities among LLM queries nor do they operate on contextual queries, leading to unacceptable false hit-and-miss rates. This paper introduces MeanCache, a user-centric semantic cache for LLM-based services that identifies semantically similar queries to determine cache hit or miss. Using MeanCache, the response to a user's semantically similar query can be retrieved from a local cache rather than re-querying the LLM, thus reducing costs, service provider load, and environmental impact. MeanCache leverages Federated Learning (FL) to collaboratively train a query similarity model without violating user privacy. By placing a local cache in each user's device and using FL, MeanCache reduces the latency and costs and enhances model performance, resulting in lower false hit rates. MeanCache also encodes context chains for every cached query, offering a simple yet highly effective mechanism to discern contextual query responses from standalone. Our experiments benchmarked against the state-of-the-art caching method, reveal that MeanCache attains an approximately 17% higher F-score and a 20% increase in precision during semantic cache hit-and-miss decisions while performing even better on contextual queries. It also reduces the storage requirement by 83% and accelerates semantic cache hit-and-miss decisions by 11%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01161v2">GazeNoter: Co-Piloted AR Note-Taking via Gaze Selection of LLM Suggestions to Match Users' Intentions</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 22 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Note-taking is critical during speeches and discussions, serving not only for later summarization and organization but also for real-time question and opinion reminding in question-and-answer sessions or timely contributions in discussions. Manually typing on smartphones for note-taking could be distracting and increase cognitive load for users. While large language models (LLMs) are used to automatically generate summaries and highlights, the content generated by artificial intelligence (AI) may not match users' intentions without user input or interaction. Therefore, we propose an AI-copiloted augmented reality (AR) system, GazeNoter, to allow users to swiftly select diverse LLM-generated suggestions via gaze on an AR headset for real-time note-taking. GazeNoter leverages an AR headset as a medium for users to swiftly adjust the LLM output to match their intentions, forming a user-in-the-loop AI system for both within-context and beyond-context notes. We conducted two user studies to verify the usability of GazeNoter in attending speeches in a static sitting condition and walking meetings and discussions in a mobile walking condition, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05449v1">LLM-based Iterative Approach to Metamodeling in Automotive</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      In this paper, we introduce an automated approach to domain-specific metamodel construction relying on Large Language Model (LLM). The main focus is adoption in automotive domain. As outcome, a prototype was implemented as web service using Python programming language, while OpenAI's GPT-4o was used as the underlying LLM. Based on the initial experiments, this approach successfully constructs Ecore metamodel based on set of automotive requirements and visualizes it making use of PlantUML notation, so human experts can provide feedback in order to refine the result. Finally, locally deployable solution is also considered, including the limitations and additional steps required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05445v1">Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05439v1">An Empirical Study of Conformal Prediction in LLM with ASP Scaffolds for Robust Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      In this paper, we examine the use of Conformal Language Modelling (CLM) alongside Answer Set Programming (ASP) to enhance the performance of standard open-weight LLMs on complex multi-step reasoning tasks. Using the StepGame dataset, which requires spatial reasoning, we apply CLM to generate sets of ASP programs from an LLM, providing statistical guarantees on the correctness of the outputs. Experimental results show that CLM significantly outperforms baseline models that use standard sampling methods, achieving substantial accuracy improvements across different levels of reasoning complexity. Additionally, the LLM-as-Judge metric enhances CLM's performance, especially in assessing structurally and logically correct ASP outputs. However, calibrating CLM with diverse calibration sets did not improve generalizability for tasks requiring much longer reasoning steps, indicating limitations in handling more complex tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20576v2">ECCOS: Efficient Capability and Cost Coordinated Scheduling for Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed as service endpoints in systems, the surge in query volume creates significant scheduling challenges. Existing scheduling frameworks mainly target at latency optimization while neglecting the capability of LLMs to serve different level of queries, which could lead to computational resource waste. This paper addresses this challenge by proposing a capability-cost coordinated scheduling framework, ECCOS, for multi-LLM serving, which explicitly constrains response quality and workload to optimize LLM inference cost. Specifically, it introduces the two-stage scheduling by designing a multi-objective predictor and a constrained optimizer. The predictor estimates both model capabilities and computational costs through training-based and retrieval-based approaches, while the optimizer determines cost-optimal assignments under quality and workload constraints. It also introduces QAServe, a dataset collected for sample-wise response quality and costs by zero-shot prompting different LLMs on knowledge QA and mathematical reasoning. Extensive experiments demonstrate that ECCOS improves success rates by 6.30% while reducing costs by 10.15% compared to existing methods, consuming less than 0.5% of LLM response time. The code is available at: https://github.com/agiresearch/ECCOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05394v1">Static Program Analysis Guided LLM Based Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      We describe a novel approach to automating unit test generation for Java methods using large language models (LLMs). Existing LLM-based approaches rely on sample usage(s) of the method to test (focal method) and/or provide the entire class of the focal method as input prompt and context. The former approach is often not viable due to the lack of sample usages, especially for newly written focal methods. The latter approach does not scale well enough; the bigger the complexity of the focal method and larger associated class, the harder it is to produce adequate test code (due to factors such as exceeding the prompt and context lengths of the underlying LLM). We show that augmenting prompts with \emph{concise} and \emph{precise} context information obtained by program analysis %of the focal method increases the effectiveness of generating unit test code through LLMs. We validate our approach on a large commercial Java project and a popular open-source Java project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05371v1">Shifting Perspectives: Steering Vector Ensembles for Robust Bias Mitigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Submitted to ACL 2025
    </div>
    <details class="paper-abstract">
      We present a novel approach to bias mitigation in large language models (LLMs) by applying steering vectors to modify model activations in forward passes. We employ Bayesian optimization to systematically identify effective contrastive pair datasets across nine bias axes. When optimized on the BBQ dataset, our individually tuned steering vectors achieve average improvements of 12.2%, 4.7%, and 3.2% over the baseline for Mistral, Llama, and Qwen, respectively. Building on these promising results, we introduce Steering Vector Ensembles (SVE), a method that averages multiple individually optimized steering vectors, each targeting a specific bias axis such as age, race, or gender. By leveraging their collective strength, SVE outperforms individual steering vectors in both bias reduction and maintaining model performance. The work presents the first systematic investigation of steering vectors for bias mitigation, and we demonstrate that SVE is a powerful and computationally efficient strategy for reducing bias in LLMs, with broader implications for enhancing AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05346v1">AutoIOT: LLM-Driven Automated Natural Language Programming for AIoT Applications</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has profoundly transformed our lives, revolutionizing interactions with AI and lowering the barrier to AI usage. While LLMs are primarily designed for natural language interaction, the extensive embedded knowledge empowers them to comprehend digital sensor data. This capability enables LLMs to engage with the physical world through IoT sensors and actuators, performing a myriad of AIoT tasks. Consequently, this evolution triggers a paradigm shift in conventional AIoT application development, democratizing its accessibility to all by facilitating the design and development of AIoT applications via natural language. However, some limitations need to be addressed to unlock the full potential of LLMs in AIoT application development. First, existing solutions often require transferring raw sensor data to LLM servers, which raises privacy concerns, incurs high query fees, and is limited by token size. Moreover, the reasoning processes of LLMs are opaque to users, making it difficult to verify the robustness and correctness of inference results. This paper introduces AutoIOT, an LLM-based automated program generator for AIoT applications. AutoIOT enables users to specify their requirements using natural language (input) and automatically synthesizes interpretable programs with documentation (output). AutoIOT automates the iterative optimization to enhance the quality of generated code with minimum user involvement. AutoIOT not only makes the execution of AIoT tasks more explainable but also mitigates privacy concerns and reduces token costs with local execution of synthesized programs. Extensive experiments and user studies demonstrate AutoIOT's remarkable capability in program synthesis for various AIoT tasks. The synthesized programs can match and even outperform some representative baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10444v2">LLM-as-BT-Planner: Leveraging LLMs for Behavior Tree Generation in Robot Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 7 pages. Accepted to ICRA 2025
    </div>
    <details class="paper-abstract">
      Robotic assembly tasks remain an open challenge due to their long horizon nature and complex part relations. Behavior trees (BTs) are increasingly used in robot task planning for their modularity and flexibility, but creating them manually can be effort-intensive. Large language models (LLMs) have recently been applied to robotic task planning for generating action sequences, yet their ability to generate BTs has not been fully investigated. To this end, we propose LLM-as-BT-Planner, a novel framework that leverages LLMs for BT generation in robotic assembly task planning. Four in-context learning methods are introduced to utilize the natural language processing and inference capabilities of LLMs for producing task plans in BT format, reducing manual effort while ensuring robustness and comprehensibility. Additionally, we evaluate the performance of fine-tuned smaller LLMs on the same tasks. Experiments in both simulated and real-world settings demonstrate that our framework enhances LLMs' ability to generate BTs, improving success rate through in-context learning and supervised fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00053v3">ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated a remarkable ability to serve as general-purpose tools for various language-based tasks. Recent works have demonstrated that the efficacy of such models can be improved through iterative dialog between multiple models. While these paradigms show promise in improving model efficacy, most works in this area treat collaboration as an emergent behavior, rather than a learned behavior. In doing so, current multi-agent frameworks rely on collaborative behaviors to have been sufficiently trained into off-the-shelf models. To address this limitation, we propose ACC-Collab, an Actor-Critic based learning framework to produce a two-agent team (an actor-agent and a critic-agent) specialized in collaboration. We demonstrate that ACC-Collab outperforms SotA multi-agent techniques on a wide array of benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00153v2">Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Probing learned concepts in large language models (LLMs) is crucial for understanding how semantic knowledge is encoded internally. Training linear classifiers on probing tasks is a principle approach to denote the vector of a certain concept in the representation space. However, the single vector identified for a concept varies with both data and training, making it less robust and weakening its effectiveness in real-world applications. To address this challenge, we propose an approach to approximate the subspace representing a specific concept. Built on linear probing classifiers, we extend the concept vectors into Gaussian Concept Subspace (GCS). We demonstrate GCS's effectiveness through measuring its faithfulness and plausibility across multiple LLMs with different sizes and architectures. Additionally, we use representation intervention tasks to showcase its efficacy in real-world applications such as emotion steering. Experimental results indicate that GCS concept vectors have the potential to balance steering performance and maintaining the fluency in natural language generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09990v2">X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Despite the rapid development of safety alignment techniques for LLMs, defending against multi-turn jailbreaks is still a challenging task. In this paper, we conduct a comprehensive comparison, revealing that some existing defense methods can improve the robustness of LLMs against multi-turn jailbreaks but compromise usability, i.e., reducing general capabilities or causing the over-refusal problem. From the perspective of mechanism interpretability of LLMs, we discover that these methods fail to establish a boundary that exactly distinguishes safe and harmful feature representations. Therefore, boundary-safe representations close to harmful representations are inevitably disrupted, leading to a decline in usability. To address this issue, we propose X-Boundary to push harmful representations away from boundary-safe representations and obtain an exact distinction boundary. In this way, harmful representations can be precisely erased without disrupting safe ones. Experimental results show that X-Boundary achieves state-of-the-art defense performance against multi-turn jailbreaks, while reducing the over-refusal rate by about 20% and maintaining nearly complete general capability. Furthermore, we theoretically prove and empirically verify that X-Boundary can accelerate the convergence process during training. Please see our code at: https://github.com/AI45Lab/X-Boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04554v1">Compositional Translation: A Novel LLM-based Approach for Low-resource Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      The ability of generative large language models (LLMs) to perform in-context learning has given rise to a large body of research into how best to prompt models for various natural language processing tasks. Machine Translation (MT) has been shown to benefit from in-context examples, in particular when they are semantically similar to the sentence to translate. In this paper, we propose a new LLM-based translation paradigm, compositional translation, to replace naive few-shot MT with similarity-based demonstrations. An LLM is used to decompose a sentence into simpler phrases, and then to translate each phrase with the help of retrieved demonstrations. Finally, the LLM is prompted to translate the initial sentence with the help of the self-generated phrase-translation pairs. Our intuition is that this approach should improve translation because these shorter phrases should be intrinsically easier to translate and easier to match with relevant examples. This is especially beneficial in low-resource scenarios, and more generally whenever the selection pool is small or out of domain. We show that compositional translation boosts LLM translation performance on a wide range of popular MT benchmarks, including FLORES 200, NTREX 128 and TICO-19. Code and outputs are available at https://github.com/ArmelRandy/compositional-translation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20984v2">UoR-NCL at SemEval-2025 Task 1: Using Generative LLMs and CLIP Models for Multilingual Multimodal Idiomaticity Representation</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      SemEval-2025 Task 1 focuses on ranking images based on their alignment with a given nominal compound that may carry idiomatic meaning in both English and Brazilian Portuguese. To address this challenge, this work uses generative large language models (LLMs) and multilingual CLIP models to enhance idiomatic compound representations. LLMs generate idiomatic meanings for potentially idiomatic compounds, enriching their semantic interpretation. These meanings are then encoded using multilingual CLIP models, serving as representations for image ranking. Contrastive learning and data augmentation techniques are applied to fine-tune these embeddings for improved performance. Experimental results show that multimodal representations extracted through this method outperformed those based solely on the original nominal compounds. The fine-tuning approach shows promising outcomes but is less effective than using embeddings without fine-tuning. The source code used in this paper is available at https://github.com/tongwu17/SemEval-2025-Task1-UoR-NCL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04474v1">Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted to the ICBINB Workshop at ICLR'25
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) based judges form the underpinnings of key safety evaluation processes such as offline benchmarking, automated red-teaming, and online guardrailing. This widespread requirement raises the crucial question: can we trust the evaluations of these evaluators? In this paper, we highlight two critical challenges that are typically overlooked: (i) evaluations in the wild where factors like prompt sensitivity and distribution shifts can affect performance and (ii) adversarial attacks that target the judge. We highlight the importance of these through a study of commonly used safety judges, showing that small changes such as the style of the model output can lead to jumps of up to 0.24 in the false negative rate on the same dataset, whereas adversarial attacks on the model generation can fool some judges into misclassifying 100% of harmful generations as safe ones. These findings reveal gaps in commonly used meta-evaluation benchmarks and weaknesses in the robustness of current LLM judges, indicating that low attack success under certain judges could create a false sense of security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04463v1">Guiding LLMs to Generate High-Fidelity and High-Quality Counterfactual Explanations for Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      The need for interpretability in deep learning has driven interest in counterfactual explanations, which identify minimal changes to an instance that change a model's prediction. Current counterfactual (CF) generation methods require task-specific fine-tuning and produce low-quality text. Large Language Models (LLMs), though effective for high-quality text generation, struggle with label-flipping counterfactuals (i.e., counterfactuals that change the prediction) without fine-tuning. We introduce two simple classifier-guided approaches to support counterfactual generation by LLMs, eliminating the need for fine-tuning while preserving the strengths of LLMs. Despite their simplicity, our methods outperform state-of-the-art counterfactual generation methods and are effective across different LLMs, highlighting the benefits of guiding counterfactual generation by LLMs with classifier information. We further show that data augmentation by our generated CFs can improve a classifier's robustness. Our analysis reveals a critical issue in counterfactual generation by LLMs: LLMs rely on parametric knowledge rather than faithfully following the classifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04412v1">Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 To appear at ICLR 2025 Workshop on Foundation Models in the Wild
    </div>
    <details class="paper-abstract">
      Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in tasks like coding. In this work, we propose $\textit{Adaptive Branching Monte Carlo Tree Search (AB-MCTS)}$, a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS consistently outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04381v1">TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Codes and models are available at https://github.com/d223302/TRACT
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm uses large language models (LLMs) for automated text evaluation, where a numerical assessment is assigned by an LLM to the input text following scoring rubrics. Existing methods for LLM-as-a-judge use cross-entropy (CE) loss for fine-tuning, which neglects the numeric nature of score prediction. Recent work addresses numerical prediction limitations of LLM fine-tuning through regression-aware fine-tuning, which, however, does not consider chain-of-thought (CoT) reasoning for score prediction. In this paper, we introduce TRACT (Two-stage Regression-Aware fine-tuning with CoT), a method combining CoT reasoning with regression-aware training. TRACT consists of two stages: first, seed LLM is fine-tuned to generate CoTs, which serve as supervision for the second stage fine-tuning. The training objective of TRACT combines the CE loss for learning the CoT reasoning capabilities, and the regression-aware loss for the score prediction. Experiments across four LLM-as-a-judge datasets and two LLMs show that TRACT significantly outperforms existing methods. Extensive ablation studies validate the importance of each component in TRACT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04377v1">How can representation dimension dominate structurally pruned LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 ICLR 2025 Workshop on Sparsity in LLMs (SLLM)
    </div>
    <details class="paper-abstract">
      Pruning assumes a subnetwork exists in the original deep neural network, which can achieve comparative model performance with less computation than the original. However, it is unclear how the model performance varies with the different subnetwork extractions. In this paper, we choose the representation dimension (or embedding dimension, model dimension, the dimension of the residual stream in the relevant literature) as the entry point to this issue. We investigate the linear transformations in the LLM transformer blocks and consider a specific structured pruning approach, SliceGPT, to extract the subnetworks of different representation dimensions. We mechanistically analyse the activation flow during the model forward passes, and find the representation dimension dominates the linear transformations, model predictions, and, finally, the model performance. Explicit analytical relations are given to calculate the pruned model performance (perplexity and accuracy) without actual evaluation, and are empirically validated with Llama-3-8B-Instruct and Phi-3-mini-4k-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04369v1">Lost in Literalism: How Supervised Training Shapes Translationese in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 19 pages;
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in machine translation, demonstrating impressive performance across diverse languages. However, translationese, characterized by overly literal and unnatural translations, remains a persistent challenge in LLM-based translation systems. Despite their pre-training on vast corpora of natural utterances, LLMs exhibit translationese errors and generate unexpected unnatural translations, stemming from biases introduced during supervised fine-tuning (SFT). In this work, we systematically evaluate the prevalence of translationese in LLM-generated translations and investigate its roots during supervised training. We introduce methods to mitigate these biases, including polishing golden references and filtering unnatural training instances. Empirical evaluations demonstrate that these approaches significantly reduce translationese while improving translation naturalness, validated by human evaluations and automatic metrics. Our findings highlight the need for training-aware adjustments to optimize LLM translation outputs, paving the way for more fluent and target-language-consistent translations. We release the data and code at https://github.com/yafuly/LLM_Translationese.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04360v1">Exploring the Multilingual NLG Evaluation Abilities of LLM-Based Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Previous research has shown that LLMs have potential in multilingual NLG evaluation tasks. However, existing research has not fully explored the differences in the evaluation capabilities of LLMs across different languages. To this end, this study provides a comprehensive analysis of the multilingual evaluation performance of 10 recent LLMs, spanning high-resource and low-resource languages through correlation analysis, perturbation attacks, and fine-tuning. We found that 1) excluding the reference answer from the prompt and using large-parameter LLM-based evaluators leads to better performance across various languages; 2) most LLM-based evaluators show a higher correlation with human judgments in high-resource languages than in low-resource languages; 3) in the languages where they are most sensitive to such attacks, they also tend to exhibit the highest correlation with human judgments; and 4) fine-tuning with data from a particular language yields a broadly consistent enhancement in the model's evaluation performance across diverse languages. Our findings highlight the imbalance in LLMs'evaluation capabilities across different languages and suggest that low-resource language scenarios deserve more attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04302v1">Malware Detection at the Edge with Lightweight LLMs: A Performance Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      The rapid evolution of malware attacks calls for the development of innovative detection methods, especially in resource-constrained edge computing. Traditional detection techniques struggle to keep up with modern malware's sophistication and adaptability, prompting a shift towards advanced methodologies like those leveraging Large Language Models (LLMs) for enhanced malware detection. However, deploying LLMs for malware detection directly at edge devices raises several challenges, including ensuring accuracy in constrained environments and addressing edge devices' energy and computational limits. To tackle these challenges, this paper proposes an architecture leveraging lightweight LLMs' strengths while addressing limitations like reduced accuracy and insufficient computational power. To evaluate the effectiveness of the proposed lightweight LLM-based approach for edge computing, we perform an extensive experimental evaluation using several state-of-the-art lightweight LLMs. We test them with several publicly available datasets specifically designed for edge and IoT scenarios and different edge nodes with varying computational power and characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04291v1">MathMistake Checker: A Comprehensive Demonstration for Step-by-Step Math Problem Mistake Finding by Prompt-Guided LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Published in AAAI 2025
    </div>
    <details class="paper-abstract">
      We propose a novel system, MathMistake Checker, designed to automate step-by-step mistake finding in mathematical problems with lengthy answers through a two-stage process. The system aims to simplify grading, increase efficiency, and enhance learning experiences from a pedagogical perspective. It integrates advanced technologies, including computer vision and the chain-of-thought capabilities of the latest large language models (LLMs). Our system supports open-ended grading without reference answers and promotes personalized learning by providing targeted feedback. We demonstrate its effectiveness across various types of math problems, such as calculation and word problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04271v1">On Fact and Frequency: LLM Responses to Misinformation Expressed with Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 4 pages, 1 figure, 3 tables, conference
    </div>
    <details class="paper-abstract">
      We study LLM judgments of misinformation expressed with uncertainty. Our experiments study the response of three widely used LLMs (GPT-4o, LlaMA3, DeepSeek-v2) to misinformation propositions that have been verified false and then are transformed into uncertain statements according to an uncertainty typology. Our results show that after transformation, LLMs change their factchecking classification from false to not-false in 25% of the cases. Analysis reveals that the change cannot be explained by predictors to which humans are expected to be sensitive, i.e., modality, linguistic cues, or argumentation strategy. The exception is doxastic transformations, which use linguistic cue phrases such as "It is believed ...".To gain further insight, we prompt the LLM to make another judgment about the transformed misinformation statements that is not related to truth value. Specifically, we study LLM estimates of the frequency with which people make the uncertain statement. We find a small but significant correlation between judgment of fact and estimation of frequency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04253v1">ADOR: A Design Exploration Framework for LLM Serving with Enhanced Latency and Throughput</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 11pages, 17 figures
    </div>
    <details class="paper-abstract">
      The growing adoption of Large Language Models (LLMs) across various domains has driven the demand for efficient and scalable AI-serving solutions. Deploying LLMs requires optimizations to manage their significant computational and data demands. The prefill stage processes large numbers of input tokens in parallel, increasing computational load, while the decoding stage relies heavily on memory bandwidth due to the auto-regressive nature of LLMs. Current hardware, such as GPUs, often fails to balance these demands, leading to inefficient utilization. While batching improves hardware efficiency, it delays response times, degrading Quality-of-Service (QoS). This disconnect between vendors, who aim to maximize resource efficiency, and users, who prioritize low latency, highlights the need for a better solution. To address this, we propose ADOR, a framework that automatically identifies and recommends hardware architectures tailored to LLM serving. By leveraging predefined architecture templates specialized for heterogeneous dataflows, ADOR optimally balances throughput and latency. It efficiently explores design spaces to suggest architectures that meet the requirements of both vendors and users. ADOR demonstrates substantial performance improvements, achieving 2.51x higher QoS and 4.01x better area efficiency compared to the A100 at high batch sizes, making it a robust solution for scalable and cost-effective LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04241v1">ThrowBench: Benchmarking LLMs by Predicting Runtime Exceptions</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Modern Large Language Models (LLMs) have shown astounding capabilities of code understanding and synthesis. In order to assess such capabilities, several benchmarks have been devised (e.g., HumanEval). However, most benchmarks focus on code synthesis from natural language instructions. Hence, such benchmarks do not test for other forms of code understanding. Moreover, there have been concerns about contamination and leakage. That is, benchmark problems (or closely related problems) may appear in training set, strongly biasing benchmark results. In this work we investigate whether large language models can correctly predict runtime program behavior. To this end, we introduce ThrowBench, a benchmark consisting of over 2,400 short user-written programs written in four different programming languages. The majority of these programs throw an exception during runtime (due to a bug). LLMs are asked to predict whether a presented program throws an exception and, if so, which one. Evaluating our benchmark on six state-of-the-art code LLMs we see modest performance ranging from 19 to 38% (F1 score). Benchmarking a wider set of code capabilities could improve the assessment of code LLMs and help identify weak points in current models. Moreover, as ground-truth answers have been determined through program execution, leakage is not a concern. We release ThrowBench as well as all of our results together with this work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13681v2">An LLM-based Agent for Reliable Docker Environment Configuration</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Environment configuration is a critical yet time-consuming step in software development, especially when dealing with unfamiliar code repositories. While Large Language Models (LLMs) demonstrate the potential to accomplish software engineering tasks, existing methods for environment configuration often rely on manual efforts or fragile scripts, leading to inefficiencies and unreliable outcomes. We introduce Repo2Run, the first LLM-based agent designed to fully automate environment configuration and generate executable Dockerfiles for arbitrary Python repositories. We address two major challenges: (1) enabling the LLM agent to configure environments within isolated Docker containers, and (2) ensuring the successful configuration process is recorded and accurately transferred to a Dockerfile without error. To achieve this, we propose atomic configuration synthesis, featuring a dual-environment architecture (internal and external environment) with a rollback mechanism to prevent environment "pollution" from failed commands, guaranteeing atomic execution (execute fully or not at all) and a Dockerfile generator to transfer successful configuration steps into runnable Dockerfiles. We evaluate Repo2Run~on our proposed benchmark of 420 recent Python repositories with unit tests, where it achieves an 86.0% success rate, outperforming the best baseline by 63.9%. Repo2Run is available at https://github.com/bytedance/Repo2Run.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01145v2">Dual Reasoning: A GNN-LLM Collaborative Framework for Knowledge Graph Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at intuitive, implicit reasoning. Guiding LLMs to construct thought chains can enhance their deliberate reasoning abilities, but also faces challenges such as hallucination. Knowledge Graphs (KGs) can provide explicit structured knowledge for LLMs to alleviate these issues. However, existing KG-enhanced methods often overlook explicit graph learning, making it challenging to efficiently provide precise reasoning chains for LLMs. Following dual-process theory, we propose Dual-Reasoning (DualR), a novel framework that integrates an external system based on Graph Neural Network (GNN) for explicit reasoning on KGs, complementing the implicit reasoning of LLMs through externalized reasoning chains. DualR designs an LLM-empowered GNN module for explicit learning on KGs, efficiently extracting high-quality reasoning chains. These reasoning chains are then refined to a knowledge-enhanced multiple-choice prompt, guiding a frozen LLM to reason thoughtfully for final answer determination. Extensive experiments on three benchmark KGQA datasets demonstrate that DualR achieves state-of-the-art performance while maintaining high efficiency and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14074v2">Investigating Non-Transitivity in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 8 pages, 6 figures, 2 tables (30 pages, 11 figures, 8 tables including references and appendices)
    </div>
    <details class="paper-abstract">
      Automatic evaluation methods based on large language models (LLMs) are emerging as the standard tool for assessing the instruction-following abilities of LLM-based agents. The most common method in this paradigm, pairwise comparisons with a baseline model, critically depends on the assumption of transitive preferences. However, the validity of this assumption remains largely unexplored. In this study, we investigate the presence of non-transitivity within the AlpacaEval framework and analyze its effects on model rankings. We find that LLM judges exhibit non-transitive preferences, leading to rankings that are sensitive to the choice of the baseline model. To mitigate this issue, we show that round-robin tournaments combined with Bradley-Terry models of preference can produce more reliable rankings. Notably, our method increases both the Spearman correlation and the Kendall correlation with Chatbot Arena (95.0% -> 96.4% and 82.1% -> 86.3% respectively). To address the computational cost of round-robin tournaments, we propose Swiss-Wise Iterative Matchmaking (Swim) tournaments, using a dynamic matching strategy to capture the benefits of round-robin tournaments while maintaining computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04130v1">Token-Efficient Long Video Understanding for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Recent advances in video-based multimodal large language models (Video-LLMs) have significantly improved video understanding by processing videos as sequences of image frames. However, many existing methods treat frames independently in the vision backbone, lacking explicit temporal modeling, which limits their ability to capture dynamic patterns and efficiently handle long videos. To address these limitations, we introduce STORM (\textbf{S}patiotemporal \textbf{TO}ken \textbf{R}eduction for \textbf{M}ultimodal LLMs), a novel architecture incorporating a dedicated temporal encoder between the image encoder and the LLM. Our temporal encoder leverages the Mamba State Space Model to integrate temporal information into image tokens, generating enriched representations that preserve inter-frame dynamics across the entire video sequence. This enriched encoding not only enhances video reasoning capabilities but also enables effective token reduction strategies, including test-time sampling and training-based temporal and spatial pooling, substantially reducing computational demands on the LLM without sacrificing key temporal information. By integrating these techniques, our approach simultaneously reduces training and inference latency while improving performance, enabling efficient and robust video understanding over extended temporal contexts. Extensive evaluations show that STORM achieves state-of-the-art results across various long video understanding benchmarks (more than 5\% improvement on MLVU and LongVideoBench) while reducing the computation costs by up to $8\times$ and the decoding latency by 2.4-2.9$\times$ for the fixed numbers of input frames. Project page is available at https://research.nvidia.com/labs/lpr/storm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02398v2">Prompting with Phonemes: Enhancing LLMs' Multilinguality for Non-Latin Script Languages</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted for NAACL 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Although multilingual LLMs have achieved remarkable performance across benchmarks, we find they continue to underperform on non-Latin script languages across contemporary LLM families. This discrepancy arises from the fact that LLMs are pretrained with orthographic scripts, which are dominated by Latin characters that obscure their shared phonology with non-Latin scripts. We propose leveraging phonemic transcriptions as complementary signals to induce script-invariant representations. Our study demonstrates that integrating phonemic signals improves performance across both non-Latin and Latin script languages, with a particularly significant impact on closing the performance gap between the two. Through detailed experiments, we show that phonemic and orthographic scripts retrieve distinct examples for in-context learning (ICL). This motivates our proposed Mixed-ICL retrieval strategy, where further aggregation from both leads to our significant performance improvements for both Latin script languages (up to 12.6%) and non-Latin script languages (up to 15.1%) compared to randomized ICL retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04113v1">Uncovering Gaps in How Humans and LLMs Interpret Subjective Language</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Published at ICLR 2025
    </div>
    <details class="paper-abstract">
      Humans often rely on subjective natural language to direct language models (LLMs); for example, users might instruct the LLM to write an enthusiastic blogpost, while developers might train models to be helpful and harmless using LLM-based edits. The LLM's operational semantics of such subjective phrases -- how it adjusts its behavior when each phrase is included in the prompt -- thus dictates how aligned it is with human intent. In this work, we uncover instances of misalignment between LLMs' actual operational semantics and what humans expect. Our method, TED (thesaurus error detector), first constructs a thesaurus that captures whether two phrases have similar operational semantics according to the LLM. It then elicits failures by unearthing disagreements between this thesaurus and a human-constructed reference. TED routinely produces surprising instances of misalignment; for example, Mistral 7B Instruct produces more harassing outputs when it edits text to be witty, and Llama 3 8B Instruct produces dishonest articles when instructed to make the articles enthusiastic. Our results demonstrate that humans can uncover unexpected LLM behavior by scrutinizing relationships between abstract concepts, without supervising outputs directly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04104v1">LLMs Can Generate a Better Answer by Aggregating Their Own Responses</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities across tasks, yet they often require additional prompting techniques when facing complex problems. While approaches like self-correction and response selection have emerged as popular solutions, recent studies have shown these methods perform poorly when relying on the LLM itself to provide feedback or selection criteria. We argue this limitation stems from the fact that common LLM post-training procedures lack explicit supervision for discriminative judgment tasks. In this paper, we propose Generative Self-Aggregation (GSA), a novel prompting method that improves answer quality without requiring the model's discriminative capabilities. GSA first samples multiple diverse responses from the LLM, then aggregates them to obtain an improved solution. Unlike previous approaches, our method does not require the LLM to correct errors or compare response quality; instead, it leverages the model's generative abilities to synthesize a new response based on the context of multiple samples. While GSA shares similarities with the self-consistency (SC) approach for response aggregation, SC requires specific verifiable tokens to enable majority voting. In contrast, our approach is more general and can be applied to open-ended tasks. Empirical evaluation demonstrates that GSA effectively improves response quality across various tasks, including mathematical reasoning, knowledge-based problems, and open-ended generation tasks such as code synthesis and conversational responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10069v3">A Survey on LLM Test-Time Compute via Search: Tasks, LLM Profiling, Search Algorithms, and Relevant Frameworks</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      LLM test-time compute (or LLM inference) via search has emerged as a promising research area with rapid developments. However, current frameworks often adopt distinct perspectives on three key aspects (task definition, LLM profiling, and search procedures), making direct comparisons challenging. Moreover, the search algorithms employed often diverge from standard implementations, and their specific characteristics are not thoroughly specified. In this survey, we provide a comprehensive technical review that unifies task definitions and provides modular definitions of LLM profiling and search procedures. The definitions enable precise comparisons of various LLM inference frameworks while highlighting their departures from conventional search algorithms. We also discuss the applicability, performance, and efficiency of these methods. We have updated our content to include the latest papers, and the differences between versions are highlighted in the appendix. For further details and ongoing updates, please refer to our GitHub repository: https://github.com/xinzhel/LLM-Agent-Survey/blob/main/search.md
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04099v1">Disparities in LLM Reasoning Accuracy and Explanations: A Case Study on African American English</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 ARR Under Review, First two authors contribute equally
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning tasks, leading to their widespread deployment. However, recent studies have highlighted concerning biases in these models, particularly in their handling of dialectal variations like African American English (AAE). In this work, we systematically investigate dialectal disparities in LLM reasoning tasks. We develop an experimental framework comparing LLM performance given Standard American English (SAE) and AAE prompts, combining LLM-based dialect conversion with established linguistic analyses. We find that LLMs consistently produce less accurate responses and simpler reasoning chains and explanations for AAE inputs compared to equivalent SAE questions, with disparities most pronounced in social science and humanities domains. These findings highlight systematic differences in how LLMs process and reason about different language varieties, raising important questions about the development and deployment of these systems in our multilingual and multidialectal world. Our code repository is publicly available at https://github.com/Runtaozhou/dialect_bias_eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17399v2">MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      We present MultiChallenge, a pioneering benchmark evaluating large language models (LLMs) on conducting multi-turn conversations with human users, a crucial yet underexamined capability for their applications. MultiChallenge identifies four categories of challenges in multi-turn conversations that are not only common and realistic among current human-LLM interactions, but are also challenging to all current frontier LLMs. All 4 challenges require accurate instruction-following, context allocation, and in-context reasoning at the same time. We also develop LLM as judge with instance-level rubrics to facilitate an automatic evaluation method with fair agreement with experienced human raters. Despite achieving near-perfect scores on existing multi-turn evaluation benchmarks, all frontier models have less than 50% accuracy on MultiChallenge, with the top-performing Claude 3.5 Sonnet (June 2024) achieving just a 41.4% average accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04076v1">Beyond Memorization: Evaluating the True Type Inference Capabilities of LLMs for Java Code Snippets</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Type inference is a crucial task for reusing online code snippets, often found on platforms like StackOverflow, which frequently lack essential type information such as fully qualified names (FQNs) and required libraries. Recent studies have leveraged Large Language Models (LLMs) for type inference on code snippets, showing promising results. However, these results are potentially affected by data leakage, as the benchmark suite (StatType-SO) has been public on GitHub since 2017 (full suite in 2023). Thus, it is uncertain whether LLMs' strong performance reflects genuine code semantics understanding or a mere retrieval of ground truth from training data. To comprehensively assess LLMs' type inference capabilities on Java code snippets, we conducted a three-pronged evaluation. First, utilizing Thalia, a program synthesis technique, we created ThaliaType--a new, unseen dataset for type inference evaluation. On unseen snippets, LLM performance dropped significantly, with up to a 59% decrease in precision and 72% in recall. Second, we developed semantic-preserving transformations that significantly degraded LLMs' type inference performance, revealing weaknesses in understanding code semantics. Third, we used delta debugging to identify the minimal syntax elements sufficient for LLM inference. While type inference primarily involves inferring FQNs for types in the code snippet, LLMs correctly infer FQNs even when the types were absent from the snippets, suggesting a reliance on knowledge from training instead of thoroughly analyzing the snippets. Our findings indicate that LLMs' strong past performance likely stemmed from data leakage, rather than a genuine understanding of the semantics of code snippets. Our findings highlight the crucial need for carefully designed benchmarks using unseen code snippets to assess the true capabilities of LLMs for type inference tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03459v2">Unified Mind Model: Reimagining Autonomous Agents in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated remarkable capabilities across domains, tasks, and languages (e.g., ChatGPT and GPT-4), reviving the research of general autonomous agents with human-like cognitive abilities. Such human-level agents require semantic comprehension and instruction-following capabilities, which exactly fall into the strengths of LLMs. Although there have been several initial attempts to build human-level agents based on LLMs, the theoretical foundation remains a challenging open problem. In this paper, we propose a novel theoretical cognitive architecture, the Unified Mind Model (UMM), which offers guidance to facilitate the rapid creation of autonomous agents with human-level cognitive abilities. Specifically, our UMM starts with the global workspace theory and further leverage LLMs to enable the agent with various cognitive abilities, such as multi-modal perception, planning, reasoning, tool use, learning, memory, reflection and motivation. Building upon UMM, we then develop an agent-building engine, MindOS, which allows users to quickly create domain-/task-specific autonomous agents without any programming effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16466v2">On the Feasibility of Using LLMs to Execute Multistage Network Attacks</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 16 pages, 14 figures
    </div>
    <details class="paper-abstract">
      LLMs have shown preliminary promise in some security tasks and CTF challenges. However, it is unclear whether LLMs are able to realize multistage network attacks, which involve executing a wide variety of actions across multiple hosts such as conducting reconnaissance, exploiting vulnerabilities to gain initial access, leveraging internal hosts to move laterally, and using multiple compromised hosts to exfiltrate data. We evaluate LLMs across 10 multistage networks and find that popular LLMs are unable to realize these attacks. To enable LLMs to realize these attacks, we introduce Incalmo, an LLM-agnostic high-level attack abstraction layer that sits between an LLM and the environment. Rather than LLMs issuing low-level command-line instructions, which can lead to incorrect implementations, Incalmo allows LLMs to specify high-level tasks (e.g., infect a host, scan a network), which are then carried out by Incalmo. Incalmo realizes these tasks by translating them into low-level primitives (e.g., commands to exploit tools). Incalmo also provides an environment state service and an attack graph service to provide structure to LLMs in selecting actions relevant to a multistage attack. Across 9 out of 10 realistic emulated networks (from 25 to 50 hosts), LLMs using Incalmo can successfully autonomously execute multistage attacks. We also conduct an ablation analysis to show the key role the high-level abstractions play. For instance, we find that both Incalmo's high-level tasks and services are crucial. Furthermore, even smaller-parameter LLMs with Incalmo can fully succeed in 5 of 10 environments, while larger-parameter LLMs without Incalmo do not fully succeed in any.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06916v2">SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 ICLR 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      Speculative decoding (SD) has emerged as a widely used paradigm to accelerate LLM inference without compromising quality. It works by first employing a compact model to draft multiple tokens efficiently and then using the target LLM to verify them in parallel. While this technique has achieved notable speedups, most existing approaches necessitate either additional parameters or extensive training to construct effective draft models, thereby restricting their applicability across different LLMs and tasks. To address this limitation, we explore a novel plug-and-play SD solution with layer-skipping, which skips intermediate layers of the target LLM as the compact draft model. Our analysis reveals that LLMs exhibit great potential for self-acceleration through layer sparsity and the task-specific nature of this sparsity. Building on these insights, we introduce SWIFT, an on-the-fly self-speculative decoding algorithm that adaptively selects intermediate layers of LLMs to skip during inference. SWIFT does not require auxiliary models or additional training, making it a plug-and-play solution for accelerating LLM inference across diverse input data streams. Our extensive experiments across a wide range of models and downstream tasks demonstrate that SWIFT can achieve over a 1.3x-1.6x speedup while preserving the original distribution of the generated text. We release our code in https://github.com/hemingkx/SWIFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03455v2">Watson: A Cognitive Observability Framework for the Reasoning of LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      As foundation models (FMs) play an increasingly prominent role in complex software systems, such as agentic software, they introduce significant observability and debuggability challenges. Although recent Large Reasoning Models (LRMs) generate their thought processes as part of the output, in many scenarios fast-thinking Large Language Models (LLMs) are still preferred due to latency constraints. LLM-powered agents operate autonomously with opaque implicit reasoning, making it difficult to debug their unexpected behaviors or errors. In this paper, we introduce Watson, a novel framework that provides reasoning observability into the implicit reasoning processes of agents driven by fast-thinking LLMs, allowing the identification and localization of errors and guidance for corrections. We demonstrate the accuracy of the recovered implicit reasoning trace by Watson and its usefulness through debugging and improving the performance of LLM-powered agents in two scenarios: Massive Multitask Language Understanding (MMLU) benchmark and SWE-bench-lite. Using Watson, we were able to observe and identify the implicit reasoning errors, and automatically provide targeted corrections at runtime that improve the Pass@1 of agents on MMLU and SWE-bench-lite by 7.58 (13.45% relative improvement) and 7.76 (12.31% relative improvement) percentage points, respectively, without updates to models or the cognitive architecture of the agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04006v1">DSV-LFS: Unifying LLM-Driven Semantic Cues with Visual Features for Robust Few-Shot Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Few-shot semantic segmentation (FSS) aims to enable models to segment novel/unseen object classes using only a limited number of labeled examples. However, current FSS methods frequently struggle with generalization due to incomplete and biased feature representations, especially when support images do not capture the full appearance variability of the target class. To improve the FSS pipeline, we propose a novel framework that utilizes large language models (LLMs) to adapt general class semantic information to the query image. Furthermore, the framework employs dense pixel-wise matching to identify similarities between query and support images, resulting in enhanced FSS performance. Inspired by reasoning-based segmentation frameworks, our method, named DSV-LFS, introduces an additional token into the LLM vocabulary, allowing a multimodal LLM to generate a "semantic prompt" from class descriptions. In parallel, a dense matching module identifies visual similarities between the query and support images, generating a "visual prompt". These prompts are then jointly employed to guide the prompt-based decoder for accurate segmentation of the query image. Comprehensive experiments on the benchmark datasets Pascal-$5^{i}$ and COCO-$20^{i}$ demonstrate that our framework achieves state-of-the-art performance-by a significant margin-demonstrating superior generalization to novel classes and robustness across diverse scenarios. The source code is available at \href{https://github.com/aminpdik/DSV-LFS}{https://github.com/aminpdik/DSV-LFS}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08177v2">SycEval: Evaluating LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in educational, clinical, and professional settings, but their tendency for sycophancy -- prioritizing user agreement over independent reasoning -- poses risks to reliability. This study introduces a framework to evaluate sycophantic behavior in ChatGPT-4o, Claude-Sonnet, and Gemini-1.5-Pro across AMPS (mathematics) and MedQuad (medical advice) datasets. Sycophantic behavior was observed in 58.19% of cases, with Gemini exhibiting the highest rate (62.47%) and ChatGPT the lowest (56.71%). Progressive sycophancy, leading to correct answers, occurred in 43.52% of cases, while regressive sycophancy, leading to incorrect answers, was observed in 14.66%. Preemptive rebuttals demonstrated significantly higher sycophancy rates than in-context rebuttals (61.75% vs. 56.52%, $Z=5.87$, $p<0.001$), particularly in computational tasks, where regressive sycophancy increased significantly (preemptive: 8.13%, in-context: 3.54%, $p<0.001$). Simple rebuttals maximized progressive sycophancy ($Z=6.59$, $p<0.001$), while citation-based rebuttals exhibited the highest regressive rates ($Z=6.59$, $p<0.001$). Sycophantic behavior showed high persistence (78.5%, 95% CI: [77.2%, 79.8%]) regardless of context or model. These findings emphasize the risks and opportunities of deploying LLMs in structured and dynamic domains, offering insights into prompt programming and model optimization for safer AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16457v2">Towards Fully-Automated Materials Discovery via Large-Scale Synthesis Dataset and Expert-Level LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Materials synthesis is vital for innovations such as energy storage, catalysis, electronics, and biomedical devices. Yet, the process relies heavily on empirical, trial-and-error methods guided by expert intuition. Our work aims to support the materials science community by providing a practical, data-driven resource. We have curated a comprehensive dataset of 17K expert-verified synthesis recipes from open-access literature, which forms the basis of our newly developed benchmark, AlchemyBench. AlchemyBench offers an end-to-end framework that supports research in large language models applied to synthesis prediction. It encompasses key tasks, including raw materials and equipment prediction, synthesis procedure generation, and characterization outcome forecasting. We propose an LLM-as-a-Judge framework that leverages large language models for automated evaluation, demonstrating strong statistical agreement with expert assessments. Overall, our contributions offer a supportive foundation for exploring the capabilities of LLMs in predicting and guiding materials synthesis, ultimately paving the way for more efficient experimental design and accelerated innovation in materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03993v3">TR-LLM: Integrating Trajectory Data for Scene-Aware LLM-Based Human Action Prediction</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Accurate prediction of human behavior is crucial for AI systems to effectively support real-world applications, such as autonomous robots anticipating and assisting with human tasks. Real-world scenarios frequently present challenges such as occlusions and incomplete scene observations, which can compromise predictive accuracy. Thus, traditional video-based methods often struggle due to limited temporal and spatial perspectives. Large Language Models (LLMs) offer a promising alternative. Having been trained on a large text corpus describing human behaviors, LLMs likely encode plausible sequences of human actions in a home environment. However, LLMs, trained primarily on text data, lack inherent spatial awareness and real-time environmental perception. They struggle with understanding physical constraints and spatial geometry. Therefore, to be effective in a real-world spatial scenario, we propose a multimodal prediction framework that enhances LLM-based action prediction by integrating physical constraints derived from human trajectories. Our experiments demonstrate that combining LLM predictions with trajectory data significantly improves overall prediction performance. This enhancement is particularly notable in situations where the LLM receives limited scene information, highlighting the complementary nature of linguistic knowledge and physical constraints in understanding and anticipating human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05050v1">A Unified Framework with Novel Metrics for Evaluating the Effectiveness of XAI Techniques in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 arXiv admin note: substantial text overlap with arXiv:2501.15374
    </div>
    <details class="paper-abstract">
      The increasing complexity of LLMs presents significant challenges to their transparency and interpretability, necessitating the use of eXplainable AI (XAI) techniques to enhance trustworthiness and usability. This study introduces a comprehensive evaluation framework with four novel metrics for assessing the effectiveness of five XAI techniques across five LLMs and two downstream tasks. We apply this framework to evaluate several XAI techniques LIME, SHAP, Integrated Gradients, Layer-wise Relevance Propagation (LRP), and Attention Mechanism Visualization (AMV) using the IMDB Movie Reviews and Tweet Sentiment Extraction datasets. The evaluation focuses on four key metrics: Human-reasoning Agreement (HA), Robustness, Consistency, and Contrastivity. Our results show that LIME consistently achieves high scores across multiple LLMs and evaluation metrics, while AMV demonstrates superior Robustness and near-perfect Consistency. LRP excels in Contrastivity, particularly with more complex models. Our findings provide valuable insights into the strengths and limitations of different XAI methods, offering guidance for developing and selecting appropriate XAI techniques for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05021v1">Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05012v1">LLMs' Reshaping of People, Processes, Products, and Society in Software Development: A Comprehensive Exploration with Early Adopters</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) like OpenAI ChatGPT, Google Gemini, and GitHub Copilot are rapidly gaining traction in the software industry, but their full impact on software engineering remains insufficiently explored. Despite their growing adoption, there is a notable lack of formal, qualitative assessments of how LLMs are applied in real-world software development contexts. To fill this gap, we conducted semi-structured interviews with sixteen early-adopter professional developers to explore their use of LLMs throughout various stages of the software development life cycle. Our investigation examines four dimensions: people - how LLMs affect individual developers and teams; process - how LLMs alter software engineering workflows; product - LLM impact on software quality and innovation; and society - the broader socioeconomic and ethical implications of LLM adoption. Thematic analysis of our data reveals that while LLMs have not fundamentally revolutionized the development process, they have substantially enhanced routine coding tasks, including code generation, refactoring, and debugging. Developers reported the most effective outcomes when providing LLMs with clear, well-defined problem statements, indicating that LLMs excel with decomposed problems and specific requirements. Furthermore, these early-adopters identified that LLMs offer significant value for personal and professional development, aiding in learning new languages and concepts. Early-adopters, highly skilled in software engineering and how LLMs work, identified early and persisting challenges for software engineering, such as inaccuracies in generated content and the need for careful manual review before integrating LLM outputs into production environments. Our study provides a nuanced understanding of how LLMs are shaping the landscape of software development, with their benefits, limitations, and ongoing implications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05010v1">Leveraging Domain Knowledge at Inference Time for LLM Translation: Retrieval versus Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have been increasingly adopted for machine translation (MT), their performance for specialist domains such as medicine and law remains an open challenge. Prior work has shown that LLMs can be domain-adapted at test-time by retrieving targeted few-shot demonstrations or terminologies for inclusion in the prompt. Meanwhile, for general-purpose LLM MT, recent studies have found some success in generating similarly useful domain knowledge from an LLM itself, prior to translation. Our work studies domain-adapted MT with LLMs through a careful prompting setup, finding that demonstrations consistently outperform terminology, and retrieval consistently outperforms generation. We find that generating demonstrations with weaker models can close the gap with larger model's zero-shot performance. Given the effectiveness of demonstrations, we perform detailed analyses to understand their value. We find that domain-specificity is particularly important, and that the popular multi-domain benchmark is testing adaptation to a particular writing style more so than to a specific domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19320v2">Shh, don't say that! Domain Certification in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 10 pages, includes appendix Published in International Conference on Learning Representations (ICLR) 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often deployed to perform constrained tasks, with narrow domains. For example, customer support bots can be built on top of LLMs, relying on their broad language understanding and capabilities to enhance performance. However, these LLMs are adversarially susceptible, potentially generating outputs outside the intended domain. To formalize, assess, and mitigate this risk, we introduce domain certification; a guarantee that accurately characterizes the out-of-domain behavior of language models. We then propose a simple yet effective approach, which we call VALID that provides adversarial bounds as a certificate. Finally, we evaluate our method across a diverse set of datasets, demonstrating that it yields meaningful certificates, which bound the probability of out-of-domain samples tightly with minimum penalty to refusal behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07942v2">Symbiotic Cooperation for Web Agents: Harnessing Complementary Strengths of Large and Small LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Web browsing agents powered by large language models (LLMs) have shown tremendous potential in automating complex web-based tasks. Existing approaches typically rely on large LLMs (e.g., GPT-4o) to explore web environments and generate trajectory data, which is then used either for demonstration retrieval (for large LLMs) or to distill small LLMs (e.g., Llama3) in a process that remains decoupled from the exploration. In this paper, we propose AgentSymbiotic, an iterative framework that couples data synthesis with task-performance, yielding a "symbiotic improvement" for both large and small LLMs. Our study uncovers a complementary dynamic between LLM types: while large LLMs excel at generating high-quality trajectories for distillation, the distilled small LLMs-owing to their distinct reasoning capabilities-often choose actions that diverge from those of their larger counterparts. This divergence drives the exploration of novel trajectories, thereby enriching the synthesized data. However, we also observe that the performance of small LLMs becomes a bottleneck in this iterative enhancement process. To address this, we propose two innovations in LLM distillation: a speculative data synthesis strategy that mitigates off-policy bias, and a multi-task learning approach designed to boost the reasoning capabilities of the student LLM. Furthermore, we introduce a Hybrid Mode for Privacy Preservation to address user privacy concerns. Evaluated on the WEBARENA benchmark, AgentSymbiotic achieves SOTA performance with both LLM types. Our best Large LLM agent reaches 52%, surpassing the previous best of 45%, while our 8B distilled model demonstrates a competitive 49%, exceeding the prior best of 28%. Code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04874v1">Memory Is All You Need: Testing How Model Memory Affects LLM Performance in Annotation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs) have shown promising results in text annotation using zero-shot and few-shot learning. Yet these approaches do not allow the model to retain information from previous annotations, making each response independent from the preceding ones. This raises the question of whether model memory -- the LLM having knowledge about its own previous annotations in the same task -- affects performance. In this article, using OpenAI's GPT-4o and Meta's Llama 3.1 on two political science datasets, we demonstrate that allowing the model to retain information about its own previous classifications yields significant performance improvements: between 5 and 25\% when compared to zero-shot and few-shot learning. Moreover, memory reinforcement, a novel approach we propose that combines model memory and reinforcement learning, yields additional performance gains in three out of our four tests. These findings have important implications for applied researchers looking to improve performance and efficiency in LLM annotation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04856v1">One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Despite extensive safety enhancements in large language models (LLMs), multi-turn "jailbreak" conversations crafted by skilled human adversaries can still breach even the most sophisticated guardrails. However, these multi-turn attacks demand considerable manual effort, limiting their scalability. In this work, we introduce a novel approach called Multi-turn-to-Single-turn (M2S) that systematically converts multi-turn jailbreak prompts into single-turn attacks. Specifically, we propose three conversion strategies - Hyphenize, Numberize, and Pythonize - each preserving sequential context yet packaging it in a single query. Our experiments on the Multi-turn Human Jailbreak (MHJ) dataset show that M2S often increases or maintains high Attack Success Rates (ASRs) compared to original multi-turn conversations. Notably, using a StrongREJECT-based evaluation of harmfulness, M2S achieves up to 95.9% ASR on Mistral-7B and outperforms original multi-turn prompts by as much as 17.5% in absolute improvement on GPT-4o. Further analysis reveals that certain adversarial tactics, when consolidated into a single prompt, exploit structural formatting cues to evade standard policy checks. These findings underscore that single-turn attacks - despite being simpler and cheaper to conduct - can be just as potent, if not more, than their multi-turn counterparts. Our findings underscore the urgent need to reevaluate and reinforce LLM safety strategies, given how adversarial queries can be compacted into a single prompt while still retaining sufficient complexity to bypass existing safety measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04724v1">LLMVoX: Autoregressive Streaming Text-to-Speech Model for Any LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Recent advancements in speech-to-speech dialogue systems leverage LLMs for multimodal interactions, yet they remain hindered by fine-tuning requirements, high computational overhead, and text-speech misalignment. Existing speech-enabled LLMs often degrade conversational quality by modifying the LLM, thereby compromising its linguistic capabilities. In contrast, we propose LLMVoX, a lightweight 30M-parameter, LLM-agnostic, autoregressive streaming TTS system that generates high-quality speech with low latency, while fully preserving the capabilities of the base LLM. Our approach achieves a significantly lower Word Error Rate compared to speech-enabled LLMs, while operating at comparable latency and UTMOS score. By decoupling speech synthesis from LLM processing via a multi-queue token streaming system, LLMVoX supports seamless, infinite-length dialogues. Its plug-and-play design also facilitates extension to various tasks with different backbones. Furthermore, LLMVoX generalizes to new languages with only dataset adaptation, attaining a low Character Error Rate on an Arabic speech task. Additionally, we have integrated LLMVoX with a Vision-Language Model to create an omni-model with speech, text, and vision capabilities, without requiring additional multimodal training. Our code base and project page is available at https://mbzuai-oryx.github.io/LLMVoX .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04723v1">Shifting Long-Context LLMs Research from Input to Output</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advancements in long-context Large Language Models (LLMs) have primarily concentrated on processing extended input contexts, resulting in significant strides in long-context comprehension. However, the equally critical aspect of generating long-form outputs has received comparatively less attention. This paper advocates for a paradigm shift in NLP research toward addressing the challenges of long-output generation. Tasks such as novel writing, long-term planning, and complex reasoning require models to understand extensive contexts and produce coherent, contextually rich, and logically consistent extended text. These demands highlight a critical gap in current LLM capabilities. We underscore the importance of this under-explored domain and call for focused efforts to develop foundational LLMs tailored for generating high-quality, long-form outputs, which hold immense potential for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04722v1">Enough Coin Flips Can Make LLMs Act Bayesian</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit the ability to generalize given few-shot examples in their input prompt, an emergent capability known as in-context learning (ICL). We investigate whether LLMs utilize ICL to perform structured reasoning in ways that are consistent with a Bayesian framework or rely on pattern matching. Using a controlled setting of biased coin flips, we find that: (1) LLMs often possess biased priors, causing initial divergence in zero-shot settings, (2) in-context evidence outweighs explicit bias instructions, (3) LLMs broadly follow Bayesian posterior updates, with deviations primarily due to miscalibrated priors rather than flawed updates, and (4) attention magnitude has negligible effect on Bayesian inference. With sufficient demonstrations of biased coin flips via ICL, LLMs update their priors in a Bayesian manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11807v7">How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted to ICLR 2025; 11 pages of main text; 26 pages of appendices; Included models: GPT-3.5-{0613, 1106, 0125}, GPT-4-0125, GPT-4o-0806, Gemini-{1.0, 1.5)-Pro, LLaMA-3.1-{7, 70, 405}B, Mixtral-8x{7, 22}B, Qwen-2-72B
    </div>
    <details class="paper-abstract">
      Decision-making is a complex process requiring diverse abilities, making it an excellent framework for evaluating Large Language Models (LLMs). Researchers have examined LLMs' decision-making through the lens of Game Theory. However, existing evaluation mainly focus on two-player scenarios where an LLM competes against another. Additionally, previous benchmarks suffer from test set leakage due to their static design. We introduce GAMA($\gamma$)-Bench, a new framework for evaluating LLMs' Gaming Ability in Multi-Agent environments. It includes eight classical game theory scenarios and a dynamic scoring scheme specially designed to quantitatively assess LLMs' performance. $\gamma$-Bench allows flexible game settings and adapts the scoring system to different game parameters, enabling comprehensive evaluation of robustness, generalizability, and strategies for improvement. Our results indicate that GPT-3.5 demonstrates strong robustness but limited generalizability, which can be enhanced using methods like Chain-of-Thought. We also evaluate 13 LLMs from 6 model families, including GPT-3.5, GPT-4, Gemini, LLaMA-3.1, Mixtral, and Qwen-2. Gemini-1.5-Pro outperforms others, scoring of $69.8$ out of $100$, followed by LLaMA-3.1-70B ($65.9$) and Mixtral-8x22B ($62.4$). Our code and experimental results are publicly available at https://github.com/CUHK-ARISE/GAMABench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04693v1">UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to Forgetting Targets</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) inevitably acquire harmful information during training on massive datasets. LLM unlearning aims to eliminate the influence of such harmful information while maintaining the model's overall performance. Existing unlearning methods, represented by gradient ascent-based approaches, primarily focus on forgetting target data while overlooking the crucial impact of logically related knowledge on the effectiveness of unlearning. In this paper, through both theoretical and experimental analyses, we first demonstrate that a key reason for the suboptimal unlearning performance is that models can reconstruct the target content through reasoning with logically related knowledge. To address this issue, we propose Unlearning Improvement via Parameter Extrapolation (UIPE), a method that removes knowledge highly correlated with the forgetting targets. Experimental results show that UIPE significantly enhances the performance of various mainstream LLM unlearning methods on the TOFU benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04691v1">Quantifying the Reasoning Abilities of LLMs on Real-world Clinical Cases</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      The latest reasoning-enhanced large language models (reasoning LLMs), such as DeepSeek-R1 and OpenAI-o3, have demonstrated remarkable success. However, the application of such reasoning enhancements to the highly professional medical domain has not been clearly evaluated, particularly regarding with not only assessing the final generation but also examining the quality of their reasoning processes. In this study, we present MedR-Bench, a reasoning-focused medical evaluation benchmark comprising 1,453 structured patient cases with reasoning references mined from case reports. Our benchmark spans 13 body systems and 10 specialty disorders, encompassing both common and rare diseases. In our evaluation, we introduce a versatile framework consisting of three critical clinical stages: assessment recommendation, diagnostic decision-making, and treatment planning, comprehensively capturing the LLMs' performance across the entire patient journey in healthcare. For metrics, we propose a novel agentic system, Reasoning Evaluator, designed to automate and objectively quantify free-text reasoning responses in a scalable manner from the perspectives of efficiency, factuality, and completeness by dynamically searching and performing cross-referencing checks. As a result, we assess five state-of-the-art reasoning LLMs, including DeepSeek-R1, OpenAI-o3-mini, and others. Our results reveal that current LLMs can handle relatively simple diagnostic tasks with sufficient critical assessment results, achieving accuracy generally over 85%. However, they still struggle with more complex tasks, such as assessment recommendation and treatment planning. In reasoning, their reasoning processes are generally reliable, with factuality scores exceeding 90%, though they often omit critical reasoning steps. Our study clearly reveals further development directions for current clinical LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02800v2">RAAD-LLM: Adaptive Anomaly Detection Using LLMs and RAG Integration</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 arXiv admin note: substantial text overlap with arXiv:2411.00914
    </div>
    <details class="paper-abstract">
      Anomaly detection in complex industrial environments poses unique challenges, particularly in contexts characterized by data sparsity and evolving operational conditions. Predictive maintenance (PdM) in such settings demands methodologies that are adaptive, transferable, and capable of integrating domain-specific knowledge. In this paper, we present RAAD-LLM, a novel framework for adaptive anomaly detection, leveraging large language models (LLMs) integrated with Retrieval-Augmented Generation (RAG). This approach addresses the aforementioned PdM challenges. By effectively utilizing domain-specific knowledge, RAAD-LLM enhances the detection of anomalies in time series data without requiring fine-tuning on specific datasets. The framework's adaptability mechanism enables it to adjust its understanding of normal operating conditions dynamically, thus increasing detection accuracy. We validate this methodology through a real-world application for a plastics manufacturing plant and the Skoltech Anomaly Benchmark (SKAB). Results show significant improvements over our previous model with an accuracy increase from 70.7% to 89.1% on the real-world dataset. By allowing for the enriching of input series data with semantics, RAAD-LLM incorporates multimodal capabilities that facilitate more collaborative decision-making between the model and plant operators. Overall, our findings support RAAD-LLM's ability to revolutionize anomaly detection methodologies in PdM, potentially leading to a paradigm shift in how anomaly detection is implemented across various industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04675v1">LLM-guided Plan and Retrieval: A Strategic Alignment for Interpretable User Satisfaction Estimation in Dialogue</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted by NAACL 2025
    </div>
    <details class="paper-abstract">
      Understanding user satisfaction with conversational systems, known as User Satisfaction Estimation (USE), is essential for assessing dialogue quality and enhancing user experiences. However, existing methods for USE face challenges due to limited understanding of underlying reasons for user dissatisfaction and the high costs of annotating user intentions. To address these challenges, we propose PRAISE (Plan and Retrieval Alignment for Interpretable Satisfaction Estimation), an interpretable framework for effective user satisfaction prediction. PRAISE operates through three key modules. The Strategy Planner develops strategies, which are natural language criteria for classifying user satisfaction. The Feature Retriever then incorporates knowledge on user satisfaction from Large Language Models (LLMs) and retrieves relevance features from utterances. Finally, the Score Analyzer evaluates strategy predictions and classifies user satisfaction. Experimental results demonstrate that PRAISE achieves state-of-the-art performance on three benchmarks for the USE task. Beyond its superior performance, PRAISE offers additional benefits. It enhances interpretability by providing instance-level explanations through effective alignment of utterances with strategies. Moreover, PRAISE operates more efficiently than existing approaches by eliminating the need for LLMs during the inference phase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02067v2">AdaptBot: Combining LLM with Knowledge Graphs and Human Input for Generic-to-Specific Task Decomposition and Knowledge Refinement</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2025
    </div>
    <details class="paper-abstract">
      An embodied agent assisting humans is often asked to complete new tasks, and there may not be sufficient time or labeled examples to train the agent to perform these new tasks. Large Language Models (LLMs) trained on considerable knowledge across many domains can be used to predict a sequence of abstract actions for completing such tasks, although the agent may not be able to execute this sequence due to task-, agent-, or domain-specific constraints. Our framework addresses these challenges by leveraging the generic predictions provided by LLM and the prior domain knowledge encoded in a Knowledge Graph (KG), enabling an agent to quickly adapt to new tasks. The robot also solicits and uses human input as needed to refine its existing knowledge. Based on experimental evaluation in the context of cooking and cleaning tasks in simulation domains, we demonstrate that the interplay between LLM, KG, and human input leads to substantial performance gains compared with just using the LLM. Project website{\S}: https://sssshivvvv.github.io/adaptbot/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00799v6">Get my drift? Catching LLM Task Drift with Activation Deltas</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 SaTML 2025
    </div>
    <details class="paper-abstract">
      LLMs are commonly used in retrieval-augmented applications to execute user instructions based on data from external sources. For example, modern search engines use LLMs to answer queries based on relevant search results; email plugins summarize emails by processing their content through an LLM. However, the potentially untrusted provenance of these data sources can lead to prompt injection attacks, where the LLM is manipulated by natural language instructions embedded in the external data, causing it to deviate from the user's original instruction(s). We define this deviation as task drift. Task drift is a significant concern as it allows attackers to exfiltrate data or influence the LLM's output for other users. We study LLM activations as a solution to detect task drift, showing that activation deltas - the difference in activations before and after processing external data - are strongly correlated with this phenomenon. Through two probing methods, we demonstrate that a simple linear classifier can detect drift with near-perfect ROC AUC on an out-of-distribution test set. We evaluate these methods by making minimal assumptions about how users' tasks, system prompts, and attacks can be phrased. We observe that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Interestingly, the fact that this solution does not require any modifications to the LLM (e.g., fine-tuning), as well as its compatibility with existing meta-prompting solutions, makes it cost-efficient and easy to deploy. To encourage further research on activation-based task inspection, decoding, and interpretability, we release our large-scale TaskTracker toolkit, featuring a dataset of over 500K instances, representations from six SoTA language models, and a suite of inspection tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04636v1">Mark Your LLM: Detecting the Misuse of Open-Source Large Language Models via Watermarking</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted by the 1st Workshop on GenAI Watermarking, collocated with ICLR 2025
    </div>
    <details class="paper-abstract">
      As open-source large language models (LLMs) like Llama3 become more capable, it is crucial to develop watermarking techniques to detect their potential misuse. Existing watermarking methods either add watermarks during LLM inference, which is unsuitable for open-source LLMs, or primarily target classification LLMs rather than recent generative LLMs. Adapting these watermarks to open-source LLMs for misuse detection remains an open challenge. This work defines two misuse scenarios for open-source LLMs: intellectual property (IP) violation and LLM Usage Violation. Then, we explore the application of inference-time watermark distillation and backdoor watermarking in these contexts. We propose comprehensive evaluation methods to assess the impact of various real-world further fine-tuning scenarios on watermarks and the effect of these watermarks on LLM performance. Our experiments reveal that backdoor watermarking could effectively detect IP Violation, while inference-time watermark distillation is applicable in both scenarios but less robust to further fine-tuning and has a more significant impact on LLM performance compared to backdoor watermarking. Exploring more advanced watermarking methods for open-source LLMs to detect their misuse should be an important future direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04619v1">SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 18 pages, 17 figures
    </div>
    <details class="paper-abstract">
      User reviews on e-commerce platforms exhibit dynamic sentiment patterns driven by temporal and contextual factors. Traditional sentiment analysis methods focus on static reviews, failing to capture the evolving temporal relationship between user sentiment rating and textual content. Sentiment analysis on streaming reviews addresses this limitation by modeling and predicting the temporal evolution of user sentiments. However, it suffers from data sparsity, manifesting in temporal, spatial, and combined forms. In this paper, we introduce SynGraph, a novel framework designed to address data sparsity in sentiment analysis on streaming reviews. SynGraph alleviates data sparsity by categorizing users into mid-tail, long-tail, and extreme scenarios and incorporating LLM-augmented enhancements within a dynamic graph-based structure. Experiments on real-world datasets demonstrate its effectiveness in addressing sparsity and improving sentiment modeling in streaming reviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04615v1">HalluCounter: Reference-free LLM Hallucination Detection in the Wild!</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 30 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Response consistency-based, reference-free hallucination detection (RFHD) methods do not depend on internal model states, such as generation probabilities or gradients, which Grey-box models typically rely on but are inaccessible in closed-source LLMs. However, their inability to capture query-response alignment patterns often results in lower detection accuracy. Additionally, the lack of large-scale benchmark datasets spanning diverse domains remains a challenge, as most existing datasets are limited in size and scope. To this end, we propose HalluCounter, a novel reference-free hallucination detection method that utilizes both response-response and query-response consistency and alignment patterns. This enables the training of a classifier that detects hallucinations and provides a confidence score and an optimal response for user queries. Furthermore, we introduce HalluCounterEval, a benchmark dataset comprising both synthetically generated and human-curated samples across multiple domains. Our method outperforms state-of-the-art approaches by a significant margin, achieving over 90\% average confidence in hallucination detection across datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04596v1">The Next Frontier of LLM Applications: Open Ecosystems and Hardware Synergy</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) applications, including LLM app stores and autonomous agents, are shaping the future of AI ecosystems. However, platform silos, fragmented hardware integration, and the absence of standardized interfaces limit scalability, interoperability, and resource efficiency. While LLM app stores democratize AI, their closed ecosystems restrict modular AI reuse and cross-platform portability. Meanwhile, agent-based frameworks offer flexibility but often lack seamless integration across diverse environments. This paper envisions the future of LLM applications and proposes a three-layer decoupled architecture grounded in software engineering principles such as layered system design, service-oriented architectures, and hardware-software co-design. This architecture separates application logic, communication protocols, and hardware execution, enhancing modularity, efficiency, and cross-platform compatibility. Beyond architecture, we highlight key security and privacy challenges for safe, scalable AI deployment and outline research directions in software and security engineering. This vision aims to foster open, secure, and interoperable LLM ecosystems, guiding future advancements in AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05891v2">Affordably Fine-tuned LLMs Provide Better Answers to Course-specific MCQs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 The 40th ACM/SIGAPP Symposium On Applied Computing
    </div>
    <details class="paper-abstract">
      In education, the capability of generating human-like text of Large Language Models (LLMs) inspired work on how they can increase the efficiency of learning and teaching. We study the affordability of these models for educators and students by investigating how LLMs answer multiple-choice questions (MCQs) with respect to hardware constraints and refinement techniques. We explore this space by using generic pre-trained LLMs (the 7B, 13B, and 70B variants of LLaMA-2) to answer 162 undergraduate-level MCQs from a course on Programming Languages (PL) -- the MCQ dataset is a contribution of this work, which we make publicly available. Specifically, we dissect how different factors, such as using readily-available material -- (parts of) the course's textbook -- for fine-tuning and quantisation (to decrease resource usage) can change the accuracy of the responses. The main takeaway is that smaller textbook-based fine-tuned models outperform generic larger ones (whose pre-training requires conspicuous resources), making the usage of LLMs for answering MCQs resource- and material-wise affordable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09977v2">LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs -- No Silver Bullet for LC or RAG Routing</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Effectively incorporating external knowledge into Large Language Models (LLMs) is crucial for enhancing their capabilities and addressing real-world needs. Retrieval-Augmented Generation (RAG) offers an effective method for achieving this by retrieving the most relevant fragments into LLMs. However, the advancements in context window size for LLMs offer an alternative approach, raising the question of whether RAG remains necessary for effectively handling external knowledge. Several existing studies provide inconclusive comparisons between RAG and long-context (LC) LLMs, largely due to limitations in the benchmark designs. In this paper, we present LaRA, a novel benchmark specifically designed to rigorously compare RAG and LC LLMs. LaRA encompasses 2326 test cases across four practical QA task categories and three types of naturally occurring long texts. Through systematic evaluation of seven open-source and four proprietary LLMs, we find that the optimal choice between RAG and LC depends on a complex interplay of factors, including the model's parameter size, long-text capabilities, context length, task type, and the characteristics of the retrieved chunks. Our findings provide actionable guidelines for practitioners to effectively leverage both RAG and LC approaches in developing and deploying LLM applications. Our code and dataset is provided at: \href{https://github.com/Alibaba-NLP/LaRA}{\textbf{https://github.com/Alibaba-NLP/LaRA}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03261v1">Can Frontier LLMs Replace Annotators in Biomedical Text Mining? Analyzing Challenges and Exploring Solutions</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can perform various natural language processing (NLP) tasks through in-context learning without relying on supervised data. However, multiple previous studies have reported suboptimal performance of LLMs in biological text mining. By analyzing failure patterns in these evaluations, we identified three primary challenges for LLMs in biomedical corpora: (1) LLMs fail to learn implicit dataset-specific nuances from supervised data, (2) The common formatting requirements of discriminative tasks limit the reasoning capabilities of LLMs particularly for LLMs that lack test-time compute, and (3) LLMs struggle to adhere to annotation guidelines and match exact schemas, which hinders their ability to understand detailed annotation requirements which is essential in biomedical annotation workflow. To address these challenges, we experimented with prompt engineering techniques targeted to the above issues, and developed a pipeline that dynamically extracts instructions from annotation guidelines. Our findings show that frontier LLMs can approach or surpass the performance of state-of-the-art (SOTA) BERT-based models with minimal reliance on manually annotated data and without fine-tuning. Furthermore, we performed model distillation on a closed-source LLM, demonstrating that a BERT model trained exclusively on synthetic data annotated by LLMs can also achieve a practical performance. Based on these results, we explored the feasibility of partially replacing manual annotation with LLMs in production scenarios for biomedical text mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09601v2">TimeRefine: Temporal Grounding with Time Refining Video LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Video temporal grounding aims to localize relevant temporal boundaries in a video given a textual prompt. Recent work has focused on enabling Video LLMs to perform video temporal grounding via next-token prediction of temporal timestamps. However, accurately localizing timestamps in videos remains challenging for Video LLMs when relying solely on temporal token prediction. Our proposed TimeRefine addresses this challenge in two ways. First, instead of directly predicting the start and end timestamps, we reformulate the temporal grounding task as a temporal refining task: the model first makes rough predictions and then refines them by predicting offsets to the target segment. This refining process is repeated multiple times, through which the model progressively self-improves its temporal localization accuracy. Second, to enhance the model's temporal perception capabilities, we incorporate an auxiliary prediction head that penalizes the model more if a predicted segment deviates further from the ground truth, thus encouraging the model to make closer and more accurate predictions. Our plug-and-play method can be integrated into most LLM-based temporal grounding approaches. The experimental results demonstrate that TimeRefine achieves 3.6% and 5.0% mIoU improvements on the ActivityNet and Charades-STA datasets, respectively. Code and pretrained models will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01711v3">MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 added project repository & dataset URL
    </div>
    <details class="paper-abstract">
      Personalized product search aims to retrieve and rank items that match users' preferences and search intent. Despite their effectiveness, existing approaches typically assume that users' query fully captures their real motivation. However, our analysis of a real-world e-commerce platform reveals that users often engage in relevant consultations before searching, indicating they refine intents through consultations based on motivation and need. The implied motivation in consultations is a key enhancing factor for personalized search. This unexplored area comes with new challenges including aligning contextual motivations with concise queries, bridging the category-text gap, and filtering noise within sequence history. To address these, we propose a Motivation-Aware Personalized Search (MAPS) method. It embeds queries and consultations into a unified semantic space via LLMs, utilizes a Mixture of Attention Experts (MoAE) to prioritize critical semantics, and introduces dual alignment: (1) contrastive learning aligns consultations, reviews, and product features; (2) bidirectional attention integrates motivation-aware embeddings with user preferences. Extensive experiments on real and synthetic data show MAPS outperforms existing methods in both retrieval and ranking tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19622v2">Weaker LLMs' Opinions Also Matter: Mixture of Opinions Enhances LLM's Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 12 pages, 1 figure, 3 tables, 4 prompt/data templates
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have raised interest in their formal reasoning capabilities, particularly in mathematics. While closed LLMs like GPT-4 perform well on mathematical benchmarks, e.g., GSM8K, it remains unclear whether small to medium-sized open LLMs can achieve similar performance, questioning their reliability. To close this gap, we propose a post-training approach leveraging a mixture of opinions (MoO) from weaker ancillary LLMs to enhance a (relatively) stronger LLM's reasoning. For that, each post-training sample is augmented with Chain-of-Thought (CoT) reasoning steps and answers from ancillary LLMs, enabling the main LLM to learn from diverse perspectives. We compare MoO with standard supervised fine-tuning (SFT), few-shot prompting, and the Mixture of Agents (MoA) method on mathematical reasoning benchmarks. Our results show that incorporating weaker LLMs' opinions improves mathematical reasoning by an average of 5%, highlighting the value of diverse perspectives in reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14644v2">zsLLMCode: An Effective Approach for Code Embedding via LLM with Zero-Shot Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has greatly advanced artificial intelligence (AI) in software engineering (SE), with code embeddings playing a critical role in tasks like code-clone detection and code clustering. However, existing methods for code embedding, including those based on LLMs, often depend on costly supervised training or fine-tuning for domain adaptation. This paper proposes a novel zero-shot approach, zsLLMCode, to generate code embeddings by using LLMs and sentence embedding models. This approach attempts to eliminate the need for task-specific training or fine-tuning, and to effectively address the issue of erroneous information commonly found in LLM-generated outputs. We conducted a series of experiments to evaluate the performance of the proposed approach by considering various LLMs and embedding models. The results have demonstrated the effectiveness and superiority of our method zsLLMCode over state-of-the-art unsupervised approaches such as SourcererCC, Code2vec, InferCode, and TransformCode. Our findings highlight the potential of zsLLMCode to advance the field of SE by providing robust and efficient solutions for code embedding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03194v1">Structured Outputs Enable General-Purpose LLMs to be Medical Experts</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Medical question-answering (QA) is a critical task for evaluating how effectively large language models (LLMs) encode clinical knowledge and assessing their potential applications in medicine. Despite showing promise on multiple-choice tests, LLMs frequently struggle with open-ended medical questions, producing responses with dangerous hallucinations or lacking comprehensive coverage of critical aspects. Existing approaches attempt to address these challenges through domain-specific fine-tuning, but this proves resource-intensive and difficult to scale across models. To improve the comprehensiveness and factuality of medical responses, we propose a novel approach utilizing structured medical reasoning. Our method guides LLMs through an seven-step cognitive process inspired by clinical diagnosis, enabling more accurate and complete answers without additional training. Experiments on the MedLFQA benchmark demonstrate that our approach achieves the highest Factuality Score of 85.8, surpassing fine-tuned models. Notably, this improvement transfers to smaller models, highlighting the method's efficiency and scalability. Our code and datasets are available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03180v1">Enhancing Cybersecurity in Critical Infrastructure with LLM-Assisted Explainable IoT Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Ensuring the security of critical infrastructure has become increasingly vital with the proliferation of Internet of Things (IoT) systems. However, the heterogeneous nature of IoT data and the lack of human-comprehensible insights from anomaly detection models remain significant challenges. This paper presents a hybrid framework that combines numerical anomaly detection using Autoencoders with Large Language Models (LLMs) for enhanced preprocessing and interpretability. Two preprocessing approaches are implemented: a traditional method utilizing Principal Component Analysis (PCA) to reduce dimensionality and an LLM-assisted method where GPT-4 dynamically recommends feature selection, transformation, and encoding strategies. Experimental results on the KDDCup99 10% corrected dataset demonstrate that the LLM-assisted preprocessing pipeline significantly improves anomaly detection performance. The macro-average F1 score increased from 0.49 in the traditional PCA-based approach to 0.98 with LLM-driven insights. Additionally, the LLM generates natural language explanations for detected anomalies, providing contextual insights into their causes and implications. This framework highlights the synergy between numerical AI models and LLMs, delivering an accurate, interpretable, and efficient solution for IoT cybersecurity in critical infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07810v5">Transformer Block Coupling and its Correlation with Generalization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 Published as a conference paper at the International Conference on Learning Representations (ICLR 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant strides in natural language processing, and a precise understanding of the internal mechanisms driving their success is essential. In this work, we analyze the trajectories of token embeddings as they pass through transformer blocks, linearizing the system along these trajectories through their Jacobian matrices. By examining the relationships between these block Jacobians, we uncover the phenomenon of \textbf{transformer block coupling} in a multitude of LLMs, characterized by the coupling of their top singular vectors across tokens and depth. Our findings reveal that coupling \textit{positively correlates} with model performance, and that this relationship is stronger than with other hyperparameters such as parameter count, model depth, and embedding dimension. We further investigate how these properties emerge during training, observing a progressive development of coupling, increased linearity, and layer-wise exponential growth in token trajectories. Additionally, experiments with Vision Transformers (ViTs) corroborate the emergence of coupling and its relationship with generalization, reinforcing our findings in LLMs. Collectively, these insights offer a novel perspective on token interactions in transformers, opening new directions for studying their mechanisms as well as improving training and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14739v3">SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in mainstream academic disciplines such as mathematics, physics, and computer science. However, human knowledge encompasses over 200 specialized disciplines, far exceeding the scope of existing benchmarks. The capabilities of LLMs in many of these specialized fields-particularly in light industry, agriculture, and service-oriented disciplines-remain inadequately evaluated. To address this gap, we present SuperGPQA, a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12599v2">Kimi k1.5: Scaling Reinforcement Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17424v4">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding: it asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03108v1">SoK: Knowledge is All You Need: Last Mile Delivery for Automated Provenance-based Intrusion Detection with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Recently, provenance-based intrusion detection systems (PIDSes) have been widely proposed for endpoint threat analysis. However, due to the lack of systematic integration and utilization of knowledge, existing PIDSes still require significant manual intervention for practical deployment, making full automation challenging. This paper presents a disruptive innovation by categorizing PIDSes according to the types of knowledge they utilize. In response to the prevalent issue of ``knowledge silos problem'' in existing research, we introduce a novel knowledge-driven provenance-based intrusion detection framework, powered by large language models (LLMs). We also present OmniSec, a best practice system built upon this framework. By integrating attack representation knowledge, threat intelligence knowledge, and benign behavior knowledge, OmniSec outperforms the state-of-the-art approaches on public benchmark datasets. OmniSec is available online at https://anonymous.4open.science/r/PIDS-with-LLM-613B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03074v1">BEVDriver: Leveraging BEV Maps in LLMs for Robust Closed-Loop Driving</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Autonomous driving has the potential to set the stage for more efficient future mobility, requiring the research domain to establish trust through safe, reliable and transparent driving. Large Language Models (LLMs) possess reasoning capabilities and natural language understanding, presenting the potential to serve as generalized decision-makers for ego-motion planning that can interact with humans and navigate environments designed for human drivers. While this research avenue is promising, current autonomous driving approaches are challenged by combining 3D spatial grounding and the reasoning and language capabilities of LLMs. We introduce BEVDriver, an LLM-based model for end-to-end closed-loop driving in CARLA that utilizes latent BEV features as perception input. BEVDriver includes a BEV encoder to efficiently process multi-view images and 3D LiDAR point clouds. Within a common latent space, the BEV features are propagated through a Q-Former to align with natural language instructions and passed to the LLM that predicts and plans precise future trajectories while considering navigation instructions and critical scenarios. On the LangAuto benchmark, our model reaches up to 18.9% higher performance on the Driving Score compared to SoTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10877v2">Improving Data Efficiency via Curating LLM-Driven Rating Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Instruction tuning is critical for adapting large language models (LLMs) to downstream tasks, and recent studies have demonstrated that small amounts of human-curated data can outperform larger datasets, challenging traditional data scaling laws. While LLM-based data quality rating systems offer a cost-effective alternative to human annotation, they often suffer from inaccuracies and biases, even in powerful models like GPT-4. In this work, we introduce DS2, a Diversity-aware Score curation method for Data Selection. By systematically modeling error patterns through a score transition matrix, DS2 corrects LLM-based scores and promotes diversity in the selected data samples. Our approach shows that a curated subset (just 3.3% of the original dataset) outperforms full-scale datasets (300k samples) across various machine-alignment benchmarks, and matches or surpasses human-aligned datasets such as LIMA with the same sample size (1k samples). These findings challenge conventional data scaling assumptions, highlighting that redundant, low-quality samples can degrade performance and reaffirming that "more can be less."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03969v1">Trim My View: An LLM-Based Code Query System for Module Retrieval in Robotic Firmware</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The software compilation process has a tendency to obscure the original design of the system and makes it difficult both to identify individual components and discern their purpose simply by examining the resulting binary code. Although decompilation techniques attempt to recover higher-level source code from the machine code in question, they are not fully able to restore the semantics of the original functions. Furthermore, binaries are often stripped of metadata, and this makes it challenging to reverse engineer complex binary software. In this paper we show how a combination of binary decomposition techniques, decompilation passes, and LLM-powered function summarization can be used to build an economical engine to identify modules in stripped binaries and associate them with high-level natural language descriptions. We instantiated this technique with three underlying open-source LLMs -- CodeQwen, DeepSeek-Coder and CodeStral -- and measured its effectiveness in identifying modules in robotics firmware. This experimental evaluation involved 467 modules from four devices from the ArduPilot software suite, and showed that CodeStral, the best-performing backend LLM, achieves an average F1-score of 0.68 with an online running time of just a handful of seconds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03967v1">Model Behavior Specification by Leveraging LLM Self-Playing and Self-Improving</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Training AI models is challenging, particularly when crafting behavior instructions. Traditional methods rely on machines (supervised learning) or manual pattern discovery, which results in not interpretable models or time sink. While Large Language Models (LLMs) simplify instruction writing through natural language, articulating intended model behavior still remains difficult. We introduce Visionary Tuning, a human-in-the-loop self-playing followed by automatic self-refinement to improve behavior specification. Our system helps users clarify desired behavior through self-playing and generates prompts through self-improving, Our first evaluation involves user study conducted on a system implementation of Visionary Tuning within the context of chatbot behavior. Our system self-play itself by simulating user interactions to identify patterns and create effective prompts based on the pattern. In a within-subject study (N=12), participants pinpointed more patterns through self-playing and crafted better prompts. Surprisingly, users felt more or less success level in specifying the model behavior. Follow-up crowd studies (N=60) confirmed that the chatbot adhered to instructions without sacrificing quality. Our second evaluation is a case study on a real-world implementation using a movie rating dataset with Visionary Tuning, demonstrating its effectiveness and robustness in modeling a critic's preferences across the spectrum of low to highly rated movies. Together, these results suggest how AI improves the design process of interactive AI systems. Furthermore, they suggest how the benefits of these tools may be non-obvious to end-users. We reflect on these findings and suggest future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14973v4">GenCeption: Evaluate Vision LLMs with Unlabeled Unimodal Data</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 Published by Computer Speech & Language (https://doi.org/10.1016/j.csl.2025.101785). Source code and Leaderboard: https://github.com/llcresearch/GenCeption
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) are typically assessed using expensive annotated multimodal benchmarks, which often lag behind the rapidly evolving demands of MLLM evaluation. This paper outlines and validates GenCeption, a novel, annotation-free evaluation method that requires only unimodal data to measure inter-modality semantic coherence and inversely assesses MLLMs' tendency to hallucinate. This approach eliminates the need for costly data annotation, minimizes the risk of training data contamination, is expected to result in slower benchmark saturation, and avoids the illusion of emerging abilities. Inspired by the DrawCeption game, GenCeption begins with a non-textual sample and proceeds through iterative description and generation steps. The semantic drift across iterations is quantified using the GC@T metric. While GenCeption is principally applicable to MLLMs across various modalities, this paper focuses on its implementation and validation for Vision LLMs (VLLMs). Based on the GenCeption method, we establish the MMECeption benchmark for evaluating VLLMs, and compare the performance of several popular VLLMs and human annotators. Our empirical results validate GenCeption's effectiveness, demonstrating strong correlations with established VLLM benchmarks. VLLMs still significantly lag behind human performance and struggle especially with text-intensive tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03911v1">Safe LLM-Controlled Robots with Formal Guarantees via Reachability Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) in robotic systems presents unique safety challenges, particularly in unpredictable environments. Although LLMs, leveraging zero-shot learning, enhance human-robot interaction and decision-making capabilities, their inherent probabilistic nature and lack of formal guarantees raise significant concerns for safety-critical applications. Traditional model-based verification approaches often rely on precise system models, which are difficult to obtain for real-world robotic systems and may not be fully trusted due to modeling inaccuracies, unmodeled dynamics, or environmental uncertainties. To address these challenges, this paper introduces a safety assurance framework for LLM-controlled robots based on data-driven reachability analysis, a formal verification technique that ensures all possible system trajectories remain within safe operational limits. Our framework specifically investigates the problem of instructing an LLM to navigate the robot to a specified goal and assesses its ability to generate low-level control actions that successfully guide the robot safely toward that goal. By leveraging historical data to construct reachable sets of states for the robot-LLM system, our approach provides rigorous safety guarantees against unsafe behaviors without relying on explicit analytical models. We validate the framework through experimental case studies in autonomous navigation and task planning, demonstrating its effectiveness in mitigating risks associated with LLM-generated commands. This work advances the integration of formal methods into LLM-based robotics, offering a principled and practical approach to ensuring safety in next-generation autonomous systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03889v1">Pretrained LLMs as Real-Time Controllers for Robot Operated Serial Production Line</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 20 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The manufacturing industry is undergoing a transformative shift, driven by cutting-edge technologies like 5G, AI, and cloud computing. Despite these advancements, effective system control, which is crucial for optimizing production efficiency, remains a complex challenge due to the intricate, knowledge-dependent nature of manufacturing processes and the reliance on domain-specific expertise. Conventional control methods often demand heavy customization, considerable computational resources, and lack transparency in decision-making. In this work, we investigate the feasibility of using Large Language Models (LLMs), particularly GPT-4, as a straightforward, adaptable solution for controlling manufacturing systems, specifically, mobile robot scheduling. We introduce an LLM-based control framework to assign mobile robots to different machines in robot assisted serial production lines, evaluating its performance in terms of system throughput. Our proposed framework outperforms traditional scheduling approaches such as First-Come-First-Served (FCFS), Shortest Processing Time (SPT), and Longest Processing Time (LPT). While it achieves performance that is on par with state-of-the-art methods like Multi-Agent Reinforcement Learning (MARL), it offers a distinct advantage by delivering comparable throughput without the need for extensive retraining. These results suggest that the proposed LLM-based solution is well-suited for scenarios where technical expertise, computational resources, and financial investment are limited, while decision transparency and system scalability are critical concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16366v2">A generative approach to LLM harmfulness detection with special red flag tokens</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Most safety training methods for large language models (LLMs) based on fine-tuning rely on dramatically changing the output distribution of the model when faced with a harmful request, shifting it from an unsafe answer to a refusal to respond. These methods inherently compromise model capabilities and might make auto-regressive models vulnerable to attacks that make likely an initial token of affirmative response. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to fine-tune the model to generate this token at any time harmful content is generated or about to be generated. This novel safety training method effectively augments LLMs into generative classifiers of harmfulness at all times during the conversation. This method offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer rather than just the input prompt and provides a stronger defence against sampling-based attacks. In addition, it simplifies the evaluation of the model's robustness and reduces correlated failures when combined with a classifier. We further show an increased robustness to long contexts, and supervised fine-tuning attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09345v3">Rational Tuning of LLM Cascades via Probabilistic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Understanding the reliability of large language models (LLMs) has recently garnered significant attention. Given LLMs' propensity to hallucinate, as well as their high sensitivity to prompt design, it is already challenging to predict the performance of an individual LLM. However, the problem becomes more complex for compound LLM systems such as cascades, where in addition to each model's standalone performance, we must understand how the error rates of different models interact. In this paper, we present a probabilistic model for the joint performance distribution of a sequence of LLMs, which enables a framework for rationally tuning the confidence thresholds of a LLM cascade using continuous optimization. Compared to selecting confidence thresholds using grid search, our parametric Markov-copula model significantly improves runtime scaling with respect to the length of the cascade and the desired resolution of the cost-error curve, turning them from intractable into low-order polynomial. In addition, the optimal thresholds computed using our continuous optimization-based algorithm increasingly outperform those found via grid search as cascade length grows, improving the area under the cost-error curve by 1.9% on average for cascades consisting of at least three models. Overall, our Markov-copula model provides a rational basis for tuning LLM cascade performance and points to the potential of probabilistic methods in analyzing LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04840v1">Framing the Game: How Context Shapes LLM Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed across diverse contexts to support decision-making. While existing evaluations effectively probe latent model capabilities, they often overlook the impact of context framing on perceived rational decision-making. In this study, we introduce a novel evaluation framework that systematically varies evaluation instances across key features and procedurally generates vignettes to create highly varied scenarios. By analyzing decision-making patterns across different contexts with the same underlying game structure, we uncover significant contextual variability in LLM responses. Our findings demonstrate that this variability is largely predictable yet highly sensitive to framing effects. Our results underscore the need for dynamic, context-aware evaluation methodologies for real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04830v1">Cite Before You Speak: Enhancing Context-Response Grounding in E-commerce Conversational LLM-Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      With the advancement of conversational large language models (LLMs), several LLM-based Conversational Shopping Agents (CSA) have been developed to help customers answer questions and smooth their shopping journey in e-commerce domain. The primary objective in building a trustworthy CSA is to ensure the agent's responses are accurate and factually grounded, which is essential for building customer trust and encouraging continuous engagement. However, two challenges remain. First, LLMs produce hallucinated or unsupported claims. Such inaccuracies risk spreading misinformation and diminishing customer trust. Second, without providing knowledge source attribution in CSA response, customers struggle to verify LLM-generated information. To address these challenges, we present an easily productionized solution that enables a "citation experience" utilizing In-context Learning (ICL) and Multi-UX-Inference (MUI) to generate responses with citations to attribute its original sources without interfering other existing UX features. With proper UX design, these citation marks can be linked to the related product information and display the source to our customers. In this work, we also build auto-metrics and scalable benchmarks to holistically evaluate LLM's grounding and attribution capabilities. Our experiments demonstrate that incorporating this citation generation paradigm can substantially enhance the grounding of LLM responses by 13.83% on the real-world data. As such, our solution not only addresses the immediate challenges of LLM grounding issues but also adds transparency to conversational AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04828v1">Beyond Next Word Prediction: Developing Comprehensive Evaluation Frameworks for measuring LLM performance on real world applications</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) are fundamentally next-token prediction systems, their practical applications extend far beyond this basic function. From natural language processing and text generation to conversational assistants and software use, LLMs have numerous use-cases, and have already acquired a significant degree of enterprise adoption. To evaluate such models, static evaluation datasets, consisting of a set of prompts and their corresponding ground truths, are often used to benchmark the efficacy of the model for a particular task. In this paper, we provide the basis for a more comprehensive evaluation framework, based upon a traditional game and tool-based architecture that enables a more overarching measurement of a model's capabilities. For simplicity, we provide a generalized foundation that can be extended, without significant alteration, to numerous scenarios, from specific use cases such as supply chain management or financial reasoning, to abstract measurements such as ethics or safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01048v3">Personalize Your LLM: Fake it then Align it</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Personalizing large language models (LLMs) is essential for delivering tailored interactions that improve user experience. Many existing personalization methods require fine-tuning LLMs for each user, rendering them prohibitively expensive for widespread adoption. Although retrieval-based approaches offer a more compute-efficient alternative, they still depend on large, high-quality datasets that are not consistently available for all users. To address this challenge, we propose CHAMELEON, a scalable and efficient personalization approach that uses (1) self-generated personal preference data and (2) representation editing to enable quick and cost-effective personalization. Our experiments on various tasks, including those from the LaMP personalization benchmark, show that CHAMELEON efficiently adapts models to personal preferences, improving instruction-tuned models and outperforms two personalization baselines by an average of 40% across two model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07132v2">Interactive Data Harmonization with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-05
    </div>
    <details class="paper-abstract">
      Data harmonization is an essential task that entails integrating datasets from diverse sources. Despite years of research in this area, it remains a time-consuming and challenging task due to schema mismatches, varying terminologies, and differences in data collection methodologies. This paper presents the case for agentic data harmonization as a means to both empower experts to harmonize their data and to streamline the process. We introduce Harmonia, a system that combines LLM-based reasoning, an interactive user interface, and a library of data harmonization primitives to automate the synthesis of data harmonization pipelines. We demonstrate Harmonia in a clinical data harmonization scenario, where it helps to interactively create reusable pipelines that map datasets to a standard format. Finally, we discuss challenges and open problems, and suggest research directions for advancing our vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11620v2">Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-03-05
      | 💬 18 pages and 3 References Pages
    </div>
    <details class="paper-abstract">
      In this work, we explore uncertainty estimation as a proxy for correctness in LLM-generated code. To this end, we adapt two state-of-the-art techniques from natural language generation -- one based on entropy and another on mutual information -- to the domain of code generation. Given the distinct semantic properties of code, we introduce modifications, including a semantic equivalence check based on symbolic execution. Our findings indicate a strong correlation between the uncertainty computed through these techniques and correctness, highlighting the potential of uncertainty estimation for quality assessment. Additionally, we propose a simplified version of the entropy-based method that assumes a uniform distribution over the LLM's responses, demonstrating comparable effectiveness. Using these techniques, we develop an abstention policy that prevents the model from making predictions when uncertainty is high, reducing incorrect outputs to near zero. Our evaluation on the LiveCodeBench shows that our approach significantly outperforms a baseline relying solely on LLM-reported log-probabilities.
    </details>
</div>
